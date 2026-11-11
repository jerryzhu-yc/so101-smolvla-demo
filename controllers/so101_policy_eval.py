"""SO-101 SmolVLA 策略推理控制器：加载训练好的 checkpoint 闭环执行分拣任务。

与 so101_pico_teleop.py 的区别：那边是人用 PICO 手柄遥操作 + 录数据；这边完全
由策略网络自己出动作，用来评估训练结果。

链路：
    robot.camera(wrist_cam/top) + robot.obs["qpos"][:6]
        -> lerobot 预处理管线（rename/归一化/tokenize）
        -> SmolVLA 推理，一次得到 50 步 action chunk（未来 1.6s 的关节目标）
        -> 逐拍 set_all，队列空了再推一次

为什么不是每拍都推理：单次推理约 116ms，而控制周期只有 32ms，每拍推会让仿真
一直追赶（内核按墙钟 realtime 推进，控制器慢就补位）。50 步 chunk 摊薄后每拍只
摊 2.3ms，中间各拍纯执行，机械臂动作才连续。

环境自举：内核固定用 sys.executable 拉起控制器，若 GUI 是从没装 lerobot 的
base 环境启动的，这里检测不到 lerobot 就用 os.execv 换成装了 lerobot 的解释器
重新执行本文件。PID 不变，ZMQ 连接与 stdout 管道全部保留，内核无感。

世界文件中通过 robots[].args 覆盖：
    model:       pretrained_model 目录（默认 ~/桌面/trained_models/so101_sort2_20k）
    task:        语言指令，必须与采集时写进数据集的 task 完全一致
    cameras:     "camera1=wrist_cam,camera2=top"，按训练 rename_map 的槽位顺序映射
    max_seconds: 最长墙钟运行秒数（默认 60）
    chunk_steps: 每个 chunk 实际执行步数（默认全部，取小可提高闭环频率）

环境变量：
    SIMOVIA_POLICY_PY  指定带 lerobot 的解释器（默认 smolvla 环境）
"""
from __future__ import annotations

import os
import pathlib
import sys
import time

# ---------------- 解释器自举（必须在 import lerobot 之前） ----------------
try:
    import lerobot  # noqa: F401
except ImportError:
    _py = os.environ.get("SIMOVIA_POLICY_PY") or str(
        pathlib.Path.home() / "miniconda3" / "envs" / "smolvla" / "bin" / "python")
    if not pathlib.Path(_py).exists():
        sys.exit(f"[policy] 找不到带 lerobot 的解释器: {_py}\n"
                 f"[policy] 请设置环境变量 SIMOVIA_POLICY_PY 指向它")
    print(f"[policy] 当前解释器无 lerobot，切换到 {_py}", flush=True)
    os.execv(_py, [_py, str(pathlib.Path(__file__).resolve()), *sys.argv[1:]])

# 权重与 tokenizer 都在本地目录，禁掉联网探测，避免没网时卡在 Hub 请求上
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import torch

from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from simovia.controller import Robot

# ---------------- 默认参数 ----------------
DEFAULT_MODEL = pathlib.Path.home() / "桌面" / "trained_models" / "so101_sort2_20k"
# 必须与 worlds/so101_pico_sort.yaml 里 args.task 一字不差，否则语言条件对不上
DEFAULT_TASK = ("pick up the red cube and put it into the red box, "
                "then pick up the blue cube and put it into the blue box")
DEFAULT_CAMERAS = "camera1=wrist_cam,camera2=top"
CAM_WH = (480, 640)      # 与采集时的 cam_height/cam_width 一致
# 采集起始姿态：遥操作每次开录前都 ramp 到这里（工具朝下悬在红块上方、
# 夹爪张开），训练数据第一帧全是这个构型。评估必须复刻同样的起步，
# 否则策略拿到分布外状态，且从零位过去的路径可能扫翻方块。
PRE_Q5 = np.array([0.21400, -0.16915, 0.11906, 1.62081, 0.03055])
GRIP_OPEN = 0.90         # 与遥操作 GRIP_OPEN 一致
PRE_SPEED = 1.0          # ramp 到预备姿态的关节角速度 rad/s
GRIP_SPEED = 2.0         # 夹爪 ctrl 变化率 /s
PRE_TOL = 0.03           # 到位容差 rad（与遥操作相同）


def parse_cameras(spec: str) -> dict[str, str]:
    """'camera1=wrist_cam,camera2=top' -> {'camera1': 'wrist_cam', ...}"""
    out: dict[str, str] = {}
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"cameras 项缺少 '=': {item!r}（形如 camera1=wrist_cam）")
        slot, name = (s.strip() for s in item.split("=", 1))
        out[slot] = name
    return out


def to_chw(img: np.ndarray) -> torch.Tensor:
    """HWC uint8 -> CHW uint8（lerobot 的图像约定）"""
    return torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1)))


def main() -> None:
    robot = Robot()
    args = dict(robot.args or {})

    model_dir = pathlib.Path(args.get("model", DEFAULT_MODEL)).expanduser()
    task_text = str(args.get("task", DEFAULT_TASK))
    cam_map = parse_cameras(str(args.get("cameras", DEFAULT_CAMERAS)))
    max_seconds = float(args.get("max_seconds", 60.0))
    chunk_steps = int(args.get("chunk_steps", 0))     # 0 = 用满整个 chunk

    if not (model_dir / "config.json").exists():
        raise RuntimeError(f"模型目录不完整（缺 config.json）: {model_dir}")
    if len(robot.actuators) != 6:
        raise RuntimeError(f"该控制器需要 6 个执行器（5 臂 + 夹爪），"
                           f"实际 {len(robot.actuators)}: {robot.actuators}")

    # 关节限位：模型可能输出越界目标，写进控制器前先夹住，免得位置伺服顶死
    lo = np.array([r[0] for r in robot.ranges], dtype=np.float64)
    hi = np.array([r[1] for r in robot.ranges], dtype=np.float64)

    t0 = time.monotonic()
    print(f"[policy] 加载模型: {model_dir}", flush=True)
    policy = SmolVLAPolicy.from_pretrained(model_dir)
    policy.to("cuda").eval()
    pre, post = make_pre_post_processors(policy.config, pretrained_path=model_dir)
    n_action = policy.config.n_action_steps
    print(f"[policy] 模型就绪（{time.monotonic() - t0:.1f}s），"
          f"chunk={policy.config.chunk_size} n_action_steps={n_action}", flush=True)

    # 初始观测：只读不推进，拿到零位作为 ramp 起点
    obs = robot.observe()
    q0 = np.asarray(obs["qpos"][:6], dtype=np.float64)
    ramp_q = q0[:5].copy()
    ramp_grip = float(q0[5])
    armed = False
    dt = robot.timestep * robot.ctrl_period

    def make_batch(qv: np.ndarray, imgs: dict[str, np.ndarray]) -> dict:
        batch = {
            "observation.state": torch.tensor(qv, dtype=torch.float32),
            "task": task_text,
        }
        # 直接给采集时的相机名，预处理管线的 rename step 会映射到 camera1/2
        for slot, name in cam_map.items():
            _ = slot                      # 槽位顺序由 rename step 决定，这里只取图像
            batch[f"observation.images.{name}"] = to_chw(imgs[name])
        return pre(batch)

    images = {name: robot.camera(name, width=CAM_WH[1], height=CAM_WH[0])
              for name in cam_map.values()}
    t_warm = time.monotonic()
    with torch.no_grad():
        policy.predict_action_chunk(make_batch(q0, images))   # 首次含 CUDA 预热
    print(f"[policy] 预热完成（{time.monotonic() - t_warm:.2f}s）", flush=True)

    queue: list[np.ndarray] = []
    n_infer = 0
    t_start = None
    print(f"[policy] 开始执行：task={task_text!r}", flush=True)

    while robot.step():
        q = np.asarray(robot.obs["qpos"][:6], dtype=np.float64)

        # ---- 阶段 1：ramp 到采集时的预备姿态（与遥操作开录前完全一致）。
        # 训练数据第一帧全是 PRE_Q5，策略只认这个起步；从零位直接推理
        # 属于分布外输入，且移动路径可能扫到方块。
        if not armed:
            step_j = PRE_SPEED * dt
            ramp_q = ramp_q + np.clip(PRE_Q5 - ramp_q, -step_j, step_j)
            ramp_grip += float(np.clip(GRIP_OPEN - ramp_grip,
                                       -GRIP_SPEED * dt, GRIP_SPEED * dt))
            robot.set_all([float(v) for v in ramp_q] + [ramp_grip])
            err = float(np.max(np.abs(q[:5] - PRE_Q5)))
            if err < PRE_TOL:
                armed = True
                t_start = time.monotonic()
                print("[policy] 预备姿态到位（与采集起始一致），策略接管",
                      flush=True)
            continue

        if not queue:
            images = {name: robot.camera(name, width=CAM_WH[1], height=CAM_WH[0])
                      for name in cam_map.values()}
            # 转发到 GUI 相机预览：核对策略实际看到的画面是否与采集一致
            try:
                for name, img in images.items():
                    robot.publish_camera(name, img)
            except Exception:
                pass            # 预览失败不影响推理
            batch = make_batch(q, images)
            t_inf = time.monotonic()
            with torch.no_grad():
                chunk = policy.predict_action_chunk(batch)   # (1, T, 6)
                chunk = post(chunk)
            dt_inf = time.monotonic() - t_inf
            n_infer += 1
            steps = chunk[0].float().cpu().numpy()
            if chunk_steps > 0:
                steps = steps[:chunk_steps]
            queue = list(steps)
            print(f"[policy] 推理 #{n_infer} {dt_inf*1000:.0f}ms  "
                  f"t={robot.time:.2f}s  首步={np.round(queue[0], 3)}", flush=True)

        a = np.clip(queue.pop(0), lo, hi)
        robot.set_all(a.tolist())

        if time.monotonic() - t_start > max_seconds:
            print(f"[policy] 到达运行上限 {max_seconds:.0f}s，停止", flush=True)
            robot.report_result(False, f"timeout after {max_seconds:.0f}s")
            break

    robot.close()


if __name__ == "__main__":
    main()
