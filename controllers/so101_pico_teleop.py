"""SO-101 PICO 遥操作控制器：VR 手柄 -> OSC 操作空间控制 -> 关节目标 + LeRobot 数据集录制。

链路（PICO 头显用自带浏览器打开本页即可，无需装任何 APP）：

    PICO 浏览器 --WebSocket(90Hz)--> 本控制器（内嵌 HTTP+WS 服务器）
        右手柄位置 -> TCP 位置目标（增量映射：按位移增量推动，B 键重新对中）
        扳机      -> 夹爪闭合（0=张开 1=夹紧）
        A 键      -> 开始/停止录制（写 LeRobot v2.1 / SmolVLA 训练格式），
                     停止时自动复位：物体搬回初始位姿 + 机械臂回预备姿态
        B 键      -> 重新对中（机械臂原地不动，手柄当前位置成为新的位移原点）
    本控制器每控制 tick 跑一次 OSC（操作空间控制）：任务空间速度经
    任务空间惯量分配到关节速度 -> robot.set_all(6 关节目标)

坐标映射（VR 右手系 y-up -> MuJoCo z-up）：v_mj = [-v_z, -v_x, v_y]，
T = [[0,0,-1],[-1,0,0],[0,1,0]]（真旋转，det=+1）。

世界文件里可通过 robots[].args 覆盖：
    dataset: 数据集根目录（默认 ~/桌面/so101_smolvla_dataset）
    task:    任务描述文本（语言条件）
    port:    HTTP/WS 端口（默认 8090）
    cameras: 逗号分隔的多路相机名（默认 wrist_cam；可写 wrist_cam,top 录两路）
"""
from __future__ import annotations

import asyncio
import json
import os
import pathlib
import socket
import threading
import time

import mujoco
import numpy as np

from simovia.controller import Robot
from simovia.lerobot_writer import LerobotWriter

# ---------------- 世界 YAML 可覆盖的默认参数 ----------------
REPO = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_DATASET = pathlib.Path.home() / "桌面" / "so101_smolvla_dataset"
DEFAULT_TASK = "pick the red cube and place it into the red box"
DEFAULT_PORT = 8090

# ---------------- 映射参数 ----------------
# SO-101 保持"夹爪朝下"时可用工作域只有约 12cm(x) × 28cm(y) × 8cm(z)，
# 之前 0.8 的缩放让手柄走 8cm 就把目标推出工作域，于是整场都在报
# "目标不可达"。0.4 让舒服的手部行程刚好扫完整个域。
SCALE = 0.4                # 手柄位移 -> TCP 位移 缩放
HOME_WORLD = np.array([0.20, -0.04, 0.22])   # 无手柄时的目标（世界系）
# 目标 clamp（模型局部系 = 世界 - 基座 pos）；按实测可达域收紧，
# 避免把目标推到"域内但姿态不可达"的角落（那会让 OSC 顶住限位、来回抖）
TGT_X = (0.15, 0.27)
TGT_Y = (-0.14, 0.14)
TGT_Z = (0.015, 0.095)
GRIP_OPEN, GRIP_CLOSE = 0.90, -0.15          # 扳机 0/1 对应的夹爪 ctrl
GRIP_SPEED = 4.0            # 夹爪 ctrl 变化率上限（/s）
CTRL_TIMEOUT = 0.6          # 手柄数据超过此秒数视为失联（冻结目标）

# ---------------- OSC（操作空间控制）参数 ----------------
# 不用逐点 IK：每个 tick 在任务空间算期望速度，用任务空间惯量
# Λ=(J M⁻¹ Jᵀ)⁻¹ 分配到关节速度，再积分成位置伺服目标。目标超出可达域时
# 表现为平滑减速停边界，而不是关节锁死 + 几十毫米残差。
OSC_KP_POS = 16.0           # 位置误差 -> 任务速度（1/s）
OSC_VMAX = 0.50             # 任务空间线速度上限（m/s）
OSC_LAMBDA_REL = 0.02       # 任务空间阻尼，相对 J 最大奇异值（抗奇异位形抖动）
OSC_KP_POSTURE = 1.2        # 零空间姿态保持增益（趋向预备构型）
OSC_KD_POSTURE = 0.35       # 零空间速度阻尼（抑制边界处来回抖）
MAX_JOINT_SPEED = 1.5       # 关节速度上限（rad/s）
# 目标不可达时的"回吸"速率：残差超过该阈值就把目标朝当前 TCP 拉，避免臂
# 一直硬顶限位（硬顶会让冗余解在两个 IK 分支间每 tick 翻转，表现为抽搐）
BACKOFF_ERR = 0.025
BACKOFF_GAIN = 2.5
# 预备姿态（工具轴朝下悬在方块上方，来自标定过的可达构型）
PRE_Q5 = np.array([0.21400, -0.16915, 0.11906, 1.62081, 0.03055])
PRE_SPEED = 1.0             # 预备段关节角速度（rad/s）

_TRACE = bool(os.environ.get("SIMOVIA_PICO_TRACE"))   # 调参用：打印目标/实测
# VR 右手系（x 右、y 上、z 朝用户）-> MuJoCo（x 前、y 左、z 上）：
# VR -z(前)->mj x(前)，VR x(右)->mj -y(右)，VR y(上)->mj z(上)；
# v_mj = [-v_z, -v_x, v_y]，T 为真旋转（det=+1）。
_T = np.array([[0.0, 0.0, -1.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])


def vr_to_mj(v: np.ndarray) -> np.ndarray:
    """VR 坐标 -> MuJoCo 坐标（向量）。"""
    return _T @ v


def quat_axes_ok() -> None:
    assert abs(np.linalg.det(_T) - 1.0) < 1e-9


class HandTracker:
    """最新手柄状态（WS 线程写，控制循环读，GIL 下小字典安全）。

    多个 WS 连接时只认"最新连上且在传数据"的那一路（owner）：头显浏览器
    刷新/重连会留下旧连接，两路数据交替喂进来会让增量目标来回跳。
    owner 变更时 generation 自增，控制循环据此重置增量基准，避免一次大跳。
    """

    STALE = 1.0        # owner 静默超过此秒数后允许别的连接接管

    def __init__(self):
        self.latest: dict | None = None
        self.anchor_pos: np.ndarray | None = None   # mj 系
        self.connected = False
        self.owner = None
        self.generation = 0
        self.press_a = 0            # 待消费的 A 键按下沿计数
        self.press_b = 0
        self._prev_a = False
        self._prev_b = False
        self._last: dict = {}       # 连接 -> 最近一次收包时间
        # 最近一次收到手柄数据的本机时间。不能用消息里的 t：那是浏览器
        # performance.now()（页面加载后秒数），和本进程 monotonic 不同源，
        # 相减恒为"超大值"，fresh 永远 False -> 手柄完全无响应。
        self.recv = 0.0

    def claim(self, cid) -> None:
        """新连接申请接管（最新连上的那一路优先）。"""
        if self.owner != cid:
            self.owner = cid
            self._last[cid] = time.monotonic()   # 刚连上视为"在线"，防止被旧连接抢回
            self.generation += 1

    def release(self, cid) -> None:
        """连接断开：owner 走了就交给最近还在传数据的那一路。"""
        self._last.pop(cid, None)
        if self.owner != cid:
            return
        self.owner = None
        if self._last:
            live = max(self._last, key=self._last.get)
            self.owner = live
            self.generation += 1
        else:
            self.latest = None      # 全断：冻结目标
            self.connected = False

    def feed(self, msg: dict, cid) -> None:
        self._last[cid] = time.monotonic()
        if self.owner is not None and cid != self.owner:
            # 非 owner 只在 owner 静默过久时才允许抢占
            if self._last.get(self.owner, 0.0) > self._last[cid] - self.STALE:
                return
            self.owner = cid
            self.generation += 1
        self.latest = msg
        self.recv = time.monotonic()
        self.connected = True
        if self.anchor_pos is None:
            self.recalibrate()
        # 按键按下沿在这里锁存：控制循环 64ms 才采样一次，短促按键不能漏
        a, b = bool(msg.get("a")), bool(msg.get("b"))
        if a and not self._prev_a:
            self.press_a += 1
        if b and not self._prev_b:
            self.press_b += 1
        self._prev_a, self._prev_b = a, b

    def recalibrate(self) -> None:
        m = self.latest
        if m is None:
            return
        self.anchor_pos = vr_to_mj(np.asarray(m["p"], dtype=float))


def _local_ips() -> list[str]:
    ips = set()
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ips.add(s.getsockname()[0])
        s.close()
    except OSError:
        pass
    try:
        for ip in socket.gethostbyname_ex(socket.gethostname())[2]:
            if not ip.startswith("127."):
                ips.add(ip)
    except socket.gaierror:
        pass
    return sorted(ips)


class PicoServer:
    """HTTP(页面) + WebSocket(手柄数据) 单端口服务器（后台线程）。"""

    def __init__(self, port: int, tracker: HandTracker,
                 html_path: pathlib.Path):
        self.port = port
        self.tracker = tracker
        self.html = html_path.read_bytes()
        self.rec = False
        self._clients: set = set()
        self._seq = 0              # 连接自增号（手柄数据来源仲裁用）

    def start(self) -> None:
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self) -> None:
        try:
            from websockets.asyncio.server import serve
        except ImportError:
            print("[pico] 缺 websockets 库：pip install websockets", flush=True)
            return

        async def handler(ws):
            self._seq += 1
            cid = self._seq
            self._clients.add(ws)
            self.tracker.claim(cid)
            peer = getattr(ws, "remote_address", None)
            print(f"[pico] 手柄已连接 {peer}（共 {len(self._clients)} 路）",
                  flush=True)
            try:
                async for raw in ws:
                    try:
                        msg = json.loads(raw)
                    except (ValueError, TypeError):
                        continue
                    if isinstance(msg, dict) and "p" in msg:
                        self.tracker.feed(msg, cid)
            finally:
                self._clients.discard(ws)
                self.tracker.release(cid)
                print(f"[pico] 手柄断开（剩 {len(self._clients)} 路）",
                      flush=True)

        async def process_request(connection, request):
            # 非 WebSocket 升级请求（PICO 浏览器打开页面）-> 伺服 HTML
            if request.headers.get("Upgrade", "").lower() != "websocket":
                from websockets.datastructures import Headers
                from websockets.http11 import Response
                headers = Headers([
                    ("Content-Type", "text/html; charset=utf-8"),
                    ("Content-Length", str(len(self.html))),
                    ("Cache-Control", "no-store"),
                    ("Connection", "close"),
                ])
                return Response(200, "OK", headers, self.html)
            return None

        async def main():
            async with serve(handler, "0.0.0.0", self.port,
                             process_request=process_request,
                             max_size=65536, ping_interval=10):
                await asyncio.get_running_loop().create_future()

        asyncio.run(main())

    def broadcast(self, payload: dict) -> None:
        """向所有已连接手柄推送状态（录制标志/震动指令）。"""
        if not self._clients:
            return

        async def push():
            data = json.dumps(payload)
            for ws in list(self._clients):
                try:
                    await ws.send(data)
                except Exception:
                    self._clients.discard(ws)

        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                asyncio.run_coroutine_threadsafe(push(), loop)
        except RuntimeError:
            pass


# ---------------------------------------------------------------- OSC

class ArmOSC:
    """SO-101 操作空间控制（OSC，位置任务 + 零空间姿态保持）。

    每个控制 tick：
      1. 任务空间误差 ep = 目标位置 - 当前 TCP 位置
      2. 期望任务速度 v = Kp_p·ep（限幅）
      3. 任务空间惯量 Λ = (J M⁻¹ Jᵀ + λ²I)⁻¹（M 为机械臂 5×5 质量矩阵）
      4. 关节速度 dq = Λ 加权伪逆 · v + N·Kp_posture·(q_pref - q)
         （N 为精确零空间投影，用于保持构型、抑制手腕乱转）
      5. q_cmd = clip(q + dq·dt, 限位)，交给位置伺服

    相比逐点 IK：惯量加权让末端轻的关节先动（动作自然）、奇异位形附近
    自动降速（不抽搐）、不可达时平滑停在边界（不会关节锁死 + 大残差）。
    姿态不单独跟踪（会与位置任务抢权重，把腕关节顶到限位），只靠零空间
    姿态项维持"夹爪朝下"。
    """

    def __init__(self, model_path: pathlib.Path, base_pos: np.ndarray):
        self.model = mujoco.MjSpec.from_file(str(model_path)).compile()
        self.data = mujoco.MjData(self.model)
        self.nv = self.model.nv
        self.site = mujoco.mj_name2id(self.model,
                                      mujoco.mjtObj.mjOBJ_SITE, "gripperframe")
        if self.site < 0:
            raise RuntimeError("SO-101 模型缺 site 'gripperframe'")
        self.base = np.asarray(base_pos, dtype=float)
        self._jacp = np.zeros((3, self.nv))
        self._jacr = np.zeros((3, self.nv))
        self._mask = np.zeros((self.nv, self.nv))   # mj_fullM 的输出缓冲
        self._lo = np.array([self.model.jnt_range[
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n)][0]
            for n in JOINTS5])
        self._hi = np.array([self.model.jnt_range[
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n)][1]
            for n in JOINTS5])

    def solve(self, q5: np.ndarray, qd5: np.ndarray,
              tgt_local: np.ndarray, dt: float) -> tuple[np.ndarray, float]:
        """返回 (关节位置目标, 位置残差 m)。"""
        d = self.data
        d.qpos[:5] = q5
        d.qvel[:5] = qd5
        mujoco.mj_forward(self.model, d)      # qM / jac / 运动学都要最新

        pos = d.site_xpos[self.site].copy()
        ep = tgt_local - pos
        pos_err = float(np.linalg.norm(ep))

        # ---- 期望任务速度（线速度限幅）
        vp = OSC_KP_POS * ep
        n = float(np.linalg.norm(vp))
        if n > OSC_VMAX:
            vp *= OSC_VMAX / n

        # ---- 位置任务雅可比 3x5
        # 只跟位置，不跟工具轴朝向：5 自由度平面臂在"夹爪朝下"位形下姿态
        # 任务与位置任务互相抢权重（1 rad/s 比 1 m/s 便宜得多），等权最小二乘
        # 会牺牲位置、把腕关节慢慢顶到限位（表现为残差几十毫米、越走越歪）。
        # 实测纯位置任务：静止 32s 残差 0.2mm 无漂移，工具轴倾角 10~18°，
        # 夹爪仍能正常抓取。腕部朝向由下面的零空间姿态项维持。
        mujoco.mj_jacSite(self.model, d, self._jacp, self._jacr, self.site)
        J = self._jacp[:, :5].copy()          # 3x5

        # ---- 任务空间惯量 Λ = (J M⁻¹ Jᵀ + λ²I)⁻¹ 的白化实现
        mujoco.mj_fullM(self.model, d, self._mask)   # 稀疏 qM -> 稠密
        Minv = np.linalg.inv(self._mask[:5, :5])
        # 秩亏处理：平面对称性下 wrist_roll 对末端位置几乎无贡献，J 的近零
        # 奇异方向若保留，反解会给出数十 rad/s 的关节速度（限幅后整条臂只剩
        # 手腕乱转）。在惯量白化坐标下做 SVD，只保留可达方向。
        # M⁻¹ = L·Lᵀ，故 Jw = J·L 时 Jw·Jwᵀ = J M⁻¹ Jᵀ；
        # J# = M⁻¹JᵀΛ = L·V·diag(σ/(σ²+λ²))·Uᵀ（回代乘 L，不是 Lᵀ）
        L = np.linalg.cholesky(Minv)
        Jw = J @ L
        # full_matrices=False：Vh 是 3x5，V 只取行空间那 3 个方向（够用且省算）
        Uw, sw, Vhw = np.linalg.svd(Jw, full_matrices=False)
        s_max = max(float(sw[0]), 1e-12)
        keep = sw > s_max * 1e-2
        keep[0] = True
        lam = OSC_LAMBDA_REL * s_max
        gain_task = np.where(keep, sw / (sw ** 2 + lam ** 2), 0.0)
        gain_pinv = np.where(keep, 1.0 / np.maximum(sw, 1e-12), 0.0)
        dq_task = L @ (Vhw.T @ (gain_task * (Uw.T @ vp)))
        # 零空间投影用"精确伪逆 + 秩截断"，保证 J·N≈0（阻尼版不是真投影，
        # 会把姿态项漏回任务空间，同样引起漂移）
        Jsharp = L @ (Vhw.T @ np.diag(gain_pinv) @ Uw.T)
        N = np.eye(5) - Jsharp @ J
        # 零空间：姿态保持 + 速度阻尼（后者抑制目标不可达时在限位边界来回抖）
        dq_null = N @ (OSC_KP_POSTURE * (PRE_Q5 - q5) - OSC_KD_POSTURE * qd5)
        dq = dq_task + dq_null

        n = float(np.linalg.norm(dq))
        if n > MAX_JOINT_SPEED:
            dq *= MAX_JOINT_SPEED / n
        q_cmd = q5 + dq * dt
        # 抗饱和：已经贴住限位还要往外推的关节，速度直接清零，避免积分离限位
        blocked = ((q_cmd <= self._lo) & (dq < 0)) | \
                  ((q_cmd >= self._hi) & (dq > 0))
        dq[blocked] = 0.0
        q_cmd = np.clip(q5 + dq * dt, self._lo, self._hi)
        return q_cmd, pos_err

    def tool_pose(self, q5: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        d = self.data
        d.qpos[:5] = q5
        mujoco.mj_kinematics(self.model, d)
        pos = d.site_xpos[self.site] + self.base
        axis = d.site_xmat[self.site].reshape(3, 3)[:, 0]
        return pos, axis


JOINTS5 = ["shoulder_pan", "shoulder_lift", "elbow_flex",
           "wrist_flex", "wrist_roll"]


# ---------------------------------------------------------------- 主程序

def _resolve_model(world_spec: dict) -> tuple[pathlib.Path, np.ndarray]:
    """从 hello 下发的 world_spec 找机器人模型与基座位置。"""
    robots = world_spec.get("robots") or []
    rd = robots[0] if robots else {}
    model_ref = rd.get("model") or ""
    base = np.asarray(rd.get("pos") or [0, 0, 0], dtype=float)
    if model_ref.startswith("zoo://"):
        name = model_ref[len("zoo://"):]
        for ext in (".xml", ".urdf", ".mjcf"):
            hits = sorted((REPO / "models" / "zoo" / name).glob(f"*{ext}"))
            if hits:
                return hits[0], base
    p = pathlib.Path(model_ref)
    if not p.is_absolute():
        p = REPO / p
    if p.exists():
        return p, base
    raise RuntimeError(f"无法解析机器人模型: {model_ref!r}")


def main() -> None:
    quat_axes_ok()
    robot = Robot()
    args = dict(robot.args or {})
    dataset_dir = pathlib.Path(args.get("dataset", DEFAULT_DATASET))
    task_text = str(args.get("task", DEFAULT_TASK))
    port = int(args.get("port", DEFAULT_PORT))
    # 相机：逗号分隔可多路（腕部 + 世界固定机位），每路一个 LeRobot 视频键
    cam_names = [c.strip() for c in
                 str(args.get("cameras", args.get("camera", "wrist_cam"))).split(",")
                 if c.strip()]
    cam_wh = (int(args.get("cam_height", 480)), int(args.get("cam_width", 640)))

    model_path, base_pos = _resolve_model(robot.world_spec or {})
    osc = ArmOSC(model_path, base_pos)
    dt = robot.timestep * robot.ctrl_period
    fps = 1.0 / dt

    writer = LerobotWriter(
        dataset_dir, fps=fps, dim=6,
        cameras={f"observation.images.{n}": cam_wh for n in cam_names},
        task=task_text, robot_type="so101",
        state_names=JOINTS5 + ["gripper"],
        action_names=JOINTS5 + ["gripper"])

    tracker = HandTracker()
    server = PicoServer(port, tracker,
                        pathlib.Path(__file__).with_name("so101_pico_webxr.html"))
    server.start()
    t_start = time.monotonic()
    for ip in _local_ips():
        print(f"[pico] PICO 浏览器打开: http://{ip}:{port}", flush=True)
    print(f"[pico] USB 直连（adb reverse tcp:{port} tcp:{port}）时用: "
          f"http://localhost:{port}", flush=True)

    # 目标状态（世界系）。raw_tgt 按手柄增量累积，但每 tick 都被夹回工作域
    # 盒内——不留漂移，手一往回走立刻跟（无死区），也不会攒出离谱的目标。
    raw_tgt = HOME_WORLD.copy()
    lo_w = base_pos + np.array([TGT_X[0], TGT_Y[0], TGT_Z[0]])
    hi_w = base_pos + np.array([TGT_X[1], TGT_Y[1], TGT_Z[1]])
    prev_hand = None           # 上一帧手柄位置（增量映射用）
    gen = tracker.generation   # 数据来源代次（换手柄/重连时重置基准）
    grip_ctrl = GRIP_OPEN
    recording = False
    armed = False              # 预备姿态是否到位（到位前只 ramp 不 OSC）
    armed_at = 0.0
    ramp_q = None
    wait_logged = False         # 只提示一次"还没收到手柄数据"
    got_logged = False
    print("[pico] 遥操作就绪：A=录制 B=重新对中 | 启动后先自动走到预备姿态",
          flush=True)

    while robot.step():
        q = np.asarray(robot.obs["qpos"], dtype=float)
        now = time.monotonic()
        m = tracker.latest
        # 新鲜度按"本进程收到数据的时间"判断，不能用消息里的 t（浏览器时钟）
        fresh = m is not None and (now - tracker.recv < CTRL_TIMEOUT)

        # ---- 阶段 1：ramp 到预备姿态（从零位直接 OSC 逼近会顶到限位）
        if not armed:
            if ramp_q is None:
                ramp_q = q[:5].copy()
            step_j = PRE_SPEED * dt
            ramp_q = ramp_q + np.clip(PRE_Q5 - ramp_q, -step_j, step_j)
            err = float(np.max(np.abs(q[:5] - PRE_Q5)))
            robot.set_all([float(v) for v in ramp_q] + [float(grip_ctrl)])
            if err < 0.03:
                armed = True
                armed_at = now
                print("[pico] 预备姿态到位，OSC 接管（右手柄即可开始移动）",
                      flush=True)
            continue

        # ---- 阶段 2：手柄 -> 目标（增量映射：按手柄位移增量推动，绝对位置
        # 只决定朝向；这样手在身前任意位置都从当前位形起步，永远不打满工作域）
        if tracker.generation != gen:
            # 数据源换了（重连/抢占）：当前位置作为新基准，目标不跳
            gen = tracker.generation
            prev_hand = None
        # 状态提示：浏览器普通页面阶段不发送手柄位姿，必须先进入 VR 会话
        if fresh and not got_logged:
            got_logged = True
            print("[pico] 已收到手柄数据，开始跟随手柄", flush=True)
        elif not fresh and not wait_logged and now - armed_at > 2.0:
            wait_logged = True
            print("[pico] 等待手柄数据：请在头显页面点『进入 VR 遥操作』"
                  "（停留在普通网页时不发送手柄位姿）", flush=True)
        if fresh:
            p = vr_to_mj(np.asarray(m["p"], dtype=float))
            if prev_hand is None:
                prev_hand = p.copy()
            raw_tgt = raw_tgt + (p - prev_hand) * SCALE
            prev_hand = p
            # 夹回工作域盒内：不留漂移，手往回走立刻跟手
            raw_tgt = np.clip(raw_tgt, lo_w, hi_w)
            # ---- 夹爪扳机
            trig = float(np.clip(m.get("trig", 0.0), 0.0, 1.0))
            grip_target = GRIP_OPEN + (GRIP_CLOSE - GRIP_OPEN) * trig
            grip_ctrl += float(np.clip(grip_target - grip_ctrl,
                                       -GRIP_SPEED * dt, GRIP_SPEED * dt))

        # ---- A/B 按键：按下沿在 WS 线程锁存，这里消费（64ms 采样也不漏按）
        while tracker.press_a > 0:
            tracker.press_a -= 1
            try:
                if not recording:
                    writer.start_episode()
                    recording = True
                    print(f"[pico] ▶ 开始录制 episode "
                          f"{writer.episode_index} -> {dataset_dir}",
                          flush=True)
                else:
                    info = writer.end_episode()
                    recording = False
                    print(f"[pico] ■ 停止录制: {info}", flush=True)
                    # 每集收尾自动复位：物体搬回初始位姿，机械臂 ramp 回预备姿态
                    try:
                        robot.reset_objects()
                        print("[pico] 已复位：物体回初始位置，机械臂回预备姿态",
                              flush=True)
                    except Exception as e:
                        print(f"[pico] 复位失败: {e}", flush=True)
                    armed = False
                    ramp_q = q[:5].copy()
                    raw_tgt = HOME_WORLD.copy()
                    grip_ctrl = GRIP_OPEN
                    prev_hand = None
                    tracker.recalibrate()
                server.broadcast({"rec": recording, "buzz": 150})
            except Exception as e:
                recording = False
                print(f"[pico] 录制出错: {e}", flush=True)
        if tracker.press_b > 0:
            tracker.press_b = 0
            tracker.recalibrate()
            prev_hand = None        # 下一帧重取基准：目标原地不动，只换手部原点
            server.broadcast({"buzz": 60})
            print("[pico] 已重新对中：机械臂保持原位，手柄当前位置为新基准",
                  flush=True)

        # ---- OSC（本地系目标）-> 关节位置目标
        tgt_local = np.clip(raw_tgt - base_pos,
                            [TGT_X[0], TGT_Y[0], TGT_Z[0]],
                            [TGT_X[1], TGT_Y[1], TGT_Z[1]])
        q5, pos_err = osc.solve(q[:5], np.asarray(robot.obs["qvel"][:5],
                                                  dtype=float),
                                tgt_local, dt)
        # 姿态把目标卡在可达边界时，把目标朝当前 TCP 轻拉，避免关节顶死
        if pos_err > BACKOFF_ERR:
            tcp_world = osc.tool_pose(q[:5])[0]
            raw_tgt = raw_tgt + (tcp_world - raw_tgt) * min(
                1.0, BACKOFF_GAIN * dt)
        if pos_err > BACKOFF_ERR and robot.tick % 160 == 0:
            print(f"[pico] 已到可达边界（残差 {pos_err*1000:.0f}mm），"
                  "目标自动回吸，往回收一下手柄即可", flush=True)
        ctrl6 = [float(v) for v in q5] + [float(grip_ctrl)]
        robot.set_all(ctrl6)
        if _TRACE and robot.tick % 16 == 0:
            tcp = osc.tool_pose(q[:5])[0]
            print(f"[trace] t={time.monotonic()-t_start:5.2f}s tick={robot.tick} "
                  f"dt={dt:.3f} tgt={np.round(raw_tgt,3)} tcp={np.round(tcp,3)} "
                  f"q={np.round(q[:5],3)} err={pos_err*1000:.0f}mm "
                  f"grip={grip_ctrl:.2f}", flush=True)

        # ---- 录制：state=当前实际关节，action=本 tick 发出的控制目标
        #（LeRobot 语义：action 是此刻下发的指令，下一状态将趋向它）
        if recording:
            try:
                images = {f"observation.images.{n}": robot.camera(
                    n, width=cam_wh[1], height=cam_wh[0]) for n in cam_names}
                writer.add_frame(q, ctrl6, images)
            except Exception as e:
                print(f"[pico] 采样失败，丢弃该帧: {e}", flush=True)

    if recording:
        try:
            print(f"[pico] 内核结束，收尾录制: {writer.end_episode()}",
                  flush=True)
        except Exception as e:
            print(f"[pico] 收尾失败: {e}", flush=True)

    # 退出即把整份数据集升级成 lerobot >= 0.6 能直接训练的 v3.0；
    # v2.1 原目录留作 <root>_v21，下次启动会自动换回来继续采。
    # 调参阶段不想每次退出都重拼视频时，可设 SIMOVIA_NO_V30=1 跳过。
    if not os.environ.get("SIMOVIA_NO_V30"):
        try:
            print(f"[pico] 数据集升级 v3.0: {writer.finalize_v30()}",
                  flush=True)
        except Exception as e:
            print(f"[pico] v3.0 升级失败（v2.1 数据完好，可手动重试）: {e}",
                  flush=True)
    print("[pico] 遥操作结束", flush=True)


if __name__ == "__main__":
    main()
