# so101-smolvla-demo

用 PICO 头显遥操作 SO-101 机械臂完成桌面双物体分拣，同步录制 LeRobot 数据集，
退出时自动升级成 lerobot >= 0.6 可直接训练的 v3.0 格式，直接喂 SmolVLA 微调。

基于 [Simovia](https://github.com/jerryzhu-yc/simovia)（World as Code + 进程级控制器 + MuJoCo）。

## 链路

```
PICO 浏览器 (WebXR 手柄 90Hz)
        │  WebSocket
        ▼
so101_pico_teleop.py  ──OSC 操作空间控制──▶  Simovia 内核 (MuJoCo)
        │                                        │
        │  每控制周期采两路画面                    │  每集结束复位物体与机械臂
        ▼                                        ▼
LeRobot v2.1 数据集  ──退出时就地升级──▶  v3.0（lerobot 0.7.0 可直接加载）
```

## 依赖

- Python 3.10+、`ffmpeg` / `ffprobe`（视频编码与拼接）
- Simovia 内核：

```bash
pip install "simovia @ git+https://github.com/jerryzhu-yc/simovia.git"
# 或本地开发：git clone 后 pip install -e /path/to/simovia
```

- PICO 4 头显（自带浏览器即可，不需要装 APP）
- 可选：`adb`（USB 直连时做端口转发）

## 快速开始

```bash
# 1. 启动世界（控制器由 YAML 自动拉起，同时开 HTTP/WS 服务）
python -m simovia run worlds/so101_pico_sort.yaml

# 2. 连 PICO（二选一）
#    同一局域网：浏览器打开 http://<PC 的 IP>:8090
#    USB 直连：
adb reverse tcp:8090 tcp:8090
adb shell am start -a android.intent.action.VIEW -d "http://localhost:8090"

# 3. 页面里点「进入 VR 遥操作」，机械臂会自动走到预备姿态
```

数据集默认写到 `~/桌面/so101_smolvla_sort2`，可在 world YAML 的
`robots[].args.dataset` 改；`task` 是写进每一集的语言指令。

## 操作

| 输入 | 作用 |
| --- | --- |
| 右手柄位移 | 增量映射到 TCP 目标（B 键重新对中） |
| 扳机 | 夹爪开合 |
| A 键 | 开始 / 停止录制一集 |
| B 键 | 原地重新对中（机械臂不动） |

每集停止后自动复位：红色/蓝色方块搬回初始位姿，机械臂 ramp 回预备姿态，
手柄重新对中，避免下一集从漂移的目标开始。

## 数据集格式

采集期按 LeRobot **v2.1** 逐帧写（每集一个 mp4 + 一个 parquet），这是能
安全续写、能手动删坏集的格式。程序退出时自动**就地升级到 v3.0**：

```
meta/info.json                     v3.0（视频特征带 names，否则建 policy 会 TypeError）
meta/stats.json                    全局统计
meta/tasks.parquet
meta/episodes/chunk-000/file-000.parquet   每集一行：视频时间戳 + 统计
data/chunk-000/file-000.parquet            所有集标量列拼接（不含视频 struct 列）
videos/<camera>/chunk-000/file-000.mp4     各路相机按集顺序无损拼接
```

原 v2.1 目录保留为 `<root>_v21`；下次对同一路径启动采集时会自动换回来继续
续写，所以「采完即 v3.0、再采自动回退」可以反复进行。

调参阶段不想每次退出都重拼视频，可设 `SIMOVIA_NO_V30=1` 跳过。

也可以手动转换：

```bash
python -m simovia.lerobot_v30 ~/桌面/so101_smolvla_sort2
```

> 不要用 lerobot 官方的 `convert_dataset_v21_to_v30`：它假设 v2.1 的相机是
> 抽帧 PNG（`dtype=image`），对真视频版 v2.1 会残留 video struct 列、把视频
> 特征的 `names` 写成 null，导致加载和建 policy 双双失败。

## 微调 SmolVLA

在装了 lerobot 的 GPU 机器上（示例为 0 号卡）：

```bash
export CUDA_VISIBLE_DEVICES=0
lerobot-train \
  --policy.path=lerobot/smolvla_base \
  --policy.device=cuda \
  --policy.use_amp=true \
  --policy.push_to_hub=false \
  --dataset.repo_id=local/so101_sort2 \
  --dataset.root=/path/to/so101_smolvla_sort2 \
  --rename_map='{"observation.images.wrist_cam":"observation.images.camera1","observation.images.top":"observation.images.camera2"}' \
  --batch_size=32 --steps=20000 \
  --save_checkpoint=true --save_freq=5000 \
  --output_dir=~/outputs/so101_sort2
```

两点注意：

- `smolvla_base` 的相机槽位是 `camera1/2/3`，本采集只有 `wrist_cam`、`top`
  两路，用 `--rename_map` 映射过去；**不要**用 `--policy.empty_cameras`，
  那个参数是「额外追加空相机」，会让期望路数变成 4 路反而对不上。
- 推理时必须用与训练一致的相机顺序（按 key 字母序：`top` → camera2，
  `wrist_cam` → camera1）。

## 策略推理评估

训练完成后把 checkpoint 放回本机，用 `controllers/so101_policy_eval.py` 闭环执行分拣。

### 取得权重

`model.safetensors`（865MB）超过 GitHub 单文件 100MB 限制，作为 Release 资产分发；
配置、归一化统计与 tokenizer 已在本仓库 `models/` 下：

```bash
git clone https://github.com/jerryzhu-yc/so101-smolvla-demo.git
cd so101-smolvla-demo
curl -L -o models/so101_sort2_20k/model.safetensors \
  https://github.com/jerryzhu-yc/so101-smolvla-demo/releases/download/v1.0.0/model.safetensors
```

### 运行

```bash
# 无头
python -m simovia run worlds/so101_policy_eval.yaml

# 带画面：GUI 里加载 worlds/so101_policy_eval.yaml
python -m simovia gui
```

推理需要 lerobot 0.7.0 + CUDA 版 torch。若 GUI 是从没装 lerobot 的解释器启动的，
控制器会自动 `os.execv` 切到装了 lerobot 的解释器（PID 不变，ZMQ 连接保留）。
目标解释器默认 `~/miniconda3/envs/smolvla/bin/python`，可用环境变量
`SIMOVIA_POLICY_PY` 覆盖。

> smolvla 环境还需要 `pyzmq`（控制器与内核通信）：
> `pip install pyzmq`

### 推理链路要点

- 一次推理出 50 步 action chunk（未来 1.6s），逐拍执行，队列空再推：单次推理
  116ms，摊薄后每拍只占 2.3ms。**不能每拍推理**——116ms 对 32ms 控制周期会让
  仿真持续补位、动作卡顿
- action 是**绝对关节目标**（5 臂 + 夹爪 ctrl），不是增量，反归一化后直接
  `set_all`，无积分无累积
- 启动先 ramp 到 `PRE_Q5` 预备姿态，与采集时每集起始帧完全一致。从零位直接推理
  属于分布外输入，且移动路径可能扫翻方块
- 相机走 checkpoint 内置 rename 预处理：`wrist_cam → camera1`、`top → camera2`，
  与训练一致。`camera3` 槽位因 `empty_cameras=0` 会被自动跳过，只喂两路即可
- 每次推理把两路相机帧转发到 GUI 预览面板，可直接核对策略眼中的画面

### 实测表现与已知限制

| 项 | 结果 |
| --- | --- |
| 训练 | 50 集 / 20247 帧 / 20000 步，31.61 epoch，final loss 0.035 |
| 推理耗时 | 116ms 每 chunk（RTX 5070，显存 0.91GB） |
| 固定布局 | 红、蓝方块均可完成分拣 |

**位置泛化是当前主要短板**：50 集数据的物体初始位置完全固定
（`reset_objects` 严格恢复 `qpos0`，无任何随机化），模型学到的是「固定布局上的
轨迹 + 视觉微调」，而不是「搜索任意位置的彩色物体」。把红蓝方块互换后，红块
仍能被抓起（说明视觉通路有效），但蓝块会失败并反复在原红块位置重试。

治本方案是在采集时做域随机化（物体 XY 偏移 + 随机 yaw），重采 100 集左右再训练。

## 目录

```
worlds/so101_pico_sort.yaml        采集世界：桌面、双方块、双盒子、固定俯视机位
worlds/so101_policy_eval.yaml      评估世界：场景与采集一致，控制器换成策略推理
controllers/so101_pico_teleop.py   PICO 遥操作 + OSC + 录制 + 自动复位
controllers/so101_pico_webxr.html  头显页面（WebXR 手柄采集与回传）
controllers/so101_policy_eval.py   SmolVLA 推理控制器（chunk 执行 + 相机预览）
models/so101_sort2_20k/            训练产出的策略配置、归一化统计、tokenizer
```

## 场景参数

- 控制周期 `ctrl_period: 16` × `timestep: 0.002` → 31.25 Hz 录制帧率
- 工作域 clamp 与手柄缩放见 `controllers/so101_pico_teleop.py` 顶部
  （`SCALE`、`TGT_X/Y/Z`、`PRE_Q5`）
- 腕部相机装在 `camera_mount`，固定俯视机位在 world YAML 的 `scene.cameras`
