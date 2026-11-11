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

## 目录

```
worlds/so101_pico_sort.yaml        世界：桌面、双方块、双盒子、固定俯视机位
controllers/so101_pico_teleop.py   PICO 遥操作 + OSC + 录制 + 自动复位
controllers/so101_pico_webxr.html  头显页面（WebXR 手柄采集与回传）
```

## 场景参数

- 控制周期 `ctrl_period: 16` × `timestep: 0.002` → 31.25 Hz 录制帧率
- 工作域 clamp 与手柄缩放见 `controllers/so101_pico_teleop.py` 顶部
  （`SCALE`、`TGT_X/Y/Z`、`PRE_Q5`）
- 腕部相机装在 `camera_mount`，固定俯视机位在 world YAML 的 `scene.cameras`
