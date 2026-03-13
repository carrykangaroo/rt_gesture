# RT-Gesture Linux DL 服务器环境说明

> 更新时间：2026-03-09  
> 适用项目：`RT-Gesture` 实时离散手势识别系统  
> 项目根目录：`/home/rxb/rt_gesture`  
> 工作目录：`/home/rxb/rt_gesture/workspace`

## 1. 当前结论

- 当前 Linux 服务器环境已经可以用于本项目开发、测试、训练与评估。
- 已确认可用 Conda 环境为：`/home/rxb/.conda/envs/torch2.0.1`。
- 已确认 `torch` 与 CUDA 路径正常：`torch=2.0.1+cu117`、`cuda=True`、`gpus=2`。
- 项目依赖已补齐，并且项目包已通过 `pip` 以 editable 方式安装。
- 数据集真实位置已确认：`/Data/CTRL_LAB/Discrete Gestures`。
- 服务器无显示器时，GUI 相关检查建议使用 `QT_QPA_PLATFORM=offscreen`。
- 按 `doc/Test/Linux_DL服务器_测试清单.md` 执行后，当前结果为：`31 通过 / 10 失败 / 10 跳过`（日期：2026-03-09）。
- 当前主要阻塞项为：benchmark 延迟阈值超标、`SIGTERM/SIGKILL` 恢复流程不够稳健、以及默认配置路径与本机数据目录不一致。

## 2. 服务器实际配置

| 项目 | 实际情况 |
|---|---|
| 操作系统 | Ubuntu 22.04.1 LTS |
| 内核 | `Linux 5.15.0-105-generic x86_64` |
| CPU | `2 × Intel Xeon E5-2620 v3 @ 2.40GHz` |
| CPU 核心/线程 | `12` 物理核心，`24` 逻辑线程 |
| 内存 | `125 GiB` |
| GPU | `2 × NVIDIA RTX A5000` |
| 单卡显存 | `24564 MiB`（约 `25.4 GB`） |
| NVIDIA 驱动 | `515.105.01` |
| CUDA 驱动侧版本 | `11.7` |
| Python | `3.10.19` |
| PyTorch | `2.0.1+cu117` |

## 3. Python/Conda 环境信息

### 3.1 指定环境

本机当前确认使用以下环境：

```bash
conda activate /home/rxb/.conda/envs/torch2.0.1
```

对应 Python 路径：

```bash
/home/rxb/.conda/envs/torch2.0.1/bin/python
```

### 3.2 安装原则

- 仅使用 `pip install` 安装 Python 依赖。
- 不使用 `conda install` 安装项目依赖。
- 项目当前有效安装方式为先安装 `requirements.txt`，再执行 editable 安装。
- 当前仓库不建议使用 `pip install -e ".[dev]"` 作为标准命令。

### 3.3 当前推荐安装命令

```bash
cd /home/rxb/rt_gesture/workspace
/home/rxb/.conda/envs/torch2.0.1/bin/python -m pip install -r requirements.txt
/home/rxb/.conda/envs/torch2.0.1/bin/python -m pip install -e .
```

## 4. 已安装关键依赖版本

| 包名 | 版本 |
|---|---|
| `torch` | `2.0.1+cu117` |
| `numpy` | `1.26.4` |
| `h5py` | `3.10.0` |
| `pandas` | `2.2.2` |
| `numba` | `0.63.1` |
| `tqdm` | `4.67.1` |
| `PyYAML` | `6.0.3` |
| `pyzmq` | `27.1.0` |
| `msgpack` | `1.1.2` |
| `PyQt6` | `6.10.2` |
| `pyqtgraph` | `0.13.7` |
| `pytest` | `8.4.2` |
| `rt_gesture` | `0.2.0` |

## 5. 已完成验证

### 5.1 依赖导入验证

已完成以下导入验证：

- `torch`
- `zmq`
- `msgpack`
- `h5py`
- `yaml`
- `numpy`
- `pandas`
- `PyQt6`
- `pyqtgraph`
- `pytest`
- `rt_gesture`

### 5.2 CUDA 验证结果

已在目标环境中完成验证，实际结果如下：

```text
All imports OK
torch=2.0.1+cu117
cuda=True
gpus=2
  gpu[0]=NVIDIA RTX A5000, vram=25.4 GB
  gpu[1]=NVIDIA RTX A5000, vram=25.4 GB
```

说明：

- 当前服务器能够被 PyTorch 正常识别为 CUDA 环境。
- 当前共有 `2` 张可用 GPU。
- 适合执行训练、评估、benchmark 与稳定性验证。

## 6. 数据集位置

数据集根目录已经确认：

```text
/Data/CTRL_LAB/Discrete Gestures
```

目录与规模如下：

| 数据集 | 真实路径 | 文件数 | 体量 |
|---|---|---:|---:|
| `mini` | `/Data/CTRL_LAB/Discrete Gestures/mini` | `3` 个 `.hdf5` | `1.1G` |
| `full` | `/Data/CTRL_LAB/Discrete Gestures/full` | `100` 个 `.hdf5` | `32G` |

建议检查命令：

```bash
ls -lh '/Data/CTRL_LAB/Discrete Gestures/mini'/*.hdf5 | head -5
ls '/Data/CTRL_LAB/Discrete Gestures/full'/*.hdf5 | wc -l
```

注意：数据集路径中包含空格，命令行使用时建议始终加引号。

## 7. GUI 无头环境说明

该服务器通常作为无头 Linux 环境使用，推荐采用以下方式执行 GUI 相关验证：

```bash
export QT_QPA_PLATFORM=offscreen
python scripts/run_gui.py --config config/debug_short.yaml
```

说明：

- `offscreen` 适合验证窗口创建、进程管理和程序不崩溃。
- 若后续需要完整交互，再额外考虑 X11 forwarding 或虚拟显示方案。

## 8. 常用命令

### 8.1 安装与校验

```bash
cd /home/rxb/rt_gesture/workspace
/home/rxb/.conda/envs/torch2.0.1/bin/python -m pip install -r requirements.txt
/home/rxb/.conda/envs/torch2.0.1/bin/python -m pip install -e .
/home/rxb/.conda/envs/torch2.0.1/bin/python -m pytest -q
```

### 8.2 实时链路

```bash
cd /home/rxb/rt_gesture/workspace
/home/rxb/.conda/envs/torch2.0.1/bin/python scripts/run_realtime.py --config config/debug_short.yaml
```

说明：若默认 `config/*.yaml` 仍使用历史路径 `/mnt/data/...`，需先改为本机真实路径 `/Data/CTRL_LAB/Discrete Gestures/...` 再执行。

### 8.3 正式训练

```bash
cd /home/rxb/rt_gesture/workspace
/home/rxb/.conda/envs/torch2.0.1/bin/python scripts/run_training.py --config config/training.yaml
```

## 9. 备注

- 本说明基于 2026-03-09 的实际检查结果整理。
- `doc/Test/Linux_DL服务器_测试清单.md` 已同步改成真实数据集路径。
- 若后续升级 `torch`、CUDA 或驱动版本，应重新核对训练、benchmark 与 GUI 兼容性。
- 本次测试中对 `workspace/rt_gesture/checkpoint_utils.py` 和 `workspace/rt_gesture/data.py` 做了兼容性修复（旧 checkpoint 模块路径兼容、CSV 数据集名重复 `.hdf5` 兼容）。
