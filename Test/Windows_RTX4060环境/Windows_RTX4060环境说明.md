# Windows RTX 4060 本机环境说明

> 采集日期：2026-03-09  
> 采集主机：`LAPTOP-P15OOT02` / `LENOVO 83DG`  
> 对照文档：`doc/Test/Windows_RTX4060_测试清单.md`、`doc/技术实现文档.md`、`doc/review/RT-Gesture完整流程功能文档.md`

## 1. 验证结论

**结论：本机硬件、CUDA 运行时、核心 Python 依赖、GUI 依赖、数据集和 checkpoint 已基本具备 RT-Gesture 在 Windows + RTX 4060 环境下的测试前提；当前仅剩 1 个阻塞项，因此暂不能判定为“完整具备测试清单要求”。**

阻塞项如下：

1. `workspace/config/*.yaml` 中仍保留 Linux 风格的 `/mnt/...` 数据路径，尚未切换为本机 Windows 路径。

补齐以上路径项后，即可进入 `doc/Test/Windows_RTX4060_测试清单.md` 的阶段 A～H 测试。

---

## 2. 本机环境快照

| 类别 | 本机实际情况 | 对照要求 | 结论 |
|---|---|---|---|
| 操作系统 | Windows 11 家庭中文版，版本 `10.0.26200`，64 位 | Windows 10/11 64-bit | ✅ 满足 |
| CPU | `Intel(R) Core(TM) i7-14650HX`，16 核 24 线程 | Intel / AMD 移动处理器 | ✅ 满足 |
| 内存 | 系统可见 `15.71 GiB`（对应 16 GB 标称内存） | ≥ 16 GB | ✅ 基本满足 |
| GPU | `NVIDIA GeForce RTX 4060 Laptop GPU` | RTX 4060 Laptop 8 GB VRAM | ✅ 满足 |
| 显存 | `8.00 GiB` | ≥ 7.5 GiB | ✅ 满足 |
| 驱动 / CUDA | `nvidia-smi` 显示 Driver `591.86`，CUDA Version `13.1` | CUDA 驱动 ≥ 12.1 | ✅ 满足 |
| 磁盘剩余空间 | `C:` 剩余 `69.60 GB`；`D:` 剩余 `120.79 GB` | ≥ 50 GB 可用空间 | ✅ 满足 |
| Conda | `conda 24.11.3` | 需具备 Conda 环境 | ✅ 满足 |
| 目标 Conda 环境 | `D:\env\condaEnv\torch2.4.1` | Python 3.12 + torch 2.4.1 | ✅ 已存在 |
| Python（系统） | `3.12.7` | Python 3.12 | ✅ 满足 |
| Python（目标环境） | `3.12.12` | Python 3.12 | ✅ 满足 |
| PyTorch（目标环境） | `2.4.1+cu124` | `torch==2.4.1` CUDA 版 | ✅ 满足 |
| CUDA 可用性 | `torch.cuda.is_available() == True` | 需可识别 CUDA | ✅ 满足 |
| 自动设备选择 | `auto_select_device() == "cuda"` | A-5 返回 `cuda` | ✅ 满足 |

---

## 3. 目标环境依赖检查

### 3.1 已验证通过的核心依赖

以下依赖已在 `D:\env\condaEnv\torch2.4.1` 中验证可正常导入：

| 包 | 版本 | 结论 |
|---|---|---|
| `torch` | `2.4.1+cu124` | ✅ |
| `numpy` | `1.26.4` | ✅ |
| `h5py` | `3.12.1` | ✅ |
| `pandas` | `2.2.2` | ✅ |
| `numba` | `0.60.0` | ✅ |
| `PyYAML` | `6.0.2` | ✅ |
| `pyzmq` | `27.1.0` | ✅ |
| `msgpack` | `1.1.2` | ✅ |
| `pytest` | `8.3.3` | ✅ |
| `PyQt6` | `6.10.2` | ✅ |
| `pyqtgraph` | `0.13.7` | ✅ |

### 3.2 已补齐的 GUI 依赖

| 包 | 测试清单用途 | 当前状态 |
|---|---|---|
| `PyQt6` | GUI 主界面、状态栏、配置面板 | ✅ 已安装 |
| `pyqtgraph` | GUI 实时波形绘制 | ✅ 已安装 |

补充说明：当前已验证 `PyQt6=6.10.2`、`Qt=6.10.0`、`pyqtgraph=0.13.7`，GUI 依赖不再构成阶段 A / D 的阻塞项。

---

## 4. 本机测试资产检查

### 4.1 数据集

本机已发现数据集目录：

- `D:\Dataset\neuromotor\discrete_gestures`（仅full数据集）

检查结果：

| 项目 | 实际情况 | 结论 |
|---|---|---|
| HDF5 文件数量 | `100` 个 | ✅ 满足训练/评估需要 |
| CSV 划分文件 | `discrete_gestures_corpus.csv` 存在 | ✅ 满足 |
| 样例 HDF5 | `discrete_gestures_user_000_dataset_000.hdf5` 可读 | ✅ 满足 |
| 数据总量 | 约 `31.07 GB` | ✅ 满足 |

样例文件抽查结果：

- 根键：`data`, `prompts`, `stages`
- `task=discrete_gestures`
- `prompts` 表共 `1900` 行，列为 `name`, `time`

### 4.2 Checkpoint 与模型配置

本机已发现以下模型资产：

| 资源 | 路径 | 结论 |
|---|---|---|
| 预训练 checkpoint | `workspace/checkpoints/discrete_gestures/model_checkpoint.ckpt` | ✅ 存在 |
| 模型配置 | `workspace/checkpoints/discrete_gestures/model_config.yaml` | ✅ 存在 |
| 参数形状清单 | `workspace/checkpoints/discrete_gestures/STATE_DICT_SHAPES.tsv` | ✅ 存在 |

---

## 5. 与测试清单环境要求的逐项对照

### 5.1 阶段 A：环境验证

| 检查项 | 本机结果 | 结论 |
|---|---|---|
| A-1 核心依赖可导入 | `zmq/msgpack/h5py/yaml/numpy/pandas/PyQt6/pyqtgraph` 均正常 | ✅ |
| A-2 `torch.cuda.is_available()` | `True` | ✅ |
| A-3 GPU 名称包含 `RTX 4060` | `NVIDIA GeForce RTX 4060 Laptop GPU` | ✅ |
| A-4 VRAM ≥ 7.5 GB | `8.00 GiB` | ✅ |
| A-5 `auto_select_device()` 返回 `cuda` | 返回 `cuda` | ✅ |

### 5.2 阶段 B～H 的环境前提

| 前提项 | 本机结果 | 结论 |
|---|---|---|
| `pytest` 可用 | `pytest 8.3.3` | ✅ |
| CUDA 训练 / 推理基础条件 | 已满足 | ✅ |
| 数据集存在 | 已满足 | ✅ |
| checkpoint 存在 | 已满足 | ✅ |
| GUI 依赖存在 | 已满足 | ✅ |
| YAML 数据路径为本机可用路径 | 不满足 | ❌ |

**结论：** 阶段 A 的环境验证前提已具备；阶段 B～H 的硬件和依赖条件也已基本具备，但在修改 YAML 数据路径前仍不能直接按文档执行。

---

## 6. 当前配置缺口

### 6.1 Linux 路径仍未替换为 Windows 本机路径

当前以下配置文件仍包含 `/mnt/...` 路径，不能在本机直接使用：

| 文件 | 当前问题 |
|---|---|
| `workspace/config/default.yaml` | `data_simulator.hdf5_path` 指向 Linux 路径 |
| `workspace/config/debug_short.yaml` | `data_simulator.hdf5_path` 指向 Linux 路径 |
| `workspace/config/evaluation.yaml` | `hdf5_path` 指向 Linux 路径 |
| `workspace/config/evaluation_benchmark.yaml` | `hdf5_path` 指向 Linux 路径 |
| `workspace/config/training.yaml` | `data_location` / `split_csv` 指向 Linux 路径 |
| `workspace/config/training_debug.yaml` | `data_location` / `split_csv` 指向 Linux 路径 |

### 6.2 本机数据布局与文档示例不完全一致

文档示例使用：

- `.../Discrete Gestures/mini/...`
- `.../Discrete Gestures/full/...`

本机实际为单层目录：

- `D:\Dataset\neuromotor\discrete_gestures\*.hdf5`
- `D:\Dataset\neuromotor\discrete_gestures\discrete_gestures_corpus.csv`

这不会阻止测试，但需要在 YAML 中使用本机真实路径，而不能照搬文档里的目录结构。

### 6.3 `install_deps.py` 默认前缀是 Linux 路径

`workspace/scripts/install_deps.py` 的默认 Conda 前缀为：

- `/mnt/ext_drive/workspace/env/conda/torch2.4.1`

在 Windows 本机使用时，不能直接使用默认参数，否则不会安装到本机目标环境。

---

## 7. 建议的本机修正方式

### 7.1 GUI 依赖已补齐

当前已安装到目标环境：

- `PyQt6 6.10.2`
- `pyqtgraph 0.13.7`

本次使用的安装命令如下，可作为复现命令：

推荐直接安装到目标环境：

```powershell
conda run -p D:\env\condaEnv\torch2.4.1 python -m pip install "PyQt6>=6.5,<7" "pyqtgraph>=0.13,<0.14"
```

如果要使用仓库脚本，请显式传入 Windows 前缀，或在激活后的环境中执行 `--no-conda-run`：

```powershell
python workspace\scripts\install_deps.py --prefix D:\env\condaEnv\torch2.4.1
```

或：

```powershell
conda run -p D:\env\condaEnv\torch2.4.1 python workspace\scripts\install_deps.py --no-conda-run
```

### 7.2 将 YAML 修改为本机真实路径

建议替换为以下路径：

| 文件 | 建议值 |
|---|---|
| `workspace/config/default.yaml` | `data_simulator.hdf5_path: D:/Dataset/neuromotor/discrete_gestures/discrete_gestures_user_000_dataset_000.hdf5` |
| `workspace/config/debug_short.yaml` | `data_simulator.hdf5_path: D:/Dataset/neuromotor/discrete_gestures/discrete_gestures_user_000_dataset_000.hdf5` |
| `workspace/config/evaluation.yaml` | `hdf5_path: D:/Dataset/neuromotor/discrete_gestures/discrete_gestures_user_000_dataset_000.hdf5` |
| `workspace/config/evaluation_benchmark.yaml` | `hdf5_path: D:/Dataset/neuromotor/discrete_gestures/discrete_gestures_user_000_dataset_000.hdf5` |
| `workspace/config/training.yaml` | `data_location: D:/Dataset/neuromotor/discrete_gestures`；`split_csv: D:/Dataset/neuromotor/discrete_gestures/discrete_gestures_corpus.csv` |
| `workspace/config/training_debug.yaml` | `data_location: D:/Dataset/neuromotor/discrete_gestures`；`split_csv: D:/Dataset/neuromotor/discrete_gestures/discrete_gestures_corpus.csv` |

说明：`checkpoint_path` 当前使用相对路径 `../checkpoints/discrete_gestures/model_checkpoint.ckpt`，本机仓库中已存在对应文件，通常无需调整。

---

## 8. 最终判定

### 8.1 已满足项

1. Windows 11 + RTX 4060 Laptop GPU + 8 GB VRAM 环境满足测试清单硬件要求。
2. CUDA 驱动、PyTorch CUDA 版、GPU 可见性、自动设备选择均满足要求。
3. 核心 Python 依赖和 GUI 依赖均已齐备。
4. 数据集、CSV、checkpoint、模型配置文件均已具备。

### 8.2 未满足项

1. YAML 仍指向 Linux 数据路径，阶段 C / E / F / G 不能直接在本机执行。

### 8.3 总结

**当前本机“硬件条件、CUDA 运行环境、核心依赖和 GUI 依赖已具备”，但“测试文档要求的完整本机测试环境尚未完全具备”。**  
当前只需把 `workspace/config/*.yaml` 的数据路径切换到 `D:/Dataset/neuromotor/discrete_gestures`，即可达到 Windows RTX 4060 测试清单的环境要求。
