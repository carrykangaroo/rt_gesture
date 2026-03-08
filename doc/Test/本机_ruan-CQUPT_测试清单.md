# RT-Gesture 测试清单 — 本机 ruan-CQUPT（CPU-only 开发机）

> 创建日期：2026-03-08  
> 目标：在主开发机上验证全部 CPU 路径的功能可靠性和完整性

---

## 1. 环境概要

| 项目 | 值 |
|------|-----|
| 主机名 | `ruan-CQUPT` |
| 操作系统 | Ubuntu 24.04.4 LTS (Noble), kernel 6.17.0-14-generic |
| CPU | Intel Core i5-10400 @ 2.90GHz, 6 核 12 线程 |
| 内存 | 16 GB DDR4 |
| GPU | Intel UHD Graphics 630 (集显, **无 NVIDIA GPU**) |
| 磁盘 | `/mnt/ext_drive` 295 GB (14% used) |
| Conda 环境 | `/mnt/ext_drive/workspace/env/conda/torch2.4.1` |
| Python | 3.12.7 |
| PyTorch | 2.4.1 (compiled with CUDA 12.4, **运行时 `cuda_available=False`**) |
| 关键包 | PyQt6=6.10.2, pyqtgraph=0.13.7, pyzmq=27.1.0, numba=0.60.0 |

### 数据资源

| 资源 | 路径 | 状态 |
|------|------|------|
| Mini HDF5 (3 文件, 1.1GB) | `/mnt/data/Dataset/.../Discrete Gestures/mini/` | ✅ 可用 |
| Full HDF5 (100 文件) | `/mnt/data/Dataset/.../Discrete Gestures/full/` | ✅ 可用 |
| 预训练 Checkpoint | `workspace/checkpoints/discrete_gestures/model_checkpoint.ckpt` | ✅ 可用 |

### 本机限制

- **无 CUDA 加速**：所有推理/训练均在 CPU 上运行，延迟指标仅作 CPU 基线参考
- **内存 16GB**：full-recording CLER 评估需使用 `full_chunk_size` 分块避免 OOM
- **CPU 6 核**：多进程测试（DataSimulator + InferenceEngine + GUI）共享有限核心

---

## 2. 测试流程

### 阶段 A：环境验证

```bash
# A-1: 确认 conda 环境激活正常
cd /mnt/ext_drive/workspace/Your\ WorkSpace/sEMG/CTRL_lab/imporve-generic-neuromotor-interface/workspace
conda run -p /mnt/ext_drive/workspace/env/conda/torch2.4.1 python -c "
import torch, zmq, msgpack, h5py, yaml, numpy, pandas, numba
from PyQt6.QtWidgets import QApplication
import pyqtgraph
print('All imports OK')
print(f'torch={torch.__version__}, cuda={torch.cuda.is_available()}')
"
```

| 检查项 | 预期结果 | 通过 |
|--------|---------|------|
| A-1: 所有核心依赖可导入 | `All imports OK` | ☐ |
| A-2: `torch.cuda.is_available()` 返回 `False` | `cuda=False` | ☐ |
| A-3: `auto_select_device()` 返回 `"cpu"` | — | ☐ |

---

### 阶段 B：自动化测试套件

```bash
# B-1: 全量快速测试
conda run -p /mnt/ext_drive/workspace/env/conda/torch2.4.1 \
  python -m pytest -q --tb=short 2>&1

# B-2: verbose 模式（排查失败）
conda run -p /mnt/ext_drive/workspace/env/conda/torch2.4.1 \
  python -m pytest -v --tb=long 2>&1

# B-3: 性能基准测试
conda run -p /mnt/ext_drive/workspace/env/conda/torch2.4.1 \
  python -m pytest -v -m benchmark --tb=short 2>&1

# B-4: CLER 基线对齐（slow 标记，需要真实数据）
RT_GESTURE_RUN_SLOW=1 conda run -p /mnt/ext_drive/workspace/env/conda/torch2.4.1 \
  python -m pytest -v -m slow --tb=long 2>&1
```

| 检查项 | 预期结果 | 通过 |
|--------|---------|------|
| B-1: 全量通过 | `59 passed, 1 skipped` (60 collected) | ☐ |
| B-2: 无 ERROR/FAILED | 全绿 | ☐ |
| B-3: benchmark 4 用例全通过 | streaming < 5ms, event_detector < 1ms, zmq < 2ms, pipeline < 50ms (CPU 基线) | ☐ |
| B-4: CLER 基线 slow 测试通过 | full_cler ≈ streaming_cler (abs_diff < 0.01) | ☐ |

---

### 阶段 C：实时后端功能验证（CLI）

```bash
# C-1: 使用 debug 短链路配置（限制消息数，快速结束）
conda run -p /mnt/ext_drive/workspace/env/conda/torch2.4.1 \
  python scripts/run_realtime.py --config config/debug_short.yaml 2>&1

# C-2: 使用 default 配置运行 mini 数据（完整回放）
conda run -p /mnt/ext_drive/workspace/env/conda/torch2.4.1 \
  python scripts/run_realtime.py --config config/default.yaml 2>&1
```

| 检查项 | 预期结果 | 通过 |
|--------|---------|------|
| C-1: debug_short 运行正常退出 | `RT-Gesture shutdown complete` 日志出现，exit code 0 | ☐ |
| C-2: 有 GESTURE 事件输出 | 日志中出现 `Detected ... conf=...` | ☐ |
| C-3: logs 目录产物完整 | `runtime.log` + `events.jsonl` + `predictions.npz` 均生成 | ☐ |
| C-4: 进程优雅退出 | 无僵尸进程残留（`ps aux | grep rt_gesture` 无结果） | ☐ |
| C-5: ZMQ 端口释放 | `ss -tlnp | grep -E "555[5-8]"` 无残留 | ☐ |
| C-6: HEARTBEAT 日志 | 推理端日志中出现 heartbeat 发送记录 | ☐ |
| C-7: 延迟 header 字段 | `events.jsonl` 中事件含 `transport_ms`、`infer_ms`、`pipeline_ms` | ☐ |

---

### 阶段 D：GUI 功能验证

> 注意：本机有显示器，可直接运行 GUI。若为远程 SSH 则需 X11 forwarding 或 VNC。

```bash
# D-1: 启动 GUI
conda run -p /mnt/ext_drive/workspace/env/conda/torch2.4.1 \
  python scripts/run_gui.py --config config/default.yaml
```

| 检查项 | 预期结果 | 通过 |
|--------|---------|------|
| D-1: GUI 窗口正常显示 | MainWindow 打开，含 EMG 波形区域 + 概率展示 + 配置面板 | ☐ |
| D-2: 配置面板完整 | HDF5 路径、Checkpoint 路径、Device 选择器（auto/cpu/cuda）、阈值滑块 | ☐ |
| D-3: Start System 启动后端 | 状态栏显示 `Running`，波形开始滚动 | ☐ |
| D-4: EMG 16ch 波形实时更新 | 可见信号波动，非全黑/全白 | ☐ |
| D-5: 概率 Bars 视图正常 | 9 类手势概率条有变化 | ☐ |
| D-6: 概率 Heatmap 视图正常 | 切换到 Heatmap tab，热力图有颜色变化 | ☐ |
| D-7: 事件列表更新 | 检测到手势时事件列表新增条目 | ☐ |
| D-8: GT 列表正常 | Ground-truth prompts 出现在 GT 列表中 | ☐ |
| D-9: FPS/延迟/心跳状态 | 状态栏 FPS > 0, 延迟显示数值, 心跳 `ok` | ☐ |
| D-10: Pause/Resume | 暂停后波形冻结，恢复后继续更新 | ☐ |
| D-11: Stop System | 后端停止，状态栏 `Stopped`，波形停止 | ☐ |
| D-12: 窗口关闭 | 关闭窗口后所有子进程退出，端口释放 | ☐ |

---

### 阶段 E：训练功能验证

```bash
# E-1: 调试训练（1 epoch，快速验证管线）
conda run -p /mnt/ext_drive/workspace/env/conda/torch2.4.1 \
  python scripts/run_training.py --config config/training_debug.yaml 2>&1

# E-2（可选）: 正式训练（250 epochs，耗时较长）
conda run -p /mnt/ext_drive/workspace/env/conda/torch2.4.1 \
  python scripts/run_training.py --config config/training.yaml 2>&1
```

| 检查项 | 预期结果 | 通过 |
|--------|---------|------|
| E-1: 调试训练完成 | 1 epoch 完成，无异常退出 | ☐ |
| E-2: checkpoint 产出 | `checkpoints/discrete_gestures/<timestamp>/best.ckpt` + `last.ckpt` | ☐ |
| E-3: summary 产出 | `training_summary.json` 含 `best_val_loss` 和 `best_val_multiclass_accuracy` | ☐ |
| E-4: 日志中有 multiclass_acc | 训练日志含 `train_mc_acc` 和 `val_mc_acc` 字段 | ☐ |
| E-5: 训练后 checkpoint 可加载 | `load_model_from_lightning_checkpoint(best.ckpt)` 不报错 | ☐ |

---

### 阶段 F：CLER 评估验证

```bash
# F-1: 标准评估
conda run -p /mnt/ext_drive/workspace/env/conda/torch2.4.1 \
  python scripts/run_evaluation.py --config config/evaluation.yaml 2>&1

# F-2: full recording 基准评估
conda run -p /mnt/ext_drive/workspace/env/conda/torch2.4.1 \
  python scripts/run_evaluation.py --config config/evaluation_benchmark.yaml 2>&1
```

| 检查项 | 预期结果 | 通过 |
|--------|---------|------|
| F-1: 评估正常完成 | JSON 报告输出到终端 | ☐ |
| F-2: full CLER 与 streaming CLER 一致 | `cler_abs_diff < 0.01` | ☐ |
| F-3: 报告文件生成 | `logs/evaluation_report.json` 存在 | ☐ |
| F-4: full_chunk_size 分块无 OOM | CPU 16GB 内存内完成，无 MemoryError | ☐ |

---

### 阶段 G：稳定性验证

```bash
# G-1: 3 分钟稳定性跑
conda run -p /mnt/ext_drive/workspace/env/conda/torch2.4.1 \
  python scripts/run_stability_validation.py \
  --config config/default.yaml \
  --duration-sec 180 \
  --report-path logs/stability_3min_report.json 2>&1
```

| 检查项 | 预期结果 | 通过 |
|--------|---------|------|
| G-1: 完整运行 180s 无崩溃 | `duration_observed_sec >= 178` | ☐ |
| G-2: 消息计数正常 | `counts.probabilities > 0` 且 `counts.heartbeat > 0` | ☐ |
| G-3: 内存无明显泄漏 | `memory_stats.delta_mb < 50` (3 分钟内 RSS 增长 < 50MB) | ☐ |
| G-4: 延迟 p95 在预算内 | `latency_stats.pipeline_ms.p95 < 80` (CPU 环境) | ☐ |
| G-5: 进程正常退出 | `inference_exit_code == 0`, `simulator_exit_code == 0` | ☐ |
| G-6: 报告文件 | `logs/stability_3min_report.json` 生成 | ☐ |

---

### 阶段 H：异常场景验证

| 检查项 | 操作方式 | 预期结果 | 通过 |
|--------|---------|---------|------|
| H-1: Ctrl+C 中断 | 运行 `run_realtime.py` 后按 Ctrl+C | 优雅退出，无僵尸进程 | ☐ |
| H-2: 无 HDF5 时启动推理 | config 中 `hdf5_path` 指向不存在的文件 | 报错退出，不崩溃 | ☐ |
| H-3: 无 Checkpoint 时推理 | config 中 `checkpoint_path` 为空 | 使用随机权重运行（warning 日志） | ☐ |
| H-4: 端口冲突 | 先占用 5555 端口，再启动系统 | 启动失败报 ZMQ bind error | ☐ |

---

## 3. 本机特有注意事项

1. **CPU 延迟基线**：本机无 GPU，`forward_streaming` 延迟约 2-5ms，benchmark 测试的延迟预算是按 CPU 基线设定的。
2. **内存瓶颈**：16GB RAM 运行 full-recording 评估时，需确保 `full_chunk_size` 参数足够小（evaluation_benchmark.yaml 默认 16000 已调优）。
3. **GUI 测试**：本机有物理显示器，GUI 测试可直接进行；若远程 SSH 需设置 `QT_QPA_PLATFORM=offscreen` 或 X11 forwarding。
4. **数据路径**：本机数据在 `/mnt/data/Dataset/`，与配置文件中的默认路径一致，无需修改。

---

## 4. 测试结果汇总表

| 阶段 | 总项数 | 通过 | 失败 | 跳过 | 测试人 | 日期 |
|------|--------|------|------|------|--------|------|
| A: 环境验证 | 3 | | | | | |
| B: 自动化测试 | 4 | | | | | |
| C: 实时后端 | 7 | | | | | |
| D: GUI 功能 | 12 | | | | | |
| E: 训练功能 | 5 | | | | | |
| F: CLER 评估 | 4 | | | | | |
| G: 稳定性 | 6 | | | | | |
| H: 异常场景 | 4 | | | | | |
| **合计** | **45** | | | | | |
