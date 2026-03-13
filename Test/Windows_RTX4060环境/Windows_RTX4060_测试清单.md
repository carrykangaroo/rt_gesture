# RT-Gesture 测试清单 — Windows 笔记本 (RTX 4060 GPU)

> 创建日期：2026-03-08  
> 目标：在 Windows + CUDA GPU 环境下验证 CUDA 路径、跨平台兼容性和端到端功能

---

## 1. 环境概要（参考配置）

| 项目 | 推荐最低值 |
|------|-----------|
| 操作系统 | Windows 10/11 (64-bit) |
| CPU | Intel / AMD 移动处理器 |
| 内存 | ≥ 16 GB DDR5 |
| GPU | NVIDIA GeForce RTX 4060 Laptop (8 GB VRAM) |
| CUDA 驱动 | ≥ 12.1 (需匹配 torch 2.4.1 要求) |
| 磁盘 | ≥ 50 GB 可用空间（数据 + 模型 + 环境） |

---

## 2. 环境搭建

### 2.1 安装 Miniconda

```powershell
# 从 https://docs.conda.io/en/latest/miniconda.html 下载 Windows 64-bit 安装包
# 安装后打开 Anaconda PowerShell Prompt
conda --version
```

### 2.2 创建 Conda 环境

```powershell
# 创建 Python 3.12 环境
conda create -n rt_gesture python=3.12 -y
conda activate rt_gesture
```

### 2.3 安装 CUDA 版 PyTorch

```powershell
# ⚠ 严禁使用 conda install，只用 pip
pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu124
```

### 2.4 安装项目依赖

```powershell
cd <workspace_root>
pip install -e ".[dev]"
# 或使用安装脚本
python scripts/install_deps.py
```

### 2.5 验证安装

```powershell
python -c "import torch; print(f'torch={torch.__version__}, cuda={torch.cuda.is_available()}, device={torch.cuda.get_device_name(0)}')"
python -c "from PyQt6.QtWidgets import QApplication; print('PyQt6 OK')"
```

---

## 3. 测试流程

### 阶段 A：环境验证

```powershell
python -c "
import torch, zmq, msgpack, h5py, yaml, numpy, pandas
from PyQt6.QtWidgets import QApplication
import pyqtgraph
print('All imports OK')
print(f'torch={torch.__version__}')
print(f'cuda={torch.cuda.is_available()}')
print(f'gpu={torch.cuda.get_device_name(0)}')
print(f'vram={torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB')
"
```

| 检查项 | 预期结果 | 通过 |
|--------|---------|------|
| A-1: 所有核心依赖可导入 | `All imports OK` | ☐ |
| A-2: `torch.cuda.is_available()` 返回 `True` | `cuda=True` | ☐ |
| A-3: GPU 名称包含 "RTX 4060" | `gpu=NVIDIA GeForce RTX 4060 Laptop GPU` | ☐ |
| A-4: VRAM ≥ 7.5 GB | `vram>=7.5 GB` | ☐ |
| A-5: `auto_select_device()` 返回 `"cuda"` | 通过代码验证 | ☐ |

---

### 阶段 B：自动化测试套件

```powershell
# B-1: 全量快速测试
python -m pytest -q --tb=short

# B-2: verbose 模式
python -m pytest -v --tb=long

# B-3: 性能基准测试
python -m pytest -v -m benchmark --tb=short

# B-4: CLER slow 标记测试（需真实数据）
$env:RT_GESTURE_RUN_SLOW="1"
python -m pytest -v -m slow --tb=long
```

| 检查项 | 预期结果 | 通过 |
|--------|---------|------|
| B-1: 全量通过 | `59 passed, 1 skipped` (60 collected) | ☐ |
| B-2: 无 ERROR/FAILED | 全绿 | ☐ |
| B-3: benchmark GPU 延迟 | `streaming < 2ms` (GPU 显著优于 CPU 5ms) | ☐ |
| B-4: CLER 基线对齐 | `cler_abs_diff < 0.01` | ☐ |

---

### 阶段 C：实时后端功能验证（CLI）

> ⚠ **Windows 注意事项**：
> - Windows 不支持 `signal.SIGTERM`；代码已使用 `terminate()` + fallback `kill()` 处理，需验证此路径。
> - 路径使用 `\` 或 `/`，确保 config 中路径正确。
> - 建议将数据放在无空格、无中文的路径下（如 `D:\Data\`）。

```powershell
# C-1: debug 短链路
python scripts/run_realtime.py --config config/debug_short.yaml

# C-2: 完整 mini 数据回放
python scripts/run_realtime.py --config config/default.yaml
```

| 检查项 | 预期结果 | 通过 |
|--------|---------|------|
| C-1: debug_short 正常退出 | `RT-Gesture shutdown complete` 日志出现 | ☐ |
| C-2: 有 GESTURE 事件 | 日志中出现 `Detected ... conf=...` | ☐ |
| C-3: 产物完整 | `runtime.log` + `events.jsonl` + `predictions.npz` 生成 | ☐ |
| C-4: 进程清理（Windows） | `Get-Process *python*` 无残留 RT-Gesture 进程 | ☐ |
| C-5: 端口释放 | `netstat -ano | findstr "555"` 无占用 | ☐ |
| C-6: GPU 推理 | 日志中显示 `device=cuda` | ☐ |
| C-7: 延迟 header | `events.jsonl` 中含 `transport_ms`/`infer_ms`/`pipeline_ms` | ☐ |
| C-8: Windows 信号处理 | Ctrl+C 后优雅退出，无弹窗错误 | ☐ |

---

### 阶段 D：GUI 功能验证

```powershell
python scripts/run_gui.py --config config/default.yaml
```

| 检查项 | 预期结果 | 通过 |
|--------|---------|------|
| D-1: GUI 窗口正常显示 | MainWindow 打开，DPI 缩放正确（无模糊） | ☐ |
| D-2: 配置面板完整 | HDF5 路径、Checkpoint 路径、Device 选择器、阈值滑块 | ☐ |
| D-3: Device 选择器 | 下拉菜单含 `auto`/`cpu`/`cuda` 三个选项 | ☐ |
| D-4: Start System | 状态栏 `Running`，波形开始 | ☐ |
| D-5: EMG 16ch 波形 | 可见信号波动 | ☐ |
| D-6: 概率 Bars 视图 | 9 类手势概率条有变化 | ☐ |
| D-7: 概率 Heatmap 视图 | 热力图有颜色变化 | ☐ |
| D-8: 事件列表 | 检测到手势时有新条目 | ☐ |
| D-9: FPS/延迟/心跳 | 状态栏 FPS > 0, 延迟 < 15ms (GPU), 心跳 ok | ☐ |
| D-10: Pause/Resume | 暂停/恢复操作正常 | ☐ |
| D-11: Stop System | 停止后状态 `Stopped`，波形停止 | ☐ |
| D-12: DPI 适配 | 高分辨率屏幕字体/控件不模糊 | ☐ |
| D-13: 窗口关闭 | 关闭后所有子进程退出 | ☐ |

---

### 阶段 E：训练功能验证

```powershell
# E-1: 调试训练（1 epoch, GPU）
python scripts/run_training.py --config config/training_debug.yaml

# E-2（可选）: 正式训练（250 epochs, GPU, ~30-60min on RTX 4060）
python scripts/run_training.py --config config/training.yaml
```

| 检查项 | 预期结果 | 通过 |
|--------|---------|------|
| E-1: 调试训练完成 | 1 epoch 完成，GPU 被使用 (`cuda:0`) | ☐ |
| E-2: GPU 利用率 | 任务管理器→性能→GPU 显示利用率 > 0% | ☐ |
| E-3: checkpoint 产出 | `best.ckpt` + `last.ckpt` 生成 | ☐ |
| E-4: summary 产出 | `training_summary.json` 含 `best_val_loss` | ☐ |
| E-5: Windows 路径兼容 | checkpoint 路径使用 `\` 或 `/` 均能加载 | ☐ |

---

### 阶段 F：CLER 评估验证

```powershell
# F-1: 标准评估
python scripts/run_evaluation.py --config config/evaluation.yaml

# F-2: full recording 基准评估（GPU 加速）
python scripts/run_evaluation.py --config config/evaluation_benchmark.yaml
```

| 检查项 | 预期结果 | 通过 |
|--------|---------|------|
| F-1: 评估完成 | JSON 报告输出 | ☐ |
| F-2: full CLER ≈ streaming CLER | `cler_abs_diff < 0.01` | ☐ |
| F-3: GPU 加速评估 | 评估耗时 < CPU 的 1/3 | ☐ |
| F-4: 无 CUDA OOM | 8GB VRAM 足够完成评估 | ☐ |

---

### 阶段 G：稳定性验证

```powershell
python scripts/run_stability_validation.py `
  --config config/default.yaml `
  --duration-sec 180 `
  --report-path logs/stability_3min_report.json
```

| 检查项 | 预期结果 | 通过 |
|--------|---------|------|
| G-1: 完整运行 180s | `duration_observed_sec >= 178` | ☐ |
| G-2: 消息计数 | `counts.probabilities > 0`, `counts.heartbeat > 0` | ☐ |
| G-3: 内存无泄漏 | `memory_stats.delta_mb < 50` | ☐ |
| G-4: GPU 延迟 p95 | `latency_stats.pipeline_ms.p95 < 30` (GPU 环境) | ☐ |
| G-5: GPU 内存稳定 | VRAM 占用无持续增长 | ☐ |
| G-6: 进程退出 | 所有子进程正常退出 | ☐ |

---

### 阶段 H：Windows 特有场景验证

| 检查项 | 操作方式 | 预期结果 | 通过 |
|--------|---------|---------|------|
| H-1: Ctrl+C 中断 | 运行 run_realtime.py 后按 Ctrl+C | 优雅退出（Windows 无 SIGTERM 走 fallback 路径） | ☐ |
| H-2: 中文/空格路径 | 数据放在含中文的目录下测试 | 正常读取或报可读错误 | ☐ |
| H-3: 反斜杠路径 | config 中使用 `D:\Data\test.h5` | 路径正确解析 | ☐ |
| H-4: GPU ↔ CPU 切换 | Device 设为 `cpu`，验证 CPU 路径在有 GPU 的机器上也工作 | 运行正常 | ☐ |
| H-5: 多显示器 DPI | 拖动 GUI 窗口到不同 DPI 的显示器 | 无崩溃，布局合理 | ☐ |
| H-6: 防火墙 ZMQ | Windows 防火墙弹窗出现 | 允许后通信正常 | ☐ |
| H-7: 端口冲突 | 先占用 5555 端口再启动 | 启动失败报 ZMQ bind error | ☐ |
| H-8: 长时间待机唤醒 | 笔记本合盖待机后恢复 | 管道需要重启（预期行为） | ☐ |

---

## 4. Windows 特有注意事项

1. **CUDA 驱动匹配**：RTX 4060 Laptop GPU 需安装 ≥ 528.xx 驱动（支持 CUDA 12.x）。通过 `nvidia-smi` 确认驱动版本。
2. **信号处理差异**：Windows 不支持 `SIGTERM`，代码通过 `process.terminate()` + 超时 `process.kill()` 的 fallback 路径处理。测试时需重点关注 H-1 和 C-8。
3. **路径分隔符**：Windows 路径使用 `\`，但 Python 的 `pathlib.Path` 已在代码中统一处理。仍需测试带空格和中文字符的路径。
4. **防火墙**：ZMQ 使用 TCP 端口 5555-5558，Windows 防火墙可能拦截。首次运行时需允许通过。
5. **DPI 缩放**：4K 笔记本屏幕 + 150%/200% DPI 缩放下，PyQt6 的 `AA_EnableHighDpiScaling` 已默认开启（Qt6 内置），但需目测 UI 元素是否正常。
6. **GPU 内存**：RTX 4060 Laptop 有 8GB VRAM，推理模型 (~50MB) 仅占很小比例，训练时 batch_size 可适当增大。
7. **性能基线**：RTX 4060 的推理延迟预期 < 1ms，整体管道延迟 < 15ms (对比 CPU 的 ~50ms)。

---

## 5. 测试结果汇总表

| 阶段 | 总项数 | 通过 | 失败 | 跳过 | 测试人 | 日期 |
|------|--------|------|------|------|--------|------|
| A: 环境验证 | 5 | | | | | |
| B: 自动化测试 | 4 | | | | | |
| C: 实时后端 | 8 | | | | | |
| D: GUI 功能 | 13 | | | | | |
| E: 训练功能 | 5 | | | | | |
| F: CLER 评估 | 4 | | | | | |
| G: 稳定性 | 6 | | | | | |
| H: Windows 特有 | 8 | | | | | |
| **合计** | **53** | | | | | |
