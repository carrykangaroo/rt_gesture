# RT-Gesture 测试清单 — 高性能 Linux 深度学习服务器

> 创建日期：2026-03-08  
> 目标：在高性能 GPU 服务器上进行完整训练、大规模评估、长时间稳定性验证和性能极限测试

---

## 1. 环境概要（参考配置）

| 项目 | 推荐最低值 |
|------|-----------|
| 操作系统 | Ubuntu 20.04/22.04 LTS |
| CPU | ≥ 16 核 (Xeon / EPYC) |
| 内存 | ≥ 64 GB |
| GPU | ≥ 1× NVIDIA A100/A6000/RTX 3090/4090 (≥ 24 GB VRAM) |
| CUDA 驱动 | ≥ 12.1（匹配 torch 2.4.1） |
| 磁盘 | ≥ 200 GB 可用 SSD（数据 + 训练产物） |
| 网络 | 可访问数据存储 / NFS |

---

## 2. 环境搭建

### 2.1 创建 Conda 环境

```bash
# 如果没有 Miniconda，先安装
# wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
# bash Miniconda3-latest-Linux-x86_64.sh

conda create -n rt_gesture python=3.12 -y
conda activate rt_gesture
```

### 2.2 安装 CUDA 版 PyTorch

```bash
# ⚠ 严禁使用 conda install，只用 pip
pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu124
```

### 2.3 安装项目依赖

```bash
cd <workspace_root>
pip install -e ".[dev]"
# 或
python scripts/install_deps.py
```

### 2.4 验证安装

```bash
python -c "
import torch, zmq, msgpack, h5py, yaml, numpy, pandas
print('All imports OK')
print(f'torch={torch.__version__}')
print(f'cuda={torch.cuda.is_available()}')
print(f'gpus={torch.cuda.device_count()}')
for i in range(torch.cuda.device_count()):
    print(f'  gpu[{i}]={torch.cuda.get_device_name(i)}, vram={torch.cuda.get_device_properties(i).total_mem / 1e9:.1f} GB')
"
```

### 2.5 部署数据集

```bash
# 确保 mini 和 full 数据集可用
ls -lh /path/to/data/mini/*.h5 | head -5
ls /path/to/data/full/*.h5 | wc -l  # 应为 100
```

### 2.6（可选）无头模式配置

> 服务器通常无显示器，GUI 测试需特殊处理：

```bash
# 方案 1: offscreen 后端（可验证窗口创建但无交互）
export QT_QPA_PLATFORM=offscreen

# 方案 2: Xvfb 虚拟显示（可验证完整交互）
sudo apt install -y xvfb
Xvfb :99 -screen 0 1920x1080x24 &
export DISPLAY=:99

# 方案 3: X11 forwarding（通过 SSH -X 连接，可在本地看到画面）
ssh -X user@server
```

---

## 3. 测试流程

### 阶段 A：环境与 GPU 验证

```bash
# A-1: nvidia-smi
nvidia-smi

# A-2: PyTorch GPU 验证
python -c "
import torch
print(f'cuda={torch.cuda.is_available()}')
print(f'device_count={torch.cuda.device_count()}')
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    print(f'  [{i}] {p.name}, {p.total_mem / 1e9:.1f} GB VRAM, SM {p.major}.{p.minor}')
# 快速计算测试
x = torch.randn(1024, 1024, device='cuda')
y = x @ x
print(f'matmul OK, result shape={y.shape}')
"

# A-3: auto_select_device
python -c "
from rt_gesture.utils import auto_select_device
dev = auto_select_device()
print(f'auto_device={dev}')
assert dev == 'cuda', f'Expected cuda, got {dev}'
print('OK')
"
```

| 检查项 | 预期结果 | 通过 |
|--------|---------|------|
| A-1: nvidia-smi 正常输出 | 显示 GPU 型号、驱动版本、CUDA 版本 | ☐ |
| A-2: `cuda_available=True` | — | ☐ |
| A-3: device_count ≥ 1 | — | ☐ |
| A-4: VRAM ≥ 24 GB | — | ☐ |
| A-5: CUDA matmul 计算正确 | `result shape=torch.Size([1024, 1024])` | ☐ |
| A-6: `auto_select_device()` → `"cuda"` | — | ☐ |
| A-7: 所有依赖正常导入 | `All imports OK` | ☐ |

---

### 阶段 B：自动化测试套件

```bash
# B-1: 全量测试
python -m pytest -q --tb=short

# B-2: benchmark 性能测试（GPU）
python -m pytest -v -m benchmark --tb=short

# B-3: CLER slow 测试（需真实数据）
RT_GESTURE_RUN_SLOW=1 python -m pytest -v -m slow --tb=long
```

| 检查项 | 预期结果 | 通过 |
|--------|---------|------|
| B-1: 全量通过 | `59 passed, 1 skipped` | ☐ |
| B-2: benchmark GPU 延迟 | `streaming_forward < 1ms` (A100 级别 ≈ 0.3ms) | ☐ |
| B-3: pipeline 端到端 | `pipeline_ms < 20ms` | ☐ |
| B-4: CLER 基线对齐 | `cler_abs_diff < 0.01` | ☐ |

---

### 阶段 C：实时后端功能验证（CLI）

```bash
# C-1: debug 短链路
python scripts/run_realtime.py --config config/debug_short.yaml

# C-2: 完整 mini 数据回放
python scripts/run_realtime.py --config config/default.yaml
```

| 检查项 | 预期结果 | 通过 |
|--------|---------|------|
| C-1: debug_short 正常退出 | `RT-Gesture shutdown complete` | ☐ |
| C-2: GESTURE 事件产出 | `Detected ... conf=...` | ☐ |
| C-3: 产物文件完整 | `runtime.log` + `events.jsonl` + `predictions.npz` | ☐ |
| C-4: GPU 推理 | 日志 `device=cuda` | ☐ |
| C-5: 延迟 header | events.jsonl 含 `transport_ms`/`infer_ms`/`pipeline_ms` | ☐ |
| C-6: 推理延迟 < 1ms | `infer_ms < 1` (高性能 GPU) | ☐ |
| C-7: ZMQ 端口释放 | `ss -tlnp | grep -E "555[5-8]"` 无残留 | ☐ |
| C-8: SIGTERM 处理 | `kill -SIGTERM <pid>` 后优雅退出 | ☐ |

---

### 阶段 D：GUI 功能验证（无头环境）

> 服务器无显示器时使用 Xvfb 或 `QT_QPA_PLATFORM=offscreen`：

```bash
# 使用 Xvfb
export DISPLAY=:99
python scripts/run_gui.py --config config/default.yaml &
sleep 10
# 截图验证
import -window root /tmp/gui_screenshot.png
kill %1

# 或 offscreen 模式（仅验证窗口不崩溃）
QT_QPA_PLATFORM=offscreen python scripts/run_gui.py --config config/debug_short.yaml
```

| 检查项 | 预期结果 | 通过 |
|--------|---------|------|
| D-1: GUI 创建不崩溃 | offscreen 模式无异常退出 | ☐ |
| D-2: Xvfb 截图 | 截图可见 MainWindow 布局 | ☐ |
| D-3: Start/Stop 正常 | 日志显示启动和停止序列 | ☐ |
| D-4: 远程 X11 forwarding（可选） | SSH -X 模式下 GUI 可交互 | ☐ |

---

### 阶段 E：完整训练（🔥 服务器重点项）

```bash
# E-1: 调试训练（1 epoch，验证管线）
python scripts/run_training.py --config config/training_debug.yaml

# E-2: 正式训练（250 epochs，GPU 加速）
# 预计耗时：RTX 3090 ~20min, A100 ~10min
python scripts/run_training.py --config config/training.yaml

# E-3: 监控 GPU 利用率（另一终端）
watch -n 1 nvidia-smi
```

| 检查项 | 预期结果 | 通过 |
|--------|---------|------|
| E-1: 调试训练完成 | 1 epoch, `device=cuda:0` | ☐ |
| E-2: **正式训练 250 epochs 完成** | 无崩溃，loss 收敛 | ☐ |
| E-3: GPU 利用率 | 训练期间 GPU Util > 50% | ☐ |
| E-4: `best.ckpt` 质量 | `best_val_multiclass_accuracy > 0.85` (参考值) | ☐ |
| E-5: `training_summary.json` | 含完整训练曲线数据 | ☐ |
| E-6: checkpoint 可加载 | 新训练的 best.ckpt 可被推理管道加载 | ☐ |
| E-7: 训练后推理验证 | 用新 checkpoint 运行 `run_realtime.py` | ☐ |

---

### 阶段 F：CLER 完整评估（🔥 服务器重点项）

```bash
# F-1: 标准 streaming CLER 评估
python scripts/run_evaluation.py --config config/evaluation.yaml

# F-2: full-recording 基准评估（用 GPU 加速，大内存支撑）
python scripts/run_evaluation.py --config config/evaluation_benchmark.yaml

# F-3: 新训练模型的评估
python scripts/run_evaluation.py --config config/evaluation.yaml \
  --checkpoint_path checkpoints/discrete_gestures/<timestamp>/best.ckpt
```

| 检查项 | 预期结果 | 通过 |
|--------|---------|------|
| F-1: streaming CLER 完成 | JSON 报告输出 | ☐ |
| F-2: full CLER 完成 | 无 OOM (≥64GB RAM + GPU chunk) | ☐ |
| F-3: **streaming ≈ full CLER** | `cler_abs_diff < 0.01` | ☐ |
| F-4: 新模型 CLER | CLER 在合理范围 (< 1.0) | ☐ |
| F-5: full 数据集评估 | 100 个 HDF5 文件全部处理 | ☐ |
| F-6: 评估报告保存 | `logs/evaluation_report.json` 生成 | ☐ |

---

### 阶段 G：长时间稳定性验证（🔥 服务器重点项）

```bash
# G-1: 3 分钟稳定性验证
python scripts/run_stability_validation.py \
  --config config/default.yaml \
  --duration-sec 180 \
  --report-path logs/stability_3min_report.json

# G-2: 30 分钟深度稳定性验证
python scripts/run_stability_validation.py \
  --config config/default.yaml \
  --duration-sec 1800 \
  --report-path logs/stability_30min_report.json

# G-3: GPU 内存监控（另一终端）
watch -n 5 "nvidia-smi --query-gpu=memory.used --format=csv,noheader"
```

| 检查项 | 预期结果 | 通过 |
|--------|---------|------|
| G-1: 3min 完成无崩溃 | `duration_observed_sec >= 178` | ☐ |
| G-2: **30min 完成无崩溃** | `duration_observed_sec >= 1798` | ☐ |
| G-3: 消息计数正常 | `counts.probabilities > 0` | ☐ |
| G-4: CPU 内存无泄漏 | `memory_stats.delta_mb < 100` (30min) | ☐ |
| G-5: **GPU 内存无泄漏** | VRAM 占用 30min 内波动 < 100MB | ☐ |
| G-6: 延迟 p95 | `pipeline_ms.p95 < 20ms` (GPU 环境) | ☐ |
| G-7: 延迟 p99 | `pipeline_ms.p99 < 30ms` (GPU 环境) | ☐ |
| G-8: 进程正常退出 | `inference_exit_code == 0`, `simulator_exit_code == 0` | ☐ |

---

### 阶段 H：高级场景与压力测试

| 检查项 | 操作方式 | 预期结果 | 通过 |
|--------|---------|---------|------|
| H-1: SIGTERM 信号 | `kill -15 <pid>` | 优雅退出，所有子进程终止 | ☐ |
| H-2: SIGKILL 恢复 | `kill -9 <pid>`，再次启动 | 新实例正常启动（端口可能需等待释放） | ☐ |
| H-3: NFS 路径数据读取 | 数据在 NFS 挂载点 | 读取正常，无卡死 | ☐ |
| H-4: 多用户并行 | 两个用户同时运行（不同端口） | 修改端口后两个实例互不干扰 | ☐ |
| H-5: CPU fallback | `CUDA_VISIBLE_DEVICES=""` 强制 CPU | 自动回退到 CPU 推理 | ☐ |
| H-6: GPU 切换 | `CUDA_VISIBLE_DEVICES=1` 指定第二块 GPU | 使用指定 GPU | ☐ |
| H-7: OOM 恢复 | 人为耗尽 GPU 内存后运行推理 | 报 CUDA OOM 错误，不崩溃 | ☐ |

---

## 4. 服务器特有注意事项

1. **无头测试**：服务器通常无显示器，GUI 测试使用 `Xvfb` 或 `QT_QPA_PLATFORM=offscreen`。offscreen 模式可验证窗口创建和后端逻辑，但无法验证视觉效果。
2. **完整训练**：这是唯一有足够算力完成 250 epoch 正式训练的平台。训练产出的 checkpoint 可复制到其他平台使用。
3. **Full 数据集评估**：100 个 HDF5 文件约 30GB+，确保磁盘空间和内存充足。`full_chunk_size` 在大内存服务器上可适当增大以加速。
4. **长时间稳定性**：服务器可运行 30min+ 稳定性测试，这是检测内存泄漏的关键条件。
5. **多 GPU**：如果有多卡，当前代码默认使用 `cuda:0`。可通过 `CUDA_VISIBLE_DEVICES` 环境变量选择 GPU。
6. **SSH 会话保持**：长时间训练/测试时使用 `tmux` 或 `screen` 防止 SSH 断开导致进程终止：
   ```bash
   tmux new -s rt_gesture
   # 在 tmux 内运行训练
   python scripts/run_training.py --config config/training.yaml
   # Ctrl+B, D 分离；tmux attach -t rt_gesture 重新连接
   ```
7. **GPU 温度监控**：长时间运行时监控温度：
   ```bash
   nvidia-smi --query-gpu=temperature.gpu --format=csv -l 10
   ```

---

## 5. 服务器产出物清单

完成所有测试后，以下产出物应从服务器取回：

| 产出物 | 路径 | 用途 |
|--------|------|------|
| 正式训练 checkpoint | `checkpoints/discrete_gestures/<timestamp>/best.ckpt` | 部署到其他平台 |
| 训练 summary | `checkpoints/.../training_summary.json` | 训练记录 |
| CLER 评估报告 | `logs/evaluation_report.json` | 精度基线 |
| 3min 稳定性报告 | `logs/stability_3min_report.json` | 短期稳定性记录 |
| 30min 稳定性报告 | `logs/stability_30min_report.json` | 长期稳定性记录 |

---

## 6. 测试结果汇总表

| 阶段 | 总项数 | 通过 | 失败 | 跳过 | 测试人 | 日期 |
|------|--------|------|------|------|--------|------|
| A: 环境与 GPU | 7 | | | | | |
| B: 自动化测试 | 4 | | | | | |
| C: 实时后端 | 8 | | | | | |
| D: GUI (无头) | 4 | | | | | |
| E: 完整训练 | 7 | | | | | |
| F: CLER 评估 | 6 | | | | | |
| G: 稳定性 | 8 | | | | | |
| H: 高级场景 | 7 | | | | | |
| **合计** | **51** | | | | | |
