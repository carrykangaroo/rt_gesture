# RT-Gesture 完整流程功能文档（优化后）

> 文档类型：实现评审文档  
> 适用目录：`workspace/`  
> 更新日期：2026-03-09  
> 对应需求文档：`doc/实时离散手势识别系统需求规格说明书.md`（v0.3）  
> 对应技术文档：`doc/技术实现文档.md`（v1.3）

---

## 1. 文档目标

本文件描述当前代码版本的完整流程能力，并同步 2026-03-07 完成的优化项（O-01~O-13，N-01~N-11）。

覆盖范围：

1. 实时主链路（数据模拟 -> 流式推理 -> 事件输出）。
2. GUI 监控链路（后端进程控制 + 可视化 + 心跳监控）。
3. 训练与评估链路（训练、CLER 对齐评估）。
4. 可观测性、日志产物、测试与当前验收状态。

---

## 2. 当前项目结构（workspace）

```text
workspace/
├── config/
│   ├── default.yaml
│   ├── debug_short.yaml
│   ├── training.yaml
│   ├── training_debug.yaml
│   ├── evaluation.yaml
│   └── evaluation_benchmark.yaml
├── checkpoints/
│   └── .gitkeep
├── logs/
│   └── .gitkeep
├── rt_gesture/
│   ├── constants.py
│   ├── config.py
│   ├── networks.py
│   ├── checkpoint_utils.py
│   ├── event_detector.py
│   ├── zmq_transport.py
│   ├── data_simulator.py
│   ├── inference_engine.py
│   ├── logger.py
│   ├── main.py
│   ├── data.py
│   ├── transforms.py
│   ├── cler.py
│   ├── train.py
│   ├── evaluate.py
│   ├── gui/
│   │   ├── main_window.py
│   │   ├── process_manager.py
│   │   ├── emg_plot_widget.py
│   │   ├── gesture_display.py
│   │   ├── config_panel.py
│   │   └── status_bar.py
│   └── tests/
│       ├── conftest.py
│       └── test_*.py
├── scripts/
│   ├── run_realtime.py
│   ├── run_gui.py
│   ├── run_training.py
│   ├── run_evaluation.py
│   ├── run_stability_validation.py
│   └── install_deps.py
├── requirements.txt
├── environment.yml
├── CHANGELOG.md
└── README.md
```

---

## 3. 关键优化已落地能力

1. HEARTBEAT 机制：推理端周期发送 `HEARTBEAT`；GUI 监听并做超时告警。
2. Python 回调接口：`InferenceEngine.register_event_callback()` / `unregister_event_callback()`。
3. 延迟可观测性：消息头统一携带 `transport_ms`、`infer_ms`、`post_ms`、`pipeline_ms`。
4. 延迟阈值告警：推理端对 transport/infer/post/pipeline 分阶段告警。
5. 训练指标补齐：训练流程新增 `MulticlassAccuracy` 并写入 metrics/summary。
6. GUI 增强：配置面板新增设备选择器，概率展示新增 Heatmap 视图，支持 GT prompt 可视化列表。
7. 配置增强：新增 `training_debug.yaml`，`training.yaml` 默认 `max_epochs=250`，checkpoint 路径切换到 `workspace/checkpoints/`。
8. CLER 评估优化：`evaluate.py` 新增 `full_chunk_size` 分块 full-forward，解决 CPU OOM。
9. 稳定性验证脚本：新增 `scripts/run_stability_validation.py`，自动输出延迟/内存趋势报告。
10. 文档与工程同步：README、CHANGELOG、需求/技术文档版本与约束已回刷。
11. checkpoint 兼容增强：`checkpoint_utils.py` 新增 legacy 模块别名映射，兼容历史 `generic_neuromotor_interface.*` 序列化路径。
12. 训练数据路径兼容：`data.py` 支持 split CSV 中已带 `.hdf5` 后缀的数据集名，避免重复拼接扩展名导致的文件找不到。

---

## 4. 全流程说明

### 4.1 流程 A：实时后端（CLI）

入口命令：

```bash
cd workspace
conda run -p /mnt/ext_drive/workspace/env/conda/torch2.4.1 \
  python scripts/run_realtime.py --config config/default.yaml
```

执行时序：

1. `main.py` 加载配置并创建本次 `logs/<timestamp>/` 运行目录。
2. 启动 `InferenceEngine` 进程，再启动 `DataSimulator` 进程。
3. `DataSimulator` 读取 HDF5，按 2kHz 节奏分块发布 `EMG_CHUNK`；可选发布 `GROUND_TRUTH`。
4. `InferenceEngine` 订阅 EMG 数据并执行 `forward_streaming`，维护 `conv_history + lstm_state`。
5. `EventDetector` 执行静息抑制、阈值上穿、拒识和去抖，输出 `GestureEvent`。
6. 推理端发布 `PROBABILITIES`、`GESTURE_EVENT`、`HEARTBEAT`，并写入 `events.jsonl` / `predictions.npz`。
7. 数据结束后主进程发送 `SHUTDOWN`，推理进程退出并清理资源；超时则强制终止。

### 4.2 流程 B：GUI 实时监控

入口命令：

```bash
cd workspace
conda run -p /mnt/ext_drive/workspace/env/conda/torch2.4.1 \
  python scripts/run_gui.py --config config/default.yaml
```

执行时序：

1. 点击 `Start System` 后，GUI 将面板参数写回 YAML。
2. `ProcessManager` 启动 `python -m rt_gesture.main`。
3. `ZmqReaderThread` 订阅 EMG/GT/结果流，驱动波形、概率条/热力图、事件列表与 GT 列表更新。
4. 状态栏显示 FPS、延迟、设备和心跳状态。
5. 心跳状态流转：`--` -> `waiting` -> `ok`；超时后显示 `timeout` 并标记 `Heartbeat timeout`。
6. 点击 `Stop System` 后发送 `SHUTDOWN`，等待后端退出，超时则 terminate/kill。

### 4.3 流程 C：离线训练

入口命令（正式训练）：

```bash
cd workspace
conda run -p /mnt/ext_drive/workspace/env/conda/torch2.4.1 \
  python scripts/run_training.py --config config/training.yaml
```

入口命令（快速调试）：

```bash
cd workspace
conda run -p /mnt/ext_drive/workspace/env/conda/torch2.4.1 \
  python scripts/run_training.py --config config/training_debug.yaml
```

执行步骤：

1. 加载 `TrainingConfig` 并构建 train/val dataloader。
2. 使用 `DiscreteGesturesArchitecture + BCEWithLogitsLoss`。
3. `FingerStateMaskGenerator` 对 release 类别进行掩码加权。
4. 记录 `frame_accuracy` 与 `multiclass_accuracy`。
5. 每轮写 `last.ckpt`，按最优 `val_loss` 更新 `best.ckpt`。
6. 输出 `training_summary.json`（含 `best_val_multiclass_accuracy`）。

### 4.4 流程 D：CLER 一致性评估

入口命令：

```bash
cd workspace
conda run -p /mnt/ext_drive/workspace/env/conda/torch2.4.1 \
  python scripts/run_evaluation.py --config config/evaluation.yaml
```

执行步骤：

1. 加载 checkpoint 与评估 HDF5。
2. 分别执行分块 `run_full_forward(full_chunk_size)` 与 `run_streaming_forward(chunk_size)`。
3. 计算 full/streaming CLER 及 `cler_abs_diff`。
4. 输出 `logs/evaluation_report.json`。

---

## 5. 消息与可观测性

### 5.1 默认端口

1. EMG：`5555`
2. Ground-truth：`5556`
3. Result：`5557`
4. Control：`5558`

### 5.2 消息类型

1. `emg_chunk`
2. `ground_truth`
3. `probabilities`
4. `gesture_event`
5. `heartbeat`
6. `shutdown`

### 5.3 关键 Header 字段

1. 协议基础：`version`, `seq`, `timestamp`, `msg_type`。
2. 概率消息：`time_start`, `time_end`, `time_start_rel`, `time_end_rel`, `transport_ms`, `infer_ms`, `pipeline_ms`。
3. 事件消息：`gesture`, `event_time`, `event_time_rel`, `confidence`, `transport_ms`, `infer_ms`, `post_ms`, `pipeline_ms`。
4. 心跳消息：`device`。

### 5.4 延迟阈值配置

`inference` 配置项：

1. `latency_warn_transport_ms`
2. `latency_warn_infer_ms`
3. `latency_warn_post_ms`
4. `latency_warn_pipeline_ms`

超过阈值时推理端记录 `warning`。

---

## 6. 配置体系

### 6.1 运行配置

1. `config/default.yaml`：标准运行配置。
2. `config/debug_short.yaml`：短链路调试配置（限制 `max_chunks` / `max_messages`）。

### 6.2 训练配置

1. `config/training.yaml`：正式训练（默认 `max_epochs: 250`）。
2. `config/training_debug.yaml`：调试训练（默认 `max_epochs: 1`，限制 step）。

### 6.3 评估配置

1. `config/evaluation.yaml`：实时 chunk 对齐评估配置（`chunk_size=40`）。
2. `config/evaluation_benchmark.yaml`：full recording 基准配置（较大 chunk，避免 CPU 长时运行）。
3. `full_chunk_size` 用于控制 full-forward 分块大小，避免单次前向触发 OOM。
4. `evaluate.py` 会将 `checkpoint_path`、`hdf5_path`、`report_path` 按配置文件目录解析为绝对路径。

---

## 7. 运行产物

### 7.1 实时运行

1. `logs/<timestamp>/runtime.log`
2. `logs/<timestamp>/events.jsonl`
3. `logs/<timestamp>/predictions.npz`

### 7.2 训练运行

1. `checkpoints/<run>/<timestamp>/best.ckpt`
2. `checkpoints/<run>/<timestamp>/last.ckpt`
3. `checkpoints/<run>/<timestamp>/training_config.yaml`
4. `checkpoints/<run>/<timestamp>/training_summary.json`

### 7.3 评估运行

1. `logs/evaluation_report.json`
2. `logs/evaluation_benchmark_report.json`
3. `logs/stability_3min_report.json`

---

## 8. 测试覆盖与最新结果

### 8.1 覆盖范围

1. 常量与配置：`test_constants.py`, `test_config.py`
2. 模型与权重：`test_networks.py`, `test_checkpoint_utils.py`
3. 事件检测：`test_event_detector.py`
4. 通信与数据：`test_zmq_transport.py`, `test_data_simulator.py`
5. 推理与集成：`test_inference.py`, `test_integration.py`
6. GUI：`test_gui.py`, `test_process_manager.py`
7. 训练与评估：`test_training.py`, `test_evaluate.py`, `test_cler_wrapper.py`, `test_transforms.py`
8. 日志：`test_logger.py`
9. 性能基准：`test_benchmark.py`

### 8.2 最近一次全量结果（2026-03-09）

1. 命令：`/home/rxb/.conda/envs/torch2.0.1/bin/python -m pytest -q --tb=short`
2. 结果：`58 passed, 1 skipped, 1 failed`
3. 失败项：`rt_gesture/tests/test_benchmark.py::test_event_detector_latency_budget`
4. 补充执行：
   - `-m benchmark`：`2 failed, 2 passed`
   - `RT_GESTURE_RUN_SLOW=1 -m slow`：`1 passed`

### 8.3 端到端实测结果（2026-03-09）

1. 评估报告：`logs/evaluation_report.json`，`cler_abs_diff=0.0002876879589208403`。
2. Full 基准报告：`logs/evaluation_benchmark_report.json`，`cler_abs_diff=0.0`。
3. 稳定性报告：`logs/stability_3min_report.json`，`duration_observed_sec=181.94`，`pipeline_ms.p95=1.711`。
4. 实时链路产物示例：`logs/2026-03-09_12-45-14/`（包含 `runtime.log` / `events.jsonl` / `predictions.npz`）。

---

## 9. 优化清单对应状态

### 9.1 已完成

1. O-01 ~ O-13
2. N-01 ~ N-11（含 full recording CLER 基准与 180s 稳定性实跑）

### 9.2 待外部环境验证

1. N-09：Windows + CUDA 实机验收。
2. N-12：full 数据集 250 epoch 正式训练与产物验证。

---

## 10. 常用命令

```bash
# 全量测试
cd workspace
/home/rxb/.conda/envs/torch2.0.1/bin/python -m pytest -q

# 仅 benchmark
/home/rxb/.conda/envs/torch2.0.1/bin/python -m pytest -q -m benchmark

# 实时后端
/home/rxb/.conda/envs/torch2.0.1/bin/python scripts/run_realtime.py --config config/default.yaml

# GUI
/home/rxb/.conda/envs/torch2.0.1/bin/python scripts/run_gui.py --config config/default.yaml

# 正式训练
/home/rxb/.conda/envs/torch2.0.1/bin/python scripts/run_training.py --config config/training.yaml

# 调试训练
/home/rxb/.conda/envs/torch2.0.1/bin/python scripts/run_training.py --config config/training_debug.yaml

# 评估
/home/rxb/.conda/envs/torch2.0.1/bin/python scripts/run_evaluation.py --config config/evaluation.yaml

# full recording CLER 基准
/home/rxb/.conda/envs/torch2.0.1/bin/python scripts/run_evaluation.py --config config/evaluation_benchmark.yaml

# 3 分钟稳定性验证
/home/rxb/.conda/envs/torch2.0.1/bin/python scripts/run_stability_validation.py \
  --config config/default.yaml --duration-sec 180 --report-path logs/stability_3min_report.json
```
