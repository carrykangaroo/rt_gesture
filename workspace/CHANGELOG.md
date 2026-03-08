# Changelog

## [0.2.0] - 2026-03-07

### Added
- 推理引擎事件回调 API：`register_event_callback()` / `unregister_event_callback()`。
- HEARTBEAT 机制：推理端周期发送 `MsgType.HEARTBEAT`，GUI 状态栏增加 heartbeat 状态监控。
- 延迟观测字段：`transport_ms`、`infer_ms`、`post_ms`、`pipeline_ms`。
- 延迟阈值告警配置与日志告警。
- 共享 pytest fixtures：`rt_gesture/tests/conftest.py`。
- 训练调试配置：`config/training_debug.yaml`。
- 工程占位目录：`checkpoints/.gitkeep`、`logs/.gitkeep`。
- GUI 配置面板新增设备选择器（auto/cpu/cuda）。
- GUI 新增 ground-truth prompts 可视化列表。
- GUI 新增概率 Heatmap 视图（Bars/Heatmap 双 tab）。
- GUI 组件测试：`test_gui.py`。
- 数据中断恢复集成测试：`test_integration.py::test_inference_recovers_after_data_timeout_gap`。
- shutdown 端口释放验证：`test_process_manager.py::test_control_port_can_be_rebound_after_stop`。
- transforms 单元测试：`test_transforms.py`。
- CLER 覆盖增强测试（debounce、边界事件、对齐索引、误差断言）。
- 稳定性验证脚本：`scripts/run_stability_validation.py`（时长可配、延迟统计、RSS 趋势报告）。
- CLER 基准配置：`config/evaluation_benchmark.yaml`。

### Changed
- `config/training.yaml` 默认 `max_epochs` 调整为 `250`。
- 训练管线新增 `MulticlassAccuracy` 指标并写入 metrics。
- benchmark 测试增强：forward_streaming、event_detector、ZMQ、端到端 pipeline 延迟预算断言。
- README 扩展：架构图、模块/API 概览、配置说明、开发测试指南。
- 需求文档与技术文档同步到最新实现（版本、配置结构、pyzmq 约束）。
- `setup.py` 版本读取改为单源（`rt_gesture.__version__`）。
- `InferenceEngine` 输出时间戳补充 `time_start_rel/time_end_rel/event_time_rel`。
- `evaluate.py` 增加 `full_chunk_size` 参数，full-forward 改为分块执行，避免整段前向内存峰值。

### Fixed
- 推理输出消息头中概率帧与事件帧的延迟字段一致性。
- 评估命令在 CPU 上因整段 LSTM 前向导致的 OOM（改为分块 full-forward）。
