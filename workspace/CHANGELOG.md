# Changelog

## 2026-03-07

### Added
- 推理引擎事件回调 API：`register_event_callback()` / `unregister_event_callback()`。
- HEARTBEAT 机制：推理端周期发送 `MsgType.HEARTBEAT`，GUI 状态栏增加 heartbeat 状态监控。
- 延迟观测字段：`transport_ms`、`infer_ms`、`post_ms`、`pipeline_ms`。
- 延迟阈值告警配置与日志告警。
- 共享 pytest fixtures：`rt_gesture/tests/conftest.py`。
- 训练调试配置：`config/training_debug.yaml`。
- 工程占位目录：`checkpoints/.gitkeep`、`logs/.gitkeep`。

### Changed
- `config/training.yaml` 默认 `max_epochs` 调整为 `250`。
- 训练管线新增 `MulticlassAccuracy` 指标并写入 metrics。
- benchmark 测试增强：forward_streaming、event_detector、ZMQ、端到端 pipeline 延迟预算断言。
- README 扩展：架构图、模块/API 概览、配置说明、开发测试指南。
- 需求文档与技术文档同步到最新实现（版本、配置结构、pyzmq 约束）。

### Fixed
- 推理输出消息头中概率帧与事件帧的延迟字段一致性。
