# C03 - 实时管道与MVP闭环

## 1. 周期目标

跑通最小可用系统：离线 HDF5 模拟流 -> 流式推理 -> 事件输出与结果存档。

## 2. 输入与边界

输入：
- 需求文档 §2.1 F1/F2/F4/F7、§4.1/§4.2/§4.3
- 技术文档 §7、§8、§9、§11、§15

本周期包含：
- ZMQ 协议与传输封装
- DataSimulator 进程
- InferenceEngine 进程
- 主进程编排与基础关闭流程
- 事件与概率日志落盘

本周期不包含：
- GUI 展示功能
- 训练流程

## 3. 任务清单

1. `zmq_transport.py`
- multipart 协议：header(msgpack) + payload(bytes)。
- 通用 `ZmqPublisher/ZmqSubscriber` 封装。
- 序列号 gap 检测与日志告警。

2. `data_simulator.py`
- 读取 HDF5 `data` 与 `prompts`。
- 默认 chunk_size=40（20ms）按 2kHz 节奏推送。
- 发布 `emg_chunk` 与 `ground_truth`。

3. `inference_engine.py`
- 订阅 EMG 并调用 `forward_streaming`。
- 发布 `probabilities` 与 `gesture_event`。
- 数据超时中断后执行 LSTM reset + warm-up。

4. `main.py`
- 启动顺序：先推理进程，再数据进程。
- 控制通道支持 `shutdown/heartbeat` 基础消息。
- 退出时执行资源回收。

5. `logger.py` 联调
- 运行日志。
- 事件 `events.jsonl`。
- 预测 `predictions.npz`。

6. 集成验证
- mini 数据集端到端跑通。
- 控制台可观察手势事件输出。

## 4. 交付物

1. MVP 可通过单命令启动并完成一次完整播放。
2. 生成标准日志目录：
- `runtime.log`
- `events.jsonl`
- `predictions.npz`
3. 端到端集成测试初版。

## 5. 验收标准

1. mini 数据集运行中可持续收到非空事件流。
2. 无数据时触发中断处理并暂停事件输出。
3. 手动触发 `SHUTDOWN` 时三进程可结束。

## 6. 风险与缓解

风险：
- ZMQ 高频小包导致吞吐抖动，触发误判中断。

缓解：
- 默认 20ms 推送粒度，预留可配置区间（10~50ms）。

