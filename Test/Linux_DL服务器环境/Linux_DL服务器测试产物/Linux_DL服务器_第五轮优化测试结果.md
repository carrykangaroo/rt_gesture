# RT-Gesture 第五轮优化实施与测试结果（Linux DL 服务器）

> 日期：2026-03-10  
> 项目目录：`/home/rxb/rt_gesture`  
> 工作目录：`/home/rxb/rt_gesture/workspace`  
> 对照清单：`doc/review/第五轮优化清单_Linux服务器测试.md`

## 1. 优化项落地情况

| 清单项 | 状态 | 说明 | 主要文件 |
|---|---|---|---|
| S-01 main.py 缺少 SIGTERM 处理 | 已完成 | `main.py` 增加 `SIGTERM/SIGINT` 统一优雅关闭逻辑；join 循环可响应关闭请求 | `workspace/rt_gesture/main.py` |
| S-02 SIGKILL 后端口残留 | 已完成 | 增加端口可用性等待、ZMQ `IMMEDIATE/TCP_KEEPALIVE/REUSEADDR`、子进程 `PR_SET_PDEATHSIG`（Linux）避免父进程被 kill 后子进程残留 | `workspace/rt_gesture/main.py`、`workspace/rt_gesture/zmq_transport.py`、`workspace/rt_gesture/gui/process_manager.py` |
| S-03 events.jsonl 缺少延迟字段 | 已完成 | `EventLogger.log_event` 扩展 `transport_ms/infer_ms/post_ms/pipeline_ms` 并在推理事件落盘时写入 | `workspace/rt_gesture/logger.py`、`workspace/rt_gesture/inference_engine.py` |
| S-04 稳定性内存统计不准确 | 已完成 | 稳定性脚本增加 warm-up 分段统计：`startup_delta_mb` / `steady_state_delta_mb`；延迟统计新增 `p99` | `workspace/scripts/run_stability_validation.py` |
| S-05 benchmark 阈值不适配多平台 | 已完成 | benchmark 阈值环境变量化；新增小 batch event_detector 基准；大 batch 改为 `xfail` 警告项 | `workspace/rt_gesture/tests/test_benchmark.py` |
| S-06 training_summary 缺少 epoch 曲线 | 已完成 | `training_summary.json` 新增 `epoch_history`（逐 epoch 指标） | `workspace/rt_gesture/train.py` |
| P-04 路径兼容建议（环境变量） | 已完成 | 配置读取支持 `${RT_GESTURE_DATA_ROOT}`，并将 `config/*.yaml` 数据路径改为占位写法 | `workspace/rt_gesture/config.py`、`workspace/rt_gesture/train.py`、`workspace/rt_gesture/evaluate.py`、`workspace/config/*.yaml` |

## 2. 自动化测试结果

执行命令：

```bash
cd /home/rxb/rt_gesture/workspace
/home/rxb/.conda/envs/torch2.0.1/bin/python -m pytest -q
```

结果：

- `62 passed`
- `1 skipped`
- `1 xfailed`

说明：

- `skipped`：`test_evaluate.py` 中慢速 CLER 场景（需显式设置 `RT_GESTURE_RUN_SLOW=1`）。
- `xfailed`：`test_event_detector_latency_budget_large_batch_xfail`（按新策略保留为跨硬件告警项，不阻断 CI）。

## 3. 脚本级功能验证

### 3.1 S-01：SIGTERM 优雅关闭

执行（2026-03-10）：启动 `config/default.yaml` 后发送 `SIGTERM` 给主进程。

关键结果：

- 主进程退出码：`0`
- 日志出现：`Received signal 15, initiating graceful shutdown`
- 日志出现：`RT-Gesture shutdown complete`
- 结束后无残留 `rt_gesture.main / DataSimulator / InferenceEngine` 进程

### 3.2 S-02：SIGKILL 后同端口重启

执行（2026-03-10）：

1. 启动 `config/default.yaml`
2. 对主进程执行 `kill -9`
3. 立即使用同端口配置再次启动

关键结果：

- 首次进程退出码：`137`（符合 `SIGKILL`）
- 二次启动退出码：`0`
- 未出现 `Address already in use` 报错
- 二次启动日志正常进入并完成：`RT-Gesture shutdown complete`

### 3.3 S-03：events.jsonl 延迟字段落盘

执行：使用临时配置（阈值置 0）强制产生事件。

产物：`workspace/logs/2026-03-10_13-37-56/events.jsonl`

关键结果：

- 事件行包含字段：`transport_ms`、`infer_ms`、`post_ms`、`pipeline_ms`
- 样例首行：

```json
{"gesture": "index_press", "timestamp": 1633014930.533361, "confidence": 0.001951, "wall_time": "2026-03-10T13:37:57.713846", "transport_ms": 1.213, "infer_ms": 36.085, "post_ms": 0.548, "pipeline_ms": 39.61}
```

### 3.4 S-04：稳定性报告 warm-up 分段与 p99

执行命令：

```bash
cd /home/rxb/rt_gesture/workspace
/home/rxb/.conda/envs/torch2.0.1/bin/python scripts/run_stability_validation.py \
  --config config/debug_short.yaml \
  --duration-sec 20 \
  --sample-interval-sec 1 \
  --report-path logs/stability_20s_report.json
```

关键结果（`logs/stability_20s_report.json`）：

- 延迟统计含 `p99`（如 `pipeline_ms.p99 = 6.54024`）
- 内存统计含分段字段：
  - `startup_delta_mb = 284.36328125`
  - `steady_state_delta_mb = 29.3046875`
  - `warmup_elapsed_sec = 2.614681629987899`

### 3.5 S-06：training_summary.json 增加 epoch_history

执行：使用 mini 数据构造轻量训练配置（2 epoch，1 step/epoch）。

产物：`workspace/checkpoints/discrete_gestures_debug_tiny/20260310_134350/training_summary.json`

关键结果：

- `epoch_history` 存在
- `epoch_history` 长度：`2`
- `epoch` 列表：`[1, 2]`

## 4. 关联测试增强

新增/更新测试覆盖：

- `workspace/rt_gesture/tests/test_logger.py`：验证 events.jsonl 延迟字段
- `workspace/rt_gesture/tests/test_integration.py`：端到端验证事件文件包含延迟字段
- `workspace/rt_gesture/tests/test_config.py`：验证 `${RT_GESTURE_DATA_ROOT}` 占位解析
- `workspace/rt_gesture/tests/test_training.py`：验证 `epoch_history` 写入与路径解析行为
- `workspace/rt_gesture/tests/test_benchmark.py`：阈值参数化 + 大 batch xfail 策略

## 5. 结论

本轮按清单要求的 S-01 ~ S-06（及 P-04 路径兼容）均已落地并完成验证。

当前状态（2026-03-10）：

- 功能回归：通过
- 自动化测试：通过（62 passed, 1 skipped, 1 xfailed）
- 关键问题修复（信号关闭、端口重启、观测字段、稳定性统计、训练 summary）：通过
