# RT-Gesture Linux DL 服务器第六轮回归测试清单（无 GUI）

> 日期：2026-03-13  
> 目的：在第五轮优化（S-01~S-06, P-04）基础上进行实现验收与回归测试  
> 适用范围：Linux 深度学习服务器（不包含 GUI 测试）

---

## 1. 第五轮优化实现验收结论

### 1.1 代码验收结果（S-01 ~ S-06）

| 项目 | 验收结论 | 证据 |
|---|---|---|
| S-01: main.py 信号优雅关闭 | 已实现 | `workspace/rt_gesture/main.py` 已含 `SIGINT/SIGTERM` handler、`Received signal ... initiating graceful shutdown`、统一 `_stop_process` 关闭流程 |
| S-02: SIGKILL 后端口恢复 | 已实现 | `workspace/rt_gesture/main.py` 增加 `_wait_port_available()` 与 `_ensure_runtime_ports_available()`；`workspace/rt_gesture/zmq_transport.py` 增加 `IMMEDIATE/TCP_KEEPALIVE/REUSEADDR` |
| S-03: events.jsonl 延迟字段 | 已实现 | `workspace/rt_gesture/logger.py` 的 `log_event()` 支持 `transport_ms/infer_ms/post_ms/pipeline_ms`；`workspace/rt_gesture/inference_engine.py` 已传参写入 |
| S-04: 稳定性 warm-up + p99 | 已实现 | `workspace/scripts/run_stability_validation.py` 已含 `warmup_elapsed_sec/startup_delta_mb/steady_state_delta_mb`，`_series_stats` 已含 `p99` |
| S-05: benchmark 多平台适配 | 已实现 | `workspace/rt_gesture/tests/test_benchmark.py` 已区分 small/large batch，并保留 large batch xfail 策略 |
| S-06: training_summary epoch 曲线 | 已实现 | `workspace/rt_gesture/train.py` 已写入 `epoch_history` |
| P-04: 数据路径环境变量化 | 已实现 | `workspace/config/default.yaml` 已使用 `${RT_GESTURE_DATA_ROOT}`，`workspace/rt_gesture/config.py` 含 `os.path.expandvars` |

### 1.2 自动化验收结果

执行命令：

```bash
cd workspace
python -m pytest -q
```

实测结果：

- `63 passed, 1 skipped`（总计 64 项）
- 结论：第五轮优化未引入回归，核心实现有效。

---

## 2. 第六轮回归测试清单（无 GUI）

> 说明：以下清单用于 Linux 服务器持续回归。GUI 相关项明确排除。

### A. 环境与配置校验

| 编号 | 检查项 | 通过标准 | 状态 |
|---|---|---|---|
| A-01 | Python/torch 环境可用 | `torch.cuda.is_available()==True` 且可识别 GPU | ☐ |
| A-02 | 配置路径展开正常 | `${RT_GESTURE_DATA_ROOT}` 在运行时正确展开 | ☐ |
| A-03 | checkpoints 路径可读 | checkpoint 文件存在且可加载 | ☐ |
| A-04 | 端口预检查生效 | 启动时端口冲突会被明确报错 | ☐ |

建议命令：

```bash
python -m pytest -q rt_gesture/tests/test_config.py
```

### B. 信号与进程生命周期（S-01/S-02 回归）

| 编号 | 检查项 | 通过标准 | 状态 |
|---|---|---|---|
| B-01 | SIGTERM 优雅关闭 | 主进程日志含 graceful shutdown，退出码 0，无残留子进程 | ☐ |
| B-02 | Ctrl+C 优雅关闭 | 与 B-01 等价，端口释放完整 | ☐ |
| B-03 | SIGKILL 后同端口重启 | 立即重启不出现 `Address already in use` | ☐ |
| B-04 | 父进程异常退出保护 | 子进程不会长期残留（Linux PDEATHSIG 生效） | ☐ |

建议命令：

```bash
python scripts/run_realtime.py --config config/debug_short.yaml &
PID=$!
sleep 8
kill -SIGTERM $PID

python scripts/run_realtime.py --config config/debug_short.yaml &
PID=$!
sleep 5
kill -9 $PID
sleep 1
python scripts/run_realtime.py --config config/debug_short.yaml
```

### C. 事件与延迟可观测性（S-03 回归）

| 编号 | 检查项 | 通过标准 | 状态 |
|---|---|---|---|
| C-01 | events.jsonl 写入完整 | 每条事件包含 `transport_ms/infer_ms/post_ms/pipeline_ms` | ☐ |
| C-02 | 概率消息延迟字段存在 | `PROBABILITIES` header 含 `transport_ms/infer_ms/pipeline_ms` | ☐ |
| C-03 | 延迟值合法 | 延迟字段均为非负数 | ☐ |

建议命令：

```bash
python -m pytest -q rt_gesture/tests/test_logger.py rt_gesture/tests/test_integration.py
```

### D. 稳定性统计（S-04 回归）

| 编号 | 检查项 | 通过标准 | 状态 |
|---|---|---|---|
| D-01 | warm-up 分段字段存在 | 报告中有 `warmup_elapsed_sec` | ☐ |
| D-02 | 启动/稳态分段内存存在 | 有 `startup_delta_mb` 与 `steady_state_delta_mb` | ☐ |
| D-03 | p99 延迟统计存在 | 各延迟序列统计包含 `p99` | ☐ |
| D-04 | 稳态内存增长可控 | `steady_state_delta_mb` 在阈值内（由团队定义） | ☐ |

建议命令：

```bash
python scripts/run_stability_validation.py \
  --config config/debug_short.yaml \
  --duration-sec 60 \
  --sample-interval-sec 1 \
  --report-path logs/stability_60s_report.json
```

### E. benchmark 与性能回归（S-05 回归）

| 编号 | 检查项 | 通过标准 | 状态 |
|---|---|---|---|
| E-01 | benchmark 全流程可执行 | benchmark 测试可运行且无异常中断 | ☐ |
| E-02 | small-batch 延迟预算 | small-batch event_detector 通过预算阈值 | ☐ |
| E-03 | large-batch 策略正确 | large-batch 为 xfail/告警，不阻断回归 | ☐ |
| E-04 | 阈值可环境变量调节 | 设置环境变量后阈值生效 | ☐ |

建议命令：

```bash
python -m pytest -q -m benchmark
```

### F. 训练产物回归（S-06 回归）

| 编号 | 检查项 | 通过标准 | 状态 |
|---|---|---|---|
| F-01 | training_summary 结构完整 | 含 `best_*` 与 `epoch_history` 字段 | ☐ |
| F-02 | epoch_history 连续 | epoch 序列为 1..N 连续递增 | ☐ |
| F-03 | 指标字段完整 | 每 epoch 含 train/val loss、accuracy、lr | ☐ |

建议命令（轻量回归）：

```bash
python -m pytest -q rt_gesture/tests/test_training.py
```

### G. 总回归

| 编号 | 检查项 | 通过标准 | 状态 |
|---|---|---|---|
| G-01 | 全量自动化回归 | `python -m pytest -q` 全通过（允许 slow 场景 skip） | ☐ |
| G-02 | 关键缺陷未复发 | S-01~S-06 均无回归证据 | ☐ |

建议命令：

```bash
python -m pytest -q
```

---

## 3. 执行记录模板

| 日期 | 执行人 | 分支/提交 | A | B | C | D | E | F | G | 结论 |
|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | |

---

## 4. 本轮备注

1. 本清单按“Linux 服务器暂不做 GUI 测试”设计，D-2/D-4 等 GUI 项不纳入本轮回归目标。
2. 若后续恢复 GUI 验证，建议单独维护 `Linux_DL服务器_GUI补充清单.md`，避免与主回归清单混淆。
