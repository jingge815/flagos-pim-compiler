# 运行时 DMA 与算子缓存并发修复（2026-10-07）

## 修改内容

`NumpyBackend` 的 host 通信命令会通过 `DmaEngine` 调用
`dpu_prepare_xfer` 和 `dpu_push_xfer`。底层 NumPy SDK 把预备的 host
缓冲登记在整台模拟机共享的 `xfer_buffers` 中。两个没有依赖关系的 host
命令并行执行时，后一个命令会覆盖前一个命令登记的缓冲，随后把错误数据写回
MRAM。张量并行度为 8 时通信最密集，因此会间歇性破坏激活、KV 缓存和 logits。

在 `backend/hal_numpy.py` 中增加 host 命令锁。host callable 的完整 DMA
事务现在串行执行，DPU launch 仍保持并行。新增测试把 prepare 到 push 之间的
窗口固定拉大，确认两组传输分别写入各自的 MRAM 区域。

`opcompiler_bridge/driver.py` 的磁盘缓存按请求 key 增加跨进程文件锁。锁内再次
检查缓存，`.meta` 和 `.pimir.mlir` 通过临时文件和原子替换发布；缓存三件套
（`.so`、`.meta`、`.pimir.mlir`）任一缺失都重编。这样同时运行的 pytest 不会
把不同编译结果混成一份缓存。新增多进程互斥测试。

全流程脚本补充了模型骨架 IR、PIMIR 和 sidecar 的编号一致性检查，避免重装
FlagTree 后新 sidecar 配旧 PIMIR 时调度器静默退回模板。

## 关键位置

| 文件 | 作用 |
| --- | --- |
| `backend/hal_numpy.py` | 串行化共享 SDK 传输缓冲的 host DMA 临界区。 |
| `opcompiler_bridge/driver.py` | 按缓存 key 锁住检查、编译和发布流程。 |
| `scripts/run_full_pipeline.py` | 校验仿真 IR 与 sidecar 属于同一套算子编号。 |
| `tests/test_hal_numpy.py` | 覆盖并发 host DMA 缓冲覆盖问题。 |
| `tests/test_opcompiler_linear.py` | 覆盖跨进程同 key 缓存锁。 |

## 验证

- `python -m pytest tests/test_hal_numpy.py tests/test_comm_lowering.py -q`：29 passed。
- `python -m pytest tests/test_opcompiler_linear.py -q -k 'cache_key_lock or compiler_fingerprint or compile_result_carries'`：6 passed。
- `python -m pytest tests/ -q -k "llama2_7b"`：已完成三次，均为 `48 passed, 1307 deselected`。其中覆盖 `tp8_pp1` 的端到端生成、KV 区一致性和非 2 次幂局部宽度。

FlagTree 已安装版本与 `bf4504c77` 一致。检查 `M=6, N=1376, K=4096` 的生成
C 和 PIMIR 后，尾块边界与 tasklet 划分正确，不需要修改 FlagTree 或 genesim。
