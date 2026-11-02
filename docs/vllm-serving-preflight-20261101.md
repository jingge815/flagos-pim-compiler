# serve_pim 的 PIM 参数与启动预检（2026-11-01）

## 改了什么

此前 `scripts/serve_pim.py` 只做两件事：换执行后端、关前缀缓存。PIM 的
DPU 数、MRAM 大小、槽池、窗口数、prefill 图长度全靠环境变量手工设，配错了
要等 7B 权重加载 + 图编译完才炸。

本次把 PIM 开关做成 `--pim-*` 命令行选项，并在 `run_server` 之前做一次几何预检。

| 文件 | 改动 |
| --- | --- |
| `scripts/serve_pim.py` | 新增 `PIM_FLAGS`、`add_pim_args`、`export_env`、`preflight`、`_kv_bytes_per_dpu`；`main()` 串上预检 |
| `tests/test_serve_pim_preflight.py` | 新增 9 例单测 |
| `docs/vllm-backend-20261023.md` | 「启动方式」补 `--pim-*` 表与预检说明 |

## 命令行选项

| 选项 | 环境变量 | 默认值 |
| --- | --- | --- |
| `--pim-num-dpus` | `PIM_NUM_DPUS` | 4 |
| `--pim-mram-bytes` | `PIM_MRAM_BYTES` | 4 GiB |
| `--pim-kv-pool-slots` | `PIM_KV_POOL_SLOTS` | `max_num_seqs × 每请求块数 + 1` |
| `--pim-parallel-slots` | `PIM_PARALLEL_SLOTS` | 2（受 `max_num_seqs` 限制） |
| `--pim-prefill-seq-len` | `PIM_PREFILL_SEQ_LEN` | `max_model_len` |

选项默认值为 `None`：没写就保留已有环境变量，环境变量也没有时由 `PIMWorker`
用上表的默认值。

vLLM 0.14 的 `--help` 只打印配置组名，这组选项挂在 `PIMConfig` 组下，
用 `--help=PIMConfig` 查看。

## 预检

```text
main()
  ├─ parse_args + PIMExecutor / 关前缀缓存
  ├─ validate_parsed_serve_args
  ├─ export_env            --pim-* → 环境变量
  ├─ preflight(EngineArgs.from_cli_args(args).create_engine_config())
  │    └─ 有问题：列出全部问题 → SystemExit（退出码 1）
  └─ asyncio.run(run_server(args))
```

`preflight()` 吃的是 vLLM **已解析**的配置（`max_model_len`、`block_size`、
`max_num_seqs`、`max_num_batched_tokens` 都已填好默认值），检查四项：

| 检查 | 判据 | 建议 |
| --- | --- | --- |
| prefill 图长度 | `PIM_PREFILL_SEQ_LEN ≤ max_model_len` | 加 `--pim-prefill-seq-len <max_model_len>` |
| 槽池容量 | `PIM_KV_POOL_SLOTS ≥ ceil(max_model_len / block_size)` | 给出所需槽数（`每请求块数 × max_num_seqs + 1`） |
| 分块 prefill 预算 | `max_num_batched_tokens ≥ prefill_seq_len × max_num_seqs` | 加 `--max-num-batched-tokens N`，或加 `--no-enable-chunked-prefill` |
| MRAM 预算 | KV 区每台 DPU 字节数 ≤ `MRAM − 系统预留` | 调小 `--pim-kv-pool-slots` 或调大 `--pim-mram-bytes` |

KV 字节数按 `memory/kv_layout.KVRegionSpec` 的口径估算：

```text
slot_bytes        = 2(K,V) × 层数 × 本机 KV 头数 × 头维 × 2(fp16) × block_size
block_table_bytes = align_up(每请求块数 × 4, 256)
每台 DPU          = 窗口数 × block_table_bytes + 槽池大小 × slot_bytes
```

本机 KV 头数取 `ceil(num_key_value_heads / PIM_NUM_DPUS)`。**权重没有计入**，
所以这只是一道粗筛；真正的上限仍由编译期 `plan_dpu()` 判定，超出时编译报
per-DPU 错误。

多个问题会**一次性列全**，改一轮配置就能全部修掉：

```text
PIM 启动预检失败：
  - PIM_KV_POOL_SLOTS=0 装不下一个满长请求：ceil(max_model_len=16 / block_size=16) = 1 块；建议加 --pim-kv-pool-slots 3
  - max_num_batched_tokens=1 放不下一步的整段 prompt：...
```

## 已实施的测试

| 测试 | 验证点 |
| --- | --- |
| `test_supported_config_passes` | 默认配置通过 |
| `test_pool_too_small_reports_required_slots` | 槽池不足时退出，并给出建议槽数 |
| `test_chunked_prefill_budget_advice` | 预算不足时同时给出两种修法 |
| `test_prefill_len_beyond_max_model_len` | prefill 图长于 `max_model_len` 报错 |
| `test_mram_budget_exceeded` | KV 超过 MRAM 预算报错 |
| `test_problems_reported_together` | 多个问题一次列全 |
| `test_export_env_writes_all_flags` / `test_export_env_leaves_unset_flags_alone` | 选项 ↔ 环境变量翻译 |
| `test_parser_accepts_pim_flags` | 参数解析器识别 `--pim-*` |

实测：`tests/test_serve_pim_preflight.py` 9 passed（3.95s）。

端到端：

| 场景 | 结果 |
| --- | --- |
| `--help=PIMConfig` | 五个 `--pim-*` 选项与说明齐全 |
| `--pim-kv-pool-slots 0` 启动 | `run_server` 之前退出，退出码 1，消息给出所需槽数 |
| 微型模型起服务 + 两个并发 `/v1/completions` | 各自 `completion_tokens=3`、`finish_reason=length` |

## 当前不足

- KV 估算不含权重，MRAM 检查偏松；权重与激活的真实上限由编译期规划器负责。
- `create_engine_config()` 调了两次（预检一次、引擎一次），各约 0.1~1 秒；
  换来的是「权重加载前就报错」。
- 预检只覆盖几何，不检查采样参数、结构化输出等；那些在 `PIMWorker`
  的 `validate_engine_config` / `validate_new_request` 里。
- `--pim-*` 只接了 int 型参数；以后新增 bool 型要扩展 `PIM_FLAGS` 的类型处理。
