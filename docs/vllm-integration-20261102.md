# vLLM 接入现状汇总（2026-11-02）

本文是 vLLM 后端的**索引**：哪项能力在哪个文件、由哪份文档描述、用哪个测试文件验收。
实现细节不在这里重复。

## 一条路径

```text
vLLM V1 引擎
  └─ PIMExecutor（UniProcExecutor，单进程）
       └─ PIMWorker（vLLM 接口 ⇄ 编译器运行时）
            ├─ 启动期：validate_engine_config → compile_llama2 → 自动定容
            └─ 每步：execute_model 翻译调度结果 → 分 wave 跑 _step → 主机采样
```

## 能力 → 代码 → 文档 → 测试

| 能力 | 代码入口 | 文档 | 测试 |
| --- | --- | --- | --- |
| 引擎接入（执行器/worker） | `backend/vllm_executor.py`、`backend/vllm_worker.py` | `vllm-backend-20261023.md` | `test_vllm_executor.py` |
| 调度结果翻译 | `PIMWorker.execute_model` | 本文 | `test_vllm_translate.py` |
| 能力矩阵与启动预检 | `validate_engine_config`、`scripts/serve_pim.py` | `vllm-capabilities-20261101.md`、`vllm-serving-preflight-20261101.md` | `test_vllm_capabilities.py`、`test_serve_pim_preflight.py` |
| 主机侧采样 | `backend/sampling.py` | `vllm-sampling-20261101.md` | `test_vllm_sampling.py` |
| 块号以 vLLM 为唯一真源 | `runtime/executor.py::KVBlockTable` | `vllm-block-table-20261101.md` | `test_vllm_block_table.py` |
| 前缀缓存 | `_sync_cached`、共享块只读 | `vllm-prefix-cache-20261101.md` | `test_prefix_cache.py` |
| 分块 prefill | `_prefill_step`、`runtime/prefill.py` | `vllm-chunked-prefill-20261101.md` | `test_chunked_prefill.py` |
| prefill 分桶 | `compile_llama2(prefill_seq_lens=)` | `vllm-prefill-buckets-20261101.md` | `test_prefill_buckets.py` |
| 融合 prefill | `runtime/kernels.py` 的融合路径 | `vllm-fused-prefill-20261101.md` | 见该文档 |
| 抢占恢复 | `_sync_cached` 的 `resumed` 分支 | `vllm-preemption-20261102.md` | `test_preemption.py` |
| 自动定容 | `runtime/compile.py::_max_slots` | `kv-auto-capacity-20261102.md` | `test_auto_capacity.py` |
| 运行指标 | `backend/metrics.py`、`PIM_LOG_STEPS` | `vllm-metrics-20261102.md` | `test_step_stats.py` |
| 架构白名单与 GQA | `backend/architectures.py` | `model-architectures-20261102.md` | `test_architectures.py` |
| 版本锁定 | `backend/vllm_compat.py` | `vllm-031-gap-20261102.md` | `test_vllm_compat.py` |
| 跨请求并行 | `runtime/binding.py`、`hal_numpy` 上下文 | `executor-parallel-phase1-20261026.md` | `test_parallel_requests.py` |

## 一次性冒烟

```bash
cd flagos-pim-compiler
python -m pytest tests/test_vllm_compat.py tests/test_vllm_translate.py \
    tests/test_vllm_capabilities.py -q      # 约 6 秒，27 条
```

根目录 `scripts/verify.sh --vllm` 跑同一组用例；不加参数时默认不跑，避免拖慢快测。

## 当前不足

见 [`vllm-backend-20261023.md`](./vllm-backend-20261023.md) 的「当前不足」一节
（2026-11-02 已按现状重写）。要点：只支持 0.14.x 与 Llama 架构、只收 fp16、
单进程 `NumpyBackend`、vLLM 层并行度必须为 1。
