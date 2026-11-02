"""把 PIM 静态编译后端接入 vLLM V1 的 Worker。

分工：vLLM 负责 HTTP/调度/连续批处理，本模块负责把每步的
`SchedulerOutput` 翻译成 PIM 的逐请求执行。核心约束：

- KV 分成固定大小的块，每块容纳 `block_size` 个 token；每台 DPU 只有一份块表
  缓冲，每步由 host 覆盖，请求通过它间接寻址。**块号由 vLLM 分配**：新请求用
  `block_ids`，cached 请求追加 `new_block_ids`，抢占恢复时整段替换。PIM 只按
  这些块号写块表，不再自建物理槽池。
- 块池大小默认按 MRAM 剩余空间自动定容（上限 `max_num_seqs × 每请求块数 + 1`，
  多出的一块留给 vLLM 启动时永久占用的 null 块），`PIM_KV_POOL_SLOTS` 可覆盖；
  结果通过 `determine_available_memory` 报给 vLLM，使其分到的块号恰好落在池内。
- 采样在主机侧完成（`backend/sampling.py`）：支持贪心、温度、top-k/top-p、
  三种惩罚、seed 与 logprobs；未实现的采样参数显式报错。
- 能力矩阵在启动期（`validate_engine_config`）与请求期
  （`validate_new_request` / `validate_scheduler_output`）校验，不支持的能力
  不会静默算错。
- 架构白名单在 `backend/architectures.py`：目前有 Llama 与 Qwen2（含 GQA 与标准 RoPE），
  其它架构、位置缩放与滑窗注意力在启动期拒绝。
- 运行时指标由 `PIM_LOG_STEPS` 每 N 步打印一行，`PIM_PROFILE=1` 在退出时打印
  请求级调度报告与各 DPU 忙碌时间。
"""

from __future__ import annotations

import atexit
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
from vllm.logger import init_logger
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheConfig, KVCacheSpec
from vllm.v1.outputs import LogprobsLists, ModelRunnerOutput
from vllm.v1.worker.worker_base import WorkerBase

from backend.architectures import ARCHITECTURES
from backend.hal_numpy import NumpyBackend, NumpyBackendConfig
from backend.metrics import StepStats
from backend.sampling import RequestSampler, SampleResult, SamplingConfig
from contracts.op_contract import PIMHardwareConfig
from memory.mem_planner import HwBudget
from runtime.compile import causal_mask_of, compile_llama2, default_prefill_buckets, load_weights
from runtime.executor import RequestTable, prefill_chunk, step_request
from runtime.kernels import register_all

logger = init_logger(__name__)


def _unsupported(param: str, current, supported: str, advice: str) -> ValueError:
    """统一「能力不支持」的报错格式：参数、当前值、PIM 支持范围、建议做法。"""
    return ValueError(
        f"{param}={current!r} 不受支持；PIM 支持 {supported}；建议 {advice}")


def validate_engine_config(
    *, parallel_config, lora_config, speculative_config, model_config,
    scheduler_config, prefill_seq_len: int,
) -> None:
    """启动期校验引擎级能力矩阵，把「静默算错」提前成「启动即失败」。"""
    tp = parallel_config.tensor_parallel_size
    if tp != 1:
        raise _unsupported(
            "tensor_parallel_size", tp, "1",
            "去掉 --tensor-parallel-size；PIM 内部已按 DPU 做张量并行")
    pp = parallel_config.pipeline_parallel_size
    if pp != 1:
        raise _unsupported(
            "pipeline_parallel_size", pp, "1", "去掉 --pipeline-parallel-size")
    if lora_config is not None:
        raise _unsupported(
            "lora_config", "已设置", "未设置（None）",
            "去掉 --enable-lora / --lora-modules")
    if speculative_config is not None:
        raise _unsupported(
            "speculative_config", "已设置", "未设置（None）",
            "去掉 --speculative-model / --num-speculative-tokens")
    quantization = getattr(model_config, "quantization", None)
    if quantization is not None:
        raise _unsupported(
            "quantization", quantization, "未量化（None）",
            "去掉 --quantization，用 fp16 权重")
    if model_config.dtype != torch.float16:
        raise _unsupported(
            "dtype", model_config.dtype, "float16", "加 --dtype float16")
    model_type = getattr(model_config.hf_config, "model_type", None)
    if model_type not in ARCHITECTURES:
        raise _unsupported(
            "model_type", model_type, "、".join(ARCHITECTURES),
            "换用受支持架构的权重目录")
    # 编译器只实现标准 RoPE 与全注意力，带窗口或缩放位置的模型会静默算错。
    rope_scaling = getattr(model_config.hf_config, "rope_scaling", None)
    if rope_scaling:
        raise _unsupported(
            "rope_scaling", rope_scaling, "未设置（标准 RoPE）",
            "换用不做位置插值/缩放的权重目录")
    # Qwen2 配置里默认带滑窗长度，只有 `use_sliding_window` 打开时才算滑窗。
    uses_window = getattr(model_config.hf_config, "use_sliding_window", True)
    sliding_window = getattr(model_config.hf_config, "sliding_window", None)
    if sliding_window and uses_window:
        raise _unsupported(
            "sliding_window", sliding_window, "未设置（全注意力）",
            "换用全注意力模型的权重目录")
    # 分块 prefill 每步最多调度 max_num_batched_tokens 个 token，这一段的长度
    # 不能超过最长 prefill 桶，否则没有图能跑。
    if scheduler_config.enable_chunked_prefill:
        limit = scheduler_config.max_num_batched_tokens
        if limit > prefill_seq_len:
            raise ValueError(
                f"max_num_batched_tokens={limit} 超过最长 prefill 桶 "
                f"{prefill_seq_len}；一个 prefill 分块不超过这个预算，"
                f"建议调小 --max-num-batched-tokens 或调大 PIM_PREFILL_SEQ_LEN")


def validate_scheduler_output(scheduler_output) -> None:
    """每步调度结果的能力校验：结构化输出要在采样时受语法约束，PIM 做不到。"""
    if (scheduler_output.has_structured_output_requests
            or scheduler_output.pending_structured_output_tokens):
        raise _unsupported(
            "structured_output", True, "无语法约束的采样",
            "去掉 guided_decoding / response_format / structured_outputs")


def validate_new_request(req) -> None:
    """新请求接入前的能力校验，让不支持的输入在第一步之前失败。"""
    if getattr(req, "lora_request", None) is not None:
        raise _unsupported(
            "lora_request", "非空", "空（None）", "去掉请求里的 LoRA 适配器")
    if getattr(req, "prompt_token_ids", None) is None:
        raise _unsupported(
            "prompt_token_ids", None, "token id 列表",
            "改用文本或 prompt_token_ids 输入；不支持 prompt_embeds")
    if getattr(req, "mm_features", None):
        raise _unsupported(
            "mm_features", "非空", "空列表", "只提交纯文本请求")


class PIMWorker(WorkerBase):
    """用 NumpyBackend 跑编译好的模型蓝图，承接 vLLM 的调度结果。"""

    def __init__(self, vllm_config, local_rank=0, rank=0,
                 distributed_init_method="", is_driver_worker=True) -> None:
        super().__init__(vllm_config, local_rank, rank,
                         distributed_init_method, is_driver_worker)
        self.num_request_slots = int(self.scheduler_config.max_num_seqs)
        # 同一步并发的请求窗口数：不设环境变量时由编译期按 MRAM 剩余空间定，
        # 默认取 2 个窗口，放不下就减到 1。
        raw_parallel = os.environ.get("PIM_PARALLEL_SLOTS")
        self.num_parallel_slots: int | None = int(raw_parallel) if raw_parallel else None
        self.kv_slot_tokens = int(self.cache_config.block_size)
        self.max_seq = int(self.model_config.max_model_len)
        # prefill 图按长度分桶：prompt 右侧补齐到最小的能放下它的桶。
        # `PIM_PREFILL_SEQ_LEN` 是最长桶，`PIM_PREFILL_SEQ_LENS` 可显式指定桶表。
        self.prefill_seq_len = int(
            os.environ.get("PIM_PREFILL_SEQ_LEN", self.max_seq))
        if self.prefill_seq_len > self.max_seq:
            raise ValueError(
                f"PIM_PREFILL_SEQ_LEN={self.prefill_seq_len} 超过 max_model_len={self.max_seq}")
        raw_lens = os.environ.get("PIM_PREFILL_SEQ_LENS")
        self.prefill_seq_lens = (
            tuple(sorted({int(x) for x in raw_lens.split(",") if x.strip()}))
            if raw_lens else default_prefill_buckets(self.prefill_seq_len))
        if self.prefill_seq_lens[-1] > self.prefill_seq_len:
            raise ValueError(
                f"PIM_PREFILL_SEQ_LENS 最长桶 {self.prefill_seq_lens[-1]} 超过 "
                f"PIM_PREFILL_SEQ_LEN={self.prefill_seq_len}")
        self.num_dpus = int(os.environ.get("PIM_NUM_DPUS", "4"))
        self.mram_bytes = int(os.environ.get("PIM_MRAM_BYTES", str(4 << 30)))
        # 交给 vLLM 的块池大小，决定了它可能分配出的块号范围。不设环境变量时
        # 编译期按 MRAM 剩余空间自动定容，上限是「并发上限 × 每请求块数 + 1」，
        # 多出的一块留给 vLLM 启动时永久占用的 null 块。
        raw = os.environ.get("PIM_KV_POOL_SLOTS")
        self.kv_pool_slots: int | None = int(raw) if raw else None
        if self.kv_pool_slots is not None and self.kv_pool_slots <= self.blocks_per_slot():
            raise ValueError(
                f"PIM_KV_POOL_SLOTS={self.kv_pool_slots} 装不下一个满长请求"
                f"（需要 {self.blocks_per_slot()} 个块，另留一块给 null 块）")
        hf = self.model_config.hf_config
        self.eos_id = int(getattr(hf, "eos_token_id", 2) or 2)
        validate_engine_config(
            parallel_config=self.parallel_config,
            lora_config=self.lora_config,
            speculative_config=self.speculative_config,
            model_config=self.model_config,
            scheduler_config=self.scheduler_config,
            prefill_seq_len=self.prefill_seq_len,
        )

        self.hal: NumpyBackend | None = None
        self.compiled = None
        self.table: RequestTable | None = None
        self._executor: ThreadPoolExecutor | None = None
        # 每 N 步打印一行运行时指标；0 表示关闭，关闭时整步不累加。
        self.stats_interval = int(os.environ.get("PIM_LOG_STEPS", "0"))
        self.stats: StepStats | None = None
        # `PIM_PROFILE=1` 时采集请求级调度与各 DPU 忙碌时间，退出时打印报告。
        self._profiler = None
        self._backend_profiler = None
        if os.environ.get("PIM_PROFILE"):
            from scripts.profile_execution import profile_vllm

            self._profiler = profile_vllm(self)
            atexit.register(self._dump_profile)
        # 采样状态按请求号保存：原始 prompt 与已生成 token 分开记，
        # 这样被抢占重放 prefill 后，frequency/presence 惩罚仍按 vLLM 的口径算。
        self._samplers: dict[str, RequestSampler] = {}
        self._prompt_ids: dict[str, list[int]] = {}
        self._output_ids: dict[str, list[int]] = {}

    # ---- 设备与模型 ----

    def init_device(self) -> None:
        """初始化 NumpyBackend 并注册全部设备内核。"""
        # 编译前还不知道最终窗口数，先按默认值开线程池；它只影响线程数。
        self.hal = NumpyBackend(NumpyBackendConfig(
            num_dpus=self.num_dpus, mram_bytes_per_dpu=self.mram_bytes,
            num_parallel_slots=self.num_parallel_slots or min(2, self.num_request_slots)))
        register_all(self.hal)
        if os.environ.get("PIM_PROFILE"):
            from scripts.profile_execution import profile_backend

            self._backend_profiler = profile_backend(self.hal)

    def load_model(self) -> None:
        """按架构编译两张图、写入权重、建立请求槽表。"""
        arch = ARCHITECTURES[self.model_config.hf_config.model_type]
        # 不改全局梯度开关：关闭时 torch.export 会导出另一张图。
        model = arch.load(self.model_config.model)
        strategy = arch.strategy(self.num_dpus, model.config)
        hw = HwBudget(
            mram_bytes=self.mram_bytes, align=256,
            sys_reserve_bytes=int(os.environ.get("PIM_SYS_RESERVE_BYTES", str(64 << 20))),
        )
        hardware = PIMHardwareConfig(
            num_dpus=self.num_dpus, num_tasklets=4,
            mram_bytes_per_dpu=hw.mram_bytes, wram_bytes_per_dpu=65536, dma_align=64,
        )
        self.compiled = compile_llama2(
            model, strategy, prefill_seq_len=self.prefill_seq_len,
            max_seq=self.max_seq, hw=hw, hardware=hardware,
            num_request_slots=self.num_request_slots,
            kv_slot_tokens=self.kv_slot_tokens,
            kv_pool_slots=self.kv_pool_slots,
            num_parallel_slots=self.num_parallel_slots,
            prefill_seq_lens=self.prefill_seq_lens,
        )
        # 自动定容的结果以编译产物为准：vLLM 的块数与窗口数都由它决定。
        self.kv_pool_slots = self.compiled.kv_pool_slots
        self.num_parallel_slots = self.compiled.num_parallel_slots
        if self.stats_interval > 0:
            self.stats = StepStats(
                pool_total=self.kv_pool_slots,
                max_width=self.num_parallel_slots,
                window_bytes=self.compiled.act_stride + self.compiled.block_table_bytes,
            )
        load_weights(self.compiled, self.hal)
        # 块号全部由 vLLM 分配，本表只做 host 记账，不再自建物理槽池。
        self.table = RequestTable(self.num_request_slots)

    def _dump_profile(self) -> None:
        """进程退出时打印调度报告与各 DPU 忙碌时间（`PIM_PROFILE=1`）。"""
        from scripts.profile_execution import print_report, print_vllm_report

        print_vllm_report(self._profiler.stop())
        if self._backend_profiler is not None:
            print_report(self._backend_profiler.stop())

    def _record_stats(self, scheduled: dict[str, int], width: int) -> None:
        """每步累加指标，按 `PIM_LOG_STEPS` 的间隔打印一行。"""
        if self.stats is None:
            return
        self.stats.record(scheduled, width=width, slots=self.table.active_slots())
        if self.stats.steps % self.stats_interval == 0:
            logger.info(self.stats.format())

    # ---- KV 缓存记账 ----

    def get_kv_cache_spec(self) -> dict[str, KVCacheSpec]:
        """报一个满注意力的 KV 规格，块尺寸与 vLLM 调度器保持一致。"""
        hf = self.model_config.hf_config
        head_dim = getattr(hf, "head_dim", None) or hf.hidden_size // hf.num_attention_heads
        return {"pim": FullAttentionSpec(
            block_size=self.cache_config.block_size,
            num_kv_heads=hf.num_key_value_heads,
            head_size=head_dim,
            dtype=torch.float16,
        )}

    def determine_available_memory(self) -> int:
        """报告全局槽池的记账字节数，使 vLLM 分到的块数正好等于槽数。"""
        spec = next(iter(self.get_kv_cache_spec().values()))
        return self.kv_pool_slots * spec.page_size_bytes

    def blocks_per_slot(self) -> int:
        """一个请求按调度器的块尺寸记账所需的块数。"""
        return -(-self.max_seq // self.kv_slot_tokens)

    def initialize_from_config(self, kv_cache_config: KVCacheConfig) -> None:
        """KV 已在编译期静态分配，这里只校验 vLLM 记下的块数与预期一致。

        vLLM 0.14 先调 `load_model` 再调 `determine_available_memory`，但
        `kv_pool_slots` 在 `__init__` 就算好，不依赖调用顺序。显式设置
        `num_gpu_blocks_override` 会改变块数，导致这里启动失败。
        """
        expected = self.kv_pool_slots
        if kv_cache_config.num_blocks != expected:
            raise RuntimeError(
                f"vLLM 分配 {kv_cache_config.num_blocks} 个 block，"
                f"预期 {expected}（PIM_KV_POOL_SLOTS）")

    def initialize_cache(self, num_gpu_blocks: int, num_cpu_blocks: int) -> None:
        """块数校验已在 `initialize_from_config` 完成，这里无需再分配。"""

    def compile_or_warm_up_model(self) -> None:
        """静态编译模型无需预热。"""

    def get_supported_tasks(self) -> tuple[str, ...]:
        return ("generate",)

    # ---- 每步执行 ----

    def execute_model(self, scheduler_output: SchedulerOutput) -> ModelRunnerOutput:
        """翻译 vLLM 的调度结果，逐请求跑一步，返回本步采样的 token。"""
        for req_id in scheduler_output.finished_req_ids:
            self._forget_request(req_id)
            self.table.free_slot(req_id)
        validate_scheduler_output(scheduler_output)
        for req in scheduler_output.scheduled_new_reqs:
            self._add_request(req)
        self._sync_cached(scheduler_output)

        req_ids = list(scheduler_output.num_scheduled_tokens)
        scheduled = dict(scheduler_output.num_scheduled_tokens)
        for req_id in req_ids:
            if self.table.get(req_id).is_prefill_done and scheduled[req_id] != 1:
                raise RuntimeError(
                    f"请求 {req_id} 解码步调度了 {scheduled[req_id]} 个 token，只支持 1 个")

        # logprobs 的宽度按整批取最大值，与 vLLM 的 Sampler 一致。
        logprobs_width = self._max_logprobs_width(req_ids)
        # 按并行窗口数分 wave，wave 内每个请求用自己的窗口同时执行。
        sampled: list[list[int]] = [[] for _ in req_ids]
        results: list[SampleResult | None] = [None] * len(req_ids)
        width = min(self.num_parallel_slots, len(req_ids)) if req_ids else 1
        if width <= 1:
            for index, req_id in enumerate(req_ids):
                token, result = self._step(req_id, 0, logprobs_width, scheduled[req_id])
                sampled[index] = [] if token is None else [token]
                results[index] = result
        else:
            if self._executor is None:
                self._executor = ThreadPoolExecutor(max_workers=self.num_parallel_slots)
            for start in range(0, len(req_ids), width):
                wave = req_ids[start:start + width]
                futures = [
                    self._executor.submit(
                        self._step, req_id, index, logprobs_width, scheduled[req_id])
                    for index, req_id in enumerate(wave)
                ]
                for offset, future in enumerate(futures):
                    token, result = future.result()
                    sampled[start + offset] = [] if token is None else [token]
                    results[start + offset] = result

        self._record_stats(scheduled, width)
        return ModelRunnerOutput(
            req_ids=req_ids,
            req_id_to_index={rid: i for i, rid in enumerate(req_ids)},
            sampled_token_ids=sampled,
            logprobs=self._logprobs_lists(results),
            prompt_logprobs_dict={rid: None for rid in req_ids},
        )

    def _add_request(self, req) -> None:
        """注册新请求的采样状态与 vLLM 分配的块号。

        `num_computed_tokens` 是前缀缓存命中的长度：这段 KV 已经在共享块里，
        本请求只从它之后开始写，否则会覆盖别的请求正在用的块。
        """
        validate_new_request(req)
        if req.sampling_params is None:
            raise ValueError(f"请求 {req.req_id} 没有采样参数，PIM 后端只支持生成任务")
        config = SamplingConfig.from_vllm(req.sampling_params)
        slot = self.table.add_request(
            req.req_id, list(req.prompt_token_ids), eos_id=self.eos_id,
            block_ids=list(req.block_ids[0]))
        slot.valid_len = int(req.num_computed_tokens)
        self._samplers[req.req_id] = RequestSampler(config)
        self._prompt_ids[req.req_id] = list(req.prompt_token_ids)
        self._output_ids[req.req_id] = []

    def _forget_request(self, req_id: str) -> None:
        """请求结束时清掉采样状态；未注册过的请求号直接忽略。"""
        self._samplers.pop(req_id, None)
        self._prompt_ids.pop(req_id, None)
        self._output_ids.pop(req_id, None)

    def _max_logprobs_width(self, req_ids: list[str]) -> int | None:
        """本步需要返回的 top-k 宽度；没有请求要 logprobs 时返回 None。"""
        widths = [
            self._samplers[req_id].config.logprobs
            for req_id in req_ids
            if self._samplers[req_id].config.logprobs is not None
        ]
        return max(widths) if widths else None

    @staticmethod
    def _logprobs_lists(results: list[SampleResult | None]) -> LogprobsLists | None:
        """把逐请求结果拼成 vLLM 的 LogprobsLists；没有请求要时返回 None。"""
        requested = [r for r in results if r is not None and r.logprob_token_ids is not None]
        if not requested:
            return None
        width = max(len(r.logprob_token_ids) for r in requested)
        token_ids = np.zeros((len(results), width), dtype=np.int32)
        logprobs = np.zeros((len(results), width), dtype=np.float32)
        ranks = np.zeros(len(results), dtype=np.int32)
        for index, result in enumerate(results):
            if result is None or result.logprob_token_ids is None:
                continue
            count = len(result.logprob_token_ids)
            token_ids[index, :count] = result.logprob_token_ids
            logprobs[index, :count] = result.logprobs
            ranks[index] = result.rank
        return LogprobsLists(token_ids, logprobs, ranks, None)

    def _step(
        self, req_id: str, slot_index: int = 0, logprobs_width: int | None = None,
        scheduled: int | None = None,
    ) -> tuple[int | None, SampleResult | None]:
        """执行一个请求的一步，使用第 `slot_index` 号激活/块表窗口。

        prefill 未完成时只写 KV、不采样，返回 `(None, None)`：vLLM 对还在
        prefill 的请求期望空 token 列表。
        """
        slot = self.table.get(req_id)
        if not slot.is_prefill_done:
            return self._prefill_step(slot, scheduled, slot_index, logprobs_width)
        return self._decode_step(slot, slot_index, logprobs_width)

    def _prefill_step(
        self, slot, scheduled: int | None, slot_index: int,
        logprobs_width: int | None,
    ) -> tuple[int | None, SampleResult | None]:
        """跑一段 prefill；本段跑完整个 prompt 才采样。"""
        pos = slot.valid_len
        remaining = len(slot.prompt_ids) - pos
        if scheduled is None or scheduled > remaining:
            scheduled = remaining
        chunk = slot.prompt_ids[pos:pos + scheduled]
        logits = prefill_chunk(
            self.compiled, self.hal, slot, chunk, pos, causal_mask_of,
            pad_id=self.eos_id, slot_index=slot_index)
        if slot.valid_len < len(slot.prompt_ids):
            return None, None
        slot.is_prefill_done = True
        token, result = self._sample(slot, logits[0, len(chunk) - 1], logprobs_width)
        # 首个生成 token 也要记进 slot，解码步拿它当输入。
        slot.generated.append(token)
        return token, result

    def _decode_step(
        self, slot, slot_index: int, logprobs_width: int | None,
    ) -> tuple[int, SampleResult]:
        """解码一个 token：把上一步的 token 喂进去，采样出下一个。"""
        captured: list[SampleResult] = []

        def sample_fn(logits) -> int:
            token, result = self._sample(slot, logits, logprobs_width)
            captured.append(result)
            return token

        step_request(
            self.compiled, self.hal, slot, causal_mask_of,
            sample_fn=sample_fn, slot_index=slot_index)
        return captured[-1].token_id, captured[-1]

    def _sample(self, slot, logits, logprobs_width: int | None):
        """在主机侧采样一个 token，并记进该请求的输出序列。"""
        result = self._samplers[slot.request_id].sample(
            logits, self._prompt_ids[slot.request_id],
            self._output_ids[slot.request_id], logprobs_width=logprobs_width)
        self._output_ids[slot.request_id].append(result.token_id)
        return result.token_id, result

    def _sync_cached(self, scheduler_output: SchedulerOutput) -> None:
        """按 vLLM 的块号刷新 cached 请求；被抢占的请求从头或从缓存长度续算。"""
        cached = scheduler_output.scheduled_cached_reqs
        for index, req_id in enumerate(cached.req_ids):
            slot = self.table.get(req_id)
            computed = cached.num_computed_tokens[index]
            resumed = req_id in cached.resumed_req_ids
            # 抢占恢复：块被 vLLM 收回，只能按它重新给的块重算；
            # 已经算过 `computed` 个 token（命中前缀缓存）时从那里接着写。
            if resumed or computed < slot.valid_len:
                slot.reset(list(cached.all_token_ids[req_id]), computed)
            new_blocks = cached.new_block_ids[index]
            if new_blocks:
                # 外层元组按 KV 分组，PIM 只有一组，取第 0 组。
                slot.blocks.adopt(list(new_blocks[0]), replace=resumed)
