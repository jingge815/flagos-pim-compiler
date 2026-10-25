"""把 PIM 静态编译后端接入 vLLM V1 的 Worker。

分工：vLLM 负责 HTTP/调度/连续批处理，本模块负责把每步的
`SchedulerOutput` 翻译成 PIM 的逐请求执行。核心约束：

- KV 分成固定大小的物理槽，每槽容纳 `block_size` 个 token；每台 DPU 只有一份
  块表缓冲，每步由 host 覆盖，请求通过它间接寻址。物理槽池全局一份，大小由
  `PIM_KV_POOL_SLOTS` 决定，默认 `max_num_seqs × 每请求块数`。
- vLLM 的 block 只是账面数字：`num_gpu_blocks = kv_pool_slots`。并发上限仍由
  `max_num_seqs` 决定；在串行执行且没有块共享的前提下，把池调大不会提高并发。
- 采样固定为贪心（argmax）；温度等采样参数不生效。
"""

from __future__ import annotations

import os

import numpy as np
import torch
from transformers import LlamaForCausalLM

from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheConfig, KVCacheSpec
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.worker.worker_base import WorkerBase

from backend.hal_numpy import NumpyBackend, NumpyBackendConfig
from contracts.op_contract import PIMHardwareConfig
from graph.strategy import llama_strategies
from memory.mem_planner import HwBudget
from runtime.compile import causal_mask_of, compile_llama2, load_weights
from runtime.executor import RequestTable, step_request
from runtime.kernels import register_all


def _argmax(logits) -> int:
    return int(np.argmax(logits))


class PIMWorker(WorkerBase):
    """用 NumpyBackend 跑编译好的 Llama 蓝图，承接 vLLM 的调度结果。"""

    def __init__(self, vllm_config, local_rank=0, rank=0,
                 distributed_init_method="", is_driver_worker=True) -> None:
        super().__init__(vllm_config, local_rank, rank,
                         distributed_init_method, is_driver_worker)
        self.num_request_slots = int(self.scheduler_config.max_num_seqs)
        self.kv_slot_tokens = int(self.cache_config.block_size)
        self.max_seq = int(self.model_config.max_model_len)
        # prefill 图是静态导出的，prompt 右侧补齐到这个长度再执行。
        self.prefill_seq_len = int(
            os.environ.get("PIM_PREFILL_SEQ_LEN", self.max_seq))
        if self.prefill_seq_len > self.max_seq:
            raise ValueError(
                f"PIM_PREFILL_SEQ_LEN={self.prefill_seq_len} 超过 max_model_len={self.max_seq}")
        self.num_dpus = int(os.environ.get("PIM_NUM_DPUS", "4"))
        self.mram_bytes = int(os.environ.get("PIM_MRAM_BYTES", str(4 << 30)))
        # 全局物理槽池大小。默认与「并发上限 × 每请求块数」一致，调大只是留余量；
        # 这里用 max_seq 而不是实际 prompt 长度，与编译期几何保持一致。
        raw = os.environ.get("PIM_KV_POOL_SLOTS")
        self.kv_pool_slots = int(raw) if raw else self.blocks_per_slot() * self.num_request_slots
        if self.kv_pool_slots < self.blocks_per_slot():
            raise ValueError(
                f"PIM_KV_POOL_SLOTS={self.kv_pool_slots} 装不下一个满长请求"
                f"（需要 {self.blocks_per_slot()} 个块）")
        hf = self.model_config.hf_config
        self.eos_id = int(getattr(hf, "eos_token_id", 2) or 2)

        self.hal: NumpyBackend | None = None
        self.compiled = None
        self.table: RequestTable | None = None

    # ---- 设备与模型 ----

    def init_device(self) -> None:
        """初始化 NumpyBackend 并注册全部设备内核。"""
        self.hal = NumpyBackend(NumpyBackendConfig(
            num_dpus=self.num_dpus, mram_bytes_per_dpu=self.mram_bytes))
        register_all(self.hal)

    def load_model(self) -> None:
        """编译 Llama 两张图、写入权重、建立请求槽表。"""
        if self.cache_config.enable_prefix_caching:
            raise ValueError(
                "PIM 后端不支持前缀缓存，请设置 enable_prefix_caching=False")
        # 不改全局梯度开关：关闭时 torch.export 会导出另一张图。
        model = LlamaForCausalLM.from_pretrained(
            self.model_config.model, dtype=torch.float16).eval()
        cfg = model.config
        strategy = llama_strategies(
            self.num_dpus,
            num_heads=cfg.num_attention_heads,
            num_kv_heads=cfg.num_key_value_heads,
            intermediate_size=cfg.intermediate_size,
            vocab_size=cfg.vocab_size,
            num_layers=cfg.num_hidden_layers,
        )[0]
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
        )
        load_weights(self.compiled, self.hal)
        self.table = RequestTable(self.num_request_slots, self.compiled.kv_pool_slots)

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
            self.table.free_slot(req_id)
        for req in scheduler_output.scheduled_new_reqs:
            self.table.add_request(
                req.req_id, list(req.prompt_token_ids), eos_id=self.eos_id)
        self._sync_preempted(scheduler_output)

        req_ids = list(scheduler_output.num_scheduled_tokens)
        sampled: list[list[int]] = []
        for req_id in req_ids:
            slot = self.table.get(req_id)
            scheduled = scheduler_output.num_scheduled_tokens[req_id]
            if not slot.is_prefill_done and len(slot.prompt_ids) > self.prefill_seq_len:
                raise RuntimeError(
                    f"请求 {req_id} 的 prompt 长 {len(slot.prompt_ids)}，"
                    f"超过 prefill 图长度 {self.prefill_seq_len}")
            # 分块 prefill 会把一个 prompt 拆到多步，静态图不支持，直接报错。
            if not slot.is_prefill_done and scheduled != len(slot.prompt_ids):
                raise RuntimeError(
                    f"请求 {req_id} 本步调度 {scheduled} token，"
                    f"prompt 长 {len(slot.prompt_ids)}；不支持分块 prefill")
            if slot.is_prefill_done and scheduled != 1:
                raise RuntimeError(
                    f"请求 {req_id} 解码步调度了 {scheduled} 个 token，只支持 1 个")
            sampled.append([step_request(
                self.compiled, self.hal, slot, causal_mask_of,
                sample_fn=_argmax,
                pad_to=self.prefill_seq_len, pad_id=self.eos_id,
            )])

        return ModelRunnerOutput(
            req_ids=req_ids,
            req_id_to_index={rid: i for i, rid in enumerate(req_ids)},
            sampled_token_ids=sampled,
            prompt_logprobs_dict={rid: None for rid in req_ids},
        )

    def _sync_preempted(self, scheduler_output: SchedulerOutput) -> None:
        """被抢占重算的请求（num_computed_tokens 归零）按全量 token 重放 prefill。"""
        cached = scheduler_output.scheduled_cached_reqs
        for index, req_id in enumerate(cached.req_ids):
            slot = self.table.get(req_id)
            if slot.is_prefill_done and cached.num_computed_tokens[index] == 0:
                slot.reset(list(cached.all_token_ids[req_id]))
