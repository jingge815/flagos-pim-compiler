"""用 PIMExecutor 启动 vLLM 的 OpenAI 兼容服务。

引擎与 PIMWorker 必须同进程，所以关闭 vLLM 的多进程执行。
并发上限由 `--max-num-seqs` 决定，它等于编译期的 slot 数。
`--max-num-batched-tokens` 决定一步最多调度多少 token，一个 prefill 分块的
长度不能超过最长 prefill 图（`--pim-prefill-seq-len`）。

PIM 自己的开关走 `--pim-*`，解析后转成环境变量交给 `PIMWorker`。
启动前会做一次几何预检，配置冲突在起服务之前就报出来。

示例：
    python scripts/serve_pim.py --model /path/to/Llama-2-7b-hf \\
        --max-model-len 512 --max-num-seqs 8 --max-num-batched-tokens 4096 \\
        --pim-num-dpus 4
"""

from __future__ import annotations

import argparse
import asyncio
import os

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vllm.engine.arg_utils import EngineArgs
from vllm.entrypoints.openai.api_server import run_server
from vllm.entrypoints.openai.cli_args import (
    FlexibleArgumentParser,
    make_arg_parser,
    validate_parsed_serve_args,
)

from backend.vllm_executor import PIMExecutor
from contracts.op_contract import (
    DPU_MRAM_BYTES, DPU_SYS_RESERVE_BYTES)
from memory.kv_layout import KVRegionSpec, build_kv_layout
from memory.mem_planner import default_hw_budget

# PIM 开关：命令行选项、参数名、对应环境变量、帮助文本。
PIM_FLAGS = (
    ("--pim-num-dpus", "pim_num_dpus", "PIM_NUM_DPUS",
     "PIM 内部张量并行的 DPU 数（默认 4）"),
    ("--pim-mram-bytes", "pim_mram_bytes", "PIM_MRAM_BYTES",
     f"单台 DPU 的 MRAM 字节数（默认 {DPU_MRAM_BYTES // 2**30} GiB）"),
    ("--pim-kv-pool-slots", "pim_kv_pool_slots", "PIM_KV_POOL_SLOTS",
     "全局物理 KV 槽池大小（默认 max_num_seqs × 每请求块数 + 1）"),
    ("--pim-parallel-slots", "pim_parallel_slots", "PIM_PARALLEL_SLOTS",
     "同时在飞的请求窗口数（默认 2，受 max_num_seqs 限制）"),
    ("--pim-prefill-seq-len", "pim_prefill_seq_len", "PIM_PREFILL_SEQ_LEN",
     "prefill 图固定长度（默认 max_model_len）"),
    ("--pim-log-steps", "pim_log_steps", "PIM_LOG_STEPS",
     "每 N 步打印一行运行时指标（默认 0，不打印）"),
)


def add_pim_args(parser: argparse.ArgumentParser) -> None:
    """把 `--pim-*` 选项加到 vLLM 的参数解析器上。

    vLLM 的 `--help` 只列配置组名，所以这组选项用 `--help=PIMConfig` 查看。
    """
    group = parser.add_argument_group(
        "PIMConfig", "PIM 后端参数，等价于同名环境变量")
    for flag, dest, _env, help_text in PIM_FLAGS:
        group.add_argument(flag, dest=dest, type=int, default=None, help=help_text)
    group.add_argument(
        "--pim-profile", dest="pim_profile", action="store_true",
        help="进程退出时打印请求级调度报告（等价于 PIM_PROFILE=1）")


def export_env(args) -> None:
    """把 `--pim-*` 写入环境变量；未指定时保留环境变量或默认值。"""
    for _flag, dest, env, _help in PIM_FLAGS:
        value = getattr(args, dest)
        if value is not None:
            os.environ[env] = str(value)
    if getattr(args, "pim_profile", False):
        os.environ["PIM_PROFILE"] = "1"


def _env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, default))


def _kv_bytes_per_dpu(model_config, block_size: int, pool_slots: int,
                      parallel_slots: int) -> int:
    """估算每台 DPU 的 KV 区字节数（块表缓冲 + 槽池）。

    直接复用 `memory.kv_layout.build_kv_layout`，与编译期 `plan_dpu()` 同一
    公式、同一对齐。预检和真实规划不能再各写一套对齐/取整。
    """
    hf = model_config.hf_config
    head_dim = getattr(hf, "head_dim", None) or hf.hidden_size // hf.num_attention_heads
    num_dpus = _env_int("PIM_NUM_DPUS", 4)
    kv_heads = -(-hf.num_key_value_heads // num_dpus)
    spec = KVRegionSpec(
        dpu_id=0,
        layers=list(range(hf.num_hidden_layers)),
        kv_heads=list(range(kv_heads)),
        q_heads_by_kv={head: [] for head in range(kv_heads)},
        max_seq=model_config.max_model_len,
        head_dim=head_dim,
        dtype_bytes=2,
        kv_base=0,
        slot_tokens=block_size,
    )
    return build_kv_layout(
        spec, default_hw_budget().align, pool_slots=pool_slots,
        block_table_slots=parallel_slots,
    ).kv_allocated_bytes


def preflight(vllm_config) -> None:
    """启动前检查 PIM 的几何约束；有问题一次性列出并以非零码退出。"""
    scheduler = vllm_config.scheduler_config
    model = vllm_config.model_config
    cache = vllm_config.cache_config
    max_model_len = model.max_model_len
    block_size = cache.block_size
    num_request_slots = scheduler.max_num_seqs
    blocks_per_request = -(-max_model_len // block_size)
    # 不设槽池大小时编译期按 MRAM 自动定容，预检里也就没有可比的数。
    raw_pool = os.environ.get("PIM_KV_POOL_SLOTS")
    pool_slots = int(raw_pool) if raw_pool else None
    prefill_seq_len = _env_int("PIM_PREFILL_SEQ_LEN", max_model_len)
    parallel_slots = min(num_request_slots, _env_int("PIM_PARALLEL_SLOTS", 2))
    mram_bytes = _env_int("PIM_MRAM_BYTES", DPU_MRAM_BYTES)
    reserve = _env_int("PIM_SYS_RESERVE_BYTES", DPU_SYS_RESERVE_BYTES)

    problems: list[str] = []
    if prefill_seq_len > max_model_len:
        problems.append(
            f"PIM_PREFILL_SEQ_LEN={prefill_seq_len} 超过 max_model_len={max_model_len}；"
            f"建议加 --pim-prefill-seq-len {max_model_len}")
    if pool_slots is not None and pool_slots <= blocks_per_request:
        problems.append(
            f"PIM_KV_POOL_SLOTS={pool_slots} 装不下一个满长请求："
            f"ceil(max_model_len={max_model_len} / block_size={block_size}) = "
            f"{blocks_per_request} 块，另有一块被 null 块占用；建议加 "
            f"--pim-kv-pool-slots {blocks_per_request * num_request_slots + 1}，"
            f"或去掉该项交给自动定容")
    if scheduler.enable_chunked_prefill:
        budget = scheduler.max_num_batched_tokens
        if budget > prefill_seq_len:
            problems.append(
                f"max_num_batched_tokens={budget} 超过最长 prefill 图 "
                f"{prefill_seq_len}：一个 prefill 分块不超过这个预算；"
                f"建议加 --max-num-batched-tokens {prefill_seq_len}，"
                f"或把 --pim-prefill-seq-len 调到 {budget} 以上")
    if pool_slots is not None and pool_slots > blocks_per_request:
        # 槽池装不下满长请求已在上面单独报过；这里只比装得下的配置，
        # 避免以规划器的异常代替可读的预检结论。
        kv_bytes = _kv_bytes_per_dpu(model, block_size, pool_slots, parallel_slots)
        if kv_bytes > mram_bytes - reserve:
            problems.append(
                f"KV 区每台 DPU 约 {kv_bytes / 2**20:.0f}MiB，超过 MRAM 预算 "
                f"{(mram_bytes - reserve) / 2**20:.0f}MiB（权重还未计入）；"
                f"建议调小 --pim-kv-pool-slots（当前 {pool_slots}）或调大 --pim-mram-bytes")
    if problems:
        raise SystemExit("PIM 启动预检失败：\n  - " + "\n  - ".join(problems))


def main() -> None:
    parser = make_arg_parser(FlexibleArgumentParser())
    add_pim_args(parser)
    args = parser.parse_args()
    args.distributed_executor_backend = PIMExecutor
    validate_parsed_serve_args(args)
    export_env(args)
    preflight(EngineArgs.from_cli_args(args).create_engine_config())
    asyncio.run(run_server(args))


if __name__ == "__main__":
    main()
