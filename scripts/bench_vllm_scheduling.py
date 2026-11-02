"""vLLM 调度超参基准：同一微型模型按配置矩阵各跑一次，输出 Markdown 对照表。

用法（先 source 环境）：
    python scripts/bench_vllm_scheduling.py                 # 默认 6 组配置
    python scripts/bench_vllm_scheduling.py --json out.json
    python scripts/bench_vllm_scheduling.py --only baseline,block8

每组配置记录步数、每步请求构成、最大并发、并行效率、墙钟和生成 token，
各组的生成 token 必须完全相同，否则返回码非零。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from profile_execution import profile_vllm  # noqa: E402

MAX_SEQ = 32
MAX_NEW_TOKENS = 2
PREFILL_LEN = 32
# 前 16 个 token 相同，前缀缓存能整块命中；后 4 个 token 各自不同。
SHARED = list(range(1, 17))
PROMPTS = [SHARED + [101, 102, 103, 104], SHARED + [201, 202, 203, 204]]

CONFIGS = [
    {"name": "baseline", "max_num_seqs": 2, "block_size": 16, "prefix": False},
    {"name": "seqs1", "max_num_seqs": 1, "block_size": 16, "prefix": False},
    {"name": "block8", "max_num_seqs": 2, "block_size": 8, "prefix": False},
    {"name": "prefix", "max_num_seqs": 2, "block_size": 16, "prefix": True},
    {"name": "seqs4", "max_num_seqs": 4, "block_size": 16, "prefix": False},
    {"name": "budget40", "max_num_seqs": 2, "block_size": 16, "prefix": False,
     "batched_tokens": 40},
    {"name": "chunked", "max_num_seqs": 2, "block_size": 16, "prefix": False,
     "chunked": True, "batched_tokens": 16},
]


def _model_dir(root: Path) -> Path:
    """存一个微型 Llama；上下文 64，够放下 20 token 的 prompt 与前缀。"""
    import torch
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(0)
    model = LlamaForCausalLM(LlamaConfig(
        hidden_size=64, intermediate_size=128, num_hidden_layers=4,
        num_attention_heads=4, num_key_value_heads=4, vocab_size=256,
        max_position_embeddings=64,
    )).eval().half()
    path = root / "tiny-llama"
    model.save_pretrained(path)
    return path


def _run(model_dir: Path, config: dict) -> dict:
    """按一组配置跑一次生成，返回调度指标。"""
    chunked = bool(config.get("chunked", False))
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    os.environ["PIM_PARALLEL_SLOTS"] = str(config.get("parallel_slots", 2))
    # 分块预填充的预算不能超过最长 prefill 桶；整段预填充按 32 建桶。
    os.environ["PIM_PREFILL_SEQ_LEN"] = str(16 if chunked else PREFILL_LEN)
    os.environ.pop("PIM_PREFILL_SEQ_LENS", None)
    os.environ.setdefault("PIM_MRAM_BYTES", str(1 << 28))

    from vllm import LLM, SamplingParams

    from backend.vllm_executor import PIMExecutor

    llm = LLM(
        model=str(model_dir), skip_tokenizer_init=True, dtype="float16",
        max_model_len=MAX_SEQ, max_num_seqs=config["max_num_seqs"],
        max_num_batched_tokens=config.get("batched_tokens", MAX_SEQ),
        enable_chunked_prefill=chunked,
        enable_prefix_caching=config["prefix"],
        block_size=config["block_size"],
        gpu_memory_utilization=0.2,
        distributed_executor_backend=PIMExecutor,
    )
    # 先跑一条无关 prompt，把首次执行的编译与预热挡在计时之外。
    llm.generate([{"prompt_token_ids": list(range(40, 60))}],
                 SamplingParams(max_tokens=1, temperature=0.0))
    worker = llm.llm_engine.model_executor.driver_worker.worker
    prof = profile_vllm(worker)
    t0 = time.perf_counter()
    try:
        outputs = llm.generate(
            [{"prompt_token_ids": prompt} for prompt in PROMPTS],
            SamplingParams(max_tokens=MAX_NEW_TOKENS, temperature=0.0),
        )
    finally:
        wall = time.perf_counter() - t0
        result = prof.stop()
        llm.llm_engine.engine_core.shutdown()

    return {
        "config": config,
        "wall_ms": wall * 1000,
        "steps": [
            {"stage": "prefill" if s.is_prefill else "decode",
             "reqs": s.req_ids, "tokens": s.tokens,
             "efficiency": s.efficiency, "wall_ms": s.duration * 1000}
            for s in result.steps
        ],
        "max_concurrency": result.max_concurrency(),
        "decode_efficiency": result.mean_efficiency(False),
        "tokens": [list(o.outputs[0].token_ids) for o in outputs],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="vLLM 调度超参基准")
    parser.add_argument("--only", default="", help="逗号分隔的配置名")
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()
    wanted = {name for name in args.only.split(",") if name}
    unknown = wanted - {c["name"] for c in CONFIGS}
    if unknown:
        print(f"未知配置：{sorted(unknown)}")
        return 2
    configs = [c for c in CONFIGS if not wanted or c["name"] in wanted]

    results = []
    with tempfile.TemporaryDirectory() as tmp:
        model_dir = _model_dir(Path(tmp))
        for config in configs:
            print(f"运行 {config['name']} ...", flush=True)
            results.append(_run(model_dir, config))

    print("\n| 配置 | 步数 | 最大并发 | 解码效率 | 墙钟 ms | 每步请求数 |")
    print("|---|---|---|---|---|---|")
    for r in results:
        steps = "; ".join(f"{s['stage']}:{len(s['reqs'])}" for s in r["steps"])
        print(f"| {r['config']['name']} | {len(r['steps'])} | "
              f"{r['max_concurrency']} | {r['decode_efficiency']:.2f} | "
              f"{r['wall_ms']:.0f} | {steps} |")

    baseline = results[0]["tokens"]
    for r in results[1:]:
        if r["tokens"] != baseline:
            print(f"token 不一致：{r['config']['name']}")
            return 1
    print(f"\ntoken（{len(baseline)} 个请求）：{baseline}")
    if args.json is not None:
        args.json.write_text(json.dumps(results, ensure_ascii=False, indent=2))
        print(f"结果写入 {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
