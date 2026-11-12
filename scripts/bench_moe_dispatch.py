#!/usr/bin/env python3
"""MoE 动态分派的执行期基准：同一份计划、同一份权重上比较两种语义。

只给执行阶段计时（编译与权装载不计入），重复若干次取中位数，并断言两种
语义生成的 token 完全一致。跳过发生在提交前，因此动态分派省下的是未命中
专家的整批命令，而不只是内核入口的一次判断。

用法（先 source paths.json 里的 pytorch_env_script）：

    # 真实 4 层切片
    QWEN3_30B_MODEL_DIR=/data2/pengcheng-work/Qwen3-30B-A3B-4L \
        python scripts/bench_moe_dispatch.py --new-tokens 2 --repeat 3

    # 无真实权重时用同几何的小模型验证入口本身
    python scripts/bench_moe_dispatch.py --synthetic
"""

from __future__ import annotations

import argparse
import statistics
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.architectures import ARCHITECTURES
from backend.hal_numpy import NumpyBackend, NumpyBackendConfig
from memory.mem_planner import default_site_capacity
from runtime.compile import causal_mask_of, compile_llama2, load_weights
from runtime.executor import RequestTable, step_request
from runtime.kernels import register_all


def _synthetic_model_dir() -> Path:
    """同几何的小 MoE：没有真实权重时也能验证基准入口。"""
    from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM

    torch.manual_seed(5)
    model = Qwen3MoeForCausalLM(Qwen3MoeConfig(
        max_position_embeddings=16, hidden_size=64, moe_intermediate_size=32,
        num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2,
        head_dim=32, vocab_size=128, num_experts=8, num_experts_per_tok=2,
        norm_topk_prob=True)).eval().to(torch.float16)
    path = Path(tempfile.mkdtemp(prefix="moe-bench-"))
    model.save_pretrained(path)
    return path


def _load(model_dir: Path, num_dpus: int):
    model_type, arch = _architecture(model_dir)
    if model_type != "qwen3_moe":
        raise SystemExit(
            f"基准只适用于 MoE 模型，{model_dir} 的 model_type={model_type!r}")
    loaded = arch.load(str(model_dir))
    strategy = arch.strategy(num_dpus, loaded.config)
    # 与主链/真实测试同一份容量契约，执行期基准不能另算一套预算。
    hw, hardware = default_site_capacity(num_dpus=num_dpus, num_tasklets=4)
    compiled = compile_llama2(
        loaded, strategy, prefill_seq_len=8, max_seq=32, hw=hw,
        hardware=hardware, num_request_slots=1, num_parallel_slots=1,
        moe_execution="dynamic_dispatch")
    return loaded, compiled, hw, strategy


def _architecture(model_dir: Path):
    from backend.architectures import architecture_of
    return architecture_of(model_dir)


def _run_steps(compiled, backend, prompt_ids, new_tokens):
    """执行一次 prefill + (new_tokens-1) 次 decode，返回 (tokens, 时间)。"""
    table = RequestTable(1, compiled.kv_pool_slots)
    slot = table.add_request("bench", list(prompt_ids), eos_id=-1)
    tokens = []
    prefill_s = decode_s = 0.0
    for step in range(new_tokens):
        start = time.perf_counter()
        tokens.append(step_request(
            compiled, backend, slot, causal_mask_of,
            sample_fn=lambda logits: int(np.argmax(logits))))
        elapsed = time.perf_counter() - start
        if step == 0:
            prefill_s = elapsed
        else:
            decode_s += elapsed
    table.free_slot(slot.request_id)
    return tokens, prefill_s, decode_s


def benchmark(model_dir: Path, *, num_dpus: int, new_tokens: int,
              repeat: int, warmup: int) -> dict:
    """两种执行语义各跑 repeat 次，返回计时与 token 对拍结果。"""
    loaded, compiled, hw, strategy = _load(model_dir, num_dpus)
    if compiled.moe_execution != "dynamic_dispatch":
        raise SystemExit("基准要求编译时开启动态分派能力（计划携带 gate 标签）")
    print(f"模型: {model_dir}")
    print(f"策略: {strategy.name}（{num_dpus} DPU），{compiled.moe_execution}")

    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
        prompt_ids = tokenizer("The capital of France is",
                               return_tensors="pt")["input_ids"][0].tolist()
    except (OSError, TypeError, ValueError):
        # 合成小模型没带 tokenizer：用词表内的固定 token 序列即可，
        # 基准只比较两种语义的执行时间与生成结果。
        prompt_ids = [1, 2, 3, 4]

    # 同一份计划、同一个后端、同一份已装载权重：两种语义只差一个运行期开关，
    # 计时差异才能归因到分派本身。
    backend = NumpyBackend(NumpyBackendConfig(
        num_dpus=num_dpus, mram_bytes_per_dpu=hw.mram_bytes,
        moe_dispatch=False))
    register_all(backend)
    load_weights(compiled, backend)

    results = {}
    for label, dispatch in (("dense", False), ("dynamic", True)):
        backend.config.moe_dispatch = dispatch
        backend.moe_dispatch = dispatch
        for _ in range(warmup):
            _run_steps(compiled, backend, prompt_ids, new_tokens)
        # 计数只保留计时轮：逐层核对的是「这一层的命令真的挂上了 gate」，
        # 预热轮混进来会把两层的比例平均掉。
        backend.gate_stats.clear()
        runs = [_run_steps(compiled, backend, prompt_ids, new_tokens)
                for _ in range(repeat)]
        results[label] = {
            "tokens": runs[-1][0],
            "prefill_s": [r[1] for r in runs],
            "decode_s": [r[2] for r in runs],
            "hit_experts": _hit_expert_counts(backend),
            "coverage": _dispatch_coverage(backend),
        }

    dense = results["dense"]
    dynamic = results["dynamic"]
    if dense["tokens"] != dynamic["tokens"]:
        raise SystemExit(
            f"两种语义生成的 token 不一致：dense={dense['tokens']} "
            f"dynamic={dynamic['tokens']}")

    def median(values):
        return statistics.median(values)

    for label, data in results.items():
        print(f"\n[{label}] tokens={data['tokens']} "
              f"命中专家数(lane,dispatch)→专家数={data['hit_experts']}")
        print(f"  逐层命令覆盖(lane/dispatch → 放行/拦下)：{data['coverage']}")
        print(f"  prefill 中位 {median(data['prefill_s']):.4f}s "
              f"（{', '.join(f'{v:.4f}' for v in data['prefill_s'])}）")
        print(f"  decode  中位 {median(data['decode_s']):.4f}s "
              f"（{', '.join(f'{v:.4f}' for v in data['decode_s'])}）")

    prefill_speedup = median(dense["prefill_s"]) / median(dynamic["prefill_s"])
    decode_speedup = (
        median(dense["decode_s"]) / median(dynamic["decode_s"])
        if median(dynamic["decode_s"]) > 0 else float("inf"))
    print(f"\n加速比（同一计划、同一权重、只计执行期）："
          f"prefill {prefill_speedup:.3f}x，decode {decode_speedup:.3f}x")
    results["speedup"] = {
        "prefill": prefill_speedup, "decode": decode_speedup}
    return results


def _hit_expert_counts(backend) -> dict:
    """每个 (lane, dispatch) 本步跳过的专家数，便于核对分派确实生效。"""
    return {f"{key[0]}/{key[1]}": len(skipped)
            for key, skipped in sorted(backend.gate_skips.items())}


def _dispatch_coverage(backend) -> dict:
    """每个 (lane, dispatch) 实际放行/拦下的门控命令数。

    跳过的专家集合写出来了、但这一层根本没有命令挂上来时，比例看起来仍然
    「正常」；只有提交侧计数能证明每个 dispatch 都拦到了本层专家子图。
    """
    return {f"{lane}/{dispatch}": dict(stats)
            for (lane, dispatch), stats in sorted(backend.gate_stats.items())}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", default=None)
    parser.add_argument("--synthetic", action="store_true",
                        help="无真实权重时用同几何的小模型")
    parser.add_argument("--num-dpus", type=int, default=8)
    parser.add_argument("--new-tokens", type=int, default=2)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    args = parser.parse_args()
    if args.synthetic:
        model_dir = _synthetic_model_dir()
    elif args.model_dir:
        model_dir = Path(args.model_dir)
    else:
        import os
        env = os.environ.get("QWEN3_30B_MODEL_DIR")
        if not env:
            raise SystemExit(
                "需要 --model-dir、--synthetic 或 QWEN3_30B_MODEL_DIR")
        model_dir = Path(env)
    benchmark(model_dir, num_dpus=args.num_dpus,
              new_tokens=args.new_tokens, repeat=args.repeat,
              warmup=args.warmup)
    return 0


if __name__ == "__main__":
    sys.exit(main())
