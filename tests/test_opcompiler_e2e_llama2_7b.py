"""验证编译线性内核在 Llama-2-7B 各切分策略下的数值和生成结果。"""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pytest
import torch
from transformers import AutoTokenizer, LlamaForCausalLM

sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.hal_numpy import NumpyBackend, NumpyBackendConfig
from contracts.op_contract import PIMHardwareConfig
from genesim_bridge.paths import llama2_7b_model_dir
from graph.strategy import llama_strategy
from memory.mem_planner import HwBudget
from runtime.compile import causal_mask_of, compile_llama2, load_weights
from runtime.executor import run_decode_loop
from runtime.kernels import register_all

MODEL_DIR = llama2_7b_model_dir(required=False)
NUM_DPUS = 8
# build_execution_plan 的默认 tasklet 数，compile_llama2 不另传，硬件配置必须一致。
NUM_TASKLETS = 4
PROMPT = "The capital of France is"
DECODE_STEPS = 16
MAX_SEQ = 64
KV_DTYPE_BYTES = 2  # fp16
NUM_LAYERS = 32

pytestmark = [
    pytest.mark.skipif(
        MODEL_DIR is None or not MODEL_DIR.is_dir(),
        reason="需要在 paths.json 配置 llama2_7b_model_dir",
    ),
    pytest.mark.skipif(not torch.cuda.is_available(), reason="opcompiler_bridge 编译期需要 GPU"),
]


def _strategies():
    """返回要验证编译产物的三种策略：纯张量、混合、纯流水。"""
    return [
        llama_strategy(
            NUM_DPUS, num_stages=num_stages, num_heads=32, num_kv_heads=32,
            intermediate_size=11008, vocab_size=32000, num_layers=NUM_LAYERS,
        )
        for num_stages in (1, 4, 8)
    ]


def test_strategy_set_covers_tensor_hybrid_and_pipeline() -> None:
    """判据 0：三种策略分别落在张量、混合、流水三类上。"""
    kinds = {s.name: s.kind for s in _strategies()}
    assert kinds == {
        "tp8_pp1": "tensor",
        "tp2_pp4": "hybrid",
        "tp1_pp8": "pipeline",
    }


@pytest.fixture(scope="module")
def tokenizer():
    return AutoTokenizer.from_pretrained(MODEL_DIR)


@pytest.fixture(scope="module")
def llama2_model():
    torch.set_grad_enabled(False)
    return LlamaForCausalLM.from_pretrained(MODEL_DIR, dtype=torch.float16).eval()


@pytest.fixture(scope="module")
def hf_reference(tokenizer, llama2_model):
    """返回单卡 HF 贪心解码的 token 序列和文本。"""
    prompt_ids = tokenizer(PROMPT, return_tensors="pt")["input_ids"][0].tolist()
    ref_ids = llama2_model.generate(
        input_ids=torch.tensor([prompt_ids], dtype=torch.long),
        max_new_tokens=DECODE_STEPS, do_sample=False,
    )[0].tolist()
    return prompt_ids, ref_ids, tokenizer.decode(ref_ids, skip_special_tokens=True)


def _wrap_with_numpy_cross_check(orig_compiled_linear_kernel):
    """包装已编译内核并记录其与 NumPy 内核的相对误差。"""
    import runtime.kernels as km

    stats: list[tuple[tuple, str, float]] = []

    def wrapped(hal, dpu_id, cmd):
        shapes = tuple(tuple(s) for s in cmd.payload["arg_shapes"])
        dtype = str(cmd.payload["dtype"])
        if not km._compiled_linear_supports(shapes, dtype):
            return km.linear_kernel(hal, dpu_id, cmd)

        npdt = np.dtype(dtype)
        x_a, w_a = cmd.reads
        (out_a,) = cmd.writes

        # 两次计算使用相同的原始输入。
        x_backup = hal.read_local(dpu_id, x_a.offset, (x_a.length // npdt.itemsize,), npdt).copy()
        w_backup = hal.read_local(dpu_id, w_a.offset, (w_a.length // npdt.itemsize,), npdt).copy()

        km.linear_kernel(hal, dpu_id, cmd)
        ref = hal.read_local(dpu_id, out_a.offset, tuple(cmd.payload["out_shape"]), npdt).copy()

        hal.write_local(dpu_id, x_a.offset, x_backup)
        hal.write_local(dpu_id, w_a.offset, w_backup)
        orig_compiled_linear_kernel(hal, dpu_id, cmd)
        got = hal.read_local(dpu_id, out_a.offset, tuple(cmd.payload["out_shape"]), npdt)

        r, g = ref.astype(np.float32), got.astype(np.float32)
        rel = float(np.abs(g - r).max() / max(np.abs(r).max(), 1e-6))
        stats.append((shapes, dtype, rel))

        # 使用 NumPy 结果继续生成后续 token。
        hal.write_local(dpu_id, out_a.offset, np.ascontiguousarray(ref, dtype=npdt))

    return wrapped, stats


@pytest.mark.parametrize("strategy", _strategies(), ids=lambda s: s.name)
def test_compiled_linear_end_to_end_matches_hf_generate(
    monkeypatch, strategy, tokenizer, llama2_model, hf_reference
) -> None:
    """判据 1：每种策略下编译产物都被真正调用，且生成结果与单卡 HF 一致。"""
    import runtime.kernels as km

    # 包装必须在 register_all 之前完成，注册时才会取到包装后的内核。
    wrapped, stats = _wrap_with_numpy_cross_check(km.compiled_linear_kernel)
    monkeypatch.setattr(km, "compiled_linear_kernel", wrapped)

    prompt_ids_list, ref_ids, ref_text = hf_reference
    hw = HwBudget(mram_bytes=4 * 2**30, align=1024, sys_reserve_bytes=64 * 2**20)
    hardware = PIMHardwareConfig(
        num_dpus=NUM_DPUS,
        num_tasklets=NUM_TASKLETS,
        mram_bytes_per_dpu=hw.mram_bytes,
        wram_bytes_per_dpu=65536,
        # DMA 分块使用独立于 MRAM 布局的字节对齐。
        dma_align=64,
    )
    compiled = compile_llama2(
        llama2_model, strategy, prefill_seq_len=len(prompt_ids_list), max_seq=MAX_SEQ,
        hw=hw, hardware=hardware, kv_dtype_bytes=KV_DTYPE_BYTES,
    )

    backend = NumpyBackend(NumpyBackendConfig(
        num_dpus=NUM_DPUS, mram_bytes_per_dpu=hardware.mram_bytes_per_dpu,
    ))
    register_all(backend, use_compiled_linear=True)
    load_weights(compiled, backend)
    compiled.state.valid_len = 0

    prompt_ids = torch.tensor(prompt_ids_list, dtype=torch.long)
    generated = run_decode_loop(
        compiled.prefill.plan, compiled.decode.plan, backend,
        prompt_ids=prompt_ids, max_new_tokens=DECODE_STEPS,
        eos_id=tokenizer.eos_token_id, state=compiled.state,
        sample_fn=lambda logits: int(np.argmax(logits)),
        prefill_output_cmd_id=compiled.prefill.output_cmd_id,
        decode_output_cmd_id=compiled.decode.output_cmd_id,
        causal_mask_of=causal_mask_of,
    )

    our_ids = prompt_ids_list + generated
    our_text = tokenizer.decode(our_ids, skip_special_tokens=True)

    per = defaultdict(list)
    for shapes, dtype, rel in stats:
        per[(shapes, dtype)].append(rel)
    print(f"\n=== 策略 {strategy.name}（{strategy.kind}）编译产物对拍：共 {len(stats)} 次调用 ===")
    for (shapes, dtype), rels in per.items():
        a = np.array(rels)
        print(
            f"  shapes={shapes} dtype={dtype}: n={len(a)} "
            f"max_rel={a.max():.4e} mean_rel={a.mean():.4e} "
            f"超 5% 容差次数={int((a > 0.05).sum())}"
        )
    print(f"generated (含编译产物): {our_text!r}")
    print(f"generated (HF model.generate): {ref_text!r}")

    assert len(stats) > 0, (
        f"策略 {strategy.name} 没有任何调用走到编译产物路径——"
        "检查本地分片 shape 是否满足 2 的幂约束"
    )
    for (shapes, dtype), rels in per.items():
        assert max(rels) < 0.05, (
            f"{strategy.name} {shapes} ({dtype}) 编译产物与 NumPy 参考值偏差过大: "
            f"max_rel={max(rels):.4e}"
        )

    assert our_ids == ref_ids
    assert our_text == ref_text
    assert our_text.startswith(PROMPT)
