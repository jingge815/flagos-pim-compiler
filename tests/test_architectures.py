"""验证架构白名单：受支持的架构能编译运行，其余在启动期拒绝。"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from transformers import (
    LlamaConfig, LlamaForCausalLM, Qwen2Config, Qwen2ForCausalLM,
    Qwen3Config, Qwen3ForCausalLM, Qwen3MoeConfig, Qwen3MoeForCausalLM)

sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.architectures import ARCHITECTURES
from backend.hal_numpy import NumpyBackend, NumpyBackendConfig
from backend.vllm_worker import validate_engine_config
from contracts.op_contract import PIMHardwareConfig
from graph.strategy import llama_strategies
from memory.mem_planner import HwBudget
from runtime.compile import causal_mask_of, compile_llama2, load_weights
from runtime.executor import (
    RequestTable, prefill_chunk, run_decode_loop, run_one_step)
from runtime.kernels import register_all

NUM_DPUS = 4
NUM_LAYERS = 4
PREFILL_SEQ_LEN = 4
MAX_SEQ = 16
MAX_NEW_TOKENS = 3
PROMPTS = [[1, 2, 3, 4], [5, 6, 7, 8]]
# GQA：4 个 Q 头共用 2 个 KV 头。
MODEL_KWARGS = dict(
    hidden_size=64, intermediate_size=128, num_hidden_layers=NUM_LAYERS,
    num_attention_heads=4, num_key_value_heads=2, vocab_size=128,
)


def _configs(hf_config: SimpleNamespace) -> dict:
    """构造一份全绿的能力矩阵配置，只替换 hf_config。"""
    return {
        "parallel_config": SimpleNamespace(
            tensor_parallel_size=1, pipeline_parallel_size=1),
        "lora_config": None,
        "speculative_config": None,
        "model_config": SimpleNamespace(
            dtype=torch.float16, quantization=None, hf_config=hf_config),
        "scheduler_config": SimpleNamespace(
            enable_chunked_prefill=True, max_num_batched_tokens=PREFILL_SEQ_LEN),
        "prefill_seq_len": PREFILL_SEQ_LEN,
    }


def test_unknown_architecture_lists_supported() -> None:
    """白名单之外报错，并列出受支持的架构。"""
    with pytest.raises(ValueError, match="model_type") as excinfo:
        validate_engine_config(**_configs(SimpleNamespace(model_type="gpt2")))

    assert "llama" in str(excinfo.value) and "qwen2" in str(excinfo.value)
    assert "qwen3" in str(excinfo.value) and "qwen3_moe" in str(excinfo.value)


def test_qwen2_config_passes() -> None:
    """Qwen2 配置默认带滑窗长度，`use_sliding_window=False` 时必须放行。"""
    hf_config = SimpleNamespace(
        model_type="qwen2", rope_scaling=None,
        sliding_window=4096, use_sliding_window=False)
    validate_engine_config(**_configs(hf_config))


def test_qwen2_sliding_window_still_rejected() -> None:
    """真的打开滑窗注意力时仍然拒绝。"""
    hf_config = SimpleNamespace(
        model_type="qwen2", rope_scaling=None,
        sliding_window=4096, use_sliding_window=True)
    with pytest.raises(ValueError, match="sliding_window"):
        validate_engine_config(**_configs(hf_config))


def test_qwen3_moe_config_passes() -> None:
    """Qwen3 MoE（model_type=qwen3_moe）在白名单内，启动期放行。"""
    hf_config = SimpleNamespace(
        model_type="qwen3_moe", rope_scaling=None,
        sliding_window=None, use_sliding_window=False)
    validate_engine_config(**_configs(hf_config))


def test_rope_scaling_rejected() -> None:
    """位置插值/缩放会算出不同的位置编码，启动期直接拒绝。"""
    hf_config = SimpleNamespace(model_type="llama", rope_scaling={"type": "linear"})
    with pytest.raises(ValueError, match="rope_scaling"):
        validate_engine_config(**_configs(hf_config))


def test_sliding_window_rejected() -> None:
    """滑窗注意力不是全注意力，启动期直接拒绝。"""
    hf_config = SimpleNamespace(model_type="llama", sliding_window=4096)
    with pytest.raises(ValueError, match="sliding_window"):
        validate_engine_config(**_configs(hf_config))


@pytest.fixture(scope="module")
def gqa_model():
    """4 个 Q 头共用 2 个 KV 头的 Llama。"""
    torch.manual_seed(0)
    return LlamaForCausalLM(LlamaConfig(
        max_position_embeddings=MAX_SEQ, **MODEL_KWARGS)).eval().to(torch.float16)


def _compile_and_load(model, strategy, num_request_slots=2, num_parallel_slots=2,
                      *, moe_dispatch=None):
    """编译 + 建后端 + 装权重，供各架构的数值测试共用。"""
    hw = HwBudget(mram_bytes=1 << 28, align=64, sys_reserve_bytes=1 << 20)
    hardware = PIMHardwareConfig(
        num_dpus=NUM_DPUS, num_tasklets=4, mram_bytes_per_dpu=hw.mram_bytes,
        wram_bytes_per_dpu=65536, dma_align=64)
    compiled = compile_llama2(
        model, strategy, prefill_seq_len=PREFILL_SEQ_LEN, max_seq=MAX_SEQ,
        hw=hw, hardware=hardware, num_request_slots=num_request_slots,
        kv_slot_tokens=4, num_parallel_slots=num_parallel_slots)
    backend = NumpyBackend(NumpyBackendConfig(
        num_dpus=NUM_DPUS, mram_bytes_per_dpu=hw.mram_bytes,
        num_parallel_slots=compiled.num_parallel_slots,
        moe_dispatch=moe_dispatch))
    register_all(backend)
    load_weights(compiled, backend)
    return compiled, backend


@pytest.fixture(scope="module")
def gqa_compiled_and_backend(gqa_model):
    """按 2 个 KV 头编译，4 台 DPU 分成 2 段、段内张量并行宽度 2。"""
    strategy = llama_strategies(
        NUM_DPUS, num_heads=4, num_kv_heads=2, intermediate_size=128,
        vocab_size=128, num_layers=NUM_LAYERS)[0]
    return _compile_and_load(gqa_model, strategy)


def _greedy(logits):
    return int(np.argmax(logits))


def test_gqa_kv_layout_keeps_two_heads(gqa_compiled_and_backend) -> None:
    """KV 区按 2 个 KV 头切分，每台 DPU 各拿一个。"""
    compiled, _ = gqa_compiled_and_backend
    heads = {head for spec in compiled.kv_specs.values() for head in spec.kv_heads}

    assert heads == {0, 1}


def test_gqa_prefill_matches_hf(gqa_model, gqa_compiled_and_backend) -> None:
    """GQA 的 PIM prefill logits 与 HF 对齐。

    一个 KV 头服务多个 Q 头，缓存行只能写一次；重复写会让后一组的 K/V
    覆盖前一组的缓存位置，logits 直接跑偏到另一个 token。
    """
    compiled, backend = gqa_compiled_and_backend
    prompt = PROMPTS[0]
    table = RequestTable(1, compiled.kv_pool_slots)
    slot = table.add_request("r0", prompt, eos_id=-1)
    got = np.asarray(prefill_chunk(
        compiled, backend, slot, prompt, 0, causal_mask_of))

    with torch.no_grad():
        ref = gqa_model(
            input_ids=torch.tensor([prompt]),
            attention_mask=causal_mask_of(len(prompt)),
            position_ids=torch.arange(len(prompt)).unsqueeze(0)).logits.numpy()
    assert np.abs(got - ref).max() < 1e-3


def test_gqa_two_requests_match_single_runs(gqa_compiled_and_backend) -> None:
    """GQA 模型两请求并发，生成 token 与各自单独运行一致。"""
    compiled, backend = gqa_compiled_and_backend
    alone = [
        run_decode_loop(compiled, backend, prompt_ids=torch.tensor(prompt),
                        max_new_tokens=MAX_NEW_TOKENS, eos_id=-1,
                        sample_fn=_greedy, causal_mask_of=causal_mask_of)
        for prompt in PROMPTS
    ]

    table = RequestTable(2, compiled.kv_pool_slots)
    for index, prompt in enumerate(PROMPTS):
        table.add_request(f"r{index}", prompt, eos_id=-1)
    for _ in range(MAX_NEW_TOKENS):
        run_one_step(compiled, backend, table, causal_mask_of, sample_fn=_greedy)

    got = [list(table.get(f"r{index}").generated) for index in range(2)]
    assert got == alone


@pytest.fixture(scope="module")
def qwen2_model(tmp_path_factory):
    """带随机 q/k/v 偏置的小 Qwen2，存盘后按注册表路径加载。"""
    torch.manual_seed(1)
    model = Qwen2ForCausalLM(Qwen2Config(
        max_position_embeddings=MAX_SEQ, **MODEL_KWARGS)).eval().to(torch.float16)
    # 默认初始化的偏置全为 0，显式随机化才能覆盖到偏置路径。
    with torch.no_grad():
        for layer in model.model.layers:
            for name in ("q_proj", "k_proj", "v_proj"):
                getattr(layer.self_attn, name).bias.normal_(0, 0.02)
    path = tmp_path_factory.mktemp("qwen2")
    model.save_pretrained(path)
    return model, path


def test_qwen2_generate_matches_hf(qwen2_model) -> None:
    """Qwen2 走注册表的加载与切分，贪心生成的 token 与 HF `generate` 逐字一致。"""
    model, path = qwen2_model
    loaded = ARCHITECTURES["qwen2"].load(str(path))
    strategy = ARCHITECTURES["qwen2"].strategy(NUM_DPUS, loaded.config)
    compiled, backend = _compile_and_load(
        loaded, strategy, num_request_slots=1, num_parallel_slots=1)

    prompt = torch.tensor(PROMPTS[0])
    got = run_decode_loop(
        compiled, backend, prompt_ids=prompt, max_new_tokens=MAX_NEW_TOKENS,
        eos_id=-1, sample_fn=_greedy, causal_mask_of=causal_mask_of)
    ref = model.generate(prompt.unsqueeze(0), max_new_tokens=MAX_NEW_TOKENS,
                         do_sample=False)
    assert got == ref[0][len(PROMPTS[0]):].tolist()


@pytest.fixture(scope="module")
def qwen3_model(tmp_path_factory):
    """小 Qwen3：head_dim 显式且不等于 hidden/heads（Qwen3 的解耦特性），
    q/k 的 RMSNorm 权重显式随机化（默认全 1，随机化才能测到装载路径）。"""
    torch.manual_seed(2)
    model = Qwen3ForCausalLM(Qwen3Config(
        max_position_embeddings=MAX_SEQ, hidden_size=64, intermediate_size=128,
        num_hidden_layers=NUM_LAYERS, num_attention_heads=4,
        num_key_value_heads=2, head_dim=32, vocab_size=128,
    )).eval().to(torch.float16)
    with torch.no_grad():
        for layer in model.model.layers:
            layer.self_attn.q_norm.weight.normal_(1.0, 0.1)
            layer.self_attn.k_norm.weight.normal_(1.0, 0.1)
    path = tmp_path_factory.mktemp("qwen3")
    model.save_pretrained(path)
    return model, path


def test_qwen3_generate_matches_hf(qwen3_model) -> None:
    """Qwen3（qk-norm + 无偏置 qkv + 显式 head_dim）贪心生成与 HF 逐 token 一致。"""
    model, path = qwen3_model
    assert model.config.head_dim != model.config.hidden_size // model.config.num_attention_heads

    loaded = ARCHITECTURES["qwen3"].load(str(path))
    strategy = ARCHITECTURES["qwen3"].strategy(NUM_DPUS, loaded.config)
    compiled, backend = _compile_and_load(
        loaded, strategy, num_request_slots=1, num_parallel_slots=1)

    prompt = torch.tensor(PROMPTS[0])
    got = run_decode_loop(
        compiled, backend, prompt_ids=prompt, max_new_tokens=MAX_NEW_TOKENS,
        eos_id=-1, sample_fn=_greedy, causal_mask_of=causal_mask_of)
    ref = model.generate(prompt.unsqueeze(0), max_new_tokens=MAX_NEW_TOKENS,
                         do_sample=False)
    assert got == ref[0][len(PROMPTS[0]):].tolist()


def test_static_moe_forward_matches_hf() -> None:
    """静态 MoE 前向改写与 HF 原版逐元素一致（两种 norm_topk_prob 语义）。"""
    from backend.architectures import make_static_moe_forward

    for norm_topk_prob in (False, True):
        torch.manual_seed(3)
        model = Qwen3MoeForCausalLM(Qwen3MoeConfig(
            max_position_embeddings=MAX_SEQ, hidden_size=64,
            moe_intermediate_size=32, num_hidden_layers=1,
            num_attention_heads=4, num_key_value_heads=2, head_dim=32,
            vocab_size=128, num_experts=8, num_experts_per_tok=2,
            norm_topk_prob=norm_topk_prob,
        )).eval().to(torch.float16)
        blk = model.model.layers[0].mlp
        x = torch.randn(1, 5, 64, dtype=torch.float16)
        with torch.no_grad():
            ref, _ = blk(x)
            got, _ = make_static_moe_forward(blk)(x)
        diff = float((got.float() - ref.float()).abs().max())
        assert diff < 1e-3, f"norm_topk_prob={norm_topk_prob}: max|Δ|={diff}"


class _ConstantLogitsGate(torch.nn.Module):
    """按行返回固定 logits 的门控，用来制造「选中但权重下溢为 0」的场景。"""

    def __init__(self, logits: list[float]) -> None:
        super().__init__()
        self.logits = torch.nn.Parameter(torch.tensor(logits))

    def forward(self, xs: torch.Tensor) -> torch.Tensor:
        return self.logits.to(xs.dtype).expand(xs.shape[0], -1)


class _ConstantExpert(torch.nn.Module):
    """输出固定值的专家，用于构造非有限输出。"""

    def __init__(self, value: float) -> None:
        super().__init__()
        self.value = value

    def forward(self, xs: torch.Tensor) -> torch.Tensor:
        return torch.full_like(xs, self.value)


def _zero_weight_selected_block(norm_topk_prob: bool):
    """3 专家 top-2，选中 [0, 1] 且 1 号权重下溢为 0；1 号输出 inf。"""
    from transformers.models.qwen3_moe.modeling_qwen3_moe import (
        Qwen3MoeSparseMoeBlock)

    cfg = Qwen3MoeConfig(
        hidden_size=4, intermediate_size=8, moe_intermediate_size=8,
        num_hidden_layers=1, num_attention_heads=1, num_key_value_heads=1,
        head_dim=4, vocab_size=16, num_experts=3, num_experts_per_tok=2,
        norm_topk_prob=norm_topk_prob, max_position_embeddings=16)
    block = Qwen3MoeSparseMoeBlock(cfg).eval().to(torch.float16)
    block.gate = _ConstantLogitsGate([0.0, -200.0, -300.0])
    block.experts = torch.nn.ModuleList([
        _ConstantExpert(0.0), _ConstantExpert(float("inf")),
        _ConstantExpert(1.0)])
    return block


@pytest.mark.parametrize("norm_topk_prob", [False, True])
def test_static_moe_forward_masks_unselected_overflow_both_norms(
        norm_topk_prob) -> None:
    """未选中专家的 inf 在两态归一化下都必须掩零，不能以 inf×0=nan 污染。"""
    from transformers.models.qwen3_moe.modeling_qwen3_moe import (
        Qwen3MoeSparseMoeBlock)

    from backend.architectures import make_static_moe_forward

    cfg = Qwen3MoeConfig(
        hidden_size=4, intermediate_size=8, moe_intermediate_size=8,
        num_hidden_layers=1, num_attention_heads=1, num_key_value_heads=1,
        head_dim=4, vocab_size=16, num_experts=2, num_experts_per_tok=1,
        norm_topk_prob=norm_topk_prob, max_position_embeddings=16)
    block = Qwen3MoeSparseMoeBlock(cfg).eval().to(torch.float16)
    block.gate = _ConstantLogitsGate([0.0, -300.0])
    block.experts = torch.nn.ModuleList([
        _ConstantExpert(1.0), _ConstantExpert(float("inf"))])
    xs = torch.ones(1, 2, 4, dtype=torch.float16)
    with torch.no_grad():
        ref, _ = block(xs)
        got, _ = make_static_moe_forward(block)(xs)
    assert torch.equal(got, ref)
    assert torch.isfinite(got).all()


@pytest.mark.parametrize("norm_topk_prob", [False, True])
def test_static_moe_forward_zero_weight_selected_expert_matches_hf(
        norm_topk_prob) -> None:
    """top-k 选中但权重下溢为 0：HF 仍执行该专家，0×inf=nan 必须同样出现。"""
    from backend.architectures import make_static_moe_forward

    block = _zero_weight_selected_block(norm_topk_prob)
    xs = torch.ones(1, 2, 4, dtype=torch.float16)
    with torch.no_grad():
        ref, _ = block(xs)
        got, _ = make_static_moe_forward(block)(xs)
    assert int(torch.isnan(ref).sum()) == 8, "探针前提：HF 对选中专家做 0×inf"
    assert int(torch.isnan(got).sum()) == int(torch.isnan(ref).sum()), (
        "静态前向漏算了 top-k 选中但权重为 0 的专家")
    finite = torch.isfinite(ref)
    assert torch.equal(got[finite], ref[finite])


def test_bool_dtype_comment_describes_the_selected_mask() -> None:
    """bool 张量的注释必须描述真实命中口径，不能退回路由权重非零判定。

    命中只由 top-k 选中索引 scatter 出的 `selected_mask` 决定，路由权重只参与
    加权；注释若写 `routing != 0` 会把读者带回「零权重选中专家被跳过」的旧语义。
    """
    import inspect

    from contracts import dtypes

    source = inspect.getsource(dtypes)
    assert "selected_mask" in source
    assert "routing != 0" not in source


def _single_expert_config() -> Qwen3MoeConfig:
    return Qwen3MoeConfig(
        max_position_embeddings=MAX_SEQ, hidden_size=64,
        moe_intermediate_size=32, num_hidden_layers=NUM_LAYERS,
        num_attention_heads=4, num_key_value_heads=2, head_dim=32,
        vocab_size=128, num_experts=1, num_experts_per_tok=1,
        norm_topk_prob=True)


def test_single_expert_moe_keeps_moe_execution_semantics() -> None:
    """单专家也是合法 MoE：执行语义与默认分块策略不能按「非 MoE」处理。"""
    from runtime.compile import graph_tiling_enabled
    from runtime.exec_plan_gen import resolve_moe_execution

    cfg = _single_expert_config()
    # 默认语义随 2026-11-19 拍板转 exact_dispatch（1 专家同样是合法 exact 配置）。
    assert resolve_moe_execution(cfg) == "exact_dispatch"
    assert resolve_moe_execution(cfg, "dense_sim") == "dense_sim"
    assert graph_tiling_enabled(cfg, None) is False
    # 真正的 dense 配置不携带 MoE 字段，行为必须不变。
    dense = Qwen3Config(hidden_size=64, intermediate_size=128,
                        num_hidden_layers=1, num_attention_heads=4,
                        num_key_value_heads=2, vocab_size=128)
    assert resolve_moe_execution(dense) == ""
    assert graph_tiling_enabled(dense, None) is True


def test_single_expert_moe_generate_matches_hf(tmp_path_factory) -> None:
    """1 专家/top-1 的静态改写、编译与贪心生成都与 HF 稀疏前向一致。"""
    torch.manual_seed(11)
    model = Qwen3MoeForCausalLM(_single_expert_config()).eval().to(torch.float16)
    path = tmp_path_factory.mktemp("qwen3_moe_single")
    model.save_pretrained(path)
    loaded = ARCHITECTURES["qwen3_moe"].load(str(path))
    strategy = ARCHITECTURES["qwen3_moe"].strategy(NUM_DPUS, loaded.config)
    compiled, backend = _compile_and_load(loaded, strategy)
    assert compiled.moe_execution == "exact_dispatch"
    prompt = torch.tensor(PROMPTS[0])
    got = run_decode_loop(
        compiled, backend, prompt_ids=prompt, max_new_tokens=MAX_NEW_TOKENS,
        eos_id=-1, sample_fn=_greedy, causal_mask_of=causal_mask_of)
    ref = model.generate(prompt.unsqueeze(0), max_new_tokens=MAX_NEW_TOKENS,
                         do_sample=False)
    assert got == ref[0][len(PROMPTS[0]):].tolist()


@pytest.fixture(scope="module")
def qwen3_moe_overflow_model(tmp_path_factory):
    """路由只选 0 号专家；1 号专家权重放大到未选中时必然溢出 fp16。"""
    torch.manual_seed(7)
    model = Qwen3MoeForCausalLM(Qwen3MoeConfig(
        max_position_embeddings=MAX_SEQ, hidden_size=64, moe_intermediate_size=32,
        num_hidden_layers=NUM_LAYERS, num_attention_heads=4, num_key_value_heads=2,
        head_dim=32, vocab_size=128, num_experts=2, num_experts_per_tok=1,
        norm_topk_prob=True)).eval().to(torch.float16)
    with torch.no_grad():
        for layer in model.model.layers:
            # logits 全相等时 top1 取首个下标，0 号专家恒命中、1 号恒不命中。
            layer.mlp.gate.weight.zero_()
            for name in ("gate_proj", "up_proj", "down_proj"):
                getattr(layer.mlp.experts[1], name).weight.fill_(60000.0)
    path = tmp_path_factory.mktemp("qwen3_moe_overflow")
    model.save_pretrained(path)
    return model, path


def test_static_moe_forward_masks_unselected_expert_overflow(
        qwen3_moe_overflow_model) -> None:
    """未选中专家内部溢出不得以 inf×0=nan 毒化结果（评审问题 3）。"""
    from backend.architectures import make_static_moe_forward

    model, _ = qwen3_moe_overflow_model
    blk = model.model.layers[0].mlp
    x = torch.randn(1, 5, 64, dtype=torch.float16)
    with torch.no_grad():
        ref, _ = blk(x)                       # HF 稀疏前向：1 号专家根本不执行
        got, _ = make_static_moe_forward(blk)(x)
    assert torch.isfinite(ref).all(), "探针前提：HF 参考必须全有限"
    assert torch.isfinite(got).all(), "静态前向把未选中专家的 inf/nan 带进了结果"
    assert torch.equal(got, ref)


def test_qwen3_moe_unselected_overflow_matches_hf(
        qwen3_moe_overflow_model) -> None:
    """导出/执行全链：稠密模拟与动态分派两态都与 HF 原始稀疏前向一致。"""
    model, path = qwen3_moe_overflow_model
    loaded = ARCHITECTURES["qwen3_moe"].load(str(path))
    strategy = ARCHITECTURES["qwen3_moe"].strategy(NUM_DPUS, loaded.config)
    prompt = torch.tensor(PROMPTS[0])

    def generate(*, moe_dispatch=None) -> list:
        compiled, backend = _compile_and_load(
            loaded, strategy, num_request_slots=1, num_parallel_slots=1,
            moe_dispatch=moe_dispatch)
        return run_decode_loop(
            compiled, backend, prompt_ids=prompt, max_new_tokens=MAX_NEW_TOKENS,
            eos_id=-1, sample_fn=_greedy, causal_mask_of=causal_mask_of)

    dispatch = generate()
    dense = generate(moe_dispatch=False)
    ref = model.generate(prompt.unsqueeze(0), max_new_tokens=MAX_NEW_TOKENS,
                         do_sample=False)
    expected = ref[0][len(PROMPTS[0]):].tolist()
    assert dispatch == expected, "动态分派路径被未选中专家的溢出污染"
    assert dense == expected, "稠密模拟路径被未选中专家的溢出污染"


def test_moe_gate_ops_stay_on_host() -> None:
    """MoE 门控链算子（topk/softmax/sum/scatter/zeros_like）分区后留在主机。"""
    from backend.architectures import make_static_moe_forward
    from contracts.graph_meta import DEVICE_HOST, DEVICE_META_KEY
    from runtime.compile import export_annotated_graph

    torch.manual_seed(4)
    model = Qwen3MoeForCausalLM(Qwen3MoeConfig(
        max_position_embeddings=MAX_SEQ, hidden_size=64, moe_intermediate_size=32,
        num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=2,
        head_dim=32, vocab_size=128, num_experts=4, num_experts_per_tok=2,
        norm_topk_prob=True,
    )).eval().to(torch.float16)
    for layer in model.model.layers:
        layer.mlp.forward = make_static_moe_forward(layer.mlp)

    gm = export_annotated_graph(
        model, 4, torch.arange(4, dtype=torch.long).unsqueeze(0))
    gate_ops = {
        "topk.default", "softmax.int", "sum.dim_IntList",
        "scatter.src", "zeros_like.default",
    }
    seen = set()
    for node in gm.graph.nodes:
        if node.op != "call_function":
            continue
        name = str(node.target).split("aten.")[-1]
        if name in gate_ops:
            seen.add(name)
            assert node.meta[DEVICE_META_KEY] == DEVICE_HOST, f"{name} 应留主机"
    assert seen == gate_ops, f"图里应出现全部门控算子，缺 {gate_ops - seen}"


def test_moe_graph_tiling_policy(monkeypatch) -> None:
    """图级分块开关：MoE 默认关闭、显式参数与 PIM_DISABLE_TILE 都能覆盖。"""
    from runtime.compile import graph_tiling_enabled

    moe = SimpleNamespace(num_experts=128)
    dense = SimpleNamespace(intermediate_size=128)
    monkeypatch.delenv("PIM_DISABLE_TILE", raising=False)
    assert graph_tiling_enabled(moe, None) is False
    assert graph_tiling_enabled(dense, None) is True
    assert graph_tiling_enabled(moe, True) is True
    monkeypatch.setenv("PIM_DISABLE_TILE", "1")
    assert graph_tiling_enabled(dense, None) is False
    assert graph_tiling_enabled(dense, True) is True


def test_moe_compile_does_not_tile_experts(qwen3_moe_model) -> None:
    """正式编译入口下 MoE 图不被图级分块拆散，prefill/decode 都受控。"""
    model, path = qwen3_moe_model
    loaded = ARCHITECTURES["qwen3_moe"].load(str(path))
    strategy = ARCHITECTURES["qwen3_moe"].strategy(NUM_DPUS, loaded.config)
    hw = HwBudget(mram_bytes=1 << 28, align=64, sys_reserve_bytes=1 << 20)
    hardware = PIMHardwareConfig(
        num_dpus=NUM_DPUS, num_tasklets=4, mram_bytes_per_dpu=hw.mram_bytes,
        wram_bytes_per_dpu=65536, dma_align=64)
    compiled = compile_llama2(
        loaded, strategy, prefill_seq_len=PREFILL_SEQ_LEN, max_seq=MAX_SEQ,
        hw=hw, hardware=hardware, num_request_slots=1, num_parallel_slots=1)
    # 4 层 × 8 专家 × (3 linear + silu + mul) + 门控/注意力 ≈ 数百节点；
    # 图级分块会把每个线性的工作集拆成上百块，节点数放大两个数量级。
    for name, gm in (("prefill", compiled.prefill_gm), ("decode", compiled.decode_gm)):
        count = len(list(gm.graph.nodes))
        assert count < 3000, f"{name} 图有 {count} 个节点，MoE 默认分块没有关闭"
        tiled = sum(1 for node in gm.graph.nodes
                    if str(node.target) == "aten.cat.default")
        assert tiled < 100, f"{name} 图出现 {tiled} 个拼接节点，像是分块留下的子块"


def test_moe_weight_rules() -> None:
    """MoE 权重切分规则：router 列切，专家 SwiGLU 复用既有规则。"""
    from graph.strategy import llama_strategies

    strategy = llama_strategies(
        NUM_DPUS, num_heads=4, num_kv_heads=2, intermediate_size=32,
        vocab_size=128, num_layers=NUM_LAYERS)[0]
    assert strategy.match("model.model.layers.0.mlp.gate.weight") == "col"
    assert strategy.match("model.model.layers.0.mlp.experts.3.gate_proj.weight") == "col"
    assert strategy.match("model.model.layers.0.mlp.experts.3.up_proj.weight") == "col"
    assert strategy.match("model.model.layers.0.mlp.experts.3.down_proj.weight") == "row"


@pytest.fixture(scope="module")
def qwen3_moe_model(tmp_path_factory):
    """小 Qwen3 MoE：8 专家 top-2、GQA、head_dim=32、门控重归一化。"""
    torch.manual_seed(5)
    model = Qwen3MoeForCausalLM(Qwen3MoeConfig(
        max_position_embeddings=MAX_SEQ, hidden_size=64, moe_intermediate_size=32,
        num_hidden_layers=NUM_LAYERS, num_attention_heads=4, num_key_value_heads=2,
        head_dim=32, vocab_size=128, num_experts=8, num_experts_per_tok=2,
        norm_topk_prob=True,
    )).eval().to(torch.float16)
    path = tmp_path_factory.mktemp("qwen3_moe")
    model.save_pretrained(path)
    return model, path


def test_qwen3_moe_generate_matches_hf(qwen3_moe_model) -> None:
    """Qwen3 MoE 走注册表加载（含静态化）与切分，贪心生成与 HF 逐 token 一致。"""
    model, path = qwen3_moe_model
    loaded = ARCHITECTURES["qwen3_moe"].load(str(path))
    strategy = ARCHITECTURES["qwen3_moe"].strategy(NUM_DPUS, loaded.config)
    compiled, backend = _compile_and_load(
        loaded, strategy, num_request_slots=1, num_parallel_slots=1)

    prompt = torch.tensor(PROMPTS[0])
    got = run_decode_loop(
        compiled, backend, prompt_ids=prompt, max_new_tokens=MAX_NEW_TOKENS,
        eos_id=-1, sample_fn=_greedy, causal_mask_of=causal_mask_of)
    ref = model.generate(prompt.unsqueeze(0), max_new_tokens=MAX_NEW_TOKENS,
                         do_sample=False)
    assert got == ref[0][len(PROMPTS[0]):].tolist()


def test_qwen3_30b_model_dir_resolution(tmp_path, monkeypatch) -> None:
    """站点键解析不依赖真实权重：环境变量优先、缺省可跳过、必填给配置指引。

    真实目录存在性由 `tests/test_qwen3_30b_real.py` 的 skipif 守护——那才是
    需要权重的验收面，架构单测不应因为本机没配站点路径而变红。
    """
    from genesim_bridge import paths

    monkeypatch.setenv("QWEN3_30B_MODEL_DIR", str(tmp_path))
    assert paths.qwen3_30b_model_dir(required=False) == tmp_path

    monkeypatch.delenv("QWEN3_30B_MODEL_DIR")
    monkeypatch.setattr(
        paths, "_load_file_config", lambda: {"qwen3_30b_model_dir": None})
    assert paths.qwen3_30b_model_dir(required=False) is None
    with pytest.raises(RuntimeError, match="qwen3_30b_model_dir"):
        paths.qwen3_30b_model_dir()


def test_architecture_registry_drives_moe_pipeline(tmp_path) -> None:
    """主链描述从注册表派生：产物名、角色契约、切分宽度、仿真配置。"""
    from backend.architectures import architecture_of

    (tmp_path / "config.json").write_text(
        '{"model_type": "qwen3_moe"}', encoding="utf-8")
    model_type, arch = architecture_of(tmp_path)
    assert model_type == "qwen3_moe"
    assert arch.artifact == "qwen3_moe"
    assert arch.required_roles == {
        "router", "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj", "lm_head"}
    assert arch.sim_config == "sim_qwen3_moe_pimir.yaml"
    cfg = SimpleNamespace(
        num_attention_heads=32, num_key_value_heads=4,
        intermediate_size=6144, moe_intermediate_size=768,
        vocab_size=151936, num_hidden_layers=48)
    widths = arch.widths(cfg)
    # 切专家 FFN 宽，不是 dense 残留的 6144。
    assert widths["intermediate_size"] == 768
    assert arch.strategy(8, cfg).name == "tp4_pp2"


def test_architecture_registry_rejects_unknown_model(tmp_path) -> None:
    from backend.architectures import architecture_of

    (tmp_path / "config.json").write_text(
        '{"model_type": "gpt2"}', encoding="utf-8")
    with pytest.raises(ValueError, match="qwen3_moe"):
        architecture_of(tmp_path)


# ---------------------------------------------------------------------------
# 编译器导出图 ↔ GeneSim 图骨架的完整算子/权重对齐
# ---------------------------------------------------------------------------

_TINY_PARITY_CONFIGS = {
    "llama": dict(
        hidden_size=32, intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, vocab_size=100,
        max_position_embeddings=16),
    "qwen3": dict(
        hidden_size=32, intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, head_dim=8,
        vocab_size=100, max_position_embeddings=16),
    "qwen3_moe": dict(
        hidden_size=32, moe_intermediate_size=16, intermediate_size=64,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        head_dim=8, vocab_size=100, num_experts=4, num_experts_per_tok=2,
        norm_topk_prob=True, max_position_embeddings=16),
}
_TINY_PARITY_CLASSES = {
    "llama": LlamaForCausalLM,
    "qwen3": Qwen3ForCausalLM,
    "qwen3_moe": Qwen3MoeForCausalLM,
}


def _role_key(target: str):
    """编译器权重名 → 与 GeneSim IR 共用的 (角色, 层号, 专家号) 归一身份。"""
    # PositionalLlama 包装后会多一层 `model.` 前缀，按后缀认身份。
    if target.endswith("embed_tokens.weight"):
        return ("embed", -1, -1)
    if target.endswith("lm_head.weight"):
        return ("lm_head", -1, -1)
    if target.endswith("model.norm.weight"):
        return ("final_norm", -1, -1)
    match = re.search(r"\.layers\.(\d+)\.", target)
    if match is None:
        return None
    layer = int(match.group(1))
    if "input_layernorm.weight" in target or "post_attention_layernorm.weight" in target:
        return ("layer_norm", layer, -1)
    if ".self_attn." in target and target.endswith("_norm.weight"):
        return ("q_norm" if ".q_norm." in target else "k_norm", layer, -1)
    if ".self_attn." in target:
        role = target.rsplit(".", 2)[-2]
        return (role, layer, -1)
    if target.endswith(".mlp.gate.weight"):
        return ("router", layer, -1)
    expert = re.search(r"\.experts\.(\d+)\.", target)
    if expert is not None:
        return (target.rsplit(".", 2)[-2], layer, int(expert.group(1)))
    if ".mlp." in target:
        return (target.rsplit(".", 2)[-2], layer, -1)
    return None


def _ir_role_key(tensor_id: str):
    """GeneSim IR 权重 tensor_id → 同一个归一身份。"""
    if tensor_id == "model.embed_tokens.weight":
        return ("embed", -1, -1)
    if tensor_id == "model.lm_head.weight":
        return ("lm_head", -1, -1)
    if tensor_id == "model.final_norm.weight":
        return ("final_norm", -1, -1)
    match = re.search(r"layer\.(\d+)\.", tensor_id)
    if match is None:
        return None
    layer = int(match.group(1))
    name = tensor_id.rsplit(".", 2)[-2]
    if name in ("attention_norm", "ffn_norm"):
        return ("layer_norm", layer, -1)
    if name in ("q_norm", "k_norm", "q_proj", "k_proj", "v_proj", "o_proj"):
        return (name, layer, -1)
    if name == "gate" or tensor_id.endswith(".mlp.gate.weight"):
        return ("router", layer, -1)
    expert = re.search(r"\.experts\.(\d+)\.", tensor_id)
    if expert is not None:
        return (name, layer, int(expert.group(1)))
    return (name, layer, -1)


def _compiler_weight_roles(gm) -> dict:
    """编译导出图里每张权重 → (归一身份, 元素数)，按身份聚合。"""
    roles = {}
    for node in gm.graph.nodes:
        if node.op != "get_attr":
            continue
        target = str(node.target)
        if not target.endswith(".weight"):
            continue
        key = _role_key(target)
        if key is None:
            continue
        shape = tuple(int(d) for d in node.meta["val"].shape)
        count = 1
        for dim in shape:
            count *= dim
        roles[key] = roles.get(key, 0) + count
    return roles


def _genesim_weight_roles(config_path) -> dict:
    """GeneSim 图骨架里每张常驻权重 → (归一身份, 元素数)，按身份聚合。"""
    from genesim_bridge.paths import genesim_root

    root = genesim_root(required=False)
    if root is None:
        pytest.skip("需要在 paths.json 配置 genesim_root")
    src = str(root / "src")
    # 追加而不是插到最前：两仓都有 `backend` 包，GeneSim 的会遮住本仓的
    # `backend.vllm_executor`，vLLM 子进程按 sys.path 重新导入时直接失败。
    if src not in sys.path:
        sys.path.append(src)
    from ir.model_ir import HOST_OP_ID, build_from_hf_config

    ir = build_from_hf_config(str(config_path), execution_dtype="float16")
    roles = {}
    for dependency in ir.dependencies:
        if (dependency.src_op_id != HOST_OP_ID
                or dependency.first_use_scope != "model"):
            continue
        key = _ir_role_key(dependency.tensor_id)
        if key is None:
            continue
        count = int(
            dependency.first_use_bytes_coeffs["constant"]) // 2
        roles[key] = roles.get(key, 0) + count
    return roles


@pytest.mark.parametrize("model_type", sorted(_TINY_PARITY_CONFIGS))
def test_compiler_graph_and_genesim_ir_cover_the_same_weights(
        model_type, tmp_path_factory) -> None:
    """逐架构对拍：角色、形状、权重名与数量必须一一对应。

    主链只查八种投影角色，漏掉的 lm_head 与 q/k norm 照样退出码 0。
    这里从两侧的真实产物出发做集合对拍，缺一个权重就失败。
    """
    from runtime.compile import export_annotated_graph

    kwargs = _TINY_PARITY_CONFIGS[model_type]
    config = _TINY_PARITY_CLASSES[model_type].config_class(**kwargs)
    model = _TINY_PARITY_CLASSES[model_type](config).eval().to(torch.float16)
    path = tmp_path_factory.mktemp(f"parity-{model_type}")
    model.save_pretrained(path)

    loaded = ARCHITECTURES[model_type].load(str(path))
    gm = export_annotated_graph(
        loaded, 2, torch.arange(2, dtype=torch.long).unsqueeze(0),
        dtype=torch.float16)
    compiler_roles = _compiler_weight_roles(gm)
    genesim_roles = _genesim_weight_roles(path / "config.json")

    assert genesim_roles == compiler_roles, (
        f"{model_type}: 编译器导出图与 GeneSim 骨架的权重不一致；"
        f"仅编译器有 {sorted(set(compiler_roles) - set(genesim_roles))}，"
        f"仅 GeneSim 有 {sorted(set(genesim_roles) - set(compiler_roles))}")


def test_arch_strategy_constraints_cover_all_entry_points(tmp_path: Path) -> None:
    """单专家 MoE：自动选择、直接段数与 PartitionPlan 共用同一组专家约束。"""
    import json

    from contracts.partition_plan import PartitionPlan
    from scripts.export_pp_placement import resolve_strategy

    cfg = _single_expert_config()
    arch = ARCHITECTURES["qwen3_moe"]
    assert arch.strategy(4, cfg).name == "tp1_pp4"

    ir = tmp_path / "m.ir"
    ir.write_text(json.dumps({"model_id": ""}))

    def args(**overrides):
        values = {"partition_plan": None, "num_stages": None, "num_dpus": 4}
        values.update(overrides)
        return SimpleNamespace(**values)

    # 自动选择之外的两条入口：合法 TP=1 放行，非法 TP=2 在加载后报错。
    strategy, num_dpus, plan = resolve_strategy(
        args(num_stages=4), arch, cfg, ir, tmp_path)
    assert (strategy.name, num_dpus, plan) == ("tp1_pp4", 4, None)
    with pytest.raises(ValueError, match="num_experts"):
        resolve_strategy(args(num_stages=2), arch, cfg, ir, tmp_path)

    legal = tmp_path / "legal.json"
    PartitionPlan(num_dpus=4, num_stages=4).write(legal)
    strategy, _, _ = resolve_strategy(
        args(partition_plan=legal), arch, cfg, ir, tmp_path)
    assert strategy.name == "tp1_pp4"

    illegal = tmp_path / "illegal.json"
    PartitionPlan(num_dpus=4, num_stages=2).write(illegal)
    with pytest.raises(ValueError, match="num_experts"):
        resolve_strategy(args(partition_plan=illegal), arch, cfg, ir, tmp_path)

    # 稠密模型没有专家约束，行为不变。
    dense = Qwen3Config(
        hidden_size=64, intermediate_size=128, num_hidden_layers=4,
        num_attention_heads=4, num_key_value_heads=2, vocab_size=128)
    strategy, _, _ = resolve_strategy(
        args(num_stages=2), ARCHITECTURES["qwen3"], dense, ir, tmp_path)
    assert strategy.name == "tp2_pp2"


_REAL_TEST_CONTRACT_LITERALS = (
    (r"8\s*\*\s*2\s*\*\*\s*30", "8 GiB 容量字面量"),
    (r"64\s*\*\s*2\s*\*\*\s*20", "64 MiB 预留字面量"),
    (r"dma_align\s*=\s*64", "对齐字面量"),
    (r"wram_bytes_per_dpu\s*=\s*65536", "WRAM 字面量"),
    (r"HwBudget\s*\(", "手写 HwBudget"),
    (r"PIMHardwareConfig\s*\(", "手写 PIMHardwareConfig"),
)


@pytest.mark.parametrize(
    "name", ["test_qwen3_8b_real.py", "test_qwen3_30b_real.py"])
def test_real_weight_tests_derive_capacity_from_the_single_factory(
        name: str) -> None:
    """真实测试不得手写容量三件套；契约变化必须经同一工厂传播。"""
    text = (Path(__file__).parent / name).read_text()
    assert "default_site_capacity(" in text, (
        f"{name} 没有使用 default_site_capacity()：容量与对齐会出现第二套真源")
    for pattern, label in _REAL_TEST_CONTRACT_LITERALS:
        assert re.search(pattern, text) is None, (
            f"{name} 出现{label}：容量三件套必须从 default_site_capacity() 派生")
