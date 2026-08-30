"""图编译器 → 算子编译器 → numpy 执行，端到端全流程验证（真实 llama2-7b）。

跟 `tests/test_natural_prompt_llama2_7b.py` 是同一套编排器/同一个真实 prompt，
唯一区别是 `register_all(backend, use_compiled_linear=True)`——`linear` 算子
不再全部走手写 NumPy，凡是 `opcompiler_bridge` 能覆盖的 shape（M/K/N 都是 2
的幂、K>=16，见 `docs/opcompiler_bridge-20260825.md` "范围边界"一节）都走真实
编译出来的 `.so`：

    图编译器（问题 1/2/3/6/7/8）
        -> contracts/op_contract.py 的 OpCompileRequest
        -> opcompiler_bridge.driver.compile_op
             (Triton 编译 -> pim mlir -> EmitC -> C -> gcc)
        -> runtime/kernels.py 的 compiled_linear_kernel
        -> backend/hal_numpy.py 的 NumpyBackend（真实 MRAM 内存）

判据比 `test_natural_prompt_llama2_7b.py` 多一层：
1. 生成文本必须跟 HF `model.generate()` 逐字符一致（跟原测试一样）；
2. 额外插桩比对——每次调用编译产物之前，先用手写 `linear_kernel`（NumPy）
   在同一份输入上算一遍作参考，比较两者输出，确认编译产物本身数值正确，
   而不只是"生成文本凑巧对上"。

**插桩细节值得读一遍，因为它反映了这条链路一个容易踩的坑**：必须先把
`x`/`w` 的原始字节备份下来，两次计算（NumPy 和编译产物）都从同一份干净备份
读，不能"先跑 NumPy 拿参考值、再跑编译产物"——因为 `mem_planner` 会把一个
用完即死的激活的地址复用给这个算子的输出，`x` 和 `out` 可能是同一块内存，
NumPy 那一步的写入会污染编译产物随后要读的 `x`（这个 bug 在开发这条链路时
真实出现过一次，表现为"编译产物相对误差 100%"，后来定位到是插桩脚本自己的
问题，不是编译产物错——教训是验证别名安全性的工具自己也不能有别名副作用）。

跑一次大约 4-5 分钟（decode 循环里编译产物是标量三层循环，比走 BLAS 的
NumPy 慢，但慢得还在可接受范围）。需要本地 Llama-2-7b-hf 权重 + GPU（编译期
要在 GPU 上跑一次目标 kernel 拿 TTIR）。
"""

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
from comm.plan import build_comm_plan
from contracts.graph_meta import SPEC_META_KEY
from contracts.op_contract import PIMHardwareConfig
from graph.partition import partition_graph
from graph.spec_prop import llama_shard_config, propagate_specs
from memory.kv_layout import kv_specs_from_placement
from memory.mem_planner import HwBudget, plan_dpu
from runtime.compile import sdpa_layer_map, write_weight_shards
from runtime.exec_plan_gen import build_execution_plan
from runtime.executor import DecodeState, make_sdpa_handler, run_decode_loop
from runtime.kernels import register_all

MODEL_DIR = Path(
    "/media/disk/fengjingge/src/flagOS/flagOS-installed/model-inference/models/Llama-2-7b-hf"
)
NUM_DPUS = 8
PROMPT = "The capital of France is"
DECODE_STEPS = 16
MAX_SEQ = 64
KV_DTYPE_BYTES = 2  # fp16

pytestmark = [
    pytest.mark.skipif(not MODEL_DIR.is_dir(), reason="需要本地 Llama-2-7b-hf 权重"),
    pytest.mark.skipif(not torch.cuda.is_available(), reason="opcompiler_bridge 编译期需要 GPU"),
]


class _PositionalLlama(torch.nn.Module):
    """RoPE 位置显式作为图输入（问题 6 decode 循环专用）。"""

    def __init__(self, model: LlamaForCausalLM) -> None:
        super().__init__()
        self.model = model

    def forward(self, input_ids: torch.Tensor, causal_mask: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
        return self.model(
            input_ids=input_ids, attention_mask=causal_mask, position_ids=position_ids,
            use_cache=False, return_dict=True,
        ).logits


def _causal_mask_of(seq_len: int) -> torch.Tensor:
    if seq_len == 1:
        return torch.zeros(1, 1, 1, 1, dtype=torch.float16)
    blocked = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=1)
    mask = torch.zeros(1, 1, seq_len, seq_len, dtype=torch.float16)
    mask.masked_fill_(blocked, torch.finfo(mask.dtype).min)
    return mask


def _export_graph(model: LlamaForCausalLM, seq_len: int, position_ids: torch.Tensor):
    input_ids = torch.arange(seq_len, dtype=torch.long).unsqueeze(0)
    gm = torch.export.export(
        _PositionalLlama(model), (input_ids, _causal_mask_of(seq_len), position_ids), strict=True
    ).module()
    partition_graph(gm)
    return gm



def _wrap_with_numpy_cross_check(orig_compiled_linear_kernel):
    """包一层 `compiled_linear_kernel`：每次调用先备份原始输入，分别用手写
    NumPy 和编译产物各算一遍，比较，再用 NumPy 的结果写回内存继续跑（避免
    编译产物万一有问题时污染后续 token 的生成，把"验证"和"让流程走完"分开）。

    返回 `(wrapped_fn, stats)`；`stats` 是一个 list，跑完之后按
    `(shapes, dtype)` 分组打印统计。
    """
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

        # 关键：两次计算都要从同一份干净输入读，不能先算一遍再拿它的输出当
        # 参考——mem_planner 可能把 out 的地址和 x 的地址重叠（用完即死的
        # 激活内存被复用），谁先写谁就污染了另一次要读的输入。
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

        # 用 NumPy 的结果继续跑：验证到此为止，不让编译产物万一算错的地方
        # 影响后续 token 的生成——生成文本是否正确和编译产物数值是否正确，
        # 这里分开判断。
        hal.write_local(dpu_id, out_a.offset, np.ascontiguousarray(ref, dtype=npdt))

    return wrapped, stats


@pytest.mark.parametrize("num_tasklets", [4])
def test_compiled_linear_end_to_end_matches_hf_generate(monkeypatch, num_tasklets) -> None:
    """完整走一遍图编译器 -> 算子编译器 -> numpy 执行，真实 llama2-7b decode，
    编译产物覆盖到的每一次 linear 调用都跟手写 NumPy 逐次比对，且最终生成
    文本与 HF `model.generate()` 一致。

    只跑 num_tasklets=4（新默认值）：真实 llama2-7b 端到端一遍耗时已经很长，
    1/8 两个边界（退化回归、tasklet 数超过行数的空转分支）已由
    tests/test_opcompiler_linear.py（小 shape，不跑完整 7B 模型）覆盖，不需要
    在这个端到端测试里重复三遍。
    """
    import runtime.kernels as km

    wrapped, stats = _wrap_with_numpy_cross_check(km.compiled_linear_kernel)
    monkeypatch.setattr(km, "compiled_linear_kernel", wrapped)

    torch.set_grad_enabled(False)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = LlamaForCausalLM.from_pretrained(MODEL_DIR, dtype=torch.float16).eval()
    cfg = model.config

    prompt_ids_list = tokenizer(PROMPT, return_tensors="pt")["input_ids"][0].tolist()
    prefill_seq_len = len(prompt_ids_list)

    prefill_gm = _export_graph(model, prefill_seq_len, torch.arange(prefill_seq_len, dtype=torch.long).unsqueeze(0))
    decode_gm = _export_graph(model, 1, torch.tensor([[0]], dtype=torch.long))

    shard_config = llama_shard_config(
        NUM_DPUS, num_heads=cfg.num_attention_heads, num_kv_heads=cfg.num_key_value_heads,
        intermediate_size=cfg.intermediate_size, vocab_size=cfg.vocab_size,
    )
    prefill_edges = propagate_specs(prefill_gm, shard_config)
    decode_edges = propagate_specs(decode_gm, shard_config)

    head_dim = cfg.hidden_size // cfg.num_attention_heads
    (k_proj,) = [n for n in prefill_gm.graph.nodes if n.op == "get_attr" and "layers.0.self_attn.k_proj.weight" in n.target]
    kv_specs = kv_specs_from_placement(
        k_proj.meta[SPEC_META_KEY], layers=list(range(cfg.num_hidden_layers)), num_kv_heads=cfg.num_key_value_heads,
        num_q_heads=cfg.num_attention_heads, head_dim=head_dim, max_seq=MAX_SEQ, dtype_bytes=KV_DTYPE_BYTES, kv_base=0,
    )
    hw = HwBudget(mram_bytes=4 * 2**30, align=1024, sys_reserve_bytes=64 * 2**20)
    hardware = PIMHardwareConfig(
        num_dpus=NUM_DPUS,
        num_tasklets=num_tasklets,
        mram_bytes_per_dpu=hw.mram_bytes,
        wram_bytes_per_dpu=65536,
        # dma_align 是 WRAM tile 搬运的对齐要求，跟 hw.align（MRAM 里
        # 张量摆放对齐，供 memory/mem_planner.py 用）是两个不同量级的概念，
        # 不能共用同一个值——用 hw.align=1024 会让 kernel_src.py 编出的
        # tile（几百到几万字节）几乎全部不整除，实测触发
        # pim-tile-to-budget 的 DMA 对齐报错。
        dma_align=64,
    )
    prefill_nodes = list(prefill_gm.graph.nodes)
    decode_nodes = list(decode_gm.graph.nodes)
    plans = {d: plan_dpu(d, prefill_nodes, decode_nodes, kv_specs, hw) for d in range(NUM_DPUS)}

    prefill_entries = {e.edge_id: e for e in build_comm_plan(prefill_edges)}
    decode_entries = {e.edge_id: e for e in build_comm_plan(decode_edges)}
    pending_prefill, pending_decode = {}, {}
    for plan in plans.values():
        pending_prefill.update(plan.pending_readers_prefill)
        pending_decode.update(plan.pending_readers_decode)

    state = DecodeState(valid_len=0)


    def make_host_handler(layer_map):
        def host_handler_of(node):
            if "scaled_dot_product_attention" in str(node.target):
                return make_sdpa_handler(layer_map[node.name], kv_specs, state, np.dtype(np.float16))
            return None
        return host_handler_of

    prefill_compiled = build_execution_plan(
        prefill_nodes, prefill_gm, prefill_entries, pending_prefill,
        hardware=hardware,
        host_handler_of=make_host_handler(sdpa_layer_map(prefill_gm)),
        num_tasklets=num_tasklets,
    )
    decode_compiled = build_execution_plan(
        decode_nodes, decode_gm, decode_entries, pending_decode,
        hardware=hardware,
        host_handler_of=make_host_handler(sdpa_layer_map(decode_gm)),
        num_tasklets=num_tasklets,
    )

    backend = NumpyBackend(NumpyBackendConfig(num_dpus=NUM_DPUS, mram_bytes_per_dpu=hw.mram_bytes))
    register_all(backend, use_compiled_linear=True)

    write_weight_shards(prefill_gm, plans, backend)

    def greedy(logits_1d):
        return int(np.argmax(logits_1d))

    prompt_ids = torch.tensor(prompt_ids_list, dtype=torch.long)
    generated = run_decode_loop(
        prefill_compiled.plan, decode_compiled.plan, backend,
        prompt_ids=prompt_ids, max_new_tokens=DECODE_STEPS, eos_id=tokenizer.eos_token_id, state=state,
        sample_fn=greedy, prefill_output_cmd_id=prefill_compiled.output_cmd_id,
        decode_output_cmd_id=decode_compiled.output_cmd_id, causal_mask_of=_causal_mask_of,
    )

    our_ids = prompt_ids_list + generated
    our_text = tokenizer.decode(our_ids, skip_special_tokens=True)

    ref_ids = model.generate(
        input_ids=torch.tensor([prompt_ids_list], dtype=torch.long),
        max_new_tokens=DECODE_STEPS, do_sample=False,
    )[0].tolist()
    ref_text = tokenizer.decode(ref_ids, skip_special_tokens=True)

    per = defaultdict(list)
    for shapes, dtype, rel in stats:
        per[(shapes, dtype)].append(rel)
    print(f"\n=== 编译产物逐次对拍统计：共 {len(stats)} 次调用 ===")
    for (shapes, dtype), rels in per.items():
        a = np.array(rels)
        print(
            f"  shapes={shapes} dtype={dtype}: n={len(a)} "
            f"max_rel={a.max():.4e} mean_rel={a.mean():.4e} "
            f"超 5% 容差次数={int((a > 0.05).sum())}"
        )

    print(f"\nprompt: {PROMPT!r}")
    print(f"generated (our orchestrator, 含编译产物): {our_text!r}")
    print(f"generated (HF model.generate): {ref_text!r}")

    assert len(stats) > 0, "没有任何调用走到编译产物路径——检查 shape 是否满足 2 的幂约束"
    for (shapes, dtype), rels in per.items():
        assert max(rels) < 0.05, f"{shapes} ({dtype}) 编译产物与 NumPy 参考值偏差过大: max_rel={max(rels):.4e}"

    assert our_ids == ref_ids
    assert our_text == ref_text
    assert our_text.startswith(PROMPT)
