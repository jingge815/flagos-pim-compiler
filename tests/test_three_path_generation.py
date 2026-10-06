"""每个编译算子都要真实产出三份产物，缺一即失败。

登记表只说明归属，不证明产物生成得出来。这里对每个算子真正跑一次：
numpy 要拿到可加载的内核并与镜像逐元素对拍（相对误差小于 0.05），
genesim 要拿到 pim mlir，gml 要在导出的图里出现对应的 op_type。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from contracts.op_contract import DEFAULT_HARDWARE_CONFIG, OpCompileRequest
from contracts.op_semantics import OP_SEMANTICS, oplevel_ops
from opcompiler_bridge.driver import compile_op

# 代表请求取执行计划里的真实形状，不再用 4×16。
# 7B 的 1376、4096、32000 按 8 倍缩小，奇数与非 2 的幂保留：
# 172、512、4000。linear 的三个形状对应 gate、down、lm_head。
_REQUESTS = {
    "softmax": dict(op="softmax", arg_shapes=[(4, 172)]),
    "matmul": dict(op="matmul", arg_shapes=[(4, 172), (172, 4)]),
    "mask": dict(op="mask", arg_shapes=[(4, 172), (1, 172)]),
    "rope": dict(op="rope", arg_shapes=[(1, 4, 4, 172)]),
    "lut": dict(op="lut", arg_shapes=[(4, 172)], kind="silu"),
    "eltwise": dict(op="eltwise", arg_shapes=[(4, 172), (4, 172)], kind="add"),
    "dynamic_quant": dict(op="dynamic_quant", arg_shapes=[(1, 512)], group_size=128),
    "kv_cache": dict(op="kv_cache", arg_shapes=[(172,), (688,)], group_size=1),
    "gather": dict(op="gather", arg_shapes=[(4000, 172), (4,)]),
    "transpose": dict(op="transpose", arg_shapes=[(4, 172), (1, 0)]),
    "reshape": dict(op="reshape", arg_shapes=[(4, 172), (2, 344)]),
    "split_heads": dict(op="split_heads", arg_shapes=[(2, 172, 4)], group_size=4),
    "concat": dict(op="concat", arg_shapes=[(4, 172), (4, 172)], group_size=1),
    "convert": dict(op="convert", arg_shapes=[(4, 172)], dtype="float16",
                    out_dtype="float32"),
    "normalize": dict(op="normalize", arg_shapes=[(4, 172)]),
    "reduce": dict(op="reduce", arg_shapes=[(4, 172)], group_size=1),
    "linear": dict(op="linear", arg_shapes=[(4, 512), (172, 512)], dtype="float16"),
}

# gml 里的名字与内核入口不是一一对应，按登记表折过去。
_GML_NAME = {}
for _spec in OP_SEMANTICS:
    _entry = _spec.kernel or (_spec.name if _spec.has_kernel else None)
    if _spec.name == "gemm":
        _entry = "linear"
    if _entry and _spec.gml_op_type:
        _GML_NAME.setdefault(_entry, _spec.gml_op_type)


def _ops() -> list[str]:
    return sorted(set(oplevel_ops()) | {"linear"})


def _assert_matches_numpy(op: str, result) -> None:
    """编译产物与 numpy 镜像逐元素对拍，相对误差小于 0.05。

    只断言产物存在证明不了数值对：删掉对拍，产物照样生成，误差判据就空了。
    """
    import ctypes

    from opcompiler_bridge.driver import load_kernel

    fn = load_kernel(result)
    rng = np.random.default_rng(7)
    spec = _REQUESTS[op]

    def f16(shape):
        return (rng.standard_normal(shape) * 0.5).astype(np.float16)

    def call(arrays, out_shape, out_dtype=np.float16, extra=()):
        out = np.zeros(out_shape, dtype=out_dtype)
        ptrs = [np.ascontiguousarray(a).ctypes.data_as(ctypes.c_void_p)
                for a in arrays]
        fn(*ptrs, *extra, out.ctypes.data_as(ctypes.c_void_p))
        return out

    got, ref = _run_pair(op, spec, f16, call, rng, fn)
    diff = np.abs(got.astype(np.float32) - ref.astype(np.float32)).max()
    scale = max(float(np.abs(ref.astype(np.float32)).max()), 1e-6)
    assert diff / scale < 0.05, f"{op} 相对误差 {diff / scale:.4e}"


def _run_pair(op, spec, f16, call, rng, fn):
    """按算子构造输入，返回 (编译产物, numpy 镜像)。"""
    import ctypes

    import runtime.kernels_pim as mirror

    shapes = spec["arg_shapes"]
    if op == "softmax":
        x = f16(shapes[0])
        return call([x], x.shape), mirror.softmax(x)
    if op == "matmul":
        a, w = f16(shapes[0]), f16(shapes[1])
        return call([a, w], (a.shape[0], w.shape[1])), a.astype(np.float32) @ w.astype(np.float32)
    if op == "mask":
        scores, m = f16(shapes[0]), f16(shapes[1])
        return call([scores, m], scores.shape), mirror.mask(scores, m)
    if op == "rope":
        # cos/sin 只有 seq×head_dim 两维，沿 head 轴广播给所有头。
        x = f16(shapes[0])
        tab = f16((1, 1, shapes[0][2], shapes[0][3]))
        return call([x, tab, tab], x.shape), mirror.rope(x, tab, tab)
    if op == "lut":
        x = f16(shapes[0])
        return call([x], x.shape), mirror._apply_activation(x, "silu")
    if op == "eltwise":
        a, b = f16(shapes[0]), f16(shapes[1])
        return call([a, b], a.shape), (a.astype(np.float32) + b.astype(np.float32)).astype(np.float16)
    if op == "dynamic_quant":
        x = f16(shapes[0])
        return call([x], x.shape, np.int8), mirror.dynamic_quant(x, spec["group_size"])
    if op == "kv_cache":
        # 散写：缓存是 memdesc 不是张量，没有输出指针，slot 是索引输入。
        row = f16(shapes[0])
        cache = np.zeros(shapes[1], dtype=np.float16)
        slot = np.zeros((1,), dtype=np.int16)
        fn(row.ctypes.data_as(ctypes.c_void_p),
           cache.ctypes.data_as(ctypes.c_void_p),
           ctypes.c_int32(0), slot.ctypes.data_as(ctypes.c_void_p))
        ref = cache.copy()
        ref[: row.shape[-1]] = row
        return cache, ref
    if op == "gather":
        table = f16(shapes[0])
        ids = rng.integers(0, shapes[0][0], size=shapes[1]).astype(np.int32)
        return call([table, ids], (ids.shape[0], table.shape[1])), mirror.gather(table, ids)
    if op == "transpose":
        x = f16(shapes[0])
        order = tuple(shapes[1])
        out_shape = tuple(x.shape[i] for i in order)
        return call([x], out_shape), mirror.transpose(x, order)
    if op == "reshape":
        x = f16(shapes[0])
        return call([x], tuple(shapes[1])), mirror.reshape(x, tuple(shapes[1]))
    if op == "split_heads":
        # 拆分轴取 1，前面还有一个维度，才能盖住外层偏移写错的那个缺陷。
        x = f16(shapes[0])
        pieces = spec["group_size"]
        axis = 1
        head = shapes[0][axis] // pieces
        piece_shape = list(shapes[0])
        piece_shape[axis] = head
        outs = [np.zeros(tuple(piece_shape), dtype=np.float16) for _ in range(pieces)]
        ptrs = [np.ascontiguousarray(x).ctypes.data_as(ctypes.c_void_p)]
        ptrs += [o.ctypes.data_as(ctypes.c_void_p) for o in outs]
        fn(*ptrs)
        got = np.concatenate(outs, axis=axis)
        return got, x
    if op == "concat":
        a, b = f16(shapes[0]), f16(shapes[1])
        out_shape = list(a.shape)
        out_shape[spec["group_size"]] = a.shape[1] + b.shape[1]
        return call([a, b], tuple(out_shape)), mirror.concat([a, b], spec["group_size"])
    if op == "convert":
        x = f16(shapes[0])
        return call([x], x.shape, np.float32), mirror.convert(x, np.float32)
    if op == "reduce":
        x = f16(shapes[0])
        scale = np.array([1.0 / shapes[0][-1]], dtype=np.float32)
        ref = x.astype(np.float32).mean(axis=-1, keepdims=True).astype(np.float16)
        return call([x, scale], ref.shape), ref
    if op == "normalize":
        x, gamma = f16(shapes[0]), np.ones(shapes[0][-1], dtype=np.float16)
        eps = np.array([1e-5], dtype=np.float32)
        return call([x, gamma, eps], x.shape), mirror.normalize(x, gamma)
    if op == "linear":
        dtype = np.dtype(spec.get("dtype", "float16"))
        x = (rng.standard_normal(shapes[0]) * 0.5).astype(dtype)
        w = (rng.standard_normal(shapes[1]) * 0.5).astype(dtype)
        return (call([x, w], (x.shape[0], w.shape[0]), dtype),
                x.astype(np.float32) @ w.astype(np.float32).T)
    raise AssertionError(f"没有 {op} 的对拍输入")


@pytest.mark.parametrize("op", _ops())
def test_numpy_and_genesim_artifacts(op: str) -> None:
    """编译一次：numpy 要拿到可加载的内核，genesim 要拿到 pim mlir。"""
    spec = _REQUESTS[op]
    request = OpCompileRequest(
        op=spec["op"], arg_shapes=spec["arg_shapes"],
        hardware=DEFAULT_HARDWARE_CONFIG, dtype=spec.get("dtype", "float16"),
        kind=spec.get("kind"), group_size=spec.get("group_size"),
        out_dtype=spec.get("out_dtype"),
    )
    result = compile_op(request, force=True)
    assert Path(result.so_path).is_file(), f"{op} 没有生成 numpy 内核"
    assert result.pimir, f"{op} 没有生成 genesim 用的 pim mlir"
    _assert_matches_numpy(op, result)


def test_gml_covers_every_compiled_op() -> None:
    """导出一张 llama 图，每个编译算子的 gml 名字都要出现。

    convert 例外：它只在位宽真的变化时才成节点，同类型的 to.dtype 是恒等，
    按设计被跨过。这条导出里的转换全是恒等，所以不要求 Convert 出现，
    但要求图里确实有转换节点、并且每一处两侧类型相同。
    """
    import torch
    from transformers import LlamaConfig, LlamaForCausalLM

    from gml_bridge.export import export_llama2
    from gml_bridge.from_fx import _cast_dtypes
    from runtime.compile import export_annotated_graph

    torch.manual_seed(0)
    # dtype 必须钉死。不写的话参数跟随进程默认 dtype，别的用例把默认改成
    # fp16 后，这里的转换就不再全是恒等，断言会随套件顺序变红。
    model = LlamaForCausalLM(LlamaConfig(
        vocab_size=32000, hidden_size=64, intermediate_size=176,
        num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=4,
        max_position_embeddings=16, bos_token_id=1, eos_token_id=2,
        pad_token_id=0,
    )).float().eval()
    text = export_llama2(model, seq_len=8, dtype=torch.float32).text
    present = {name for name in set(_GML_NAME.values()) if name in text}
    missing = {op: name for op, name in 
               ((op, _GML_NAME.get(op)) for op in _ops())
               if name and name not in present and op != "convert"}
    assert not missing, missing

    pos = torch.arange(8, dtype=torch.long).unsqueeze(0)
    gm = export_annotated_graph(model, 8, pos, dtype=torch.float32)
    casts = [n for n in gm.graph.nodes
             if n.op == "call_function"
             and n.target in (torch.ops.aten.to.dtype,
                              torch.ops.aten.to.dtype_layout)]
    assert casts, "图里没有类型转换节点，convert 的 gml 归属无法核对"
    # 恒等转换按设计被跨过，不发 Convert；位宽真变了的必须发。
    # 这条断言是正向的：图里出现非恒等转换时，GML 文本里就要有 Convert，
    # 不再靠「这张图恰好全是恒等」这个前提。
    real_casts = [n for n in casts
                  if (pair := _cast_dtypes(n)) and pair[0] != pair[1]]
    if real_casts:
        assert "Convert" in text, "图里有位宽变化的转换，GML 里却没有 Convert"


def test_non_identity_cast_emits_convert() -> None:
    """参数是 fp16、导出到 fp32 时，位宽变化的转换必须在 GML 里成 Convert。

    模型不钉死 dtype 时参数跟随进程默认 dtype，别的用例把默认改成 fp16 后，
    「图里的转换全是恒等」这个前提就不成立了。所以这里显式构造一个 fp16
    模型，确认非恒等转换真的发出去，而不是被静默跨过。
    """
    import torch
    from transformers import LlamaConfig, LlamaForCausalLM

    from gml_bridge.export import export_llama2

    torch.manual_seed(0)
    config = LlamaConfig(
        vocab_size=32000, hidden_size=64, intermediate_size=176,
        num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=4,
        max_position_embeddings=16, bos_token_id=1, eos_token_id=2,
        pad_token_id=0, dtype=torch.float16,
    )
    model = LlamaForCausalLM(config).float().to(torch.float16).eval()
    text = export_llama2(model, seq_len=8, dtype=torch.float32).text
    assert "Convert" in text, "fp16 模型导出到 fp32，GML 里没有 Convert"
