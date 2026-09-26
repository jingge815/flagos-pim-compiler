"""算子级 mnemonic：编出的 `.so` 与 numpy 镜像逐元素对拍。

这是 B 路（整算子级 IR → EmitC → C）的验收。在此之前这条链在展开之后就断了：
算子级算子到了降级 pass 那儿被当死代码丢掉，生成的 C 什么都不算，而 numpy 镜像
"对上了"只是因为两边错得一样。

对拍口径：镜像的公式来自 `gml_bridge/phase_data.py`，所以这里比的不是"像不像
硬件"，而是"编译出来的 C 与数值真源是不是同一个函数"。
"""

from __future__ import annotations

import ctypes
import re
import sys
from pathlib import Path
from unittest import mock

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from contracts.op_contract import (
    DEFAULT_HARDWARE_CONFIG,
    OpCompileRequest,
)
from opcompiler_bridge.driver import compile_op, load_kernel
import dataclasses

import runtime.kernels_pim as mirrors


def _pim_passes_available() -> bool:
    try:
        from genesim_bridge.env import assert_pim_passes_available

        assert_pim_passes_available()
    except Exception:
        return False
    return True


pytestmark = pytest.mark.skipif(
    not _pim_passes_available(),
    reason="当前 triton 没有 PIM pass，需重跑 0-install-flagtree.sh",
)

# 单 DPU、小 MRAM：编译产物与硬件规模无关（算子级路径不分配片上缓冲），
# 但要与 A 路用同一份契约，免得两边的缓存键撞车。
_HARDWARE = dataclasses.replace(
    DEFAULT_HARDWARE_CONFIG, num_dpus=1, num_tasklets=1,
    mram_bytes_per_dpu=1 << 20)


def _compile(op: str, shape: tuple[int, ...], *,
             group_size: int | None = None) -> tuple:
    """编一个算子，返回 (ctypes 函数, .so 路径)。"""
    request = OpCompileRequest(
        op=op, arg_shapes=[shape], hardware=_HARDWARE,
        dtype="float16", group_size=group_size,
    )
    result = compile_op(request, force=True)
    assert Path(result.so_path).is_file()
    return load_kernel(result), result


def _call(fn, *arrays, out_shape, out_dtype=np.float16) -> np.ndarray:
    """按 ABI 调一次：输入在前、输出在后，都是裸指针。"""
    out = np.zeros(out_shape, dtype=out_dtype)
    args = [a.ctypes.data_as(ctypes.c_void_p) for a in arrays]
    args.append(out.ctypes.data_as(ctypes.c_void_p))
    fn(*args)
    return out


def test_softmax_kernel_matches_numpy() -> None:
    """Softmax：五相展开后编成 C，与镜像逐元素一致。"""
    fn, result = _compile("softmax", (4, 1024))
    assert result.argtypes == ["int16_t", "int16_t"], (
        "两个裸指针：输入在前、输出在后")
    assert "pim.softmax" not in result.pimir, "pimir 里该是展开后的相位链"

    rng = np.random.default_rng(7)
    x = (rng.standard_normal((4, 1024)) * 2).astype(np.float16)
    got = _call(fn, x, out_shape=(4, 1024))
    ref = mirrors.softmax(x)
    assert np.array_equal(got, ref), (
        f"{int((got != ref).sum())} 个元素不一致，"
        f"最大绝对差 {float(np.abs(got.astype(np.float32) - ref.astype(np.float32)).max())}")


def test_dynamic_quant_kernel_matches_numpy() -> None:
    """动态量化：四相展开后编成 C，int8 结果逐元素一致。"""
    fn, result = _compile("dynamic_quant", (1, 4096), group_size=128)
    # 两个裸指针：输入在前、int8 输出在后。scale 那个操作数在展开后已无人读
    # （相 1 自己产出它），所以不占形参。
    assert result.argtypes == ["int16_t", "int8_t"]

    rng = np.random.default_rng(11)
    x = rng.standard_normal((1, 4096)).astype(np.float16)
    got = _call(fn, x, out_shape=(1, 4096), out_dtype=np.int8)
    ref = mirrors.dynamic_quant(x, 128)
    assert np.array_equal(got, ref), (
        f"{int((got != ref).sum())} 个元素不一致，"
        f"最大绝对差 {int(np.abs(got.astype(int) - ref.astype(int)).max())}")


def test_dq_phase1_scale_matches_phase_data() -> None:
    """相 1 的 ÷256 只有一处真源，两边的副本必须同值。

    `gml_bridge/phase_data.py:DQ_PHASE1_SCALE` 是数值真源，C++ 侧
    （`ExpandPhases.cpp` 的 `kDqPhase1Scale`）必须与它同值才能编出同一个数。
    这个常数在两个语言里各有一份，因为没有 C++ 能读 Python；这条测试是
    唯一把两份钉在一起的东西。少了它，改一边另一边悄悄不动，而产出的 C
    算的是另一个数。
    """
    from gml_bridge.phase_data import DQ_PHASE1_SCALE

    result = _compile_dq((1, 256), group_size=128)
    match = re.search(r"fpsuScale = ([0-9.eE+-]+) : f64", result.pimir)
    assert match, f"相 1 的定标没进 IR:\n{result.pimir}"
    assert float(match.group(1)) == DQ_PHASE1_SCALE, (
        f"C++ 侧的 {float(match.group(1))} != phase_data 的 {DQ_PHASE1_SCALE}")


def test_dq_phase1_scale_reaches_the_c() -> None:
    """×1/256 要真的出现在编出来的 C 里，不能只写在 IR 上。"""
    result = _compile_dq((1, 256), group_size=128)
    from opcompiler_bridge import driver as _driver

    _, emitc = _driver._run_oplevel_triton_opt(result.pimir)
    c_source = _driver._translate_to_c(emitc)
    # 定标是 fp32 逻辑缓冲，常量按 float 字面量落进 C。
    assert "3.906250000e-03f" in c_source, (
        "相 1 的 ÷256 没编进 C——定标只写在 IR 上等于没降级")


def _compile_dq(shape, group_size):
    from contracts.op_contract import DEFAULT_HARDWARE_CONFIG as HW

    request = OpCompileRequest(op="dynamic_quant", arg_shapes=[list(shape)],
                               hardware=HW, dtype="float16",
                               group_size=group_size)
    return compile_op(request, force=True)


def test_dynamic_quant_group_size_reaches_the_kernel() -> None:
    """组大小是编译期参数：换一个必须编出另一份内核。

    缓存键漏掉它的话，第二次会命中第一次的 `.so`，而那份内核按 128 分组，
    结果全错——而且不会报错。
    """
    rng = np.random.default_rng(3)
    x = rng.standard_normal((1, 4096)).astype(np.float16)
    for group_size in (128, 1024):
        fn, _ = _compile("dynamic_quant", (1, 4096), group_size=group_size)
        got = _call(fn, x, out_shape=(1, 4096), out_dtype=np.int8)
        ref = mirrors.dynamic_quant(x, group_size)
        assert np.array_equal(got, ref), f"group_size={group_size} 不一致"


def test_unexpanded_operator_is_refused() -> None:
    """忘了展开就直接降级必须报错，不能当死代码丢掉。"""
    from opcompiler_bridge.driver import _run_passes, _triton_opt

    text = """
module {
  tt.func @sm(%s: tensor<1x64xf16>) {
    %p = pim.softmax %s {axis = 1 : i64} : tensor<1x64xf16> -> tensor<1x64xf16>
    tt.return
  }
}
"""
    with pytest.raises(RuntimeError, match="does not lower the operator-level op"):
        _run_passes(_triton_opt(), text, ["-pim-lower-to-emitc"], "emitc")


def test_gather_kernel_matches_numpy() -> None:
    """词嵌入查表：编成 C 后与 `table[indices]` 逐元素一致。

    方案明确要求它**不准回退主机 embedding**——查表是纯访存，在设备上做和在主机
    上做的区别是那 256 MiB 的表要不要搬过去，这正是编排器要规划的东西。
    """
    request = OpCompileRequest(
        op="gather", arg_shapes=[(100, 8), (1, 5)], hardware=_HARDWARE,
        dtype="float16",
    )
    result = compile_op(request, force=True)
    fn = load_kernel(result)
    # 三个裸指针：表、索引、输出。索引是 i32，不与 f16 存储混用。
    assert result.argtypes == ["int16_t", "int32_t", "int16_t"]

    rng = np.random.default_rng(5)
    table = rng.standard_normal((100, 8)).astype(np.float16)
    ids = np.array([[7, 0, 99, 42, 13]], dtype=np.int32)
    got = _call(fn, table, ids, out_shape=(1, 5, 8))
    ref = mirrors.gather(table, ids)
    assert np.array_equal(got, ref), (
        f"{int((got != ref).sum())} 个元素与 table[indices] 不一致")


def test_gather_kernel_reads_int64_indices() -> None:
    """图上的 token id 是 int64，按 int32 读会把相邻两个 id 拼成一个行号。

    索引宽度跟着 `out_dtype` 走：请求声明 int64，生成的 C 就按 8 字节读。
    """
    request = OpCompileRequest(
        op="gather", arg_shapes=[(100, 8), (1, 5)], hardware=_HARDWARE,
        dtype="float16", out_dtype="int64",
    )
    result = compile_op(request, force=True)
    assert result.argtypes == ["int16_t", "int64_t", "int16_t"], result.argtypes
    rng = np.random.default_rng(6)
    table = rng.standard_normal((100, 8)).astype(np.float16)
    ids = np.array([[7, 0, 99, 42, 13]], dtype=np.int64)
    got = _call(load_kernel(result), table, ids, out_shape=(1, 5, 8))
    assert np.array_equal(got, mirrors.gather(table, ids))


def test_lut_silu_matches_the_closed_form() -> None:
    """编译内核的 SiLU 走闭式，与 numpy 镜像同一条公式。

    31 段弦线表在 32 层里累积后，logits 与 torch 差到 1 以上。闭式在
    域外自然趋近正确的渐近线，不再需要单独的饱和处理。
    """
    fn, _ = _compile_shapes("lut", [[7]], dtype="float16")
    x = np.array([[-8.0, -4.0, -1.0, 0.0, 1.0, 4.0, 23.0]], dtype=np.float16)
    got = _call(fn, x, out_shape=(1, 7))
    xf = x.astype(np.float32)
    ref = (xf / (1.0 + np.exp(-xf))).astype(np.float16)
    assert np.array_equal(got, ref), f"编译内核 {got} 与闭式 {ref} 不一致"


def test_rope_kernel_matches_numpy() -> None:
    """RoPE：三相展开后编成 C，与镜像逐元素一致。

    这条以前编不出来：`pim-lower-to-emitc` 对带半旋转标记的乘法**显式报错**——
    半区交换是个注解，按普通乘法降级会算出形状对、数值错的结果。现在标记进了
    ODS（`rotateHalf`），降级按索引重映射实现，不另建缓冲。
    """
    request = OpCompileRequest(
        op="rope", arg_shapes=[(1, 2, 4, 8)], hardware=_HARDWARE,
        dtype="float16",
    )
    result = compile_op(request, force=True)
    fn = load_kernel(result)
    # 四个裸指针：x、cos、sin、输出。
    assert result.argtypes == ["int16_t"] * 4
    assert "pim.rope" not in result.pimir, "pimir 里该是展开后的三相"

    rng = np.random.default_rng(3)
    x = rng.standard_normal((1, 2, 4, 8)).astype(np.float16)
    cos = rng.standard_normal((1, 1, 4, 8)).astype(np.float16)
    sin = rng.standard_normal((1, 1, 4, 8)).astype(np.float16)
    got = _call(fn, x, cos, sin, out_shape=(1, 2, 4, 8))
    ref = mirrors.rope(x, cos, sin)
    assert np.array_equal(got, ref), (
        f"{int((got != ref).sum())} 个元素不一致，最大绝对差 "
        f"{float(np.abs(got.astype(np.float32) - ref.astype(np.float32)).max())}")


def test_rope_kernel_carries_the_unit_and_the_tail_card() -> None:
    """driver 路径与图路径必须发同一份 RoPE IR。

    原先 driver 只发 `{numHeads}`，图路径还发 `unit` 与（K 路的）
    `tailCardValue`。缺 `tailCardValue` 时展开 pass 把末相当成卡值 0，
    K 路写 cache 前的重定标与 `dq_contraction` 标记一起消失，而 IR 仍合法。
    """
    from opcompiler_bridge.oplevel_kernel import rope_kernel

    q = rope_kernel("q", 32, 16, 128, tail_card_value=0)
    k = rope_kernel("k", 32, 16, 128, tail_card_value=3)
    assert "unit = #pim.unit<cstl>" in q
    assert "unit = #pim.unit<cstl>" in k
    assert "tailCardValue" not in q
    assert "tailCardValue = 3 : i64" in k

    request = OpCompileRequest(
        op="rope", arg_shapes=[(1, 2, 4, 8)], hardware=_HARDWARE,
        dtype="float16",
    )
    result = compile_op(request, force=True)
    # 展开后的 IR 必须带着 unit；缺了说明文本没发。
    assert "unit = #pim.unit" in result.pimir or "unit = #pim.unit" in (result.pimir or "")


def test_rope_mirror_rounds_per_phase() -> None:
    """镜像必须**逐相**落 fp16，不能全程 f32 算完再截。

    硬件跑三次引擎遍历，中间结果真的落进缓冲。实测两种口径在 1/3 的元素上差一个
    fp16 ulp——全程 f32 的那份不是"更精确"，是少模拟了两次落盘，会让编译内核
    与镜像对不上而看起来像内核错了。
    """
    rng = np.random.default_rng(3)
    x = rng.standard_normal((1, 2, 4, 8)).astype(np.float16)
    cos = rng.standard_normal((1, 1, 4, 8)).astype(np.float16)
    sin = rng.standard_normal((1, 1, 4, 8)).astype(np.float16)

    half = x.shape[-1] // 2
    xf = x.astype(np.float32)
    rotated = np.concatenate((-xf[..., half:], xf[..., :half]), axis=-1)
    whole_f32 = (xf * cos.astype(np.float32)
                 + rotated * sin.astype(np.float32)).astype(np.float16)

    phased = mirrors.rope(x, cos, sin)
    assert not np.array_equal(phased, whole_f32), (
        "两种口径应当有差别；若相同则这条测试失去意义（换个种子或形状）")


def test_matmul_w4a8_kernel_matches_numpy() -> None:
    """w4a8 投影：按组反量化累加编成 C，与镜像逐元素一致。

    两个操作数都是 i8，所以绕开 fp16 那条检查；收缩维按 128 分组。
    """
    request = OpCompileRequest(
        op="matmul", arg_shapes=[(16, 128), (128, 8)], hardware=_HARDWARE,
        dtype="int8", group_size=128,
    )
    result = compile_op(request, force=True)
    assert Path(result.so_path).is_file()
    assert "groupDequantAccum = true, groupSize = 128" in result.pimir, (
        "组反量化累加没进 IR，编出来的还是整段累加")
    fn = load_kernel(result)
    # 四个裸指针：A、W、逐组定标（fp16 → int16_t）、int8 输出。
    assert result.argtypes == ["int8_t", "int8_t", "int16_t", "int8_t"]

    rng = np.random.default_rng(3)
    # int4 存成 i8，值域就是 [-8, 7]。
    a = rng.integers(-8, 8, size=(16, 128)).astype(np.int8)
    w = rng.integers(-8, 8, size=(128, 8)).astype(np.int8)
    scales = rng.uniform(0.5, 2.0, size=(1, 8)).astype(np.float16)
    got = _call(fn, a, w, scales, out_shape=(16, 8), out_dtype=np.int8)
    ref = mirrors.matmul(a, w, group_size=128, scales=scales)
    assert np.array_equal(got, ref), (
        f"{int((got != ref).sum())} 个元素不一致，"
        f"最大绝对差 {int(np.abs(got.astype(int) - ref.astype(int)).max())}")


def test_matmul_group_boundary_is_in_the_emitted_loops() -> None:
    """组边界必须出现在 EmitC 的循环结构里，而不是只在 IR 属性上。

    「整段累加完再统一反量化」编出来的 C 没有内层的组循环，那种内核在 IR 上
    看不出差别，跑起来数值也常常一样——所以这个检查盯的是降级后的结构本身。
    """
    from opcompiler_bridge.driver import _run_passes, _triton_opt

    request = OpCompileRequest(
        op="matmul", arg_shapes=[(4, 4), (4, 2)], hardware=_HARDWARE,
        dtype="int8", group_size=2,
    )
    result = compile_op(request, force=True)
    assert result.pimir, "需要 pimir 文本才能单独跑降级段"
    emitc = _run_passes(_triton_opt(), result.pimir, ["-pim-lower-to-emitc"],
                        "emitc")
    # 四层循环：m、n、组、k。最外层带方言前缀，其余在 region 内打印裸 `for`。
    assert emitc.count("emitc.for") == 1 and emitc.count("  for ") + emitc.count(" for ") >= 3, emitc
    # 组边界上的反量化乘法：组循环里先开一个 f32 累加器，内层 k 走完后乘定标
    # 因子再折进总数。整段累加的形式没有这个内层累加器。
    assert emitc.count("emitc.variable") == 2, (
        f"该有两个 f32 累加器（总数 + 组），数到 {emitc.count('emitc.variable')}")

    # 对照：不开组反量化时组循环退化成一趟，累加器仍在但组边界不再必要。
    plain = dataclasses.replace(request, group_size=None)
    plain_result = compile_op(plain, force=True)
    assert "groupDequantAccum" not in plain_result.pimir


def _compile_shapes(op: str, shapes: list, **kw):
    """按多个形状编一个算子，返回 (ctypes 函数, 结果)。"""
    request = OpCompileRequest(op=op, arg_shapes=shapes, hardware=_HARDWARE, **kw)
    result = compile_op(request, force=True)
    assert Path(result.so_path).is_file(), f"{op} 没出 .so"
    return load_kernel(result), result


def test_rms_normalize_kernel_matches_numpy() -> None:
    """RMS 归一化：编成 C 后与镜像逐元素一致。ε 必须显式传入。"""
    fn, result = _compile_shapes("normalize", [(4, 8)], dtype="float16")
    assert result.argtypes == ["int16_t", "int16_t", "float", "int16_t"]
    rng = np.random.default_rng(5)
    x = (rng.standard_normal((4, 8)) * 3).astype(np.float16)
    g = (rng.standard_normal(8) * 0.5 + 1).astype(np.float16)
    eps = np.array([1e-5], dtype=np.float32)
    got = _call(fn, x, g, eps, out_shape=(4, 8))
    ref = mirrors.normalize(x, g, epsilon=float(eps[0]))
    assert np.array_equal(got, ref), (
        f"{int((got != ref).sum())} 个元素不一致")
    # 换一个不是 1e-5 的 ε：判据必须能失败。默认值链路会让两边都用 1e-5。
    eps2 = np.array([1e-3], dtype=np.float32)
    got2 = _call(fn, x, g, eps2, out_shape=(4, 8))
    ref2 = mirrors.normalize(x, g, epsilon=float(eps2[0]))
    assert np.array_equal(got2, ref2)
    assert not np.array_equal(got, got2), "ε 不同结果必须不同，否则判据盖不住"


def test_mask_kernel_matches_numpy() -> None:
    """加性掩码：编成 C 后与镜像逐元素一致。"""
    fn, _ = _compile_shapes("mask", [(2, 4), (1, 4)], dtype="float16")
    rng = np.random.default_rng(13)
    scores = rng.standard_normal((2, 4)).astype(np.float16)
    m = np.array([[0, 0, -1e4, -1e4]], dtype=np.float16)
    got = _call(fn, scores, m, out_shape=(2, 4))
    ref = mirrors.mask(scores, m)
    assert np.array_equal(got, ref), (
        f"{int((got != ref).sum())} 个元素不一致")


def test_lower_rank_mask_broadcasts() -> None:
    """掩码秩低于分数时按**尾维**广播——verifier 放行，降级也必须放行。

    `MaskOp::verify` 明确允许 `maskTy.getRank() < scoresTy.getRank()`，
    而降级侧原来要求两者同秩，于是一条合法 IR 编不出 C。
    """
    fn, _ = _compile_shapes("mask", [(2, 4), [4]], dtype="float16")
    scores = np.arange(8, dtype=np.float16).reshape(2, 4)
    m = np.array([0, 0, -1e4, -1e4], dtype=np.float16)
    got = _call(fn, scores, m, out_shape=(2, 4))
    ref = mirrors.mask(scores, m)
    assert np.array_equal(got, ref), (
        f"{int((got != ref).sum())} 个元素不一致——低秩掩码没按尾维广播")


def test_prefill_mask_compiles() -> None:
    """方阵掩码是 prefill 的 causal_tril，不能被 vector 校验器拒掉。"""
    fn, _ = _compile_shapes("mask", [(4, 4), (4, 4)], dtype="float16")
    scores = np.zeros((4, 4), dtype=np.float16)
    m = np.triu(np.full((4, 4), -1e4, dtype=np.float16), k=1)
    got = _call(fn, scores, m, out_shape=(4, 4))
    assert np.array_equal(got, mirrors.mask(scores, m))


def test_transpose_kernel_matches_numpy() -> None:
    """按轴序重排：编成 C 后与镜像逐元素一致。"""
    fn, _ = _compile_shapes("transpose", [[2, 3], [1, 0]], dtype="float16")
    x = np.arange(6, dtype=np.float16).reshape(2, 3)
    got = _call(fn, x, out_shape=(3, 2))
    assert np.array_equal(got, mirrors.transpose(x, (1, 0)))


def test_reshape_and_concat_and_convert_match_numpy() -> None:
    """三个视图/转换算子各自与镜像逐元素一致。"""
    fn, _ = _compile_shapes("reshape", [[2, 3], [3, 2]], dtype="float16")
    x = np.arange(6, dtype=np.float16).reshape(2, 3)
    assert np.array_equal(_call(fn, x, out_shape=(3, 2)),
                          mirrors.reshape(x, (3, 2)))

    fn, _ = _compile_shapes("concat", [[2, 3], [1, 3]], dtype="float16",
                            group_size=0)
    a = np.arange(6, dtype=np.float16).reshape(2, 3)
    b = np.arange(3, dtype=np.float16).reshape(1, 3)
    assert np.array_equal(_call(fn, a, b, out_shape=(3, 3)),
                          mirrors.concat([a, b], 0))

    fn, _ = _compile_shapes("convert", [[2, 3]], dtype="float16", out_dtype="int8")
    x = np.array([[1.4, 1.6, -2.5], [0.5, 1.5, 127.9]], dtype=np.float16)
    got = _call(fn, x, out_shape=(2, 3), out_dtype=np.int8)
    assert np.array_equal(got, mirrors.convert(x, np.int8))


def test_convert_target_type_comes_from_the_contract() -> None:
    """目标类型走 `out_dtype`，不是写死的 i8。

    `dtype` 说的是**输入**的存储类型；换类型这件事只有 `out_dtype` 说得出来。
    原来降级侧把它写死成 `"i8"`，于是 f16→f32 也会被编成 f16→i8——形状与
    接口都对，只有数值全错。
    """
    fn, result = _compile_shapes("convert", [[2, 3]], dtype="float16",
                                 out_dtype="float32")
    assert "tensor<2x3xf32>" in result.pimir, (
        "目标类型没进 IR，编出来的还是 f16→i8")
    fn, _ = _compile_shapes("convert", [[2, 3]], dtype="float16",
                            out_dtype="float32")
    x = np.array([[1.4, 1.6, -2.5], [0.5, 1.5, 127.9]], dtype=np.float16)
    got = _call(fn, x, out_shape=(2, 3), out_dtype=np.float32)
    assert np.array_equal(got, mirrors.convert(x, np.float32))


def test_convert_narrowing_back_to_f16_is_not_an_identity() -> None:
    """f32→f16 也是转换，不能编成 f16→f16。

    图上 `to.dtype` 两个方向都有：RMSNorm 链里是 f16→f32→f16。源侧写死成
    fp16 时这个方向会发成 f16→f16，而 `pim.convert` 的 verifier 明确拒绝
    同类型——实测把三条策略扫描的端到端用例一起打挂了。
    """
    fn, result = _compile_shapes("convert", [[2, 3]], dtype="float32",
                                 out_dtype="float16")
    assert "tensor<2x3xf32> -> tensor<2x3xf16>" in result.pimir, (
        f"源侧类型没进 IR：{result.pimir.splitlines()[2] if len(result.pimir.splitlines()) > 2 else result.pimir}")
    x = np.array([[1.4003906, -2.5, 1e-8]], dtype=np.float32)
    got = _call(fn, x, out_shape=(1, 3), out_dtype=np.float16)
    assert np.array_equal(got, mirrors.convert(x, np.float16))


def test_convert_without_a_target_type_is_refused() -> None:
    """不给 `out_dtype` 就报错，不猜一个默认值编下去。"""
    with pytest.raises(ValueError, match="out_dtype"):
        _compile_shapes("convert", [[2, 3]], dtype="float16")


def test_convert_of_one_type_is_refused() -> None:
    """两侧同类型不是一次转换，发射侧就要拒绝，不留给 verifier。"""
    with pytest.raises(ValueError, match="不是一次转换"):
        _compile_shapes("convert", [[2, 3]], dtype="float16", out_dtype="float16")


def test_concat_outer_blocks_match_numpy() -> None:
    """拼接轴前面还有维度时，每一块外层块必须读源的对应块。

    上面那条用例钉的是 axis=0（外层块只有一个），读不到外层块的下标项；
    KV 缓存增长走的是 axis=2、外层块 32 个的形态（`pim.concat` 的示例用法），
    这里补上：源下标少了 `o * extent * inner` 时会静默读到第一块。
    """
    fn, _ = _compile_shapes("concat", [[1, 2, 3, 4], [1, 2, 1, 4]],
                            dtype="float16", group_size=2)
    rng = np.random.default_rng(29)
    a = rng.standard_normal((1, 2, 3, 4)).astype(np.float16)
    b = rng.standard_normal((1, 2, 1, 4)).astype(np.float16)
    got = _call(fn, a, b, out_shape=(1, 2, 4, 4))
    ref = mirrors.concat([a, b], 2)
    bad = int((got != ref).sum())
    assert np.array_equal(got, ref), (
        f"{bad}/{ref.size} 个元素不一致——外层块下标没进源地址")


def test_lut_silu_and_eltwise_match_numpy() -> None:
    """Silu 兜底路径与逐元素二元运算各自与镜像一致。"""
    fn, _ = _compile_shapes("lut", [[2, 4]], dtype="float16")
    rng = np.random.default_rng(17)
    x = (rng.standard_normal((2, 4)) * 2).astype(np.float16)
    got = _call(fn, x, out_shape=(2, 4))
    # 参照是闭式，与编译内核同一条公式。
    xf = x.astype(np.float32)
    ref = (xf / (1.0 + np.exp(-xf))).astype(np.float16)
    assert np.array_equal(got, ref)

    fn, _ = _compile_shapes("eltwise", [[2, 4], [2, 4]], dtype="float16")
    a = rng.standard_normal((2, 4)).astype(np.float16)
    b = rng.standard_normal((2, 4)).astype(np.float16)
    assert np.array_equal(_call(fn, a, b, out_shape=(2, 4)),
                          (a.astype(np.float32) + b.astype(np.float32)
                           ).astype(np.float16))


def test_eltwise_mul_compiles() -> None:
    """逐元素乘要能编出来，不是只有加法。

    门控乘与 RoPE 乘在图上都是 EltwiseMul，按方案要升到编译路径。
    原来 `kind` 在降级侧写死成 `add`，`OpCompileRequest` 上也没有任何一个
    字段能选出 `mul`——这条路根本表达不出乘法，编出来的却是加法，数值全错。
    """
    fn, result = _compile_shapes("eltwise", [[2, 4], [2, 4]], dtype="float16",
                                 kind="mul")
    assert "kind = #pim.eltwise<mul>" in result.pimir, result.pimir
    rng = np.random.default_rng(31)
    a = rng.standard_normal((2, 4)).astype(np.float16)
    b = rng.standard_normal((2, 4)).astype(np.float16)
    got = _call(fn, a, b, out_shape=(2, 4))
    ref = (a.astype(np.float32) * b.astype(np.float32)).astype(np.float16)
    assert np.array_equal(got, ref), f"{int((got != ref).sum())} 个元素不一致"
    # 判据要能失败：加法与乘法在这里的结果必须不同，否则上面那条盖不住。
    added = (a.astype(np.float32) + b.astype(np.float32)).astype(np.float16)
    assert not np.array_equal(ref, added), "夹具选得不好，加减乘除分不出来"


def test_kv_cache_scatter_writes_the_indexed_row() -> None:
    """散写：按 i16 索引写进缓存的那一行，其余行不动。"""
    fn, result = _compile_shapes("kv_cache", [(4,), (16,)], dtype="int8",
                                 group_size=1)
    # v、cache、pos、slot 四个形参；没有输出——写进缓存就是全部结果。
    assert result.argtypes == ["int8_t", "int8_t", "int32_t", "int16_t"]
    value = np.array([1, 2, 3, 4], dtype=np.int8)
    cache = np.zeros(16, dtype=np.int8)
    # 步计数器是**按值**参数（C 里是 `int32_t`，不是 `int32_t*`），
    # 所以传整数本身；另一半 `slot` 是缓冲，仍传指针。
    slot = np.array([2], dtype=np.int16)
    fn(value.ctypes.data_as(ctypes.c_void_p),
       cache.ctypes.data_as(ctypes.c_void_p),
       ctypes.c_int32(0),
       slot.ctypes.data_as(ctypes.c_void_p))
    assert np.array_equal(cache[8:12], value), "没写进索引指的那一行"
    assert not cache[:8].any() and not cache[12:].any(), "写到了别处"


def test_kv_cache_range_write_starts_at_pos() -> None:
    """整段写从 `pos` 指的那一行开始，不是永远从缓存开头写。

    `pos` 是 op 的正式操作数，主机侧镜像按 `row = pos + t` 定位
    （`runtime/kernels.py` 的 SDPA 内核）。解码循环里每生成一个 token 就
    调一次，忽略它会让第 2 个 token 起全部覆盖第 0 行。
    """
    fn, result = _compile_shapes("kv_cache", [(4,), (16,)], dtype="int8")
    # 整段写只有 v、cache、pos 三个形参。
    assert result.argtypes == ["int8_t", "int8_t", "int32_t"]
    value = np.array([1, 2, 3, 4], dtype=np.int8)

    def write(pos: int) -> np.ndarray:
        cache = np.zeros(16, dtype=np.int8)
        fn(value.ctypes.data_as(ctypes.c_void_p),
           cache.ctypes.data_as(ctypes.c_void_p),
           ctypes.c_int32(pos))
        return cache

    at0, at2 = write(0), write(2)
    assert np.array_equal(at0[:4], value), "pos=0 应写在缓存开头"
    assert not at0[4:].any()
    assert np.array_equal(at2[8:12], value), (
        f"pos=2 应写在第 2 行（偏移 8），实际缓存 {at2.tolist()}")
    assert not at2[:8].any() and not at2[12:].any(), "写到了别处"


def test_split_heads_kernel_matches_numpy() -> None:
    """按头拆开：每个头一个输出指针，值与镜像一致。"""
    fn, result = _compile_shapes("split_heads", [[1, 4, 8]], dtype="float16",
                                 group_size=2)
    # 一个输入加两个输出。
    assert result.argtypes == ["int16_t", "int16_t", "int16_t"]
    rng = np.random.default_rng(23)
    x = rng.standard_normal((1, 4, 8)).astype(np.float16)
    h0 = np.zeros((1, 2, 8), dtype=np.float16)
    h1 = np.zeros((1, 2, 8), dtype=np.float16)
    fn(x.ctypes.data_as(ctypes.c_void_p),
       h0.ctypes.data_as(ctypes.c_void_p),
       h1.ctypes.data_as(ctypes.c_void_p))
    full = np.concatenate([h0, h1], axis=1)
    assert np.array_equal(full, x), "拆开再拼起来应与原张量相同"


# --- 编译真的发生了吗（评审 20260923 的 P1-5）-------------------------------


def test_softmax_compile_failure_is_not_swallowed() -> None:
    """softmax 编不出来时只有「工具链缺失」允许回退，其余必须抛。

    原来 `_compiled_softmax` 捕获 `(ValueError, NotImplementedError,
    RuntimeError)` 三类一起回退镜像。于是上面那条「编译内核 vs numpy 镜像逐元素
    一致」在编译**静默失败**时也成立 —— 两边比的是同一份镜像，判据本身失去意义。

    实测 softmax 对 `1x16` / `4x7` / `3x1024` / `1x1` / `2x13` 全都编得出来，
    所以那两个宽泛的异常类型捕的不是形状边界。
    """
    from opcompiler_bridge.driver import ToolchainUnavailable
    from runtime import kernels as K

    # 工具链缺失：回退镜像（CI 上可能没重建 FlagTree，这是真实边界）。
    K._COMPILED_KERNEL_CACHE.clear()
    with mock.patch("opcompiler_bridge.driver.compile_op",
                    side_effect=ToolchainUnavailable("no triton-opt")):
        assert K._compiled_softmax(4, 8) is None

    # 编译器自己报的错：必须抛，不许变成"和镜像一致"。
    for boom in (ValueError("bad shape"), RuntimeError("gcc failed"),
                 NotImplementedError("no such op")):
        K._COMPILED_KERNEL_CACHE.clear()
        with mock.patch("opcompiler_bridge.driver.compile_op", side_effect=boom):
            with pytest.raises(type(boom)):
                K._compiled_softmax(4, 8)
    K._COMPILED_KERNEL_CACHE.clear()


def test_softmax_mirror_comparison_really_compiled_something() -> None:
    """对拍 softmax 时缓存里必须有一个真编出来的函数，不是 None。

    没有这条，`test_softmax_kernel_matches_numpy` 在编译回退时仍然通过。
    """
    from runtime import kernels as K

    K._COMPILED_KERNEL_CACHE.clear()
    fn = K._compiled_softmax(4, 8)
    assert fn is not None, (
        "softmax 没编出来就对拍不了：两边都会是 numpy 镜像。"
        "若这里失败，先确认 FlagTree 已重建（triton-opt 带 pim pass）")
    assert callable(fn)
    K._COMPILED_KERNEL_CACHE.clear()


def test_no_kernel_entry_names_its_own_mirror_as_the_compiled_one() -> None:
    """`KernelEntry.compiled` 必须指向与 `mirror` **不同**的函数。

    `register_all` 做 `globals()[compiled] if compiled else entry.mirror`：
    两者同名时两个分支拿到同一个函数，这个标记就是空的，而读表的人会以为
    那一项「有编译内核可用」。实测 embedding 一度就是这样
    （`KernelEntry(gather_kernel, "gather_kernel")`）。
    """
    from runtime import kernels as K

    same = [
        target for target, entry in K._KERNELS.items()
        if entry.compiled and getattr(K, entry.compiled) is entry.mirror
    ]
    assert not same, (
        f"这些条目的 compiled 名字就是它自己的 mirror，标记是空的: {same}")


def test_group_dequant_differs_from_dequantizing_once_at_the_end() -> None:
    """按组反量化与整段累加完再反量化，必须算出**不同**的数。

    这是评审 20260923 的 P1-4：方案 §5.15 写着「EmitC 必须按组反量化再累加，
    不能整段 int MAC 完再统一反量化——那会与硬件数值不符」。原来两侧在组边界
    乘的都是 `weight_binding.sfMultiplier` **一个标量**，而标量乘在组边界与乘在
    整行末尾在精确算术里相等——于是那条判据无论实现对错都会通过，它不可能失败。

    逐组带各自的因子才让两种顺序分道。这条测试先证明「两种顺序确实不同」
    （否则判据没有意义），再证明编出来的 C 走的是分组那一种。
    """
    groups, width, n = 4, 4, 3
    k = groups * width
    rng = np.random.default_rng(11)
    a = rng.integers(-8, 8, size=(5, k)).astype(np.int8)
    w = rng.integers(-8, 8, size=(k, n)).astype(np.int8)
    # 逐组各不相同的因子（2 的幂，fp16 里精确表示），组间跨 8 倍：差别足够大
    # 不会被 RNE 取整吃掉，又小到**一个元素都不饱和**——落在 ±127 上的元素
    # 两种顺序会给出同一个 127，那种"相同"是钳位造成的，不能算判据。
    scales = np.array([[0.125] * n, [0.25] * n, [0.5] * n, [1.0] * n],
                      dtype=np.float16)

    grouped = mirrors.matmul(a, w, group_size=width, scales=scales)
    # 「整段累加完再反量化」：一个标量（取第一组的因子）乘在整行末尾。
    at_the_end = mirrors.matmul(a, w, scales=float(scales[0, 0]))
    assert not np.abs(grouped).max() == 127, (
        "有元素饱和到 127：那种'相同'来自钳位而不是算法，判据会变弱")
    assert not np.array_equal(grouped, at_the_end), (
        "两种累加顺序算出了同一个数，这条判据就不可能失败——"
        "说明逐组定标没有真的逐组生效")

    # 编出来的 C 必须是分组那一种。
    request = OpCompileRequest(
        op="matmul", arg_shapes=[(5, k), (k, n)], hardware=_HARDWARE,
        dtype="int8", group_size=width,
    )
    result = compile_op(request, force=True)
    fn = load_kernel(result)
    got = _call(fn, a, w, scales, out_shape=(5, n), out_dtype=np.int8)
    assert np.array_equal(got, grouped), (
        f"编出来的内核与分组镜像不一致：{int((got != grouped).sum())} 个元素不同")
    assert not np.array_equal(got, at_the_end), (
        "编出来的内核与「整段反量化」一致，说明它没在组边界反量化")


def test_group_dequant_rejects_a_scalar_factor() -> None:
    """逐组定标必须是 `[K/group_size, N]`，形状不对就抛。

    给一个标量不会报错地退化成「整段反量化」——那正是要消掉的静默失真。
    """
    a = np.ones((2, 8), dtype=np.int8)
    w = np.ones((8, 3), dtype=np.int8)
    with pytest.raises(ValueError, match=r"\[2, 3\]"):
        mirrors.matmul(a, w, group_size=4, scales=np.ones((5, 3), np.float16))
    with pytest.raises(ValueError, match="组宽"):
        mirrors.matmul(a, w, group_size=3, scales=np.ones((2, 3), np.float16))



def test_fused_matmul_applies_the_activation() -> None:
    """带 silu 的 matmul 与不带的必须算出不同的数，且与镜像逐元素一致。

    只钉 C 里有一次 `pim_lut_silu` 挡不住「镜像从没见过激活」那种假绿。
    """
    plain = OpCompileRequest(
        op="matmul", arg_shapes=[(4, 8), (8, 4)], hardware=_HARDWARE,
        dtype="int8",
    )
    fused = dataclasses.replace(plain, activation="silu")
    plain_fn = load_kernel(compile_op(plain, force=True))
    fused_result = compile_op(fused, force=True)
    fused_fn = load_kernel(fused_result)
    assert "act_spec<kind = silu>" in (fused_result.pimir or ""), (
        "activation 没进 IR")

    rng = np.random.default_rng(5)
    a = rng.integers(-8, 8, size=(4, 8)).astype(np.int8)
    w = rng.integers(-8, 8, size=(8, 4)).astype(np.int8)
    got_plain = _call(plain_fn, a, w, out_shape=(4, 4), out_dtype=np.int8)
    got_fused = _call(fused_fn, a, w, out_shape=(4, 4), out_dtype=np.int8)
    assert not np.array_equal(got_plain, got_fused), (
        "带/不带激活结果相同，这条判据失去意义")
    ref = mirrors.matmul(a, w, activation="silu")
    assert np.array_equal(got_fused, ref)


def test_fused_matmul_activates_before_it_rounds() -> None:
    """激活在 f32 域做，量化只在最后落一次。

    上面那条用的是不分组的 int8×int8，累加值是精确整数、`rint` 是恒等，
    「先激活后取整」与「先取整后激活」恰好等价——它抓不到取整顺序。
    逐组非整数定标让累加值带小数，两种顺序才算得出不同的数：实测累加值
    0.5337 处，先激活得 silu(0.5337)=0.336→0，先取整得 rint(0.5337)=1、
    silu(1)=0.731→1。
    """
    groups, width, n = 4, 32, 64
    k = groups * width
    rng = np.random.default_rng(2024)
    a = rng.integers(-8, 8, size=(8, k)).astype(np.int8)
    w = rng.integers(-8, 8, size=(k, n)).astype(np.int8)
    # 非 2 的幂的随机因子：累加值不会正好落在整数上，两种顺序才分得开。
    scales = rng.uniform(0.05, 0.6, size=(groups, n)).astype(np.float16)

    request = OpCompileRequest(
        op="matmul", arg_shapes=[(8, k), (k, n)], hardware=_HARDWARE,
        dtype="int8", group_size=width, activation="silu",
    )
    fn = load_kernel(compile_op(request, force=True))
    got = _call(fn, a, w, scales, out_shape=(8, n), out_dtype=np.int8)
    ref = mirrors.matmul(a, w, group_size=width, scales=scales,
                         activation="silu")
    assert np.array_equal(got, ref), (
        f"编出来的 C 与镜像差 {int((got != ref).sum())} 个元素")

    # 判据本身要有分辨力：按「先取整后激活」重算一遍，必须得到不同的数。
    # 无激活的那次调用给出的就是取整后的累加值。
    rounded_first = mirrors.matmul(a, w, group_size=width, scales=scales)
    wrong = np.clip(
        np.rint(mirrors._apply_activation(rounded_first.astype(np.float32),
                                          "silu")),
        -128, 127).astype(np.int8)
    assert not np.array_equal(ref, wrong), (
        "两种取整顺序在这个输入上算出了同一个数，这条判据不可能失败——"
        "换一组定标因子")


def test_graph_and_driver_emit_the_same_rope_and_matmul_text() -> None:
    """图路径与 driver 路径对同一算子必须出同一份 IR 文本。

    只比函数体：函数名两边本来就不同（FX 节点名 vs `kernel`）。缺这条判据
    时，emitter 自己写一份、kernel 再写一份，GML 与 C 会各说各话而全绿。
    """
    import re
    from opcompiler_bridge.driver import _make_oplevel_mlir
    from opcompiler_bridge.oplevel_kernel import matmul_kernel, rope_kernel

    def body_of(text: str) -> str:
        return re.sub(r"tt\.func @\w+", "tt.func @X", text)

    q = rope_kernel("q", 32, 16, 128, tail_card_value=0)
    k = rope_kernel("k", 32, 16, 128, tail_card_value=3)
    q_drv = _make_oplevel_mlir(OpCompileRequest(
        op="rope", arg_shapes=[(1, 32, 16, 128)], hardware=_HARDWARE,
        dtype="float16", tail_card_value=0))
    k_drv = _make_oplevel_mlir(OpCompileRequest(
        op="rope", arg_shapes=[(1, 32, 16, 128)], hardware=_HARDWARE,
        dtype="float16", tail_card_value=3))
    assert body_of(q) in body_of(q_drv)
    assert body_of(k) in body_of(k_drv)
    assert "tailCardValue = 3 : i64" in k_drv
    assert "tailCardValue" not in q_drv

    fused = matmul_kernel("gate", 8, 128, 64, activation="silu")
    m_drv = _make_oplevel_mlir(OpCompileRequest(
        op="matmul", arg_shapes=[(8, 128), (128, 64)], hardware=_HARDWARE,
        dtype="int8", activation="silu"))
    assert body_of(fused) in body_of(m_drv)
    assert "nmuMode = floating_point" in fused
    assert "-> tensor<8x64xi8>" in fused


def test_unknown_op_name_is_refused_with_a_readable_error() -> None:
    """名字不认识要在这里就说清楚，不能落进 A 路去解包 `arg_shapes[1]`。

    落到 A 路时的现场是 `IndexError: list index out of range`——看着像参数
    个数不对，实际是名字拼错或漏登记，排查方向被带偏。
    """
    import dataclasses

    from contracts.op_contract import DEFAULT_HARDWARE_CONFIG

    request = OpCompileRequest(
        op="no_such_op", arg_shapes=[(4, 4)],
        hardware=dataclasses.replace(DEFAULT_HARDWARE_CONFIG, num_dpus=1),
    )
    with pytest.raises(NotImplementedError, match="没有 'no_such_op' 的编译路径"):
        compile_op(request)
