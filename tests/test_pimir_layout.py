"""P1-1：统一 IR 的四维写进下发的 pimir。

主路是 `driver._make_oplevel_mlir`（执行路径：`runtime/kernels` → `driver.compile_op`，
请求里带 `PIMHardwareConfig`），不是 GML 路径的 `oplevel_emitter` ——
后者那条路径没有 spec，拿不到切分决策。

验收用的是 printer 的省略规则：`dpusPerDevice` 一旦出现在文本里就**必然**
非默认值（全 1 会被省略），所以「字段出现」即可判定图编译器确实写入了跨 DPU
切分决策，不必解析数值。
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from dataclasses import replace
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from contracts.mlir_layout import module_attributes, tasklet_tiled
from contracts.op_contract import DpuShard, OpCompileRequest, PIMHardwareConfig


def _hw() -> PIMHardwareConfig:
    return PIMHardwareConfig(num_dpus=2, num_tasklets=16,
                             mram_bytes_per_dpu=1 << 32,
                             wram_bytes_per_dpu=1 << 16, dma_align=64)


def _request(*, shard: DpuShard | None = None, op: str = "normalize",
             arg_shapes=None, **kw) -> OpCompileRequest:
    return OpCompileRequest(
        op=op, arg_shapes=arg_shapes or [(2048, 4096)], hardware=_hw(),
        dtype="float16", shard=shard, **kw)


def test_tasklet_tiled_encodes_the_cross_dpu_decision() -> None:
    """rank-2、第 0 维分给 2 台 DPU。

    `taskletsPerDpu` 落在第 1 维：tasklet 沿布局顺序（最内层在前）铺，而
    第 1 维有 128 个元素、装得下 16 个 tasklet。
    """
    got = tasklet_tiled((16, 128), shard=DpuShard(dim=0, num_dpus=2),
                        num_tasklets=16)
    assert got == ("#pim.tasklet_tiled<{sizePerTasklet = [1, 1], "
                   "taskletsPerDpu = [1, 16], dpusPerDevice = [2, 1], "
                   "order = [1, 0]}>")


def test_single_dpu_emits_nothing() -> None:
    """单 DPU 没有跨 DPU 决策可记，编码整段不出现。

    这是「默认值下走原路径」的落点：文本与改动前逐字节相同。
    """
    assert tasklet_tiled((16, 128), shard=None, num_tasklets=16) == ""
    assert tasklet_tiled((16, 128), shard=DpuShard(dim=0, num_dpus=1),
                         num_tasklets=16) == ""


@pytest.mark.parametrize("shape,shard_dim,num_dpus", [
    ((64,), 0, 2), ((64, 128), 0, 4), ((64, 128), 1, 2),
    ((4, 16, 32), 2, 8), ((2, 4, 8, 16), 1, 2),
])
def test_encoding_satisfies_the_ods_verifier(shape, shard_dim, num_dpus) -> None:
    """四组数组同秩、取值全为正、order 是 [0, rank) 的排列。

    这三条正是 FlagTree 侧 `TaskletTiledEncodingAttr::verify` 的判据，
    违反了会被 verifier 拒。
    """
    import re

    rank = len(shape)
    text = tasklet_tiled(shape, shard=DpuShard(dim=shard_dim, num_dpus=num_dpus),
                         num_tasklets=16)
    arrays = {name: [int(v) for v in re.findall(r"\d+", body)]
              for name, body in re.findall(r"(\w+) = \[([^\]]*)\]", text)}
    assert set(arrays) == {"sizePerTasklet", "taskletsPerDpu", "dpusPerDevice",
                           "order"}
    assert all(len(values) == rank for values in arrays.values())
    for name in ("sizePerTasklet", "taskletsPerDpu", "dpusPerDevice"):
        assert all(v > 0 for v in arrays[name]), name
    assert sorted(arrays["order"]) == list(range(rank))
    assert arrays["dpusPerDevice"][shard_dim] == num_dpus
    assert sum(1 for v in arrays["dpusPerDevice"] if v != 1) == 1


def test_module_attributes_carry_the_hardware_config() -> None:
    """硬件配置进模块属性，对齐 A 路的富度。"""
    attrs = " ".join(module_attributes(_hw()))
    assert '"pim.num-dpus" = 2 : i32' in attrs
    assert '"pim.num-tasklets" = 16 : i32' in attrs
    assert '"pim.dma-align" = 64 : i32' in attrs


def test_multi_dpu_request_writes_the_field() -> None:
    """多 DPU 时文本里出现 dpusPerDevice，且不是全 1。

    printer 省略全 1 默认值，所以「出现」即证明写入成功。
    """
    from opcompiler_bridge.driver import _make_oplevel_mlir

    text = _make_oplevel_mlir(_request(shard=DpuShard(dim=0, num_dpus=2)))
    assert "dpusPerDevice = [2, 1]" in text
    assert "#pim.tasklet_tiled" in text
    assert '"pim.num-dpus" = 2 : i32' in text


def test_single_dpu_request_keeps_the_text_unchanged() -> None:
    """没有切分决策时，文本与改动前逐字节相同（编码与硬件属性都不出现）。"""
    from opcompiler_bridge.driver import _make_oplevel_mlir

    from opcompiler_bridge.oplevel_kernel import normalize_kernel

    text = _make_oplevel_mlir(_request(shard=None))
    assert "tasklet_tiled" not in text
    assert "pim.num-dpus" not in text
    # 模块头就是改动前那一句，正文是内核原样输出 —— 一字未加
    body = normalize_kernel("kernel", 2048, 4096, 4096)
    assert text == f'module attributes {{pim.target = "pim:v1"}} {{\n{body}\n}}\n'



def test_the_layout_is_attached_to_the_result_type() -> None:
    """编码贴在算子的**结果**类型上，与决策的来源同一个坐标系。

    切分决策取自输出的 `shard_map`（`exec_plan_gen._shard_decision_of`
    收的是 `out_detail`），所以描述的是输出那块数据怎么摆。原先贴在第一个
    实参上：输入输出不同形时就会标成另一块张量的另一根轴。

    注意这里查的是**算子正文**里的结果类型，不是 `tt.func` 签名行 ——
    签名行不含 `->`，拿它切箭头会得到恒真断言。
    """
    from opcompiler_bridge.driver import _make_oplevel_mlir

    text = _make_oplevel_mlir(_request(shard=DpuShard(dim=0, num_dpus=2)))
    body = next(line for line in text.splitlines() if "->" in line)
    result = body.split("->")[-1]
    assert "tasklet_tiled" in result, text


def test_the_decision_lands_on_the_output_axis_not_the_input_axis() -> None:
    """输入输出形状不同时，编码的秩与轴都按**输出**算。

    reshape 是最小反例：输入 `1x16x4096`（rank 3）、输出 `1x65536`（rank 2），
    决策是「输出第 1 维分给 2 台 DPU」。按输入贴会得到 rank-3 的
    `dpusPerDevice = [1, 2, 1]` —— 宣称长度 16 的那一维分给了 2 台 DPU，
    而真正被切的是长度 65536 的输出维。
    """
    from opcompiler_bridge.driver import _make_oplevel_mlir

    text = _make_oplevel_mlir(_request(
        op="reshape", arg_shapes=[(1, 16, 4096), (1, 65536)],
        shard=DpuShard(dim=1, num_dpus=2)))
    body = next(line for line in text.splitlines() if "->" in line)
    result = body.split("->")[-1]
    assert "dpusPerDevice = [1, 2]" in result, text
    # 输入那份类型不带编码：它不是决策描述的那块数据。
    operand = body.split("->")[0]
    assert "tasklet_tiled" not in operand, text


def test_a_shard_dim_beyond_the_output_rank_is_rejected() -> None:
    """切分维超出**输出**秩要抛错，不能静默贴到别的轴上。

    输出 rank 2 而 shard.dim=2 时，按输入（rank 3）算会正好落在合法区间内、
    静默通过 —— 这正是坐标系不一致时最难发现的那种错。
    """
    from opcompiler_bridge.driver import _make_oplevel_mlir

    with pytest.raises(ValueError, match="切分维"):
        _make_oplevel_mlir(_request(
            op="reshape", arg_shapes=[(1, 16, 4096), (1, 65536)],
            shard=DpuShard(dim=2, num_dpus=2)))


def test_every_oplevel_op_accepts_a_shard_decision() -> None:
    """15 个整算子级入口都要能把编码写出来，不能只有一两个支持。"""
    from opcompiler_bridge.driver import _make_oplevel_mlir

    requests = [
        _request(op="normalize", arg_shapes=[(2048, 4096)]),
        _request(op="softmax", arg_shapes=[(32, 1024)]),
        _request(op="dynamic_quant", arg_shapes=[(1, 65536)], group_size=128),
        _request(op="eltwise", arg_shapes=[(1, 4096), (1, 4096)]),
        _request(op="reshape", arg_shapes=[(1, 16, 4096), (1, 65536)]),
        _request(op="transpose", arg_shapes=[(1, 16, 4096), (0, 2, 1)]),
    ]
    for request in requests:
        request = replace(request, shard=DpuShard(dim=0, num_dpus=2))
        text = _make_oplevel_mlir(request)
        assert "dpusPerDevice = [2" in text, request.op


def test_rope_sub_blocks_keep_the_fixed_order() -> None:
    """`subBlocks` 必须按 ROPE_UNITS 的声明序发，且正好 6 个。

    顺序就是语义：FlagTree 的读回侧按位置展开，乱序会让整块字段错位而不报错。
    """
    from contracts.gml_hw_constants import ROPE_UNITS
    from opcompiler_bridge.oplevel_kernel import rope_kernel

    text = rope_kernel("kernel", 32, 16, 128)
    assert "subBlocks" in text
    order = [name for _, name in ROPE_UNITS]
    assert len(order) == 6
    for index, name in enumerate(order):
        assert f'"{name}"' in text
    names_in_text = [m for m in __import__("re").findall(r'"(Llama2Activation_\w+)"', text)]
    assert names_in_text == order, f"subBlocks 顺序与 ROPE_UNITS 不一致：{names_in_text}"


def test_plan_records_the_placement_decision_in_the_payload() -> None:
    """exec_plan_gen 把**三档**放置决策写进命令 payload，运行时才有东西可读。

    早先这里只带 shard 一档，Replicate 与 Partial 被压成同一个 None ——
    PIMMLIR 侧因此分不清「复制」与「单 DPU」，而两者的归约与容量口径不同。
    """
    from contracts.pim_tensor_spec import PIMTensorSpec, Placement, TensorShardDetail
    from runtime.exec_plan_gen import _shard_decision_of

    def two_dpus(placement, shard_dim, reduce_type=None):
        return PIMTensorSpec(
            "dpu", placement, "transient", None,
            {0: TensorShardDetail(0, shard_dim, 0, 2048, (2048, 4096)),
             1: TensorShardDetail(1, shard_dim, 2048, 4096, (2048, 4096))},
            reduce_type)

    sharded = two_dpus(Placement("Shard", 0), 0)
    assert _shard_decision_of(sharded, sharded.shard_map[0]) == \
        [0, 2, "shard", None]

    # 复制：每台持有完整形状，没有被切开的轴，但**要下发** —— 这是与单 DPU
    # 的区别所在。
    repl = two_dpus(Placement("Replicate"), -1)
    assert _shard_decision_of(repl, repl.shard_map[0]) == \
        [-1, 2, "replicate", None]

    # 局部和：归约方式是它的全部内容，必须带出来。
    part = two_dpus(Placement("Partial", reduce_type="sum"), -1,
                    reduce_type="sum")
    assert _shard_decision_of(part, part.shard_map[0]) == \
        [-1, 2, "partial", "sum"]

    # 单 DPU：一个字不下发，文本与改动前逐字节相同。
    single = PIMTensorSpec(
        "dpu", Placement("Replicate"), "transient", None,
        {0: TensorShardDetail(0, -1, 0, 4096, (4096,))}, None)
    assert _shard_decision_of(single, single.shard_map[0]) is None, \
        "单 DPU 不给决策，下发的文本才与改动前一致"


def test_command_context_feeds_the_request_and_the_text() -> None:
    """payload → `_ctx_of` → OpCompileRequest → pimir：整条链路是通的。"""
    from runtime.kernels import _ctx_of

    class _Cmd:
        payload = {"hardware": {"num_dpus": 2, "num_tasklets": 16,
                                "mram_bytes_per_dpu": 1 << 32,
                                "wram_bytes_per_dpu": 1 << 16,
                                "dma_align": 64},
                   "shard": [0, 2]}

    ctx = _ctx_of(_Cmd())
    assert ctx.shard == DpuShard(dim=0, num_dpus=2)
    assert ctx.hardware.num_dpus == 2

    from opcompiler_bridge.driver import _make_oplevel_mlir

    request = replace(_request(), shard=ctx.shard)
    text = _make_oplevel_mlir(request)
    assert "dpusPerDevice = [2, 1]" in text


def test_command_without_a_shard_stays_single_dpu() -> None:
    """载荷里没有 shard（或 payload 里没有硬件）时走默认上下文。"""
    from runtime.kernels import DEFAULT_OPS_CONTEXT, _ctx_of

    class _Cmd:
        payload = {"hardware": {"num_dpus": 2, "num_tasklets": 16,
                                "mram_bytes_per_dpu": 1 << 32,
                                "wram_bytes_per_dpu": 1 << 16,
                                "dma_align": 64}}

    assert _ctx_of(_Cmd()).shard is None

    class _Bare:
        payload: dict = {}

    assert _ctx_of(_Bare()) == DEFAULT_OPS_CONTEXT


# ---- 实跑：让真正的 C++ verifier 与 pass 链当最终判据 ----
#
# 上面那些是文本断言：快、但校验规则是在 Python 里重写了一遍 ODS 的
# `TaskletTiledEncodingAttr::verify`。方言侧改了规则，文本断言不会知道。
# 下面两条把真实 `triton-opt` 接进来，文本断言降为第一道快速反馈。

def _triton_opt() -> Path | None:
    from genesim_bridge.paths import flagtree_prefix

    path = flagtree_prefix() / "build" / "flagtree-cmake" / "bin" / "triton-opt"
    return path if path.is_file() else None


def _has_pim_passes() -> bool:
    binary = _triton_opt()
    if binary is None:
        return False
    proc = subprocess.run([str(binary), "--help"], capture_output=True, text=True)
    return "--pim-expand-phases" in proc.stdout


def _run_triton_opt(text: str, *passes: str) -> str:
    """把文本喂给真实 `triton-opt`，返回它的输出。非零退出即断言失败。"""
    with tempfile.NamedTemporaryFile("w", suffix=".mlir", delete=False) as handle:
        handle.write(text)
        path = handle.name
    try:
        proc = subprocess.run([str(_triton_opt()), path, *passes],
                              capture_output=True, text=True)
    finally:
        Path(path).unlink(missing_ok=True)
    assert proc.returncode == 0, f"triton-opt 拒绝了这份文本：\n{proc.stderr}\n{text}"
    return proc.stdout


# 覆盖三类形状关系：输入输出同形、不同秩、同形不同元素类型。
_LIVE_CASES = [
    ("normalize", [(2048, 4096)], {}, 0),
    ("softmax", [(32, 1024)], {}, 1),
    ("eltwise", [(1, 4096), (1, 4096)], {"kind": "add"}, 1),
    ("lut", [(1, 4096)], {}, 1),
    ("matmul", [(1, 128), (128, 64)], {}, 1),
    ("reshape", [(1, 16, 4096), (1, 65536)], {}, 1),
    ("transpose", [(1, 16, 4096), (0, 2, 1)], {}, 2),
    ("gather", [(1, 64), (1,)], {}, 2),
    ("concat", [(1, 64), (1, 64)], {"group_size": 1}, 1),
    ("convert", [(1, 64)], {"out_dtype": "int8"}, 1),
    ("dynamic_quant", [(1, 65536)], {"group_size": 128}, 1),
    ("split_heads", [(1, 4096)], {"group_size": 32}, 1),
]


@pytest.mark.skipif(not _has_pim_passes(),
                    reason="当前 triton-opt 没有 PIM pass，需重跑 0-install-flagtree.sh")
@pytest.mark.parametrize("op,arg_shapes,kw,dim", _LIVE_CASES)
def test_the_encoding_round_trips_through_triton_opt(op, arg_shapes, kw, dim) -> None:
    """往返一致：带编码的 pimir 过真实解析器再打印，`dpusPerDevice` 值不变。

    这是需求 §4.3 风险表第 5 条要守的那件事 ——「写入后 FlagTree verifier 拒绝」。
    文本断言看不出这个：它判的是我们自己拼的字符串，不是方言认不认。
    """
    from opcompiler_bridge.driver import _make_oplevel_mlir

    text = _make_oplevel_mlir(_request(op=op, arg_shapes=arg_shapes,
                                       shard=DpuShard(dim=dim, num_dpus=2), **kw))
    assert "dpusPerDevice = [" in text
    printed = _run_triton_opt(text)
    assert "dpusPerDevice = [" in printed, printed
    # 被切的那一维记 2 台 DPU，往返后数值不变。
    assert printed.count("dpusPerDevice") >= 1
    for chunk in printed.split("dpusPerDevice = [")[1:]:
        assert "2" in chunk.split("]")[0], printed


@pytest.mark.skipif(not _has_pim_passes(),
                    reason="当前 triton-opt 没有 PIM pass，需重跑 0-install-flagtree.sh")
@pytest.mark.parametrize("op,arg_shapes,kw,dim", _LIVE_CASES)
def test_the_encoding_survives_the_pass_chain(op, arg_shapes, kw, dim) -> None:
    """过 `-pim-fuse-activation -pim-expand-phases` 后编码仍在。

    展开 pass 会重建算子与结果类型。只贴结果、不贴同形的操作数时，
    `dynamic_quant` 的编码在这一步会整个丢掉（实测重建后 0 处）——
    传下去的信息为空，等于没写。
    """
    from opcompiler_bridge.driver import _make_oplevel_mlir

    text = _make_oplevel_mlir(_request(op=op, arg_shapes=arg_shapes,
                                       shard=DpuShard(dim=dim, num_dpus=2), **kw))
    expanded = _run_triton_opt(text, "-pim-fuse-activation", "-pim-expand-phases")
    assert "dpusPerDevice = [" in expanded, \
        f"{op} 的编码过 pass 链后丢了：\n{expanded}"


# ---- 无结果算子与缓存键（评审 round2 问题 4）----

def test_an_op_without_a_result_type_still_compiles_with_a_shard() -> None:
    """`kv_cache` 把结果写进 memdesc，没有张量结果可挂编码。

    改动前它在带 `DpuShard` 时抛「算子正文里没有 `->`」—— 一个指向内部
    实现细节的报错。现在按契约跳过编码，模块属性照发。
    """
    from opcompiler_bridge import oplevel_kernel
    from opcompiler_bridge.driver import _module_text

    body = oplevel_kernel.kv_cache_kernel("kernel", (1, 1, 128), 4096, False)
    text = _module_text(body, _request(op="kv_cache", arg_shapes=[(1, 1, 128)],
                                       shard=DpuShard(dim=2, num_dpus=2)))
    assert "tasklet_tiled" not in text, "没有结果类型的算子不该挂布局编码"
    assert '"pim.num-dpus"' in text, "硬件属性仍要下发"


def test_an_unregistered_op_without_a_result_type_still_raises() -> None:
    """不在 `_NO_RESULT_OPS` 里又取不到结果类型的，照旧抛错。

    否则将来任何一个漏写结果类型的算子都会静默不下发编码。
    """
    from opcompiler_bridge.driver import _module_text

    body = "  tt.func @kernel(%a: tensor<1x16xf16>) {\n    tt.return\n  }"
    with pytest.raises(ValueError, match="没有 `->`"):
        _module_text(body, _request(op="normalize",
                                    shard=DpuShard(dim=1, num_dpus=2)))


def test_the_shard_decision_is_part_of_the_cache_key() -> None:
    """切分决策改变下发文本，所以必须进缓存键。

    漏掉它：先编的单 DPU 与后来的 tp2 落到同一份 `.so` 与同一份 pimir 上，
    多 DPU 请求拿回不带编码的旧文本 —— P1-1 的「多 DPU 时 dpusPerDevice
    出现」在热缓存上静默不成立。
    """
    from opcompiler_bridge.driver import _cache_key

    keys = {
        _cache_key(_request(shard=None)),
        _cache_key(_request(shard=DpuShard(dim=0, num_dpus=2))),
        _cache_key(_request(shard=DpuShard(dim=1, num_dpus=2))),
        _cache_key(_request(shard=DpuShard(dim=0, num_dpus=4))),
    }
    assert len(keys) == 4, f"不同切分决策撞到同一个缓存键：{keys}"


# ---- A 路（linear）的切分决策下发（评审 round2 问题 1 第 2 条）----
#
# `linear` 走 A 路（TTIR → convert-triton-to-pim），它的张量编码全部由
# FlagTree 生成，而那个 builder 把 `dpusPerDevice` 固定为全 1
# （`PIMAttrDefs.td` 里的 `(void)numDpus;`）。图编译器往张量编码里写不进去，
# 所以 A 路改走模块属性。实测 llama tp2 计划里带切分决策的命令 3/4 是 linear，
# 丢掉它等于主算子上的决策全没下发。

def test_the_shard_decision_becomes_one_placement_attribute() -> None:
    """切分决策的载体是**一个** `#pim.placement`，不是两个裸整数。

    原先这里是 `pim.shard-dim` / `pim.shard-dpus`，而 FlagTree 读的是
    `pim.placement` —— 两套载体各自有测试、各自全绿，但在一条真实 tp2 链路上
    从未相遇：实测下发的模块头里有 shard-dim、没有 placement，于是
    `dpusPerDevice` 始终为空。收口成一个之后那条链路才真的接上。
    """
    from contracts.mlir_layout import PLACEMENT_ATTR, placement_attribute

    assert placement_attribute(None, rank=2) == ()
    assert placement_attribute(DpuShard(dim=0, num_dpus=1), rank=2) == (), \
        "单 DPU 没有跨 DPU 决策可记，一个字都不该加"

    got = placement_attribute(DpuShard(dim=1, num_dpus=4), rank=2)
    assert got == (f"{PLACEMENT_ATTR} = #pim.placement<kind = shard, "
                   f"dim = 1, numDpus = 4>",), got


def test_the_non_shard_kinds_do_not_leave_an_empty_encoding() -> None:
    """replicate / partial 的下发文本不能带空编码槽。

    `tasklet_tiled` 对这两档返回空串（编码全 1，与不发逐字节相同），而
    `_attach_layout` 原先无条件把编码拼进类型文本，于是产出
    `tensor<1x16x32xf16, >` —— 构造上非法，只是当前解析器恰好容忍。
    真实 tp2 计划里 B 路命令绝大多数是 replicate，这条路径是常态。
    """
    from opcompiler_bridge.driver import _make_oplevel_mlir

    for shard in (DpuShard(dim=-1, num_dpus=2, kind="replicate"),
                  DpuShard(dim=-1, num_dpus=2, kind="partial", reduce="sum")):
        text = _make_oplevel_mlir(OpCompileRequest(
            op="eltwise", arg_shapes=[(1, 16, 32), (1, 16, 32)], hardware=_hw(),
            dtype="float16", kind="mul", shard=shard))
        assert ", >" not in text, f"{shard.kind} 档拼出了空编码：\n{text}"
        assert "pim.placement" in text, f"{shard.kind} 档的模块属性丢了"

    # shard 档照旧贴编码 —— 上面那条不能靠「一概不贴」变绿。
    sharded = _make_oplevel_mlir(OpCompileRequest(
        op="eltwise", arg_shapes=[(1, 16, 32), (1, 16, 32)], hardware=_hw(),
        dtype="float16", kind="mul", shard=DpuShard(dim=1, num_dpus=2)))
    assert "dpusPerDevice" in sharded, sharded[:400]


def test_all_three_placement_kinds_are_downlinked() -> None:
    """三档都要发，不只发切分那一档。

    早先 `replicate` / `partial` 返回空元组，理由是「它们在张量编码上都是全 1，
    与不发等价」—— 那个理由只在**张量编码**那一层成立。模块级的 `#pim.placement`
    能分辨三者，而它们的口径并不相同：`partial` 每台持有一份全形状的局部和、
    要跨 DPU 归约才完整；`replicate` 不欠归约。不发就让 PIMMLIR 分不清「复制」
    与「单 DPU」，而这正是 `#pim.placement` 存在的理由。
    """
    from contracts.mlir_layout import PLACEMENT_ATTR, placement_attribute

    repl = placement_attribute(
        DpuShard(dim=-1, num_dpus=2, kind="replicate"), rank=2)
    assert repl == (f"{PLACEMENT_ATTR} = #pim.placement<kind = replicate, "
                    f"numDpus = 2>",), repl

    # `partial` 还要带 `reduce`：归约方式是这一档的全部内容，FlagTree 的
    # verifier 见到裸 partial 直接拒收（见 test_a_partial_op_actually_compiles）。
    part = placement_attribute(
        DpuShard(dim=-1, num_dpus=4, kind="partial", reduce="sum"), rank=2)
    assert part == (f"{PLACEMENT_ATTR} = #pim.placement<kind = partial, "
                    f"numDpus = 4, reduce = sum>",), part

    # 单 DPU 口径下三档一律不发 —— §5.2 产物不变判据的落点。
    for kind, reduce in (("shard", None), ("replicate", None),
                         ("partial", "sum")):
        dim = 0 if kind == "shard" else -1
        assert placement_attribute(
            DpuShard(dim=dim, num_dpus=1, kind=kind, reduce=reduce),
            rank=2) == (), f"单 DPU 的 {kind} 不该下发"


def test_an_out_of_range_shard_dim_is_refused_not_dropped() -> None:
    """维号超出被标注张量的秩必须抛，不能静默写出去。

    FlagTree 的 placement 版 builder 对越界维号静默跳过（`dpusPerDevice` 恒全
    1），决策整条丢掉而无任何诊断 —— 真实 tp2 的 `linear` 就是这么丢的：输出是
    三维、`shard_dim=2`，而 kernel 张量压平成二维。这个口子在下发侧堵住，与
    `tasklet_tiled` 同一条纪律。
    """
    from contracts.mlir_layout import placement_attribute

    with pytest.raises(ValueError, match="超出被标注张量的秩"):
        placement_attribute(DpuShard(dim=2, num_dpus=2), rank=2)


@pytest.mark.parametrize("kind,reduce", [
    ("replicate", None), ("partial", "sum"),
])
def test_the_non_shard_kinds_have_no_dim_to_translate(kind, reduce) -> None:
    """`replicate` / `partial` 的维号恒为 -1，不该被送去换算。

    `_a_path_shard` 早先无条件换算，于是 -1 被 `flatten_shard_dim` 的范围检查
    挡下 —— 那是误报：这两档本来就没有被切开的轴。实测会把真实 tp4/tp2 的
    `test_strategy_sweep` 整条打断。
    """
    from opcompiler_bridge.driver import _a_path_shard

    shard = DpuShard(dim=-1, num_dpus=2, kind=kind, reduce=reduce)
    request = OpCompileRequest(
        op="linear", arg_shapes=[(1, 16, 64), (32, 64)], hardware=_hw(),
        dtype="float16", shard=shard)
    assert _a_path_shard(request) == shard, "非 shard 档不该被改动"


@pytest.mark.parametrize("graph_rank,graph_dim,want", [
    (3, 2, 1),      # 真实 tp2 linear：图输出三维、切末维 → 压平后第 1 维
    (3, 1, 0),      # 切前导维 → 合并进 M
    (3, 0, 0),
    (2, 1, 1),      # 本来就是二维：原样
    (2, 0, 0),
])
def test_the_graph_dim_is_translated_to_the_kernel_coordinates(
        graph_rank, graph_dim, want) -> None:
    """图张量的维号换算到压平后的坐标系：末维→1，其余→0。"""
    from contracts.op_contract import flatten_shard_dim

    assert flatten_shard_dim(graph_dim, graph_rank) == want


def test_only_one_placement_carrier_exists() -> None:
    """不允许再出现第二套切分载体。

    两套载体描述同一个事实，就会各自绿、合起来断 —— 这条钉住只有一个。
    """
    from pathlib import Path as _P

    import contracts.mlir_layout as ml

    # 去掉注释行：注释里提旧名字是历史说明（解释为什么收口），不是载体。
    code = "\n".join(
        line.split("#", 1)[0] for line in
        _P(ml.__file__).read_text(encoding="utf-8").splitlines())
    for dead in ("pim.shard-dim", "pim.shard-dpus"):
        assert dead not in code, f"旧载体 {dead} 又回来了"


def test_the_ttir_module_head_takes_the_decision_in_both_forms() -> None:
    """TTIR 模块头有 `module {` 与 `module attributes {...} {` 两种形态。"""
    from opcompiler_bridge.driver import _with_placement_attr

    shard = DpuShard(dim=1, num_dpus=2)
    bare = _with_placement_attr("module {\n  tt.func @k() {\n  }\n}\n", shard,
                                rank=2)
    assert "#pim.placement<kind = shard, dim = 1, numDpus = 2>" in bare
    assert "module attributes {" in bare

    rich = _with_placement_attr(
        'module attributes {tt.foo = 1 : i32} {\n}\n', shard, rank=2)
    assert "numDpus = 2" in rich
    assert "tt.foo" in rich, "既有属性不能被挤掉"

    # 没有切分：逐字节不变。
    same = "module {\n}\n"
    assert _with_placement_attr(same, None, rank=2) == same


@pytest.mark.skipif(not _has_pim_passes(),
                    reason="当前 triton-opt 没有 PIM pass，需重跑 0-install-flagtree.sh")
def test_the_a_path_does_not_silently_drop_the_shard() -> None:
    """A 路必须真的把切分决策下发到 pimir 里，用**真实计划的形状**验。

    这条此前只 `inspect.getsource(compile_op)` 找 "shard" 这个词，不跑任何形状，
    于是在决策被静默丢掉时照样全绿：真实 tp2 的 `linear` 输出是三维、
    `shard_dim=2`，而 kernel 张量压平成二维，越界维号被 FlagTree 的 builder
    静默跳过，`dpusPerDevice` 一次都不出现。判据改成产物断言。
    """
    from opcompiler_bridge.driver import compile_op

    # `runtime/kernels.py` 在真实 tp2 计划里发出的就是这个请求形状：
    # 图输出 (1, 16, 32) 三维、按末维切 2 台。
    tp2 = compile_op(OpCompileRequest(
        op="linear", arg_shapes=[(1, 16, 64), (32, 64)], hardware=_hw(),
        dtype="float16", shard=DpuShard(dim=2, num_dpus=2)), force=True)
    assert "dpusPerDevice" in (tp2.pimir or ""), \
        f"真实 tp2 形状下切分决策被丢掉了：\n{(tp2.pimir or '')[:600]}"
    # printer 省略全 1，所以「出现」即「非全 1」。
    assert "numDpus = 2" in tp2.pimir

    plain = compile_op(OpCompileRequest(
        op="linear", arg_shapes=[(1, 16, 64), (32, 64)], hardware=_hw(),
        dtype="float16", shard=None), force=True)
    assert "dpusPerDevice" not in (plain.pimir or ""), \
        "单 DPU 不该出现跨 DPU 编码"


def test_a_partial_placement_names_its_reduce() -> None:
    """`partial` 的归约方式必须写进模块属性。

    与 `shard` 不同，`partial` 不带 `dim`（它按收缩维拆，不按结果维），
    但必须带 `reduce`：不带的话 FlagTree 直接拒收整份 pimir。
    """
    from opcompiler_bridge.driver import _with_placement_attr

    text = _with_placement_attr(
        "module {\n}\n",
        DpuShard(dim=-1, num_dpus=2, kind="partial", reduce="sum"), rank=2)
    assert "#pim.placement<kind = partial, numDpus = 2, reduce = sum>" in text


@pytest.mark.skipif(not _has_pim_passes(),
                    reason="当前 triton-opt 没有 PIM pass，需重跑 0-install-flagtree.sh")
def test_a_partial_op_actually_compiles() -> None:
    """判据落在**编译结果**上：带 partial 的算子必须编得过。

    改动前实测 rc=1：`a partial placement must say how its pieces combine;
    set reduce to sum or mean`。当时按「reduce 没有读者」不下发，这版 pass 的
    verifier 与归约暂存都要它，于是任何一个 partial 算子都编不出来。
    """
    from opcompiler_bridge.driver import compile_op

    result = compile_op(_request(
        op="linear", arg_shapes=[(16, 512), (4096, 512)],
        shard=DpuShard(dim=-1, num_dpus=2, kind="partial", reduce="sum")),
        force=True)
    assert "kind = partial" in result.pimir
    assert "reduce = sum" in result.pimir


@pytest.mark.skipif(not _has_pim_passes(),
                    reason="当前 triton-opt 没有 PIM pass，需重跑 0-install-flagtree.sh")
def test_the_placement_attribute_survives_the_a_path_pass_chain() -> None:
    """模块属性要穿过 A 路整条 pass 链存活，否则下游读不到。

    这是选模块属性而不是张量编码的依据：张量编码在 `convert-triton-to-pim`
    里被重建（`dpusPerDevice` 恒全 1），模块属性不被动。
    """
    from opcompiler_bridge.driver import (
        _a_path_shard, _make_ttir, _run_triton_opt)

    # 走真实 A 路：`_make_ttir` 出 TTIR，`_run_triton_opt` 跑
    # `convert-triton-to-pim` → `-pim-tile-to-budget` → `-pim-explicit-dma`。
    # 手搓的空 kernel 过不了 `-pim-tile-to-budget`（它要求至少一个 `tt.dot`），
    # 而这条用例要验的恰是真实链路，所以用真的 linear。
    # 用**真实计划的形状**：图输出三维、切末维。此前这里传 `DpuShard(dim=1)`
    # 配二维输出，恰好落在秩内，于是越界那条缺陷测不到。
    request = _request(op="linear", arg_shapes=[(1, 16, 64), (32, 64)],
                       shard=DpuShard(dim=2, num_dpus=2))
    pimir, _ = _run_triton_opt(_make_ttir(request), request.hardware,
                               _a_path_shard(request))
    assert "dpusPerDevice" in pimir, (
        f"placement 没落进张量编码，`dpusPerDevice` 仍是空的：\n{pimir[:600]}")
    assert "numDpus = 2" in pimir, pimir[:400]

    # 单 DPU：一个字都不加，与改动前逐字节相同。
    plain, _ = _run_triton_opt(_make_ttir(request), request.hardware, None)
    assert "pim.placement" not in plain
    assert "dpusPerDevice" not in plain


@pytest.mark.skipif(not _has_pim_passes(),
                    reason="当前 triton-opt 没有 PIM pass，需重跑 0-install-flagtree.sh")
def test_the_layout_order_changes_the_dma_stride_on_the_a_path() -> None:
    """统一 IR 的排布字段要改变 A 路交付 pimir 的 DMA 步幅。

    只改 `elem_strides`、其余全同：行主序时最内维步幅是 1，列主序时是行宽。
    交付文本里的 `elem_stride` 必须跟着变 —— 不变就说明排布没穿过
    `convert-triton-to-pim` → `-pim-explicit-dma` 这条链，写了没人读。
    """
    from opcompiler_bridge.driver import (
        _a_path_shard, _make_ttir, _run_triton_opt)

    def delivered(strides):
        request = _request(op="linear", arg_shapes=[(1, 16, 64), (32, 64)],
                           shard=DpuShard(dim=2, num_dpus=2),
                           elem_strides=strides)
        pimir, _ = _run_triton_opt(_make_ttir(request), request.hardware,
                                   _a_path_shard(request),
                                   elem_strides=request.elem_strides)
        return pimir

    import re

    def strides_of(text):
        """每个 DMA 的 (形状, 步幅)，按出现顺序。"""
        out = []
        for line in text.splitlines():
            if "dma_" not in line:
                continue
            m = re.search(r"elem_stride = (\d+).*?memdesc<([0-9x]+)", line)
            if m:
                out.append((m.group(2), int(m.group(1))))
        return out

    # 行主序：最内维步幅是 1，三条 DMA 都是 1。
    assert set(s for _, s in strides_of(delivered((64, 1)))) == {1}
    # 列主序：最内维换成行方向，步幅变成行宽（x 与 out 是 16、权重是 32）。
    assert strides_of(delivered((1, 16))) == [
        ("16x32x", 16), ("32x32x", 32), ("16x32x", 16)], \
        "排布没改变 DMA 步幅"


@pytest.mark.skipif(not _has_pim_passes(),
                    reason="当前 triton-opt 没有 PIM pass，需重跑 0-install-flagtree.sh")
def test_the_compiled_kernel_does_not_add_the_offset_twice() -> None:
    """起始地址只由调用方加一次：内核不得再加一遍。

    运行时按每个 access 的 offset 传精确地址（`runtime/kernels.py` 的
    `base + out_access.offset`）。若内核内部再把同一个 `mram_offset` 加进
    DMA 地址，结果会落到 2 倍偏移处，写到别人的缓冲上。

    同一份数据、同一个 out 指针，只改请求里的 `mram_offset`，结果必须落在
    同一处 —— 偏移由调用方负责，内核里的加法是多余的。
    """
    import ctypes

    import numpy as np

    from backend.hal_numpy import NumpyBackend, NumpyBackendConfig
    from opcompiler_bridge.driver import compile_op, load_kernel

    hw = PIMHardwareConfig(num_dpus=2, num_tasklets=4, mram_bytes_per_dpu=1 << 32,
                           wram_bytes_per_dpu=65536, dma_align=8)
    m, k, n = 16, 64, 32
    np.random.seed(0)
    x = np.random.randn(m, k).astype(np.float16)
    w = np.random.randn(n, k).astype(np.float16)
    want = (x.astype(np.float32) @ w.astype(np.float32).T).astype(np.float16)

    def result_at(mram_offset: int) -> np.ndarray:
        hal = NumpyBackend(NumpyBackendConfig(num_dpus=2, mram_bytes_per_dpu=1 << 32,
                                              wram_bytes_per_dpu=65536))
        base = hal.raw_mram_ptr(0)
        x_off, w_off, out_off = 131072, 133120, 137216
        ctypes.memset(base, 0, 300000)
        ctypes.memmove(base + x_off, x.ctypes.data_as(ctypes.c_void_p), x.nbytes)
        ctypes.memmove(base + w_off, w.ctypes.data_as(ctypes.c_void_p), w.nbytes)
        fn = load_kernel(compile_op(OpCompileRequest(
            op="linear", arg_shapes=[(m, k), (n, k)], hardware=hw,
            dtype="float16", num_tasklets=4,
            shard=DpuShard(dim=0, num_dpus=2), mram_offset=mram_offset)))
        fn(ctypes.c_void_p(base + x_off), ctypes.c_void_p(base + w_off),
           ctypes.c_void_p(base + out_off))
        return np.frombuffer(ctypes.string_at(base + out_off, m * n * 2),
                             dtype=np.float16).reshape(m, n)

    assert np.allclose(result_at(0), want), "偏移为 0 时内核本身就算错了"
    assert np.allclose(result_at(137216), want), (
        "内核把 mram_offset 又加了一遍，结果落到 2 倍偏移处")


def test_the_shard_alignment_changes_the_chosen_tile() -> None:
    """分片对齐要改变 A 路选出的分块，否则下发了没人用。

    `(16,256) x (64,256)` 上，DMA 对齐从 8 提到 1024，选出的 tile 从
    `m=8,n=64` 变成 `m=16,n=32`（实测）。`align_bytes` 是统一 IR 里这条
    对齐要求的载体，必须进到 `-pim-tile-to-budget` 的分块决策里。
    """
    from opcompiler_bridge.driver import _a_path_shard, _make_ttir, _run_triton_opt

    def tile_of(align):
        request = _request(op="linear", arg_shapes=[(16, 256), (64, 256)],
                           shard=DpuShard(dim=0, num_dpus=2), align_bytes=align)
        # WRAM 收到 8192：预算宽裕时整块矩阵本来就放得下，对齐没有可改变的分块。
        request = replace(request, hardware=replace(
            request.hardware, dma_align=8, wram_bytes_per_dpu=8192,
            num_tasklets=4))
        pimir, _ = _run_triton_opt(
            _make_ttir(request), request.hardware, _a_path_shard(request),
            align_bytes=request.align_bytes)
        import re
        got = dict(re.findall(r'pim\.tile-([mn])" = (\d+)', pimir))
        return (int(got["m"]), int(got["n"]))

    assert tile_of(0) != tile_of(1024), "分片对齐没有改变选出的分块"


def test_the_alignment_is_sent_down_now_that_it_has_a_consumer() -> None:
    """`alignBytes` 有了消费者就要下发。

    `-pim-tile-to-budget` 在它比 `pim.dma-align` 更严时按它选分块
    （见 `test_the_shard_alignment_changes_the_chosen_tile`），不再是只写不读。
    """
    from contracts.mlir_layout import placement_attribute

    text = " ".join(placement_attribute(
        DpuShard(dim=0, num_dpus=2), rank=2, align_bytes=1024))
    assert "alignBytes = 1024" in text


def test_the_address_lands_on_the_result_dma_only() -> None:
    """起始地址只盖到结果张量那条 DMA，对齐不改变行内步幅。

    `mram_offset` 描述的是结果分片，x 与 w 各有各的地址，盖同一个会把输入的
    地址也挪了。`align_bytes` 是字节单位的起始地址对齐，不是元素步幅：早先把它
    直接抬成 `elem_stride`，f16 下把搬运量放大了一倍。
    """
    from opcompiler_bridge.driver import (
        _a_path_shard, _make_ttir, _run_triton_opt)

    request = _request(op="linear", arg_shapes=[(1, 16, 64), (32, 64)],
                       shard=DpuShard(dim=2, num_dpus=2),
                       mram_offset=4096, align_bytes=64)
    pimir, _ = _run_triton_opt(
        _make_ttir(request), request.hardware, _a_path_shard(request),
        mram_offset=request.mram_offset, align_bytes=request.align_bytes)
    stores = [l for l in pimir.splitlines() if "dma_store" in l]
    loads = [l for l in pimir.splitlines() if "dma_load" in l]
    assert stores and all("mram_offset = 4096" in l for l in stores), stores
    assert loads and all("mram_offset" not in l for l in loads), loads
    assert "elem_stride = 64" not in pimir, "对齐被当成元素步幅了"
    assert "elem_stride = 1" in pimir


def test_the_cost_model_reads_the_shard_decision_without_double_dividing() -> None:
    """成本模型要读回切分决策，但**不能**再按切分数分摊搬运量。

    搬运量是从这份 pimir 自己的 `pim.dma_*` 累出来的，而这份 pimir 由图编译器
    按执行计划的 `local_shape` 生成 —— 文本里的字节数本来就是单台口径。再除一次
    就把单台流量算成实际的 1/N：实测同一份本地形状，带 placement 得 3584B、
    不带得 7168B，而真实单台搬运量是 7168B。

    这条原先断言的恰是那个错口径（`== without / 2`）。决策仍然被消费，只是落点
    不是这个除法：`shard_dim` / `shard_dpus` 进 sidecar 并在 GeneSim 侧参与容量
    核对，`placed_mram_bytes` 覆盖单台占用。
    """
    from genesim_bridge.ir_cost import analyze_ir

    body = ('  tt.func @kernel(%a: !pim.memdesc<256xf16, #pim.mram>, '
            '%w: !pim.memdesc<256xf16, #pim.wram>) {\n'
            '    pim.dma_load %w, %a : !pim.memdesc<256xf16, #pim.wram>, '
            '!pim.memdesc<256xf16, #pim.mram>\n'
            '    tt.return\n  }\n')
    bare = 'module attributes {pim.target = "pim:v1"} {\n' + body + '}\n'
    tp2 = ('module attributes {pim.target = "pim:v1", '
           'pim.placement = #pim.placement<kind = shard, dim = 1, '
           'numDpus = 2>} {\n' + body + '}\n')

    without = analyze_ir(bare, "k", (1,), {}, ir_level="pimir")
    with_shard = analyze_ir(tp2, "k", (1,), {}, ir_level="pimir")

    # 决策读回来了 —— 这是「不是只写不读」的判据。
    assert without.shard_dpus is None and without.shard_dim is None
    assert with_shard.shard_dim == 1 and with_shard.shard_dpus == 2
    assert any("跨 DPU 切分" in n for n in with_shard.notes)

    # 同一份 IR 文本：带不带 placement，搬运量必须相同。
    assert without.mram_traffic_bytes > 0, "用例前提：这份 IR 有真实搬运量"
    assert with_shard.mram_traffic_bytes == without.mram_traffic_bytes, (
        f"搬运量被按切分数多除了一次："
        f"{without.mram_traffic_bytes} -> {with_shard.mram_traffic_bytes}")


def test_the_shard_decision_reaches_the_simulation_input() -> None:
    """决策要随 sidecar 进仿真输入（需求目标四的第三个落点）。

    判据是 `_pim_kernel_dict` **产出的取值**，不是它源码里出现过这两个词：
    字段改名、或取值恒为 None，源码扫法都发现不了。
    """
    from genesim_bridge.cost_extractor import _pim_kernel_dict
    from genesim_bridge.ir_cost import analyze_ir

    body = ('  tt.func @kernel(%a: !pim.memdesc<256xf16, #pim.mram>) {\n'
            '    tt.return\n  }\n')
    tp2 = ('module attributes {pim.target = "pim:v1", '
           'pim.placement = #pim.placement<kind = shard, dim = 1, '
           'numDpus = 2>} {\n' + body + '}\n')
    bare = 'module attributes {pim.target = "pim:v1"} {\n' + body + '}\n'

    sharded = _pim_kernel_dict(analyze_ir(tp2, "k", (1,), {}, ir_level="pimir"))
    assert sharded["shard_dim"] == 1 and sharded["shard_dpus"] == 2, sharded

    # 单 DPU 时这两项为 None —— 既有字段语义不受影响，也让上面那条可失败。
    plain = _pim_kernel_dict(analyze_ir(bare, "k", (1,), {}, ir_level="pimir"))
    assert plain["shard_dim"] is None and plain["shard_dpus"] is None, plain


# ---- 第 1 步：两个载体收口成一个，断开的链路接上 ----

@pytest.mark.skipif(not _has_pim_passes(),
                    reason="当前 triton-opt 没有 PIM pass，需重跑 0-install-flagtree.sh")
def test_a_real_tp2_linear_carries_a_non_all_ones_split() -> None:
    """P1-1 的真判据：**真实** tp2 `linear` 的 pimir 里有非全 1 的 dpusPerDevice。

    这条是整轮的核心。此前图编译器下发 `pim.shard-dim`、FlagTree 读
    `pim.placement`，两套载体从未在一条真实链路上相遇 —— 两边测试都绿，
    而实算的 tp2 pimir 里 `dpusPerDevice` 一次都不出现。
    """
    from opcompiler_bridge.driver import compile_op

    hw = _hw()
    # 真实 tp2 计划里 `linear` 的输出是三维、切末维（`shard_dim=2`）——
    # 此前这里用二维 + `dim=1`，恰好落在秩内，测不到越界丢弃那条缺陷。
    base = dict(op="linear", arg_shapes=[(1, 16, 64), (32, 64)], hardware=hw,
                dtype="float16")
    plain = compile_op(OpCompileRequest(**base, shard=None), force=True)
    tp2 = compile_op(OpCompileRequest(**base, shard=DpuShard(dim=2, num_dpus=2)),
                     force=True)

    assert "dpusPerDevice" not in (plain.pimir or ""), \
        "单 DPU 不该出现跨 DPU 编码"
    assert "dpusPerDevice" in (tp2.pimir or ""), \
        f"tp2 的 dpusPerDevice 丢了：\n{(tp2.pimir or '')[:600]}"
    # printer 省略全 1，所以「出现」即「非全 1」。
    assert "numDpus = 2" in tp2.pimir


# ---- 第 2 步：下发的决策改变 PIMMLIR 自己的决策 ----

@pytest.mark.skipif(not _has_pim_passes(),
                    reason="当前 triton-opt 没有 PIM pass，需重跑 0-install-flagtree.sh")
def test_the_mram_budget_is_charged_the_real_single_dpu_footprint() -> None:
    """MRAM 判据按**本地形状**算的 footprint 比单台预算，不再按切分数分摊。

    下发到 A 路的 `arg_shapes` 取自执行计划的 `local_shape`，本来就是单台那一份
    （tp2 的 q_proj 权重是 32×64，全局是 64×64）。再除一次切分数就把判据放宽了
    N 倍：实测一个单台实际要 7168B 的 kernel 通过了 5000B 的预算，而它本不该过。

    这条用真实口径钉住：同一份本地形状，预算刚好小于真实占用时必须被拒。
    """
    import re

    from opcompiler_bridge.driver import (
        _a_path_shard, _make_ttir, _with_placement_attr)

    # tp2 的本地权重 32×64（全局 64×64）。真实单台占用由 pass 回传。
    local = dict(op="linear", arg_shapes=[(1, 16, 64), (32, 64)],
                 dtype="float16")
    shard = DpuShard(dim=2, num_dpus=2)

    def run(mram_bytes: int):
        hw = replace(_hw(), num_dpus=2, mram_bytes_per_dpu=mram_bytes)
        request = OpCompileRequest(**local, hardware=hw, shard=shard)
        ttir = _with_placement_attr(_make_ttir(request), _a_path_shard(request),
                                    rank=2)
        passes = [
            f"-convert-triton-to-pim=target=pim:v1 num-dpus={hw.num_dpus} "
            f"num-tasklets={hw.num_tasklets} wram-bytes={hw.wram_bytes_per_dpu} "
            f"mram-bytes={hw.mram_bytes_per_dpu} dma-align={hw.dma_align}",
            "-pim-tile-to-budget",
        ]
        return subprocess.run([str(_triton_opt()), "-"] + passes,
                              input=ttir, capture_output=True, text=True)

    # 先用一个宽裕的预算问出这个 kernel 单台到底占多少。
    proc = run(1 << 30)
    assert proc.returncode == 0, proc.stderr[:400]
    placed = re.search(r'"pim.placed-mram-bytes" = (\d+)', proc.stdout)
    assert placed is not None, f"回传里没有单台占用：\n{proc.stdout[:400]}"
    per_dpu = int(placed.group(1))

    # 预算刚好差 1 字节：必须被拒。分摊过的实现会把它放过去。
    proc = run(per_dpu - 1)
    assert proc.returncode != 0 and "exceeds mram-bytes" in proc.stderr, (
        f"单台真实占用 {per_dpu}B，预算 {per_dpu - 1}B 却通过了 —— "
        f"判据被按切分数放宽了：rc={proc.returncode}\n{proc.stderr[:400]}")

    # 预算刚好够：必须通过（判据不是一味从严）。
    assert run(per_dpu).returncode == 0


# ---- 第 3 步：PIMMLIR 回传，且回传改变下游产物 ----

@pytest.mark.skipif(not _has_pim_passes(),
                    reason="当前 triton-opt 没有 PIM pass，需重跑 0-install-flagtree.sh")
def test_the_pass_reports_back_what_the_split_cost() -> None:
    """回程：`pim-tile-to-budget` 把实际的单台占用与除数写回模块。

    图编译器决定怎么切，但切完一台 DPU 到底摆多少字节取决于分块 ——
    而分块是那个 pass 定的，所以这个数只能由它回传。
    """
    from contracts.ir_payloads import placement_of_module
    from opcompiler_bridge.driver import compile_op

    hw = _hw()

    def back(arg_shapes, shard):
        result = compile_op(OpCompileRequest(
            op="linear", arg_shapes=arg_shapes, hardware=hw, dtype="float16",
            shard=shard), force=True)
        return placement_of_module(result.pimir)

    # 全局权重 1024 行、单 DPU。
    whole = back([(4, 1024), (1024, 1024)], None)
    # 同一个算子切 2 台：下发的 `arg_shapes` 取执行计划的 `local_shape`，
    # 所以权重是**本地的** 512 行。
    half = back([(1, 4, 1024), (512, 1024)], DpuShard(dim=2, num_dpus=2))

    assert whole.placed_shards == 1, whole
    assert half.placed_shards == 2, half
    assert whole.placed_mram_bytes and half.placed_mram_bytes

    # 回传的是**本地形状**的真实占用：本地权重只有一半行，占用随之变小。
    # 注意这不是「拿全局占用除以切分数」—— 那个口径把判据放宽了 N 倍（实测
    # 单台实际要 7168B 的 kernel 通过了 5000B 的预算）。这里减小是因为
    # 下发的形状本来就小，pass 没有再除一次。
    assert half.placed_mram_bytes < whole.placed_mram_bytes, (
        f"本地形状更小，回传的占用却没变小：{whole} vs {half}")

    # 同一份本地形状下，带不带 placement 都回传同一个占用 —— 这钉住
    # 「不再按切分数分摊」：分摊过的实现会让带 placement 的那份小一半。
    local_only = back([(1, 4, 1024), (512, 1024)], None)
    assert half.placed_mram_bytes == local_only.placed_mram_bytes, (
        f"同一份本地形状，带 placement 的回传被多除了一次："
        f"{local_only} vs {half}")

    # 意图（下发的 numDpus）与效果（pass 实际看到的切分宽度）必须一致。
    assert half.intent_matches_effect


def test_the_feedback_changes_the_cost_product() -> None:
    """回传必须改变下游产物，否则就是只写不读（需求 P1-2 反面判据）。"""
    from genesim_bridge.ir_cost import analyze_ir

    body = ('  tt.func @kernel(%a: !pim.memdesc<256xf16, #pim.mram>) {\n'
            '    tt.return\n  }\n')
    head = ('module attributes {pim.target = "pim:v1", '
            '"pim.placed-mram-bytes" = %d : i64, '
            '"pim.placed-shards" = %d : i64} {\n')

    one = analyze_ir((head % (2_113_536, 1)) + body, "k", (1,), {},
                     ir_level="pimir")
    two = analyze_ir((head % (1_056_768, 2)) + body, "k", (1,), {},
                     ir_level="pimir")
    assert one.mram_bytes_per_dpu == 2_113_536
    assert two.mram_bytes_per_dpu == 1_056_768, \
        "回传的单台占用没有进成本模型"


def test_an_intent_effect_mismatch_is_surfaced() -> None:
    """下发意图与回传效果不符时必须说出来，不能静默按错的规模算。"""
    from genesim_bridge.ir_cost import analyze_ir

    # 下发切 4 台，回传说只除了 2 —— 两侧对切分的理解漂了。
    text = ('module attributes {pim.target = "pim:v1", '
            'pim.placement = #pim.placement<kind = shard, dim = 1, numDpus = 4>, '
            '"pim.placed-shards" = 2 : i64} {\n'
            '  tt.func @kernel(%a: !pim.memdesc<256xf16, #pim.mram>) {\n'
            '    tt.return\n  }\n}\n')
    cost = analyze_ir(text, "k", (1,), {}, ir_level="pimir")
    assert not cost.placement.intent_matches_effect
    assert any("意图与算子编译器实际除数不符" in n for n in cost.notes), cost.notes


def test_the_feedback_reaches_the_simulation_input_too() -> None:
    """回传要随 sidecar 进仿真输入（需求目标四的第三个落点）。

    判据是**产物**里有这两项，不是源码里出现过这两个词：原先这条只
    `inspect.getsource(_pim_kernel_dict)` 找字符串，字段改名或取值恒为 None
    都测不出来。
    """
    from genesim_bridge.cost_extractor import _pim_kernel_dict
    from genesim_bridge.ir_cost import analyze_ir

    text = ('module attributes {pim.target = "pim:v1", '
            'pim.placement = #pim.placement<kind = shard, dim = 1, numDpus = 2>, '
            '"pim.placed-mram-bytes" = 7168 : i64, '
            '"pim.placed-shards" = 2 : i64} {\n'
            '  tt.func @kernel(%a: !pim.memdesc<256xf16, #pim.mram>) {\n'
            '    tt.return\n  }\n}\n')
    entry = _pim_kernel_dict(analyze_ir(text, "k", (1,), {}, ir_level="pimir"))
    assert entry["placed_mram_bytes"] == 7168, entry
    assert entry["placed_shards"] == 2, entry
    assert entry["shard_dpus"] == 2 and entry["shard_dim"] == 1, entry


def test_the_simulator_actually_consumes_the_returned_fields() -> None:
    """回传在仿真侧必须有读者，不能只是 sidecar 里的一行。

    需求 P1-2 的反面判据：本轮新增的回传字段不允许只写不读。消费点是 GeneSim
    的 `_check_placed_mram_against_capacity` —— 用回传的单台占用核对常驻容量，
    并比对下发意图与实际除数。这里只钉住「那个消费方存在且读这两个键」，它自己
    的行为判据在 GeneSim 仓的 `tests/sim/test_cost_sidecar.py` 里。
    """
    import inspect
    from pathlib import Path as _P

    scheduler = _P("/media/disk/fengjingge/src/genesim/src/scheduler/"
                   "gene_sim_scheduler.py")
    if not scheduler.is_file():
        pytest.skip("GeneSim 不在位")
    source = scheduler.read_text()
    assert "_check_placed_mram_against_capacity" in source, \
        "GeneSim 侧没有回传的消费方，这两个字段又变成只写不读了"
    for key in ('"placed_mram_bytes"', '"placed_shards"'):
        assert f'kernel.get({key})' in source, \
            f"消费方没读 {key}"


# ---- 算子语义维：transpose purpose 的计费口径 ----

def test_all_transpose_purposes_are_billed_the_same() -> None:
    """三种 purpose 计费相同，这是刻意的 —— 别按 `absorbed` 少计。

    `absorbed` / `onthefly` 说的是**图层面**「这一层不单独发射节点」，而内核
    仍然要把值排到正确顺序上：输出是一个扁平缓冲，`LowerPIMToEmitC.cpp` 里
    那段逐元素置换真的在读源、写结果。按 absorbed 少计就把真实搬运漏掉了
    （本轮试过一版这样的改动，是错的，已撤回）。

    图层面的那个区分有它自己的消费方，走 `node.meta` 而不是这个 IR 属性：
    `gml_bridge/from_fx.py:233,925` 的 `ABSORBED_META_KEY` 决定发不发节点。
    两个载体回答的是不同层次的问题 —— 这条用例把口径钉住，免得下次又有人
    把「图层不发射」误读成「内核不搬」。
    """
    from genesim_bridge.ir_cost import _line_movement_bytes

    def line(purpose: str | None) -> str:
        attrs = "axes = array<i64: 1, 0>"
        if purpose:
            attrs += f", purpose = #pim.transpose_purpose<purpose = {purpose}>"
        return (f"    %0 = pim.transpose %a {{{attrs}}} "
                f": tensor<4x8xf16> -> tensor<8x4xf16>")

    billed = {p: _line_movement_bytes(line(p))
              for p in ("absorbed", "tensor_transpose", "layout_reorder", None)}
    assert billed["absorbed"] == 64.0
    assert len(set(billed.values())) == 1, billed


# ---- dtype 维的回传：元素宽度 ----

def test_the_returned_element_width_beats_the_name_table_guess() -> None:
    """元素宽度优先用回传值，不按类型名猜。

    `_DTYPE_BYTES.get(dtype, 2)` 的默认 2 是猜测：认不出的类型会让由它推出的
    每个字节数一起偏掉（w4a8 投影的操作数是 1 字节）。算子编译器定分块时手里
    就有真实宽度，回传它比猜更可靠 —— 这是 dtype 维的第一条回传。

    判据两半：回传与猜测不同时按回传并出 note；取不到回传时退回猜测。
    """
    from genesim_bridge.ir_cost import analyze_ir

    body = ('  tt.func @k(%a: !pim.memdesc<256xf16, #pim.mram>) {\n'
            '    tt.return\n  }\n')

    # 回传 1 字节，而按 `f16` 猜是 2 —— 必须按回传。
    told = analyze_ir(
        'module attributes {pim.target = "pim:v1", '
        '"pim.placed-elem-bytes" = 1 : i64} {\n' + body + '}\n',
        "k", (1,), {}, ir_level="pimir")
    assert told.element_bytes == 1, told.element_bytes
    assert any("元素宽度按回传取" in n for n in told.notes), told.notes

    # 没有回传：退回按类型名猜，与改动前一致。
    guessed = analyze_ir(
        'module attributes {pim.target = "pim:v1"} {\n' + body + '}\n',
        "k", (1,), {}, ir_level="pimir")
    assert guessed.element_bytes == 2, guessed.element_bytes
    assert not any("元素宽度按回传取" in n for n in guessed.notes)


# ---- Memory Layout 维：内核 tile 占用进容量判据 ----

@pytest.mark.skipif(not _has_pim_passes(),
                    reason="当前 triton-opt 没有 PIM pass，需重跑 0-install-flagtree.sh")
def test_the_kernel_footprint_probe_uses_the_real_arg_shapes() -> None:
    """`peak_kernel_mram_bytes` 在真实 tp2 计划上必须拿到**非 0** 的峰值。

    `linear` 的契约是 `arg_shapes=[x.shape, weight.shape]`，而 `shard_map` 挂的是
    节点**输出**的分片。拿输出当 x 会让 K 维对不上：`compile_op` 抛「K 维不一致」，
    而那个 `ValueError` 被「拿不到就不猜」的 except 吞掉 —— 峰值静默恒为 0，
    `plan_dpu` 的容量判据退化成只看三区，而测试全绿。实测真实 tp2 计划的 8 个
    linear **全部**命中这条路径。

    所以判据是「真实计划上非 0」，不是「函数能跑完」。
    """
    import torch
    from transformers import LlamaConfig, LlamaForCausalLM

    from graph.partition import partition_graph
    from graph.spec_prop import propagate_specs
    from graph.strategy import llama_strategy
    from runtime.compile import peak_kernel_mram_bytes
    from tests.test_partition import _FixedMaskLlama

    seq = 16
    model = LlamaForCausalLM(LlamaConfig(
        vocab_size=320, hidden_size=64, intermediate_size=176,
        num_hidden_layers=1, num_attention_heads=8, num_key_value_heads=8,
        max_position_embeddings=seq, bos_token_id=1, eos_token_id=2,
        pad_token_id=0)).eval()
    ids = torch.arange(seq, dtype=torch.long).unsqueeze(0)
    blocked = torch.triu(torch.ones(seq, seq, dtype=torch.bool), diagonal=1)
    mask = torch.zeros((1, 1, seq, seq))
    mask.masked_fill_(blocked, torch.finfo(torch.float32).min)
    gm = torch.export.export(_FixedMaskLlama(model), (ids, mask),
                             strict=True).module()
    partition_graph(gm)
    propagate_specs(gm, llama_strategy(
        2, num_stages=1, num_heads=8, num_kv_heads=8,
        intermediate_size=176, vocab_size=320, num_layers=1))

    hardware = replace(_hw(), num_dpus=2, num_tasklets=4)
    peak = peak_kernel_mram_bytes(list(gm.graph.nodes), hardware=hardware)
    assert peak > 0, (
        "真实 tp2 计划上拿不到内核 tile 峰值 —— 多半是按输出形状当 x 去编译，"
        "K 维对不上后异常被吞掉，容量判据静默退回只看三区")


# ---- dtype 维：量化布局要到交付的 pimir 里 ----

@pytest.mark.skipif(not _has_pim_passes(),
                    reason="当前 triton-opt 没有 PIM pass，需重跑 0-install-flagtree.sh")
@pytest.mark.parametrize("group_size,want_groups", [(32, 4), (64, 2), (128, 1)])
def test_the_quant_layout_survives_into_the_delivered_pimir(
        group_size, want_groups) -> None:
    """量化布局要在**交付的** pimir 上可断言，不能只在下发的中间态里。

    `#pim.quant_spec` 本身被 `-pim-expand-phases` 消费掉（实测交付的 pimir 里
    出现 0 次，全仓缓存同样 0 份命中），此前 dtype 维的证据引的是下发文本 ——
    那是中间态，而需求 P1-1 的验证方式是「解析 pimir 断言」。

    布局并没有丢：它落成 `global_pool` 的 `groupSize` 与分组后的结果形状，两者
    都随 spec 变。这条按真实组宽参数化，钉住「组宽变 → 交付产物变」。
    """
    import re

    from opcompiler_bridge.driver import (
        _make_oplevel_mlir, _run_oplevel_triton_opt)

    request = OpCompileRequest(
        op="dynamic_quant", arg_shapes=[(1, 128)], hardware=_hw(),
        dtype="float16", group_size=group_size)
    downlink = _make_oplevel_mlir(request)
    assert "quant_spec" in downlink, "下发侧本来就该带 quant_spec"

    pimir, _ = _run_oplevel_triton_opt(downlink)
    # 属性本身被展开 pass 消费 —— 这是事实，写在这里免得下次又把它当缺陷。
    assert "quant_spec" not in pimir

    got = re.search(r"pim\.global_pool[^\n]*groupSize = (\d+)", pimir)
    assert got is not None, f"交付的 pimir 里没有量化组宽：\n{pimir[:500]}"
    assert int(got.group(1)) == group_size

    # 分组后的形状：128 个元素按组宽切出 want_groups 组。
    assert f"tensor<1x{want_groups}xf16>" in pimir, \
        f"分组形状与组宽不符（想要 {want_groups} 组）：\n{pimir[:500]}"


# ---- 排布字段（Memory Layout 第 3 层）从统一 IR 走到交付的 pimir ----
#
# 本轮之前 `tasklet_tiled` 只收 `rank`：tasklet 无条件全铺在第 0 维，`order`
# 写死行主序。实测 `tensor<1x16x32xf16>` 会拿到 `taskletsPerDpu = [16, 1, 1]`
# —— 给一个长度为 1 的轴分配 16 个 tasklet，且与 FlagTree 自己 builder 的
# 夹取口径相反。

def test_tasklet_distribution_follows_the_shard_shape() -> None:
    """tasklet 按形状铺 —— 与 FlagTree 两个 builder 的夹取算式同口径。"""
    text = tasklet_tiled((1, 16, 32), shard=DpuShard(dim=1, num_dpus=2),
                         num_tasklets=16)
    assert "taskletsPerDpu = [1, 1, 16]" in text, text


@pytest.mark.parametrize("shape,num_tasklets,want", [
    # 手算自 FlagTree builder 的算式：order[0] 是最内层，逐维
    # `avail = shape[i] // sizePerTasklet[i]`，`tasklets[i] = min(剩余, avail)`，
    # 剩余整除后继续；循环结束仍有剩余时全乘到 order 的最后一维上。
    ((1, 16, 32), 16, [1, 1, 16]),      # 内层 32 就装得下 16 个，外层不动
    ((16, 128), 16, [1, 16]),
    ((4, 8), 16, [2, 8]),               # 内层只装 8 个，剩 2 个铺到外层
    ((2048, 4096), 16, [1, 16]),
    ((1, 1, 1), 8, [8, 1, 1]),          # 整块比 tasklet 还少，余数落最后一维
])
def test_tasklet_distribution_matches_the_flagtree_builder(
        shape, num_tasklets, want) -> None:
    """逐维手算值与实现一致 —— 与 FlagTree builder 的算式同口径。

    夹取不是可选项：给长度为 1 的轴分 16 个 tasklet 是做不到的分配，而
    FlagTree 的两个 builder 都按可用长度夹取，本仓原先无条件铺在第 0 维。
    """
    import re

    text = tasklet_tiled(shape, shard=DpuShard(dim=len(shape) - 1, num_dpus=2),
                         num_tasklets=num_tasklets)
    got = [int(v) for v in re.search(
        r"taskletsPerDpu = \[([^\]]*)\]", text).group(1).split(",")]
    assert got == want


def test_the_encoding_order_comes_from_the_layout_field() -> None:
    """`order` 由统一 IR 的排布字段推出，不是写死的行主序。

    行主序与列主序必须给出不同的 order —— 这条钉住「排布真的从 IR 来」，
    而不是恰好与写死的值相同。
    """
    row = tasklet_tiled((128, 4), shard=DpuShard(dim=0, num_dpus=2),
                        num_tasklets=16, elem_strides=(4, 1))
    # 列主序：第 0 维步幅 1、第 1 维步幅等于行数，行与行不重叠。
    col = tasklet_tiled((128, 4), shard=DpuShard(dim=0, num_dpus=2),
                        num_tasklets=16, elem_strides=(1, 128))
    assert "order = [1, 0]" in row, row
    assert "order = [0, 1]" in col, col


def test_overlapping_strides_are_rejected() -> None:
    """反例：步幅小于内层跨度时相邻元素会重叠，必须抛，不能照单下发。"""
    with pytest.raises(ValueError, match="重叠"):
        tasklet_tiled((4, 128), shard=DpuShard(dim=0, num_dpus=2),
                      num_tasklets=16, elem_strides=(1, 1))


def test_a_malformed_layout_field_is_rejected() -> None:
    """反例：步幅秩与形状不符时直接抛，不静默退回行主序。"""
    with pytest.raises(ValueError, match="秩"):
        tasklet_tiled((4, 128), shard=DpuShard(dim=0, num_dpus=2),
                      num_tasklets=16, elem_strides=(128,))


def test_the_layout_field_reaches_the_delivered_pimir() -> None:
    """统一 IR 的排布字段要真的走到交付的 pimir 上 —— 变异测试。

    同一份请求只改 `elem_strides`，模块属性 `#pim.placement` 的 `order`
    必须跟着变。张量编码的 `order` 不跟着变：B 路没有 pass 读它，让它变
    就是写一个没人读的值。
    """
    from opcompiler_bridge.driver import _make_oplevel_mlir

    row = _make_oplevel_mlir(_request(shard=DpuShard(dim=0, num_dpus=2),
                                      arg_shapes=[(4096, 2048)],
                                      elem_strides=(2048, 1)))
    col = _make_oplevel_mlir(_request(shard=DpuShard(dim=0, num_dpus=2),
                                      arg_shapes=[(4096, 2048)],
                                      elem_strides=(1, 4096)))
    assert "order = [0, 1]" in col, "排布字段没有进模块属性"
    assert "order = [0, 1]" not in row, "行主序不该显式写 order"


def test_the_layout_field_is_part_of_the_cache_key() -> None:
    """排布进缓存键：它改变下发文本（`order`），漏掉就会拿回旧文本。

    与 `shard` 同一条理由：`ctypes.CDLL` 按路径缓存句柄，两次不同排布的请求
    落到同一份 `.so` 上，第二次拿到的是第一份的代码与 pimir。
    """
    from opcompiler_bridge.driver import _cache_key

    keys = {
        _cache_key(_request(shard=DpuShard(dim=0, num_dpus=2))),
        _cache_key(_request(shard=DpuShard(dim=0, num_dpus=2),
                            elem_strides=(2048, 1))),
        _cache_key(_request(shard=DpuShard(dim=0, num_dpus=2),
                            elem_strides=(1, 2048))),
    }
    assert len(keys) == 3, "排布不同的请求落到了同一份产物上"


# ---- 没有挂载点的算子：shard 档不能只发模块属性 ----
#
# `kv_cache` 的结果写进 memdesc、没有张量结果（`_NO_RESULT_OPS`），所以贴不了
# `#pim.tasklet_tiled`。而 FlagTree 的跨载体漂移校验要求 shard 档**必须有**某个
# 张量编码记下切分，否则判「切分整条丢失」并拒绝整个模块 —— 一边刻意不贴、
# 一边要求必须有，两者直接互斥。

def _kv_request(*, shard):
    return _request(op="kv_cache", arg_shapes=[(1, 32, 128), (4096,)],
                    group_size=1, shard=shard)


def test_a_shard_without_a_mount_point_does_not_declare_the_module_placement() -> None:
    """shard 档 + 没有张量结果 → 一个 placement 字都不发。"""
    from opcompiler_bridge.driver import _make_oplevel_mlir

    text = _make_oplevel_mlir(_kv_request(shard=DpuShard(dim=1, num_dpus=2)))
    assert "pim.placement" not in text, text.splitlines()[0]


def test_a_replicated_op_without_a_mount_point_still_declares_its_placement() -> None:
    """replicate 档仍然发 —— 它没有「必须有编码」的要求，而这一档与单 DPU 的
    区别只在模块属性上表达得出（归约与容量口径不同）。"""
    from opcompiler_bridge.driver import _make_oplevel_mlir

    text = _make_oplevel_mlir(
        _kv_request(shard=DpuShard(dim=-1, num_dpus=2, kind="replicate")))
    assert "kind = replicate" in text, text.splitlines()[0]


@pytest.mark.skipif(not _has_pim_passes(),
                    reason="当前 triton-opt 没有 PIM pass，需重跑 0-install-flagtree.sh")
def test_a_sharded_kv_cache_actually_compiles() -> None:
    """判据落在**编译结果**上：带切分的 kv_cache 必须编得过。

    改动前实测 rc=1，报「no tensor layout in the module records a split」。
    """
    from opcompiler_bridge.driver import compile_op

    result = compile_op(_kv_request(shard=DpuShard(dim=1, num_dpus=2)))
    assert result.pimir


@pytest.mark.skipif(not _has_pim_passes(),
                    reason="当前 triton-opt 没有 PIM pass，需重跑 0-install-flagtree.sh")
def test_a_sharded_op_with_a_mount_point_still_declares_both() -> None:
    """有张量结果的算子不受影响：模块属性与张量编码都照发。"""
    from opcompiler_bridge.driver import compile_op

    result = compile_op(_request(shard=DpuShard(dim=0, num_dpus=2)))
    assert "pim.placement" in result.pimir
    assert "dpusPerDevice = [2, 1]" in result.pimir
