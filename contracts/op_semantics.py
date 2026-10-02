"""算子语义的唯一真源。

原先四份清单各自硬编码，其中两份真重复：`_OPLEVEL_OPS`（算子编译器的内核
入口，15 个）与 `MNEMONICS`（GeneSim 助记符，14 个）去掉 `pim.` 前缀后
**重合 14 个、只差 `convert` 一项**，却是两处独立字面量 —— 改一处另一处不报错。

本表的条目按「算子名」建立，四份视图全部由它派生：
`oplevel_ops()` / `mnemonics()` / `aten_to_gml()` / `role_to_gml()`。

两类条目的区别只在 `has_kernel` / `is_mnemonic`：有内核入口的（15 个）参与
前两个视图；GML 独有的类型（Gemm、四种 Eltwise、池化、Conv）只提供 GML 名，
供 `aten_to_gml()` 用。
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class OpSemantics:
    """一个 PIM 算子在图阶段的完整语义。

    这是算子语义维度的唯一真源：四份既有清单全部由它派生或对它加断言。
    """

    name: str                          # 算子名，不带 pim. 前缀，如 "matmul"
    gml_op_type: str | None            # 对应的 GML op_type；None = 不落 GML
    has_kernel: bool                   # 算子编译器有内核入口（→ _OPLEVEL_OPS）
    is_mnemonic: bool                  # 进 GeneSim 助记符统计（→ MNEMONICS）
    aten_targets: tuple = ()           # 映射到它的 aten 目标（→ OP_TYPES）
    role_aliases: tuple = ()           # 角色别名，优先于 aten（→ _ROLE_OP_TYPES）
    # 运行时走的内核入口名。None = 就是 `name` 自己。GML 名与内核入口不是一一
    # 对应：四种逐元素都进同一个 `eltwise` 内核，`silu` 进 `lut` 内核。
    kernel: str | None = None
    note: str = ""


# 声明序即 `mnemonics()` 的顺序 —— 与改动前的字面量一致，便于人读错误信息。
OP_SEMANTICS: tuple[OpSemantics, ...] = (
    OpSemantics("normalize", "RMSNorm_vpu", has_kernel=True, is_mnemonic=True,
                aten_targets=("rsqrt.default",),
                note="aten 里是一串 pow/mean/add/rsqrt/mul，折成一个 RMSNorm_vpu"),
    OpSemantics("matmul", "MatMul", has_kernel=True, is_mnemonic=True,
                aten_targets=("bmm.default", "matmul.default", "mm.default",
                              "scaled_dot_product_attention.default"),
                role_aliases=("matmul1", "matmul2"),
                note="attn 的两个矩阵乘经逐头展开后带角色标记，走角色别名"),
    OpSemantics("softmax", "Softmax", has_kernel=True, is_mnemonic=True,
                aten_targets=("_softmax.default",), role_aliases=("softmax",)),
    OpSemantics("mask", "Mask", has_kernel=True, is_mnemonic=True,
                aten_targets=("masked_fill.Scalar", "where.self"),
                role_aliases=("mask",)),
    OpSemantics("rope", "Llama2Activation", has_kernel=True, is_mnemonic=True,
                note="由 fuse_rope 折出来，aten 目标不是一个算子而是一条链"),
    OpSemantics("lut", "Lut", has_kernel=True, is_mnemonic=True,
                note="折进 contraction 的激活，GML 用 Lut 加 activation_op_type"),
    OpSemantics("eltwise", None, has_kernel=True, is_mnemonic=True,
                note="加/减/乘/除共用一个内核入口，GML 名各不相同，见下面四条目"),
    OpSemantics("dynamic_quant", "DynamicScaling", has_kernel=True,
                is_mnemonic=True,
                note="由 quant_pass 插入，aten 图里没有"),
    OpSemantics("kv_cache", "KV_Cache_DMA", has_kernel=True, is_mnemonic=True,
                note="由 kv_dma_pass 插入，aten 图里没有（图用 use_cache=False 导出）"),
    OpSemantics("gather", "Gather", has_kernel=True, is_mnemonic=True,
                aten_targets=("embedding.default",)),
    OpSemantics("transpose", "Transpose", has_kernel=True, is_mnemonic=True,
                aten_targets=("transpose.int", "permute.default")),
    OpSemantics("reshape", "Reshape", has_kernel=True, is_mnemonic=True,
                aten_targets=("view.default", "reshape.default")),
    OpSemantics("split_heads", "Split", has_kernel=True, is_mnemonic=True,
                aten_targets=("split.Tensor", "split_with_sizes.default"),
                note="逐头展开用 slice 隐式拆头，实物是一个多输出的 Split"),
    OpSemantics("concat", "Concat", has_kernel=True, is_mnemonic=True,
                aten_targets=("cat.default",)),
    # convert 有内核入口但不进助记符统计 —— 它是上面两份清单唯一的差集项。
    OpSemantics("convert", "Convert", has_kernel=True, is_mnemonic=False,
                aten_targets=("to.dtype", "to.dtype_layout"),
                note="类型转换是一次真实的数据运动，位宽不同必须有节点承载"),
    # 以下只提供 GML 名，不算内核入口，也不进助记符统计。
    OpSemantics("gemm", "Gemm", has_kernel=False, is_mnemonic=False,
                aten_targets=("linear.default", "addmm.default")),
    OpSemantics("eltwise_add", "EltwiseAdd", has_kernel=False, is_mnemonic=False,
                aten_targets=("add.Tensor",), kernel="eltwise"),
    OpSemantics("eltwise_sub", "EltwiseSub", has_kernel=False, is_mnemonic=False,
                aten_targets=("sub.Tensor",), kernel="eltwise"),
    OpSemantics("eltwise_mul", "EltwiseMul", has_kernel=False, is_mnemonic=False,
                aten_targets=("mul.Tensor",), kernel="eltwise"),
    OpSemantics("eltwise_div", "EltwiseDiv", has_kernel=False, is_mnemonic=False,
                aten_targets=("div.Tensor",), kernel="eltwise"),
    OpSemantics("silu", "Silu", has_kernel=False, is_mnemonic=False,
                aten_targets=("silu.default",), kernel="lut",
                note="自带 nmu_mode / fpsu_* / kantor_mode 与 contraction，是主算子"),
    OpSemantics("max_pool", "MaxPool", has_kernel=False, is_mnemonic=False,
                aten_targets=("max_pool2d.default",)),
    OpSemantics("average_pool", "AveragePool", has_kernel=False, is_mnemonic=False,
                aten_targets=("avg_pool2d.default",)),
    OpSemantics("conv", "Conv", has_kernel=False, is_mnemonic=False,
                aten_targets=("convolution.default",)),
)


def _resolve_aten(name: str):
    """把 "bmm" / "to.dtype" 解析成 `torch.ops.aten` 的重载对象。

    名字里带点的是显式重载（to.dtype）；不带点的补 `.default`。
    解析失败直接抛 —— 拼错算子名必须立刻暴露。
    """
    import torch

    parts = name.split(".")
    op = getattr(torch.ops.aten, parts[0], None)
    if op is None:
        raise ValueError(f"torch.ops.aten 没有 {parts[0]!r}")
    overload = parts[1] if len(parts) > 1 else "default"
    if not hasattr(op, overload):
        raise ValueError(f"aten.{parts[0]} 没有重载 {overload!r}")
    return getattr(op, overload)


def _validate_registry(registry: tuple = OP_SEMANTICS) -> None:
    """登记表自洽性。模块导入时跑一次 —— 契约错了就不该能 import 成功。"""
    names = [s.name for s in registry]
    dup = {n for n in names if names.count(n) > 1}
    if dup:
        raise ValueError(f"算子登记表有重名：{sorted(dup)}")
    # aten 目标不能映射到两个算子，否则派生结果取决于遍历顺序
    seen: dict[str, str] = {}
    for spec in registry:
        for target in spec.aten_targets:
            if target in seen:
                raise ValueError(f"aten 目标 {target} 同时映射到 {seen[target]} 与 {spec.name}")
            seen[target] = spec.name


_validate_registry()


def oplevel_ops() -> frozenset:
    """算子编译器内核入口集合。取代 driver.py 的字面量。"""
    return frozenset(s.name for s in OP_SEMANTICS if s.has_kernel)


def kernel_entry_of(aten_target: str) -> str | None:
    """这个 aten 目标在运行时走的算子编译器内核入口；不走的返回 None。

    与 `oplevel_ops()` 同源：入口名必须在那个集合里。GML 名与内核入口不是一一
    对应的（四种逐元素共用 `eltwise`，`silu` 走 `lut`），所以不能只看
    `has_kernel` 那一行。
    """
    entries = oplevel_ops()
    for spec in OP_SEMANTICS:
        for target in spec.aten_targets:
            if aten_target.endswith(f"aten.{target}"):
                entry = spec.kernel or (spec.name if spec.has_kernel else None)
                return entry if entry in entries else None
    return None


def mnemonics() -> tuple:
    """GeneSim 助记符。取代 op_classify.py 的字面量。

    顺序按登记表声明序 —— 它只进错误信息，不参与判定，但保持稳定顺序便于人读。
    """
    return tuple(f"pim.{s.name}" for s in OP_SEMANTICS if s.is_mnemonic)


def aten_to_gml() -> dict:
    """aten 目标 → GML op_type。取代 from_fx.py 的 OP_TYPES。"""
    out = {}
    for spec in OP_SEMANTICS:
        if spec.gml_op_type is None:
            continue
        for target in spec.aten_targets:
            out[_resolve_aten(target)] = spec.gml_op_type
    return out


def role_to_gml() -> dict:
    """角色 → GML op_type。取代 from_fx.py 的 _ROLE_OP_TYPES。

    **必须与 `aten_to_gml()` 分开**：`from_fx` 先查角色再查 aten，
    合并成一张表会丢掉这个优先级，逐头节点的 op_type 会退回按 aten 判定。
    """
    return {r: s.gml_op_type for s in OP_SEMANTICS
            for r in s.role_aliases if s.gml_op_type}
