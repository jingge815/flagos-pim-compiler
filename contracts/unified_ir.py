"""图编译阶段统一 IR 的键登记表与阶段协议。

这是四维信息（算子语义 / 数据类型 / Placement / Memory Layout）挂载点的
**唯一真源**。新增 pass 不得私建键 —— `tests/test_unified_ir_contract.py`
的集合相等断言是这条规约的执行点。

`contracts/graph_meta.py` 保留为兼容 re-export，12 个既有 import 点不动。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:                       # 仅检查期，运行时零导入
    from torch.fx import Node

    from contracts.pim_tensor_spec import PIMTensorSpec

# ── 四维 + 基础设施 ──────────────────────────────────────────────
DIM_OP_SEMANTICS = "op_semantics"
DIM_DTYPE = "dtype"
DIM_PLACEMENT = "placement"
DIM_MEM_LAYOUT = "mem_layout"
DIM_INFRA = "infra"                     # 不属四维的基础设施键

DIMENSIONS = (DIM_OP_SEMANTICS, DIM_DTYPE, DIM_PLACEMENT, DIM_MEM_LAYOUT)

# ── 键名常量 ─────────────────────────────────────────────────────
# 跨模块读同一个键必须走常量，裸字符串在改名时不会报错。
DEVICE_META_KEY = "device"
PART_ID_META_KEY = "part_id"
SPEC_META_KEY = "spec"
REDISTRIBUTE_META_KEY = "redistribute"
FUSED_TAIL_META_KEY = "fused_tail"
DQ_META_KEY = "pim_dynamic_scaling"
RMS_NORM_META_KEY = "pim_rms_norm"
ATTENTION_SCALE_META_KEY = "pim_attention_scale"
ABSORBED_META_KEY = "pim_absorbed"
KV_DMA_META_KEY = "pim_kv_cache_dma"
SPLIT_META_KEY = "pim_split"
ROPE_META_KEY = "pim_rope"
HEAD_ROLE_META_KEY = "pim_head_role"
HEAD_INDEX_META_KEY = "pim_head_index"
VAL_META_KEY = "val"
MODULE_STACK_META_KEY = "nn_module_stack"

# ── 键值 ────────────────────────────────────────────────────────
DEVICE_DPU = "dpu"
DEVICE_HOST = "host"


@dataclass(frozen=True)
class MetaKeySpec:
    """一个 node.meta 键的完整契约。"""

    key: str                            # 键名
    dimensions: tuple[str, ...]         # 所属维度，可跨维
    payload: str                        # 载荷类型名（字符串，避免循环 import）
    producer: str                       # 写入方 "模块:函数"
    consumers: tuple[str, ...]          # 读取方；空 = 仅测试读取
    note: str = ""


# 四维信息的挂载点。键集合与全仓在用集合必须恰好相等。
META_KEYS: tuple[MetaKeySpec, ...] = (
    MetaKeySpec(
        DEVICE_META_KEY, (DIM_PLACEMENT,), "str",
        producer="graph.partition:partition_graph",
        consumers=("graph.spec_prop", "runtime.exec_plan_gen"),
        note="取值 DEVICE_DPU / DEVICE_HOST"),
    MetaKeySpec(
        PART_ID_META_KEY, (DIM_PLACEMENT,), "int",
        producer="graph.partition:partition_graph",
        consumers=(),
        note="生产代码零读取，仅 tests/test_partition.py 断言。"
             "显式声明以免被误判为死字段而删除"),
    MetaKeySpec(
        SPEC_META_KEY, (DIM_PLACEMENT, DIM_MEM_LAYOUT, DIM_DTYPE), "PIMTensorSpec",
        producer="graph.spec_prop:propagate_specs",
        consumers=("memory.mem_planner", "memory.kv_layout",
                   "runtime.exec_plan_gen", "runtime.compile",
                   "genesim_bridge.placement_export"),
        note="跨三维：切分属 Placement、mram_offset/elem_strides 属 "
             "Memory Layout、dtype/quant 属数据类型（本轮新增）"),
    MetaKeySpec(
        REDISTRIBUTE_META_KEY, (DIM_PLACEMENT,), "list[RedistributeEdge]",
        producer="graph.spec_prop:propagate_specs",
        consumers=("memory.mem_planner", "runtime.exec_plan_gen"),
        note="跨 DPU 的通信需求，每条含源 / 目标 DPU 集与字节数"),
    MetaKeySpec(
        FUSED_TAIL_META_KEY, (DIM_OP_SEMANTICS,), "FusedTail",
        producer="graph.fuse:fuse_graph | graph.fuse_pim:fuse_for_pim",
        consumers=("gml_bridge.from_fx", "gml_bridge.export",
                   "opcompiler_bridge.oplevel_emitter"),
        note="融合后 node.target 不变而语义已变，靠此键表达"),
    MetaKeySpec(
        ROPE_META_KEY, (DIM_OP_SEMANTICS,), "RopeMatch",
        producer="graph.fuse_rope:fuse_rope",
        # `graph.kv_dma_pass` 用 `ROPE_META_KEY in n.meta` 判断（不是 `.meta[...]`
        # 取数），漏登记过一次 —— 这一栏是排查取数点时唯一的索引。
        consumers=("gml_bridge.from_fx", "gml_bridge.export",
                   "graph.kv_dma_pass",
                   "opcompiler_bridge.oplevel_emitter")),
    MetaKeySpec(
        DQ_META_KEY, (DIM_OP_SEMANTICS, DIM_DTYPE), "DynamicScalingSpec",
        producer="graph.quant_pass:insert_dynamic_scaling",
        consumers=("gml_bridge.from_fx", "gml_bridge.export",
                   "opcompiler_bridge.oplevel_emitter"),
        note="跨两维：量化既是算子也是类型变换"),
    MetaKeySpec(
        RMS_NORM_META_KEY, (DIM_OP_SEMANTICS,), "RmsNormFusion",
        producer="graph.fuse_pim:fuse_for_pim",
        consumers=("gml_bridge.from_fx", "gml_bridge.export",
                   "opcompiler_bridge.oplevel_emitter")),
    MetaKeySpec(
        ATTENTION_SCALE_META_KEY, (DIM_OP_SEMANTICS,), "float",
        producer="graph.fuse_pim:fuse_for_pim | graph.split_heads:split_attention_heads",
        consumers=("gml_bridge.from_fx",),
        note="attention 的 1/√head_dim 缩放因子，GML 落成 Scaling_buffer_file"),
    MetaKeySpec(
        ABSORBED_META_KEY, (DIM_OP_SEMANTICS,), "bool",
        producer="graph.fuse_pim | graph.fuse_rope | graph.quant_pass "
                 "| graph.split_heads",
        consumers=("gml_bridge.from_fx", "opcompiler_bridge.oplevel_emitter"),
        note="True = 已被别的算子折进去，不再单独发射 GML 节点"),
    MetaKeySpec(
        KV_DMA_META_KEY, (DIM_MEM_LAYOUT, DIM_OP_SEMANTICS), "KvDmaSpec",
        producer="graph.kv_dma_pass:insert_kv_dma_and_split",
        consumers=("gml_bridge.from_fx",)),
    MetaKeySpec(
        SPLIT_META_KEY, (DIM_OP_SEMANTICS,), "SplitSpec",
        producer="graph.kv_dma_pass:insert_kv_dma_and_split",
        consumers=("gml_bridge.from_fx",)),
    MetaKeySpec(
        HEAD_ROLE_META_KEY, (DIM_OP_SEMANTICS,), "str",
        producer="graph.split_heads:split_attention_heads",
        consumers=("graph.kv_dma_pass", "graph.quant_pass",
                   "gml_bridge.from_fx", "opcompiler_bridge.oplevel_emitter"),
        note="逐头展开后该节点在 attention 里的角色，取值见 graph.split_heads"),
    MetaKeySpec(
        HEAD_INDEX_META_KEY, (DIM_OP_SEMANTICS, DIM_PLACEMENT), "int",
        producer="graph.split_heads:split_attention_heads",
        consumers=("graph.kv_dma_pass", "graph.quant_pass",
                   "gml_bridge.from_fx", "opcompiler_bridge.oplevel_emitter"),
        note="头下标，GML 落成 split_channel_number；也指示本地分片落在哪台 DPU"),
    MetaKeySpec(
        VAL_META_KEY, (DIM_INFRA,), "FakeTensor",
        producer="torch.export | graph.split_heads:split_attention_heads",
        consumers=("graph.spec_prop", "gml_bridge.from_fx",
                   "gml_bridge.export", "graph.fuse_rope", "graph.kv_dma_pass",
                   "graph.quant_pass", "graph.split_heads",
                   "runtime.exec_plan_gen"),
        note="torch.export 的产物，shape/dtype 的派生来源；不属四维，"
             "本轮不取代它"),
    MetaKeySpec(
        MODULE_STACK_META_KEY, (DIM_INFRA,), "dict",
        producer="torch.export",
        consumers=("graph.spec_prop",),
        note="模块路径栈，spec_prop 靠它认出权重属于哪一层"),
)

_BY_KEY = {spec.key: spec for spec in META_KEYS}


def meta_keys_of(dimension: str) -> tuple[str, ...]:
    """某一维度登记的全部键名。给分析 pass 用。"""
    if dimension not in DIMENSIONS and dimension != DIM_INFRA:
        raise ValueError(
            f"未知维度 {dimension!r}，允许 {(*DIMENSIONS, DIM_INFRA)}")
    return tuple(s.key for s in META_KEYS if dimension in s.dimensions)


def dimensions_of(key: str) -> tuple[str, ...]:
    """键属于哪些维度。未登记的键直接抛错 —— 这是契约收口的执行点。"""
    spec = _BY_KEY.get(key)
    if spec is None:
        raise ValueError(
            f"未登记的 meta 键 {key!r}。新增键必须先登记进 "
            f"contracts/unified_ir.py::META_KEYS，当前已登记 {sorted(_BY_KEY)}")
    return spec.dimensions


def spec_of(node: Node) -> PIMTensorSpec:
    """节点的张量规格。缺失直接抛错，不返回 None。

    这样调用方不必到处写 `if spec is None` —— 缺 spec 说明 propagate_specs
    没跑或该节点是 host 节点，两种情况都该由调用方在更早处理。
    """
    spec = node.meta.get(SPEC_META_KEY)
    if spec is None:
        raise ValueError(
            f"节点 {node.name} 没有 spec。请先运行 "
            f"graph.spec_prop.propagate_specs，或先判断 device 是否为 host")
    return spec


# ── 阶段协议 ─────────────────────────────────────────────────────
# 阶段标记挂在 GraphModule 上而非 node.meta —— 它描述整张图的状态，
# 不是单个节点的属性，也避免给键登记表塞特殊条目。
STAGE_EXPORTED = "exported"        # torch.export 出来，未标注
STAGE_PARTITIONED = "partitioned"  # partition_graph 跑过：device / part_id 齐
STAGE_SPECS = "specs"              # propagate_specs 跑过：DPU 节点有 spec
STAGE_FUSED = "fused"              # fuse_for_gml 跑过：融合语义键齐
STAGE_PLANNED = "planned"          # mem_planner 跑过：mram_offset 已回填

GRAPH_STAGE_KEY = "pim_graph_stage"

# 阶段的偏序。SPECS 与 FUSED 是**两条分叉**，不是先后关系：执行/仿真路径走
# 前者（有 spec 无融合语义），GML 路径走后者（有融合语义无 spec）。
# 所以用 dict 记前驱而非线性序号，否则会得出「FUSED 隐含 SPECS 已完成」的错论。
_STAGE_PREDECESSOR = {
    STAGE_EXPORTED: None,
    STAGE_PARTITIONED: STAGE_EXPORTED,
    STAGE_SPECS: STAGE_PARTITIONED,
    STAGE_FUSED: STAGE_PARTITIONED,
    STAGE_PLANNED: STAGE_SPECS,
}

_STAGE_PRODUCER = {
    STAGE_PARTITIONED: "graph.partition.partition_graph",
    STAGE_SPECS: "graph.spec_prop.propagate_specs",
    STAGE_FUSED: "gml_bridge.export.fuse_for_gml",
    STAGE_PLANNED: "memory.mem_planner.plan_dpu（经 runtime.compile.compile_model）",
}


def stages_of(gm) -> frozenset:
    """图已达成的全部阶段。

    返回集合而非单值 —— 一张图可以同时是 SPECS 与 FUSED（若两条 pass 都跑过）。
    """
    return frozenset(getattr(gm, "meta", {}).get(GRAPH_STAGE_KEY, ()) or ()) \
        | {STAGE_EXPORTED}


def mark_stage(gm, stage: str) -> None:
    """pass 在出口处标记。重复标记同一阶段允许（幂等）。"""
    if stage not in _STAGE_PREDECESSOR:
        raise ValueError(f"未知阶段 {stage!r}，允许 {sorted(_STAGE_PREDECESSOR)}")
    pred = _STAGE_PREDECESSOR[stage]
    if pred is not None and pred not in stages_of(gm):
        raise ValueError(
            f"不能标记 {stage}：前驱阶段 {pred} 未达成。"
            f"当前 {sorted(stages_of(gm))}")
    gm.meta = getattr(gm, "meta", {})
    gm.meta[GRAPH_STAGE_KEY] = tuple(stages_of(gm) | {stage})


def require_stage(gm, stage: str, *, who: str) -> None:
    """前置断言。把注释里的顺序依赖变成可执行契约的唯一入口。

    `who` 是调用方名字，出现在错误信息里 —— 照 spec_prop.py 的既有风格。
    """
    if stage not in stages_of(gm):
        raise ValueError(
            f"{who} 要求图已达到阶段 {stage}，当前 {sorted(stages_of(gm))}。"
            f"请先运行 {_STAGE_PRODUCER.get(stage, '对应 pass')}")


# ── 跨维一致性校验 ───────────────────────────────────────────────
# 单维校验（PIMTensorSpec.validate）抓不到维度**之间**的矛盾，这里补上。
# 每个阶段应当已填哪些维度：未到阶段的维度缺失是合法的，不报错。
DIMENSION_READY_AT = {
    STAGE_PARTITIONED: (),
    STAGE_SPECS: (DIM_DTYPE, DIM_PLACEMENT),
    STAGE_PLANNED: (DIM_DTYPE, DIM_PLACEMENT, DIM_MEM_LAYOUT),
}


def validate_node_dimensions(node, *, stage: str) -> None:
    """一个节点上四维的交叉一致性。在 pass 出口按图调用。

    只校验该阶段应当已填的维度（见 `DIMENSION_READY_AT`），未到阶段的维度
    缺失是合法的。两条跨维不变式：

    跨维一（dtype × 算子语义）：`spec.dtype` 是 `meta["val"].dtype` 的派生物，
    不是第二个真源。两者漂移说明有代码绕过 `_dtype_of` 直接改了 spec ——
    正是 P0-5 要防的旁路，这是本校验最重要的一条。

    跨维二（Placement × Memory Layout）：切分维必须落在本地形状的秩内。
    构造期只校验「维号 < 全局形状的秩」，切完之后的 `local_shape` 才是
    算子编译器实际看到的那块，秩对不上就是标到了不存在的轴上。
    """
    if stage not in DIMENSION_READY_AT:
        raise ValueError(
            f"未知阶段 {stage!r}，允许 {sorted(DIMENSION_READY_AT)}")
    dims = DIMENSION_READY_AT[stage]
    spec = node.meta.get(SPEC_META_KEY)
    if spec is None:
        return                              # 未到 STAGE_SPECS，或非张量节点

    spec.validate()                         # 单维校验先跑

    if DIM_DTYPE in dims:
        val = node.meta.get(VAL_META_KEY)
        # 只比张量：`val` 还可能是元组（split_with_sizes 那类），没有 dtype。
        if spec.dtype and hasattr(val, "dtype"):
            actual = str(val.dtype).removeprefix("torch.")
            if actual != spec.dtype:
                raise ValueError(
                    f"{node.name} 的 spec.dtype={spec.dtype!r} 与 "
                    f"meta['val'].dtype={actual!r} 不符：dtype 应由 val 派生，"
                    f"不该有第二个真源")

    if DIM_PLACEMENT in dims:
        for dpu_id, detail in spec.shard_map.items():
            if 0 <= detail.shard_dim >= len(detail.local_shape):
                raise ValueError(
                    f"{node.name} DPU{dpu_id} 的 shard_dim={detail.shard_dim} "
                    f"超出 local_shape 秩 {len(detail.local_shape)}")


def validate_graph_dimensions(gm, *, stage: str) -> None:
    """按图跑一遍跨维校验。pass 出口用，O(节点数) 的单次遍历。"""
    for node in gm.graph.nodes:
        validate_node_dimensions(node, stage=stage)
