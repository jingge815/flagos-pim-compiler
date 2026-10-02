from dataclasses import dataclass
from typing import Literal

from contracts.dtypes import validate_dtype
from contracts.gml_quant import QuantLayout
from contracts.mem_layout import check_elem_strides


@dataclass(frozen=True)
class Placement:
    kind: Literal["Shard", "Replicate", "Partial"]
    dim: int | None = None
    reduce_type: str | None = None

    def validate(self) -> None:
        if self.kind == "Shard":
            if self.dim is None or self.dim < 0 or self.reduce_type is not None:
                raise ValueError("Shard placement requires a non-negative dim and no reduce_type")
        elif self.kind == "Replicate":
            if self.dim is not None or self.reduce_type is not None:
                raise ValueError("Replicate placement requires no dim or reduce_type")
        elif self.kind == "Partial":
            if self.dim is not None or self.reduce_type not in ("sum", "mean"):
                raise ValueError("Partial placement requires reduce_type 'sum' or 'mean' and no dim")
        else:
            raise ValueError(f"unsupported placement kind: {self.kind}")


@dataclass(frozen=True)
class TensorShardDetail:
    dpu_id: int
    shard_dim: int
    start_idx: int
    end_idx: int
    local_shape: tuple[int, ...]
    mram_offset: int = 0
    # 本轮新增：排布层（第 3 层）。空值语义是**已确认**，不是未知 ——
    # 图编译器是做内存规划的那一方，知道这段字节怎么摆，所以
    # elem_strides=() 表示「确认行主序紧密」、align_bytes=0 表示「无额外要求」。
    # 注意这与 PIMMLIR 侧 DMA 属性的口径相反（那边未证明就不能假设连续），
    # 两者回答的问题不同，不是不一致。
    elem_strides: tuple[int, ...] = ()   # 逐维步幅（元素数）；() = 行主序紧密
    align_bytes: int = 0                 # 该分片起始地址的额外对齐；0 = 无额外要求

    def validate(self) -> None:
        if self.dpu_id < 0:
            raise ValueError("dpu_id must be non-negative")
        if self.start_idx < 0 or self.end_idx < 0 or self.end_idx < self.start_idx:
            raise ValueError("shard range must be non-negative and ordered")
        if self.mram_offset < 0:
            raise ValueError("mram_offset must be non-negative")
        if any(dim < 0 for dim in self.local_shape):
            raise ValueError("local_shape dimensions must be non-negative")
        self._validate_layout()

    def _validate_layout(self) -> None:
        """排布层：步幅自洽，对齐合法。"""
        # 步幅必须能容纳该维的实际长度，否则相邻行会重叠。
        check_elem_strides(self.local_shape, self.elem_strides)
        if self.align_bytes:
            if self.align_bytes & (self.align_bytes - 1):
                raise ValueError(f"align_bytes 必须是 2 的幂，got {self.align_bytes}")
            if self.mram_offset % self.align_bytes:
                raise ValueError(
                    f"mram_offset {self.mram_offset} 不满足 {self.align_bytes} 字节对齐")


@dataclass
class PIMTensorSpec:
    device: Literal["host", "dpu"]
    placement: Placement
    residency: Literal["transient", "pinned"]
    pinned_dpu_id: int | None
    shard_map: dict[int, TensorShardDetail]
    reduce_type: str | None
    # 本轮新增，**必须在末尾**：graph/spec_prop.py 的三个构造点
    # （_host_spec / _dpu_spec / _weight_spec）都用位置参数，插在中间会静默错位。
    dtype: str = ""                        # 元素类型名，见 contracts/dtypes.py
    # 空串表示**尚未填充**（任何张量都有类型）；quant 为空表示**已确认未量化**。
    # 注意 dtype 是 FX 图这一层的数值类型（DQ 载体是 alias，所以继承源的类型），
    # 不是 GML 缓冲落盘类型 —— 后者由 gml_bridge 的 _stamp_dtypes 按 GML 语义补。
    quant: QuantLayout | None = None       # 量化布局；None = 未量化

    def validate(self) -> None:
        self.placement.validate()
        if self.reduce_type != self.placement.reduce_type:
            raise ValueError("spec reduce_type must match placement reduce_type")
        self._validate_dtype()
        if self.device == "host":
            if self.shard_map:
                raise ValueError("host spec must have an empty shard_map")
            return
        if not self.shard_map:
            raise ValueError("dpu spec must have a non-empty shard_map")
        for dpu_id, detail in self.shard_map.items():
            if dpu_id != detail.dpu_id:
                raise ValueError("shard_map key must match TensorShardDetail.dpu_id")
            detail.validate()
            if self.placement.kind == "Shard":
                if detail.shard_dim != self.placement.dim:
                    raise ValueError("Shard detail shard_dim must match placement dim")
            elif detail.shard_dim != -1:
                raise ValueError("Replicate and Partial details must use shard_dim=-1")

    def _validate_dtype(self) -> None:
        """数据类型维度：类型名合法，且与量化布局相容。"""
        if self.dtype:                      # 空串 = 尚未填充，不校验
            validate_dtype(self.dtype)
        if self.quant is None:
            return
        if not self.dtype:
            raise ValueError("给了 quant 布局却没有 dtype，两者必须同时填")
        # 量化布局只对定点类型有意义：fp16 权重不带 scale/zp（实测）
        if self.dtype.startswith("float"):
            raise ValueError(
                f"浮点类型 {self.dtype} 不应带量化布局 {self.quant.granularity!r}")
        if self.quant.granularity == "per_group" and self.quant.group_size <= 0:
            raise ValueError(
                f"per_group 量化的 group_size 必须为正，got {self.quant.group_size}")


@dataclass(frozen=True)
class RedistributeEdge:
    """记录生产节点和消费节点之间的一次张量重分布。"""

    edge_id: int                    # 全图唯一编号。
    src: str                        # 生产节点名称。
    dst: str                        # 消费节点名称。
    from_placement: Placement       # 生产节点的张量布局。
    to_placement: Placement         # 消费节点要求的张量布局。
    src_spec: PIMTensorSpec         # 生产节点的张量规格。
    dst_spec: PIMTensorSpec         # 消费节点的张量规格。
    type: Literal["all_reduce", "all_gather", "all_to_all", "scatter", "local_slice"]
    src_loc: dict                   # 源位置及参与的 DPU。
    dst_loc: dict                   # 目标位置及参与的 DPU。
    nbytes: int                     # 全局张量字节数。
    reduce_type: str | None = None  # `all_reduce` 的规约类型。
    shape: tuple[int, ...] = ()     # 全局张量形状。
    dtype: str = ""                 # PyTorch 数据类型名称。
