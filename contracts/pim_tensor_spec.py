from dataclasses import dataclass
from typing import Literal


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

    def validate(self) -> None:
        if self.dpu_id < 0:
            raise ValueError("dpu_id must be non-negative")
        if self.start_idx < 0 or self.end_idx < 0 or self.end_idx < self.start_idx:
            raise ValueError("shard range must be non-negative and ordered")
        if self.mram_offset < 0:
            raise ValueError("mram_offset must be non-negative")
        if any(dim < 0 for dim in self.local_shape):
            raise ValueError("local_shape dimensions must be non-negative")


@dataclass
class PIMTensorSpec:
    device: Literal["host", "dpu"]
    placement: Placement
    residency: Literal["transient", "pinned"]
    pinned_dpu_id: int | None
    shard_map: dict[int, TensorShardDetail]
    reduce_type: str | None

    def validate(self) -> None:
        self.placement.validate()
        if self.reduce_type != self.placement.reduce_type:
            raise ValueError("spec reduce_type must match placement reduce_type")
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


@dataclass(frozen=True)
class RedistributeEdge:
    """一条布局不一致边上的一次逻辑重分布（方案问题 2 二.(9)）。

    打在消费方节点的 ``node.meta["redistribute"]`` 列表里；``DPU→host→DPU``
    两跳展开为 DMA 段是问题 3 的事，这里只记录类型、端点与总量。
    """

    edge_id: int                    # 全图唯一编号，问题 3 通信计划表的主键
    src: str                        # 上游（生产方）节点名
    dst: str                        # 下游（消费方）节点名
    from_placement: Placement       # 上游实际产出布局
    to_placement: Placement         # 下游要求布局
    src_spec: PIMTensorSpec         # 上游实际产出 tensor spec
    dst_spec: PIMTensorSpec         # 下游输入要求 materialize 后的 tensor spec
    type: Literal["all_reduce", "all_gather", "all_to_all", "scatter", "local_slice"]
    src_loc: dict                   # {"device": "dpu", "dpus": [...]} 或 {"device": "host"}
    dst_loc: dict                   # 同上
    nbytes: int                     # 逻辑张量总字节数；逐 segment 字节由问题 3 按 shard_map 展开
    reduce_type: str | None = None  # 仅 all_reduce 使用，取自 Partial 的规约类型
