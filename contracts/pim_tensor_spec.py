from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class Placement:
    kind: Literal["Shard", "Replicate", "Partial"]
    dim: int | None = None
    reduce_type: str | None = None


@dataclass(frozen=True)
class TensorShardDetail:
    dpu_id: int
    shard_dim: int
    start_idx: int
    end_idx: int
    local_shape: tuple[int, ...]
    mram_offset: int = 0


@dataclass
class PIMTensorSpec:
    device: Literal["host", "dpu"]
    placement: Placement
    residency: Literal["transient", "pinned"]
    pinned_dpu_id: int | None
    shard_map: dict[int, TensorShardDetail]
    reduce_type: str | None
