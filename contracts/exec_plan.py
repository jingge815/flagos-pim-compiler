from dataclasses import dataclass, field
from typing import Literal


@dataclass(frozen=True)
class Access:
    loc: tuple[str, int | None]
    offset: int
    length: int


@dataclass
class Command:
    id: int
    op: Literal[
        "launch",
        "dma_in",
        "dma_out",
        "host_reduce",
        "host_concat",
        "host_permute",
        "host_slice",
        "host_op",
    ]
    dpu_id: int | None
    payload: dict[str, object]
    reads: list[Access] = field(default_factory=list)
    writes: list[Access] = field(default_factory=list)
    waits: list[int] = field(default_factory=list)
    # `launch` 命令在单台 DPU 内使用的 tasklet 数。
    num_tasklets: int = 4


@dataclass
class ExecutionPlan:
    commands: list[Command]
