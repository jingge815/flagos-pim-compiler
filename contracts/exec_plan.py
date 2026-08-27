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
    # 仅 op=="launch" 有意义：这条 launch 命令在 DPU 内部按几个 tasklet
    # 顺序模拟执行（backend/hal_numpy.py 的确定性顺序模拟 + hazard 检测，
    # 见该模块 HazardTracker 的说明）。默认 4，不是 1——多 tasklet 是本次
    # 要验证的主路径，不是需要显式开启的旁支。
    num_tasklets: int = 4


@dataclass
class ExecutionPlan:
    commands: list[Command]
