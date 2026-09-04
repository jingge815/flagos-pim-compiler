"""图层编译器向算子编译器传递本地算子形状和硬件参数。"""

from __future__ import annotations

from dataclasses import dataclass
from math import prod


@dataclass(frozen=True)
class PIMHardwareConfig:
    num_dpus: int
    num_tasklets: int
    mram_bytes_per_dpu: int
    wram_bytes_per_dpu: int
    dma_align: int

    def __post_init__(self) -> None:
        for name in (
            "num_dpus",
            "num_tasklets",
            "mram_bytes_per_dpu",
            "wram_bytes_per_dpu",
            "dma_align",
        ):
            value = getattr(self, name)
            if not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive int, got {value!r}")
        if self.num_dpus & (self.num_dpus - 1):
            raise ValueError(f"num_dpus must be a power of two, got {self.num_dpus}")
        if self.dma_align & (self.dma_align - 1):
            raise ValueError(f"dma_align must be a power of two, got {self.dma_align}")

    def to_payload(self) -> dict[str, int]:
        return {
            "num_dpus": self.num_dpus,
            "num_tasklets": self.num_tasklets,
            "mram_bytes_per_dpu": self.mram_bytes_per_dpu,
            "wram_bytes_per_dpu": self.wram_bytes_per_dpu,
            "dma_align": self.dma_align,
        }

    @classmethod
    def from_payload(cls, payload: object) -> "PIMHardwareConfig":
        if not isinstance(payload, dict):
            raise ValueError(f"hardware payload must be a dict, got {type(payload).__name__}")
        return cls(
            num_dpus=int(payload["num_dpus"]),
            num_tasklets=int(payload["num_tasklets"]),
            mram_bytes_per_dpu=int(payload["mram_bytes_per_dpu"]),
            wram_bytes_per_dpu=int(payload["wram_bytes_per_dpu"]),
            dma_align=int(payload["dma_align"]),
        )


# 默认 PIM 硬件配置。
DEFAULT_HARDWARE_CONFIG = PIMHardwareConfig(
    num_dpus=8,
    num_tasklets=16,
    mram_bytes_per_dpu=8 * 2**30,
    wram_bytes_per_dpu=65536,
    dma_align=8,
)


def flatten_leading_dims(shape: tuple[int, ...]) -> tuple[int, int]:
    """将末维保留为 K，其余维合并为 M，返回 `(M, K)`。"""
    if len(shape) < 2:
        raise ValueError(f"expected rank >= 2, got shape={shape!r}")
    *leading, k = shape
    return prod(leading), k


@dataclass(frozen=True)
class OpCompileRequest:
    op: str
    arg_shapes: list[tuple[int, ...]]
    hardware: PIMHardwareConfig
    # MRAM 数据类型，如 `float16` 或 `float32`。
    dtype: str = "float32"
    # 单台 DPU 使用的 tasklet 数。
    num_tasklets: int = 4


@dataclass(frozen=True)
class OpCompileResult:
    so_path: str
    symbol: str
    argtypes: list[str]
    # 算子编译产出的 pim mlir 文本。GeneSim 的代价模型靠它拿到真实分块和 DMA
    # 结构（`genesim_bridge.ir_cost.analyze_ir` 负责解析），而不是沿用
    # `conf/sim.yaml` 里拍下的 `tile_size` 常量。
    #
    # 命中编译缓存时为 None：`.so` 可以复用，pim mlir 不落盘。需要它的调用方
    # 用 `compile_op(request, force=True)` 强制重编，或读 `pimir_path`。
    pimir: str | None = None
    # pim mlir 的缓存路径（与 `.so` 同名、后缀 `.pimir.mlir`）。命中缓存时
    # `.so` 复用而 pim mlir 也在这里，直接读文件即可，不必重编。
    pimir_path: str | None = None
