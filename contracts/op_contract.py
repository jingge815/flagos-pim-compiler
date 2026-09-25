"""图层编译器向算子编译器传递本地算子形状和硬件参数。"""

from __future__ import annotations

from dataclasses import dataclass, field
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
    # 复用字段，含义随 `op` 变：dynamic_quant / matmul 的组宽、kv_cache 的
    # 是否散写、split_heads 的头数、concat 的拼接轴。A 路 linear 不看它。
    # concat 必须显式给，省略不再默认 0。
    group_size: int | None = None
    # 折进主算子的尾部激活（silu / relu / gelu）。None 表示不折。
    activation: str | None = None
    # RoPE 末相卡值：K 路写 cache 前重定标是 3，Q 路是 0。缺了展开 pass
    # 把末相当成 0，K 路的 dq_contraction 一起消失。
    tail_card_value: int = 0
    # 权值次正规保护倍数，必须是 2 的幂。1 表示不补偿。
    sf_multiplier: int = 1
    # `eltwise` 的运算种类（`add` / `mul` / `sub`）。门控乘与 RoPE 乘都是
    # `mul`，与残差加的 `add` 走同一个 mnemonic——没有这一位，降级侧只能
    # 猜一个，而猜错的后果是数值全错、形状与接口都对。
    kind: str | None = None
    # `convert` 的目标元素类型（`float16` / `float32` / `int8`）。`dtype` 是
    # **输入**的存储类型，换类型这件事只有这一位说得出来；原来把它写死在
    # 降级侧，于是 f16→f32 也会被编成 f16→i8。
    out_dtype: str | None = None


@dataclass(frozen=True)
class OpCompileResult:
    so_path: str
    symbol: str
    argtypes: list[str]
    # 与 `argtypes` 一一对应：这一位是不是**按值**传的标量。
    # `pim.kv_cache` 的步计数器是唯一一例（其余都是裸指针）。空列表表示
    # 「全是指针」，即旧产物。少了这一位，`load_kernel` 只能把标量也当成
    # 指针塞进去，C 侧读到的就是那个地址的低 32 位——一个数当指针用。
    by_value: list[bool] = field(default_factory=list)
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
