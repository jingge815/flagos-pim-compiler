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


def flatten_shard_dim(shard_dim: int, rank: int) -> int:
    """把图张量的切分维号换算到 `flatten_leading_dims` 压平后的坐标系。

    A 路的 kernel 张量是二维的（x 是 (M, K)、out 是 (M, N)），而切分决策的维号
    取自**图张量**：llama tp2 的 linear 输出是三维、`shard_dim` 是 2。压平规则是
    「末维留下、前导维合并」，所以末维换算成 1、其余换算成 0。

    不换算就会把一个秩 2 张量没有的维号写进去：越界的那一位被 FlagTree 的 builder
    静默跳过，`dpusPerDevice` 恒为全 1 —— 决策整条丢掉而无任何诊断。
    """
    if rank < 2:
        raise ValueError(f"压平后的秩至少是 2，got rank={rank}")
    if not 0 <= shard_dim < rank:
        raise ValueError(f"切分维 {shard_dim} 超出秩 {rank}")
    return 1 if shard_dim == rank - 1 else 0


# Placement 的三档与 partial 的归约方式。名字与统一 IR 的 `Placement.kind`
# （`contracts/pim_tensor_spec.py`）以及 FlagTree 的 `PlacementKind` /
# `PartialReduce` 枚举逐字对应 —— 三处是一条契约，改名要一起改。
PLACEMENT_KINDS = frozenset({"shard", "replicate", "partial"})
PARTIAL_REDUCES = frozenset({"sum", "mean"})


@dataclass(frozen=True)
class DpuShard:
    """一块张量的跨 DPU 放置决策（Placement 维的下发载体）。

    由 `runtime/exec_plan_gen` 从统一 IR 的 `spec.placement` 与 `shard_map`
    算出，随请求下发，落成 pimir 的 `#pim.placement` 与
    `#pim.tasklet_tiled.dpusPerDevice`。

    三档与统一 IR 的 `Placement.kind` 一一对应，不另立一套词：

    - `shard`：按 `dim` 这一维切开，每台 DPU 持有其中一片
    - `replicate`：每台 DPU 持有完整形状（`dim = -1`）
    - `partial`：每台 DPU 持有一份**全形状的局部和**，要按 `reduce` 跨 DPU
      归约才是完整值（`dim = -1`，`reduce` 必须给）

    早先这个结构只能表达 `shard` 一档，`replicate` / `partial` 在下发侧被整条
    丢掉（`placement_attribute` 对它们返回空元组），于是 PIMMLIR 侧分不清
    「复制」与「单 DPU」—— 而这两者的归约与容量口径并不相同。
    """

    dim: int              # 切分维；-1 表示不按维切（replicate / partial）
    num_dpus: int         # 参与这块张量的 DPU 数
    kind: str = "shard"   # shard / replicate / partial
    reduce: str | None = None   # 仅 partial：sum / mean

    def __post_init__(self) -> None:
        if self.kind not in PLACEMENT_KINDS:
            raise ValueError(
                f"未知的 placement 档位 {self.kind!r}，可选：{sorted(PLACEMENT_KINDS)}")
        if self.kind == "shard":
            if self.dim < 0:
                raise ValueError(
                    f"shard 要指明切哪一维，got dim={self.dim}")
            if self.reduce is not None:
                raise ValueError("shard 不欠归约，reduce 必须为 None")
        else:
            # replicate / partial 都是「每台持有完整形状」，没有被切开的轴。
            if self.dim >= 0:
                raise ValueError(
                    f"{self.kind} 不按某一维切开，dim 必须为 -1，got {self.dim}")
        if self.kind == "partial":
            if self.reduce not in PARTIAL_REDUCES:
                raise ValueError(
                    f"partial 的全部内容就是「怎么归约」，reduce 必须是 "
                    f"{sorted(PARTIAL_REDUCES)} 之一，got {self.reduce!r}")
        elif self.kind == "replicate" and self.reduce is not None:
            raise ValueError("replicate 不欠归约，reduce 必须为 None")

    @property
    def splits(self) -> bool:
        """这块张量是否真的被切开（而非复制或局部和）。"""
        return self.kind == "shard" and self.num_dpus > 1

    def to_payload(self) -> list:
        """命令 payload 里的形态。

        旧形态是 `[dim, num_dpus]` 两元素列表，新增两项**追加在尾部**，
        读取侧按长度兼容 —— 蓝图是编译期产物、可能与运行时版本不同步。
        """
        return [self.dim, self.num_dpus, self.kind, self.reduce]

    @classmethod
    def from_payload(cls, raw) -> "DpuShard":
        """从 payload 还原。两元素的旧形态按 `shard` 读。"""
        dim, num_dpus = int(raw[0]), int(raw[1])
        kind = str(raw[2]) if len(raw) > 2 else "shard"
        reduce = raw[3] if len(raw) > 3 else None
        return cls(dim=dim, num_dpus=num_dpus, kind=kind,
                   reduce=None if reduce is None else str(reduce))


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
    # 跨 DPU 切分决策。None = 单 DPU（不下发 `dpusPerDevice`，文本与改动前相同）。
    shard: DpuShard | None = None
    # 结果分片的排布（`TensorShardDetail.elem_strides`，元素数）。空元组 =
    # 没有排布信息，按行主序紧密处理。它决定布局编码的 `order`：哪一维步幅
    # 最小，哪一维就是最内层。
    elem_strides: tuple[int, ...] = ()
    # 结果分片在 MRAM 的起始字节与额外对齐（`TensorShardDetail` 的同名字段）。
    # 0 表示从基址起、没有额外对齐，下发文本与改动前逐字节相同。
    mram_offset: int = 0
    align_bytes: int = 0


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
