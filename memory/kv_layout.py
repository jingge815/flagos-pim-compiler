"""问题 7：KV cache 管理——编译期布局 + 运行时本地追加/读取 + mask（方案问题 7）。

编译期（图层编译器一侧）：`KVRegionSpec` 记录单台 DPU 持有哪些层、哪些 KV head；
`build_kv_layout` 按 max_seq 把 KV 区切成定长 [max_seq, head_dim] 子块、算死每块
MRAM offset，产出含对齐 padding 的真实占用 `kv_allocated_bytes` 喂问题 8（方案三.②）。
`kv_bytes` 只是未对齐下界估算，不作分配依据。`kv_specs_from_placement` 从问题 2 的
k_proj 切分结果推出每台 DPU 驻留的 (layer, kv_head)——KV 按 head 切是硬约束（三.①）。

运行时（编排器一侧，问题 6 调用）：`PIMStaticKVCache` 无内部序列指针，读写位置由
编排器 `DecodeState.valid_len` 显式传入（唯一真值源，方案三.④）；追加与读取全部落在
本 DPU 本地 MRAM，KV 本地驻留、永不跨 DPU 搬。mask 只是数据、不改变张量形状，
不破坏"大小/位置/布局编译期定死"的静态性。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from contracts.exec_plan import Access
from contracts.pim_tensor_spec import PIMTensorSpec

_NP_DTYPE = {2: np.dtype(np.float16), 4: np.dtype(np.float32)}


def align_up(n: int, align: int) -> int:
    """向上对齐到 DMA 对齐边界（方案问题 8 三 的同名 helper，问题 7/8 共用）。"""
    if align <= 0:
        raise ValueError(f"align 必须为正，got {align}")
    return (n + align - 1) // align * align


@dataclass
class KVRegionSpec:
    """单台 DPU 的 KV 区规格（方案问题 7 三 的代码骨架）。

    kv_off / kv_allocated_bytes 是 `build_kv_layout` 填的输出字段：
    kv_off 为 (layer, kv_head, 'k'|'v') -> 绝对 MRAM offset；kv_allocated_bytes 为
    含对齐 padding 的真实占用，问题 8 排 arena 必须用它推进激活区起点，不得用
    kv_bytes 公式重算（否则 padding 累积会让 KV 写入侵入激活区）。
    """

    dpu_id: int
    layers: list[int]         # 本 DPU 持有哪些层的 KV（第 1 阶段不按层切，为全部层）
    kv_heads: list[int]       # 本 DPU 持有哪些 KV head（GQA 按 KV head，不是 Q head）
    q_heads_by_kv: dict[int, list[int]]  # kv_head -> 它服务的 q_head 列表；MHA 退化为 {h: [h]}
    max_seq: int              # 编译期定长，KV 区按它预留
    head_dim: int
    dtype_bytes: int
    kv_base: int              # KV 区基址，来自问题 8 的 plan.kv_base
    kv_off: dict[tuple[int, int, str], int] = field(default_factory=dict)
    kv_allocated_bytes: int = 0

    def validate(self) -> None:
        if self.dpu_id < 0:
            raise ValueError("dpu_id must be non-negative")
        if not self.layers or not self.kv_heads:
            raise ValueError("layers / kv_heads 不能为空")
        if self.max_seq <= 0 or self.head_dim <= 0:
            raise ValueError("max_seq / head_dim 必须为正")
        if self.dtype_bytes not in _NP_DTYPE:
            raise ValueError(f"unsupported dtype_bytes {self.dtype_bytes}")
        if self.kv_base < 0:
            raise ValueError("kv_base must be non-negative")
        missing = [h for h in self.kv_heads if h not in self.q_heads_by_kv]
        if missing:
            raise ValueError(f"q_heads_by_kv 缺 kv head 映射: {missing}")


def kv_bytes(spec: KVRegionSpec) -> int:
    """KV 区未对齐下界估算（2 = K/V 两份），仅供容量粗估，【不是分配依据】。

    真实分配量由 build_kv_layout 的 kv_allocated_bytes 给出（方案问题 7 三.②）。
    """
    return (2 * len(spec.layers) * spec.max_seq
            * len(spec.kv_heads) * spec.head_dim * spec.dtype_bytes)


def build_kv_layout(spec: KVRegionSpec, align: int) -> KVRegionSpec:
    """把 KV 区切成 [max_seq, head_dim] 定长子块，编译期算死每块 offset（方案问题 7 三.②）。

    入: spec（含 kv_base）、align（DMA 对齐，来自硬件）。
    出: 同一个 spec，kv_off 填好 (layer, head, k|v) -> 绝对 MRAM offset，
        kv_allocated_bytes 填好含全部对齐 padding 的真实占用。循环顺序固定，结果可复现；
        重复调用先清空 kv_off 重算，因此改 layers/kv_heads/kv_base 后重建即可。
    """
    spec.validate()
    spec.kv_off = {}
    off = spec.kv_base
    block = spec.max_seq * spec.head_dim * spec.dtype_bytes  # 一个 (layer, head, k|v) 子块
    for layer in spec.layers:
        for head in spec.kv_heads:
            for which in ("k", "v"):
                spec.kv_off[(layer, head, which)] = off
                off += align_up(block, align)
    spec.kv_allocated_bytes = off - spec.kv_base
    return spec


def kv_specs_from_placement(
    k_proj_spec: PIMTensorSpec,
    *,
    num_layers: int,
    num_kv_heads: int,
    num_q_heads: int,
    head_dim: int,
    max_seq: int,
    dtype_bytes: int,
    kv_base: int,
) -> dict[int, KVRegionSpec]:
    """从问题 2 的 k_proj 切分结果推出每台 DPU 的 KVRegionSpec——KV 跟着 head 切（方案问题 7 三.①）。

    入: k_proj_spec —— 任一层 self_attn.k_proj.weight 的 PIMTensorSpec（第 1 阶段
        Megatron 列切 Shard(0) 且各层切法一致）；num_* / head_dim 取自模型 config；
        kv_base 来自问题 8 的 plan.kv_base（规划前可先填 0 做容量粗估）。
    出: dpu_id -> KVRegionSpec。每个 DPU 的 kv_heads 由其 k_proj 分片行区间
        [start_idx, end_idx) 按 head_dim 换算；q_heads_by_kv 按 GQA 通用写法生成
        （一个 KV head 服务连续 group 个 Q head），MHA 时 group=1 退化为恒等映射。
        切点未对齐 head 边界（KV 本地驻留的前提被破坏）直接抛错。
    """
    if k_proj_spec.placement.kind != "Shard" or k_proj_spec.placement.dim != 0:
        raise ValueError("KV 按 head 切要求 k_proj 为列切 Shard(0)，got "
                         f"{k_proj_spec.placement}")
    if num_q_heads % num_kv_heads != 0:
        raise ValueError(f"num_q_heads={num_q_heads} 不能整除 num_kv_heads={num_kv_heads}")
    group = num_q_heads // num_kv_heads  # GQA：一个 KV head 服务的 Q head 数
    specs: dict[int, KVRegionSpec] = {}
    for dpu_id, det in sorted(k_proj_spec.shard_map.items()):
        if det.start_idx % head_dim != 0 or det.end_idx % head_dim != 0:
            raise ValueError(
                f"dpu{dpu_id} k_proj 切点 [{det.start_idx},{det.end_idx}) 未对齐 head_dim={head_dim}")
        heads = list(range(det.start_idx // head_dim, det.end_idx // head_dim))
        specs[dpu_id] = KVRegionSpec(
            dpu_id=dpu_id,
            layers=list(range(num_layers)),  # 第 1 阶段不按层切，各层 KV 都驻留本 DPU 的 head
            kv_heads=heads,
            q_heads_by_kv={h: list(range(h * group, (h + 1) * group)) for h in heads},
            max_seq=max_seq,
            head_dim=head_dim,
            dtype_bytes=dtype_bytes,
            kv_base=kv_base,
        )
    return specs


def kv_access(
    spec: KVRegionSpec, layer: int, head: int, which: Literal["k", "v"], pos_start: int, pos_end: int
) -> Access:
    """单个 (layer, head, k|v) 子块内 [pos_start, pos_end) 行的字节区间（绝对 MRAM 地址）。

    供问题 6 生成 ExecutionPlan 时填 Command.reads / writes：K/V 投影 kernel 的 writes
    与注意力 QK^T / weights@V kernel 的 reads 都取本函数算的区间，保证与 update /
    read_tile 实际访问的地址完全一致（同一处 offset 数学，不各算各的）。
    """
    if not (0 <= pos_start < pos_end <= spec.max_seq):
        raise ValueError(f"区间 [{pos_start},{pos_end}) 越界 [0,{spec.max_seq})")
    row = spec.head_dim * spec.dtype_bytes
    return Access(
        loc=("dpu", spec.dpu_id),
        offset=spec.kv_off[(layer, head, which)] + pos_start * row,
        length=(pos_end - pos_start) * row,
    )


class PIMStaticKVCache:
    """面向存算一体的静态 KV cache：定长预留、跨 step 驻留、本地读写、永不跨 DPU 搬。

    接口形态借 HF StaticCache。【无内部序列指针】：位置唯一真值源是编排器
    DecodeState.valid_len（问题 6），每次读写由调用方显式传 pos，本类不自持指针、
    不做推进（方案问题 7 三.④：避免多处真值源不同步导致 mask 位置与写入位置错位）。
    本类也不持有数据——数据在各 DPU 的 MRAM 里，删掉本对象重建一个，读回的值不变。
    """

    def __init__(self, backend, specs: dict[int, KVRegionSpec], wram_budget_bytes: int) -> None:
        # backend: HAL（NumpyBackend 或厂商 SDK）；wram_budget_bytes: 单 tile 的 WRAM
        # 占用上界，真机上由问题 5 的桥按 WRAM 容量定，此处作 read_tile 的校验预算。
        self.backend = backend
        self.specs = specs
        self.wram_budget_bytes = wram_budget_bytes
        for spec in specs.values():
            spec.validate()
            if not spec.kv_off:
                raise ValueError(f"dpu{spec.dpu_id} 的 KVRegionSpec 未过 build_kv_layout")

    def update(self, layer: int, pos: int, k_by_dpu: dict, v_by_dpu: dict) -> None:
        """把本步各 DPU 本地算出的新 K/V 写到位置 pos（真机在 K/V 投影 kernel 内做）。

        入: layer；pos —— 写入格（= 调用方 DecodeState.valid_len，唯一真值源）；
            k_by_dpu / v_by_dpu —— {dpu_id: {head: [head_dim] 向量}}，本步新算的 K/V。
        出: 无（原地写各 DPU 本地 MRAM；空间编译期已定长预留，这里不涉及分配）。
        """
        for dpu_id, spec in self.specs.items():
            assert 0 <= pos < spec.max_seq, f"pos={pos} 越界 [0,{spec.max_seq})"  # 写前校验
            row = spec.head_dim * spec.dtype_bytes
            for head in spec.kv_heads:
                for which, by_dpu in (("k", k_by_dpu), ("v", v_by_dpu)):
                    vec = np.asarray(by_dpu[dpu_id][head])
                    if vec.size != spec.head_dim or vec.dtype.itemsize != spec.dtype_bytes:
                        raise ValueError(
                            f"dpu{dpu_id} layer{layer} head{head} {which}: 期望 "
                            f"[{spec.head_dim}]x{spec.dtype_bytes}B，got shape={vec.shape} dtype={vec.dtype}")
                    off = spec.kv_off[(layer, head, which)] + pos * row
                    self.backend.write_local(dpu_id, off, vec)  # 本地写，零跨 DPU

    def read_tile(
        self, layer: int, dpu_id: int, head: int, tile_start: int, tile_end: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """按 key token 分块读本 DPU 某层某 head 的 K/V tile（不满块搬入 WRAM）。

        整块 [max_seq, head_dim] 远超单 DPU WRAM，由 QK^T / weights@V 两个 GEMV kernel
        逐 tile 读入处理；无效尾部（pos >= valid_len）由 host softmax 前的 mask 盖掉。
        入: layer/dpu_id/head；[tile_start, tile_end) —— 本次读取的 key token 区间。
        出: (K_tile, V_tile)，各为 [tile_end-tile_start, head_dim]。
        """
        spec = self.specs[dpu_id]
        if head not in spec.kv_heads:
            raise KeyError(f"head {head} 不在 dpu{dpu_id} 的 kv_heads {spec.kv_heads}")
        if not (0 <= tile_start < tile_end <= spec.max_seq):
            raise ValueError(f"tile [{tile_start},{tile_end}) 越界 [0,{spec.max_seq})")
        row = spec.head_dim * spec.dtype_bytes
        if (tile_end - tile_start) * row > self.wram_budget_bytes:
            raise ValueError("单 tile 超 WRAM 预算，调小 tile（尺寸由问题 5 的桥按 WRAM 容量定）")
        koff = spec.kv_off[(layer, head, "k")] + tile_start * row
        voff = spec.kv_off[(layer, head, "v")] + tile_start * row
        shape = (tile_end - tile_start, spec.head_dim)
        dtype = _NP_DTYPE[spec.dtype_bytes]
        return (self.backend.read_local(dpu_id, koff, shape, dtype),
                self.backend.read_local(dpu_id, voff, shape, dtype))


def prefill_mask(prompt_len: int, max_seq: int) -> np.ndarray:
    """prefill 用 mask：scores 是 [max_seq, max_seq] 方阵（满预留 + 满算）。

    可见当且仅当 不看未来（因果 j <= i）且 是真实 token（j < prompt_len，非预留位）；
    可见处为 0、其余为 -inf，softmax 在 host 做，mask 随 scores 一起送 host（方案问题 7 三）。
    """
    i = np.arange(max_seq)[:, None]   # query 位置
    j = np.arange(max_seq)[None, :]   # key 位置
    visible = (j <= i) & (j < prompt_len)
    return np.where(visible, 0.0, -np.inf).astype(np.float32)


def decode_mask(valid_len: int, max_seq: int) -> np.ndarray:
    """decode 用 mask：只有一个新 token 在 valid_len 处，scores 是 [max_seq] 向量。

    [0, valid_len] 为 0（含本步刚写入的位置）、其余预留位为 -inf；只有一个 query
    位置时因果自动满足。valid_len = DecodeState.valid_len（唯一真值源）。
    """
    j = np.arange(max_seq)
    return np.where(j <= valid_len, 0.0, -np.inf).astype(np.float32)


def format_kv_layout(specs: dict[int, KVRegionSpec]) -> str:
    """把各 DPU 的 KV 布局打印成可读文本，便于核对 offset 与容量（对齐 format_comm_plan 惯例）。"""
    lines = []
    for dpu_id, spec in sorted(specs.items()):
        heads = spec.kv_heads
        lines.append(
            f"dpu{dpu_id}: layers={len(spec.layers)} kv_heads=[{heads[0]}..{heads[-1]}] "
            f"max_seq={spec.max_seq} head_dim={spec.head_dim} kv_base={spec.kv_base} "
            f"kv_allocated_bytes={spec.kv_allocated_bytes} (kv_bytes 下界={kv_bytes(spec)}) "
            f"blocks={len(spec.kv_off)}"
        )
    return "KV layout:\n" + "\n".join(lines)
