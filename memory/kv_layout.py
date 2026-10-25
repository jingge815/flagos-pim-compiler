"""规划静态 KV 缓存布局，并提供本地读写和注意力掩码。"""

from __future__ import annotations

from dataclasses import dataclass, field
from numbers import Integral

import numpy as np

from contracts.mem_layout import align_up
from contracts.pim_tensor_spec import PIMTensorSpec

_NP_DTYPE = {2: np.dtype(np.float16), 4: np.dtype(np.float32)}


def _check_position(name: str, value: int, upper: int, *, inclusive: bool) -> None:
    """序列位置必须是整数，避免小数改变 mask 中可见 token 的集合。"""
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name}={value!r} 必须是整数")
    in_range = 0 <= value <= upper if inclusive else 0 <= value < upper
    if not in_range:
        bound = f"[0,{upper}]" if inclusive else f"[0,{upper})"
        raise ValueError(f"{name}={value} 越界 {bound}")


@dataclass
class KVRegionSpec:
    """描述单台 DPU 的层、KV head 和按固定大小槽组织的 KV 内存区域。"""

    dpu_id: int
    layers: list[int]         # 本 DPU 持有 KV 缓存的层号。
    kv_heads: list[int]       # 本 DPU 持有的 KV head 编号。
    q_heads_by_kv: dict[int, list[int]]  # 每个 KV head 对应的 Q head 编号。
    max_seq: int              # 单个请求的最大序列长度。
    head_dim: int
    dtype_bytes: int
    kv_base: int              # KV 区起始地址（页表区起点）。
    slot_tokens: int = 16     # 一个槽容纳的 token 数。
    kv_off: dict[tuple[int, int, str], int] = field(default_factory=dict)
    kv_allocated_bytes: int = 0
    blocks_per_request: int = 0   # 每个请求槽的页表项数。
    pool_slots: int = 0           # 物理槽总数。
    page_table_base: int = 0      # 页表区起始地址。
    page_table_stride: int = 0    # 每个请求槽的页表占用字节数。
    slots_base: int = 0           # 物理槽池起始地址。
    slot_bytes: int = 0           # 单个物理槽的字节数。

    def validate(self) -> None:
        if self.dpu_id < 0:
            raise ValueError("dpu_id must be non-negative")
        if not self.layers or not self.kv_heads:
            raise ValueError("layers / kv_heads 不能为空")
        if self.max_seq <= 0 or self.head_dim <= 0 or self.slot_tokens <= 0:
            raise ValueError("max_seq / head_dim / slot_tokens 必须为正")
        if self.dtype_bytes not in _NP_DTYPE:
            raise ValueError(f"unsupported dtype_bytes {self.dtype_bytes}")
        if self.kv_base < 0:
            raise ValueError("kv_base must be non-negative")
        missing = [h for h in self.kv_heads if h not in self.q_heads_by_kv]
        if missing:
            raise ValueError(f"q_heads_by_kv 缺 kv head 映射: {missing}")


def kv_bytes(spec: KVRegionSpec) -> int:
    """返回一个请求在未对齐前需要的 KV 字节数（逻辑下界）。"""
    return (2 * len(spec.layers) * spec.max_seq
            * len(spec.kv_heads) * spec.head_dim * spec.dtype_bytes)


def build_kv_layout(spec: KVRegionSpec, align: int, num_request_slots: int = 1) -> KVRegionSpec:
    """规划页表区和固定大小物理槽池，`kv_off` 存槽 0 内的绝对地址。

    每个物理槽存本 DPU 全部本地 KV 平面的 `slot_tokens` 个 token；槽内按
    `layer -> kv_head -> k/v` 顺序排列。页表按请求槽分区，池由所有请求共享。
    """
    spec.validate()
    if num_request_slots < 1:
        raise ValueError(f"num_request_slots 必须 >= 1，got {num_request_slots}")
    row = spec.head_dim * spec.dtype_bytes
    plane_stride = align_up(spec.slot_tokens * row, align)
    spec.slot_bytes = 2 * len(spec.layers) * len(spec.kv_heads) * plane_stride
    spec.blocks_per_request = -(-spec.max_seq // spec.slot_tokens)
    spec.pool_slots = num_request_slots * spec.blocks_per_request
    spec.page_table_base = spec.kv_base
    spec.page_table_stride = align_up(spec.blocks_per_request * 4, align)
    spec.slots_base = spec.kv_base + num_request_slots * spec.page_table_stride

    spec.kv_off = {}
    off = spec.slots_base
    for layer in spec.layers:
        for head in spec.kv_heads:
            for which in ("k", "v"):
                spec.kv_off[(layer, head, which)] = off
                off += plane_stride
    spec.kv_allocated_bytes = (
        num_request_slots * spec.page_table_stride + spec.pool_slots * spec.slot_bytes)
    return spec


def write_page_table(spec: KVRegionSpec, hal, request_slot: int,
                     block_table: list[int]) -> None:
    """把逻辑块到物理槽的映射写成 int32 页表，不足的项补 -1。"""
    if not spec.page_table_stride:
        raise ValueError(f"dpu{spec.dpu_id} 的 KVRegionSpec 未过 build_kv_layout")
    num_request_slots = spec.pool_slots // spec.blocks_per_request
    if not 0 <= request_slot < num_request_slots:
        raise ValueError(
            f"request_slot={request_slot} 越界 [0,{num_request_slots})")
    if len(block_table) > spec.blocks_per_request:
        raise ValueError(
            f"块表长 {len(block_table)} 超过每请求页表项数 {spec.blocks_per_request}")
    entries = np.full((spec.blocks_per_request,), -1, dtype=np.int32)
    entries[:len(block_table)] = block_table
    offset = spec.page_table_base + request_slot * spec.page_table_stride
    hal.write_local(spec.dpu_id, offset, entries)


def kv_specs_from_placement(
    k_proj_spec: PIMTensorSpec,
    *,
    layers: list[int],
    num_kv_heads: int,
    num_q_heads: int,
    head_dim: int,
    max_seq: int,
    dtype_bytes: int,
    kv_base: int,
    slot_tokens: int = 16,
) -> dict[int, KVRegionSpec]:
    """从 `k_proj` 分片构造每台 DPU 的 KV 缓存规格。"""
    if k_proj_spec.placement.kind != "Shard" or k_proj_spec.placement.dim != 0:
        raise ValueError("KV 按 head 切要求 k_proj 为列切 Shard(0)，got "
                         f"{k_proj_spec.placement}")
    if num_q_heads % num_kv_heads != 0:
        raise ValueError(f"num_q_heads={num_q_heads} 不能整除 num_kv_heads={num_kv_heads}")
    group = num_q_heads // num_kv_heads  # 每个 KV head 服务的 Q head 数。
    specs: dict[int, KVRegionSpec] = {}
    for dpu_id, det in sorted(k_proj_spec.shard_map.items()):
        if det.start_idx % head_dim != 0 or det.end_idx % head_dim != 0:
            raise ValueError(
                f"dpu{dpu_id} k_proj 切点 [{det.start_idx},{det.end_idx}) 未对齐 head_dim={head_dim}")
        heads = list(range(det.start_idx // head_dim, det.end_idx // head_dim))
        specs[dpu_id] = KVRegionSpec(
            dpu_id=dpu_id,
            layers=list(layers),
            kv_heads=heads,
            q_heads_by_kv={h: list(range(h * group, (h + 1) * group)) for h in heads},
            max_seq=max_seq,
            head_dim=head_dim,
            dtype_bytes=dtype_bytes,
            kv_base=kv_base,
            slot_tokens=slot_tokens,
        )
    assigned_heads = [head for spec in specs.values() for head in spec.kv_heads]
    expected_heads = set(range(num_kv_heads))
    if len(assigned_heads) != len(set(assigned_heads)) or set(assigned_heads) != expected_heads:
        raise ValueError(
            "k_proj 分片必须恰好覆盖每个 KV head 一次，got "
            f"{sorted(assigned_heads)}，expected {sorted(expected_heads)}"
        )
    return specs


def kv_specs_from_strategy(
    gm,
    strategy,
    *,
    num_layers: int,
    num_kv_heads: int,
    num_q_heads: int,
    head_dim: int,
    max_seq: int,
    dtype_bytes: int,
    kv_base: int = 0,
    slot_tokens: int = 16,
) -> dict[int, KVRegionSpec]:
    """按切分策略为每台 DPU 生成层和 KV head 的缓存规格。"""
    specs: dict[int, KVRegionSpec] = {}
    for stage in range(strategy.num_stages):
        layers = strategy.layers_of_stage(stage, num_layers)
        stage_specs = [_k_proj_spec_of_layer(gm, layer) for layer in layers]
        first = stage_specs[0]
        for layer, spec in zip(layers[1:], stage_specs[1:]):
            if set(spec.shard_map) != set(first.shard_map) or spec.placement != first.placement:
                raise ValueError(
                    f"stage{stage} 内 layer{layer} 的 k_proj 切法与 layer{layers[0]} 不一致："
                    f"{sorted(spec.shard_map)}@{spec.placement} vs "
                    f"{sorted(first.shard_map)}@{first.placement}"
                )
        specs.update(
            kv_specs_from_placement(
                first,
                layers=layers,
                num_kv_heads=num_kv_heads,
                num_q_heads=num_q_heads,
                head_dim=head_dim,
                max_seq=max_seq,
                dtype_bytes=dtype_bytes,
                kv_base=kv_base,
                slot_tokens=slot_tokens,
            )
        )
    missing = [dpu_id for dpu_id in strategy.dpu_ids if dpu_id not in specs]
    if missing:
        raise ValueError(f"策略里的 DPU {missing} 没有分到 KV 区（stage 划分与 k_proj 切分不一致）")
    return specs


def _k_proj_spec_of_layer(gm, layer: int) -> PIMTensorSpec:
    """取某一层 self_attn.k_proj.weight 的 PIMTensorSpec（KV 按 head 切的依据）。"""
    from contracts.graph_meta import SPEC_META_KEY

    pattern = f"layers.{layer}.self_attn.k_proj.weight"
    matches = [
        node for node in gm.graph.nodes
        if node.op == "get_attr" and pattern in str(node.target)
    ]
    if len(matches) != 1:
        raise ValueError(f"{pattern} 应恰好匹配 1 个 get_attr 节点，实际 {len(matches)} 个")
    return matches[0].meta[SPEC_META_KEY]


def prefill_mask(prompt_len: int, max_seq: int) -> np.ndarray:
    """返回 prefill 的因果掩码，屏蔽未来位置和预留位置。"""
    _check_position("prompt_len", prompt_len, max_seq, inclusive=True)
    i = np.arange(max_seq)[:, None]   # 查询位置。
    j = np.arange(max_seq)[None, :]   # 键位置。
    visible = (j <= i) & (j < prompt_len)
    return np.where(visible, 0.0, -np.inf).astype(np.float32)


def decode_mask(valid_len: int, max_seq: int) -> np.ndarray:
    """返回 decode 的掩码，保留已写入 token 和当前位置。"""
    _check_position("valid_len", valid_len, max_seq, inclusive=False)
    j = np.arange(max_seq)
    return np.where(j <= valid_len, 0.0, -np.inf).astype(np.float32)


def format_kv_layout(specs: dict[int, KVRegionSpec]) -> str:
    """把各 DPU 的槽几何打印成可读文本，便于核对槽池与页表占用。"""
    lines = []
    for dpu_id, spec in sorted(specs.items()):
        heads = spec.kv_heads
        num_request_slots = spec.pool_slots // spec.blocks_per_request
        lines.append(
            f"dpu{dpu_id}: layers={len(spec.layers)} kv_heads=[{heads[0]}..{heads[-1]}] "
            f"max_seq={spec.max_seq} head_dim={spec.head_dim} "
            f"slot_tokens={spec.slot_tokens} slot_bytes={spec.slot_bytes} "
            f"blocks_per_request={spec.blocks_per_request} "
            f"pool_slots={spec.pool_slots} "
            f"page_table_stride={spec.page_table_stride} "
            f"request_slots={num_request_slots} "
            f"kv_base={spec.kv_base} slots_base={spec.slots_base} "
            f"kv_allocated_bytes={spec.kv_allocated_bytes} "
            f"(kv_bytes 下界={kv_bytes(spec)})"
        )
    return "KV layout:\n" + "\n".join(lines)
