"""规划静态 KV 缓存布局，并提供本地读写和注意力掩码。"""

from __future__ import annotations

from dataclasses import dataclass, field
from numbers import Integral
from typing import Literal

import numpy as np

from contracts.exec_plan import Access
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


def align_up(n: int, align: int) -> int:
    """向上对齐到 DMA 边界。"""
    if align <= 0:
        raise ValueError(f"align 必须为正，got {align}")
    return (n + align - 1) // align * align


@dataclass
class KVRegionSpec:
    """描述单台 DPU 的层、KV head 和 KV 内存区域。"""

    dpu_id: int
    layers: list[int]         # 本 DPU 持有 KV 缓存的层号。
    kv_heads: list[int]       # 本 DPU 持有的 KV head 编号。
    q_heads_by_kv: dict[int, list[int]]  # 每个 KV head 对应的 Q head 编号。
    max_seq: int              # KV 缓存的最大序列长度。
    head_dim: int
    dtype_bytes: int
    kv_base: int              # KV 区起始地址。
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
    """返回未包含对齐填充的 KV 缓存字节数。"""
    return (2 * len(spec.layers) * spec.max_seq
            * len(spec.kv_heads) * spec.head_dim * spec.dtype_bytes)


def build_kv_layout(spec: KVRegionSpec, align: int) -> KVRegionSpec:
    """为每个层、KV head 和 K/V 张量分配对齐后的 MRAM 偏移。"""
    spec.validate()
    spec.kv_off = {}
    off = spec.kv_base
    block = spec.max_seq * spec.head_dim * spec.dtype_bytes  # 单个 K 或 V 子块。
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
    layers: list[int],
    num_kv_heads: int,
    num_q_heads: int,
    head_dim: int,
    max_seq: int,
    dtype_bytes: int,
    kv_base: int,
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


def kv_access(
    spec: KVRegionSpec, layer: int, head: int, which: Literal["k", "v"], pos_start: int, pos_end: int
) -> Access:
    """返回指定 K/V 子块中 token 区间的绝对 MRAM 地址。"""
    if not (0 <= pos_start < pos_end <= spec.max_seq):
        raise ValueError(f"区间 [{pos_start},{pos_end}) 越界 [0,{spec.max_seq})")
    row = spec.head_dim * spec.dtype_bytes
    return Access(
        loc=("dpu", spec.dpu_id),
        offset=spec.kv_off[(layer, head, which)] + pos_start * row,
        length=(pos_end - pos_start) * row,
    )


class PIMStaticKVCache:
    """管理预分配在 DPU 本地 MRAM 中的定长 KV 缓存。"""

    def __init__(self, backend, specs: dict[int, KVRegionSpec], wram_budget_bytes: int) -> None:
        # 保存后端、KV 规格和 WRAM 上限。
        self.backend = backend
        self.specs = specs
        self.wram_budget_bytes = wram_budget_bytes
        for spec in specs.values():
            spec.validate()
            if not spec.kv_off:
                raise ValueError(f"dpu{spec.dpu_id} 的 KVRegionSpec 未过 build_kv_layout")

    def update(self, layer: int, pos: int, k_by_dpu: dict, v_by_dpu: dict) -> None:
        """将指定位置的 K/V 向量写入持有该层的 DPU。"""
        for dpu_id, spec in self.specs.items():
            if layer not in spec.layers:
                continue
            if not 0 <= pos < spec.max_seq:
                raise ValueError(f"pos={pos} 越界 [0,{spec.max_seq})")
            row = spec.head_dim * spec.dtype_bytes
            for head in spec.kv_heads:
                for which, by_dpu in (("k", k_by_dpu), ("v", v_by_dpu)):
                    vec = np.asarray(by_dpu[dpu_id][head])
                    expected_dtype = _NP_DTYPE[spec.dtype_bytes]
                    if vec.shape != (spec.head_dim,) or vec.dtype != expected_dtype:
                        raise ValueError(
                            f"dpu{dpu_id} layer{layer} head{head} {which}: 期望 "
                            f"shape=({spec.head_dim},) dtype={expected_dtype}，got "
                            f"shape={vec.shape} dtype={vec.dtype}")
                    off = spec.kv_off[(layer, head, which)] + pos * row
                    self.backend.write_local(dpu_id, off, vec)  # 写入 DPU 本地 MRAM。

    def read_tile(
        self, layer: int, dpu_id: int, head: int, tile_start: int, tile_end: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """读取一个 KV head 的 token 区间，返回 K 和 V 瓦片。"""
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
