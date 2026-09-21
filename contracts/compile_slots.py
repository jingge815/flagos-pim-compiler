"""编译期槽位：prepare_out 的尺寸真源。

导出图用 `--seq-len 16` 只提供拓扑；txt / bin / L2 一律按 decode 槽位算
（llama2-7B：H=4096、I=11008、hd=128、S=1024、nh=32）。
依据：参考产物 IO_info 与 `docs/prepare_out-代码评审4-20260921.md` §2.1。
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CompileSlots:
    """decode block 的编译期几何。"""

    hidden: int = 4096
    intermediate: int = 11008
    head_dim: int = 128
    seq: int = 1024
    heads: int = 32

    @classmethod
    def from_config(cls, config, *, max_seq: int = 1024) -> "CompileSlots":
        hidden = int(config.hidden_size)
        heads = int(config.num_attention_heads)
        return cls(
            hidden=hidden,
            intermediate=int(config.intermediate_size),
            head_dim=hidden // heads,
            seq=int(max_seq),
            heads=heads,
        )

    @property
    def kv_cache_elems(self) -> int:
        return self.heads * self.seq * self.head_dim

    @property
    def kv_index_elems(self) -> int:
        return self.heads * 3

    @property
    def kv_new_elems(self) -> int:
        return self.heads * self.head_dim

    @property
    def bmm_weight_elems(self) -> int:
        return self.seq * self.head_dim


DEFAULT_SLOTS = CompileSlots()
