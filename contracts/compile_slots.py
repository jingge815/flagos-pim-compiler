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

    def dq_layout(self, last_dim: int, *, is_attention_scores: bool,
                  group_size: int) -> tuple[int, int]:
        """DQ 一行（一个 token）的量化口径：(元素数, 组宽)。

        注意力分数**整行一组** —— 元素数与组宽都按 `seq` 算（导出图那一行
        只有 16 或 256 个元素，不能拿去当组宽）；其余按最后一维落到 MLP
        中间态或 hidden，组宽沿用图侧给的 128。

        `output_buffer_<self>.bin`（numel 个 int8）、`output_sf_<self>.bin`
        （numel/组宽 个 fp16）与 GML 里的形状字段、spc/spg 都取自这里 ——
        两处各写一份规则就会出现「声明 [1,16,512,128] 而盘上只有 32 个
        scale」这种 16 倍的错位。
        """
        if is_attention_scores:
            return self.seq, self.seq
        if last_dim == 0:
            raise ValueError(
                "dq_layout 的 last_dim 是 0：上游没给出被量化张量的末维，"
                "猜 hidden 会让 original_shape 与盘上 scale 对不上")
        if last_dim == self.intermediate:
            return self.intermediate, group_size
        if last_dim in (self.hidden, self.head_dim):
            # head_dim 出现在 K 路 RoPE-DQ：一行仍按 hidden 落盘（参考
            # original_shape `[1, 1, 1, 4096]`），不是按 128。
            return self.hidden, group_size
        # 不是 llama2-7B 那三个已知宽度：用张量自己的末维，不许猜成 hidden。
        return last_dim, group_size


DEFAULT_SLOTS = CompileSlots()
