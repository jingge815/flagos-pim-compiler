"""每步运行时指标：wave 宽度、槽池占用与 attention 路由计数。"""

from __future__ import annotations

from dataclasses import dataclass

from runtime.kernels import route_counts


@dataclass
class StepStats:
    """累计每步调度指标；`record` 每步调一次，`format` 拼成一行日志。"""

    pool_total: int = 0          # 物理槽总数，定容后写入
    steps: int = 0
    waves: int = 0
    widest_wave: int = 0
    requests: int = 0
    tokens: int = 0
    last_requests: int = 0
    pool_used: int = 0           # 当前被请求占用的物理块数，共享块只算一次
    attn_hit: int = 0            # 融合 paged_attn 的累计调用次数
    attn_fallback: int = 0       # 退回主机两遍路径的累计次数

    def record(self, scheduled: dict[str, int], *, width: int, slots) -> None:
        """记一步：请求数、token 数、wave 个数、块占用与路由累计值。"""
        count = len(scheduled)
        self.steps += 1
        self.last_requests = count
        self.requests += count
        self.tokens += sum(scheduled.values())
        if count:
            width = max(1, width)
            self.waves += -(-count // width)
            self.widest_wave = max(self.widest_wave, width)
        self.pool_used = len({block for slot in slots for block in slot.blocks.slots})
        routes = route_counts()
        self.attn_hit = routes.get(("paged_attn", "hit"), 0)
        self.attn_fallback = routes.get(("paged_attn", "fallback"), 0)

    def format(self) -> str:
        """一行日志：本步与累计的请求/token、wave、槽池占用、注意力路径。"""
        return (
            f"PIM 步 {self.steps}：本步请求 {self.last_requests}，"
            f"累计 wave {self.waves}（最宽 {self.widest_wave}），"
            f"token {self.tokens}，槽池 {self.pool_used}/{self.pool_total}，"
            f"paged_attn 命中 {self.attn_hit} / 退回 {self.attn_fallback}")
