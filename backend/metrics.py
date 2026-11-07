"""每步运行时指标：wave 宽度、槽池占用与 attention 路由计数。"""

from __future__ import annotations

from dataclasses import dataclass, field

from runtime.kernels import route_counts


@dataclass
class StepStats:
    """累计每步调度指标；`record` 每步调一次，`format` 拼成一行日志。"""

    pool_total: int = 0          # 物理槽总数，定容后写入
    max_width: int = 0           # 编译期窗口上限，即最多能同时跑几个请求
    window_bytes: int = 0        # 单个窗口在每台 DPU 上占的字节（激活 + 块表）
    steps: int = 0
    waves: int = 0
    widest_wave: int = 0
    requests: int = 0
    tokens: int = 0
    last_requests: int = 0
    last_width: int = 0          # 本步选中的窗口数
    width_steps: dict[int, int] = field(default_factory=dict)  # 每档窗口数各用了多少步
    pool_used: int = 0           # 当前被请求占用的物理块数，共享块只算一次
    attn_hit: int = 0            # 融合 paged_attn 的累计调用次数
    attn_fallback: int = 0       # 退回主机两遍路径的累计次数

    def record(self, scheduled: dict[str, int], *, width: int, slots) -> None:
        """记一步：请求数、token 数、选中窗口、wave 个数、块占用与路由累计值。"""
        count = len(scheduled)
        self.steps += 1
        self.last_requests = count
        self.requests += count
        self.tokens += sum(scheduled.values())
        if count:
            width = max(1, width)
            self.last_width = width
            self.width_steps[width] = self.width_steps.get(width, 0) + 1
            self.waves += -(-count // width)
            self.widest_wave = max(self.widest_wave, width)
        self.pool_used = len({block for slot in slots for block in slot.blocks.slots})
        routes = route_counts()
        self.attn_hit = routes.get(("paged_attn", "hit"), 0)
        self.attn_fallback = routes.get(("paged_attn", "fallback"), 0)

    def format(self) -> str:
        """一行日志：本步与累计的请求/token、窗口分布与占用、wave、槽池、注意力路径。"""
        widths = "/".join(
            f"{w}×{n}" for w, n in sorted(self.width_steps.items())) or "无"
        return (
            f"PIM 步 {self.steps}：本步请求 {self.last_requests}、窗口 {self.last_width}，"
            f"累计 wave {self.waves}（最宽 {self.widest_wave}），窗口分布 {widths}，"
            f"窗口占用 {self.last_width * self.window_bytes}/"
            f"{self.max_width * self.window_bytes} B/DPU，"
            f"token {self.tokens}，槽池 {self.pool_used}/{self.pool_total}，"
            f"paged_attn 命中 {self.attn_hit} / 退回 {self.attn_fallback}")
