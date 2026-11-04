"""PWLA（分段线性激活）查找表生成器，产出与厂商逐字节一致的 288 字节表。

算法复刻厂商 PWLA 拟合流程：按指数 x 尾数网格生成候选端点，贪心扩展分段并做最小二乘直线拟合，
误差超 tol 就截断成一段；再调 tol 直到恰好 32 段，最后重排段索引并打包成硬件表。

288 字节布局：

    [0:64]     32 个 fp16 slope（小端）
    [64:128]   32 个 fp16 intercept
    [128]      0
    [129:209]  128 个 5 bit 段索引（LUT2 -> 段号）
    [209:288]  0

全程显式用 float64，不改 torch 全局默认 dtype。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Callable

import numpy as np
import torch

LUT_BYTES = 288
LUT_SEGMENTS = 32
LUT2_ENTRIES = 128
_F64 = torch.float64


@dataclass(frozen=True)
class PwlaParams:
    func: Callable[[torch.Tensor], torch.Tensor]
    segments_count: int  # LUT2 候选端点数（双边时正负各一半）
    tol: float  # 初始最大绝对误差，拟合时会被自动调整
    num_points: int  # 每个端点区间的采样点数
    min_exp: int
    max_exp: int
    two_sided: bool = True
    zero_entry: bool = True
    side: str = "positive"  # 单边时建模哪一侧："positive" 或 "negative"
    zero_start_seg: bool = False  # 第 0 段强制为零（slope=intercept=0）
    clip_start: bool = False  # 首段清零
    first_slope_one: bool = False  # 首段 slope 强制为 1（恒等表专用）
    activation_mode: int = 0  # 0 普通，1 奇函数；只影响尾数位宽，不影响表内容


# 每种表一行参数。新激活 = 加一行 + 一个函数。softmax 倒数与 DQ 倒数参数相同，共用 "reciprocal"。
PWLA_PARAMS: dict[str, PwlaParams] = {
    "identity": PwlaParams(lambda x: x, 128, 1.42e-3, 100, -4, 3, activation_mode=1, first_slope_one=True),
    "reciprocal": PwlaParams(lambda x: torch.reciprocal(x + 1.0e-12), 128, 8.0e-5, 100, 1, 1,
                             two_sided=False, zero_entry=False, activation_mode=1),
    "softmax_exp": PwlaParams(lambda x: torch.exp(2 * x), 64, 5e-4, 100, -5, 2,
                              two_sided=False, side="negative", zero_start_seg=True),
    "silu": PwlaParams(torch.nn.functional.silu, 128, 1.42e-3, 100, -4, 3, clip_start=True),
}


@dataclass
class Segment:
    start_entry: int
    end_entry: int  # 不含
    slope: float
    intercept: float


class _ToleranceExceeded(Exception):
    pass


def _exp_range(p: PwlaParams) -> tuple[int, int]:
    """指数区间对齐到 2 的幂个指数。"""
    min_exp = math.floor(p.min_exp)
    span = math.ceil(p.max_exp) - min_exp + 1
    target = 1 if span == 0 else 2 ** math.ceil(math.log2(span))
    return min_exp, min_exp + target - 1


def _entries(p: PwlaParams) -> torch.Tensor:
    """候选端点：每个指数 e 下尾数 0.5..1 均分，端点 = m * 2^e，再按单双边拼负半轴和 0。"""
    min_exp, max_exp = _exp_range(p)
    exps = torch.arange(min_exp, max_exp + 1)
    n_seg = p.segments_count // 2 if p.two_sided else p.segments_count
    per_exp = n_seg // len(exps)
    exp_full = torch.tile(exps.reshape(-1, 1), (per_exp,)).reshape(n_seg)
    mant = torch.tile(torch.arange(0.5, 1.0, 0.5 / per_exp, dtype=_F64), (len(exps),))
    pos = mant * torch.exp2(exp_full.to(_F64))
    neg = torch.flip(-pos, (-1,))
    zero = torch.zeros(1, dtype=_F64)
    if p.two_sided:
        return torch.cat((neg, zero, pos))
    if p.side == "positive":
        return torch.cat((zero, pos)) if p.zero_entry else pos
    return torch.cat((neg, zero)) if p.zero_entry else neg


def _linregress(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """最小二乘直线，与 scipy.stats.linregress 相同：slope = 总体协方差 / 总体方差。"""
    ssxm, ssxym, _, _ = np.cov(x, y, bias=True).flat
    slope = ssxym / ssxm
    return slope, np.mean(y) - slope * np.mean(x)


def _fit(p: PwlaParams, tol: float) -> list[Segment]:
    """贪心分段：从当前起点不断并入下一个端点区间，误差超 tol 时把上一次的拟合定为一段。"""
    n_seg = p.segments_count if p.zero_entry else p.segments_count - 1
    ent = _entries(p)
    segs: list[Segment] = []
    index0, index = 0, 1
    x = torch.zeros(0, dtype=_F64)
    while index <= n_seg:
        # 厂商每个区间的采样点并入两次，最小二乘结果依赖这一点，必须保留
        cur = torch.linspace(float(ent[index - 1]), float(ent[index]), p.num_points, dtype=_F64)
        x = torch.hstack((x, cur, cur))
        with torch.no_grad():
            y = p.func(x)
        slope, intercept = _linregress(x.numpy(), y.numpy())
        err = torch.max(torch.abs(y - (x * slope + intercept)))
        if index == index0 + 1 and err > tol:
            raise _ToleranceExceeded
        if err <= tol:
            seg = Segment(index0, index, float(slope), float(intercept))
            index += 1
        first_zero = index0 == 0 and p.zero_start_seg
        if err > tol or index > n_seg or first_zero:
            if first_zero:
                seg = Segment(index0, index - 1, 0.0, 0.0)
            segs.append(seg)
            index0 = index - 1
            x = torch.zeros(0, dtype=_F64)
    segs[-1].end_entry = n_seg + 1
    if p.clip_start:
        segs[0].slope = segs[0].intercept = 0.0
    return segs


def _fit_32(p: PwlaParams) -> list[Segment]:
    """调整 tol 直到恰好 32 段，步长与回退规则照搬厂商 gen_LUT1。"""
    tol, prev_tol, prev_n = p.tol, None, LUT_SEGMENTS
    while True:
        try:
            segs = _fit(p, tol)
            n = len(segs)
        except _ToleranceExceeded:
            n = -1
        if n == -1 and prev_n == LUT_SEGMENTS:
            tol += 1e-3
        elif n == -1 or (prev_n < LUT_SEGMENTS and n > LUT_SEGMENTS):
            return _fit(p, prev_tol)
        elif prev_n > LUT_SEGMENTS and n < LUT_SEGMENTS:
            return segs
        if n == LUT_SEGMENTS:
            return segs
        step = 0.1e-3 if abs(n - LUT_SEGMENTS) > 1 else 0.01e-3
        prev_tol, prev_n = tol, n
        tol = tol + step if n > LUT_SEGMENTS else tol - step


def _lut2(p: PwlaParams, segs: list[Segment]) -> np.ndarray:
    """128 项 LUT2 -> 段号。双边时硬件先放正半轴再放倒序的负半轴，未覆盖项为 -1。"""
    lut = -np.ones(LUT2_ENTRIES, dtype=np.int32)
    if p.side == "positive" and not p.two_sided:
        for idx, s in enumerate(segs):
            lut[s.start_entry:min(s.end_entry, 128)] = idx
        return lut
    for idx, s in enumerate(segs):
        for ii in range(s.start_entry + 1, min(s.end_entry, 127) + 1):
            if ii in (1, 65):
                lut[ii - 1] = idx
            lut[ii] = idx
    return np.concatenate((lut[64:128], lut[0:64][::-1]))


def _pack(slopes: torch.Tensor, intercepts: torch.Tensor, lut2: np.ndarray) -> bytes:
    """打包成 288 字节。段索引按 5 bit 连续写入，-1 转 uint8 为 255，高位溢出照硬件截断。"""
    blob = bytearray(LUT_BYTES)
    n = len(slopes)
    blob[0:2 * n] = slopes.numpy().astype("<f2").tobytes()
    blob[64:64 + 2 * n] = intercepts.numpy().astype("<f2").tobytes()
    for i, v in enumerate(lut2.astype(np.uint8).tolist()):
        bit = i * 5
        off, out = bit % 8, 129 + bit // 8
        blob[out] |= (v << off) & 0xFF
        if off > 3:
            blob[out + 1] |= v >> (8 - off)
    return bytes(blob)


@lru_cache(maxsize=None)
def pwla_table(kind: str) -> bytes:
    """生成一张 288 字节 PWLA 表。kind 取 PWLA_PARAMS 的键；未知 kind 抛 KeyError。"""
    p = PWLA_PARAMS[kind]
    segs = _fit_32(p)
    # 先 float64 -> fp16 一次舍入，与厂商 torch.tensor(..., dtype=float16) 一致
    slopes = torch.tensor([s.slope for s in segs], dtype=torch.float16)
    intercepts = torch.tensor([s.intercept for s in segs], dtype=torch.float16)
    if p.first_slope_one:
        slopes[0] = 1.0
    return _pack(slopes, intercepts, _lut2(p, segs))
