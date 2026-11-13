"""地址区间依赖判定：静态计划与运行期调度共用同一套 RAW/WAW/WAR 语义。"""

from __future__ import annotations

from contracts.exec_plan import Access


def overlap(a: Access, b: Access) -> bool:
    """判断两个访问区间是否相交。"""
    return a.loc == b.loc and a.offset < b.offset + b.length and b.offset < a.offset + a.length


def deps_of(reads: list[Access], writers: dict[tuple, list[tuple[Access, int]]]) -> list[int]:
    """返回与读区间相交的历史写命令。

    同一地址区间被反复写（各层复用激活缓冲）时，后写者生成时已经等过前写者，
    所以只留最后一个写者；前面那些沿等待链传递，不必重复列出。
    """
    last: dict[tuple, int] = {}
    for a in reads:
        for wa, cid in writers.get(a.loc, []):
            if not overlap(a, wa):
                continue
            key = (wa.loc, wa.offset, wa.length)
            if cid > last.get(key, -1):
                last[key] = cid
    return list(last.values())


def war_waits(
    loc: tuple, offset: int, pending_readers: dict[tuple, list[str]],
    reader_cmds: dict[tuple, list[int]],
) -> list[int]:
    """返回目标地址旧值的读取命令。"""
    return [cid for rn in pending_readers.get((loc, offset), [])
            for cid in reader_cmds.get((rn, loc[1]), [])]
