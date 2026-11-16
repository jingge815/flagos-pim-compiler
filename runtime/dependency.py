"""地址区间依赖判定：静态计划与运行期调度共用同一套 RAW/WAW/WAR 语义。"""

from __future__ import annotations

from typing import Iterable

from contracts.exec_plan import Access, Command


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


def fold_waits(waits: Iterable[int], lane_of: dict[int, int | None]) -> list[int]:
    """同 DPU 只保留最后一条前驱；host 命令逐条保留。

    静态计划生成（`_PlanBuilder.append`）与运行期 shadow 校验
    （`address_waits`）共用本实现，折叠规则是唯一真源，不得另写副本。
    """
    last: dict[tuple, int] = {}
    for cid in waits:
        lane = lane_of.get(cid)
        key = ("dpu", lane) if lane is not None else ("host", cid)
        if cid > last.get(key, -1):
            last[key] = cid
    return sorted(last.values())


def address_waits(commands: Iterable[Command]) -> dict[int, list[int]]:
    """按计划顺序重算地址区间依赖（RAW/WAW/WAR），供运行期调度做 shadow 校验。

    这里只覆盖由 `reads/writes` 能推出的依赖；host 闭包经 `resolve_node` 产生的
    数据依赖仍以计划的 `waits` 为准，二者在调用方合并。
    """
    writers: dict[tuple, list[tuple[Access, int]]] = {}
    readers: dict[tuple, list[tuple[Access, int]]] = {}
    lane_of: dict[int, int | None] = {}
    result: dict[int, list[int]] = {}
    for cmd in commands:
        waits = set(deps_of(cmd.reads, writers))
        waits.update(deps_of(cmd.writes, writers))
        for write in cmd.writes:
            for read, cid in readers.get(write.loc, []):
                if overlap(write, read):
                    waits.add(cid)
        result[cmd.id] = fold_waits(waits, lane_of)
        for read in cmd.reads:
            readers.setdefault(read.loc, []).append((read, cmd.id))
        for write in cmd.writes:
            writers.setdefault(write.loc, []).append((write, cmd.id))
            # 新写覆盖旧值后，旧读者不再影响后续写者的 WAR 判定。
            readers[write.loc] = [
                (read, cid) for read, cid in readers.get(write.loc, [])
                if not overlap(write, read)]
        lane_of[cmd.id] = cmd.dpu_id
    return result


def address_wait_gaps(commands: Iterable[Command],
                      runtime: dict[int, list[int]]) -> list[str]:
    """返回运行期地址依赖中不在静态 `waits` 里的命令；空表示 shadow 通过。"""
    gaps = []
    for cmd in commands:
        extra = set(runtime.get(cmd.id, ())) - set(cmd.waits)
        if extra:
            gaps.append(f"cmd {cmd.id} 的运行期地址依赖 {sorted(extra)} 不在静态 waits 里")
    return gaps
