"""编译子任务的并行执行工具。

被并行化的只有子进程型编译（triton-opt / gcc）：每个任务写各自的内容寻址缓存
文件，`compile_op` 对同 key 还有文件锁，线程只负责等待，互不影响。并行度由
环境变量 `OPCOMPILER_JOBS` 控制，默认 32。
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Iterable, List, Sequence, TypeVar

DEFAULT_JOBS = 32

T = TypeVar("T")
R = TypeVar("R")


def job_workers() -> int:
    """读并行度：OPCOMPILER_JOBS，非法或缺失时取默认 32，至少 1。"""
    raw = os.environ.get("OPCOMPILER_JOBS")
    try:
        value = int(raw) if raw else DEFAULT_JOBS
    except ValueError:
        return DEFAULT_JOBS
    return max(1, value)


def run_jobs(fn: Callable[[T], R], items: Iterable[T]) -> List[R]:
    """对 items 逐个执行 fn，按提交顺序返回结果。

    并行度 1 或任务数不超过 1 时直接串行，不开线程池。异常按首个失败的任务
    抛出（executor.map 的语义），不吞。
    """
    sequence: Sequence[T] = list(items)
    if len(sequence) <= 1 or job_workers() <= 1:
        return [fn(item) for item in sequence]
    workers = min(job_workers(), len(sequence))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(fn, sequence))
