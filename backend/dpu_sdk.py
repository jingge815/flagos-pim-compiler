"""厂商 SDK（pre-g-driver-api，api/include/dpu.h）的 Python 镜像——numpy 伪硬件实现。

函数名与语义逐一对齐 C API，通信库（comm/lowering.py）与编排器（runtime/，问题 6）
只面对这层签名；换真硬件时以同签名绑定厂商库（backend/hal_vendor.py），上层不动。

与 C API 的对应：

- ``struct dpu_set_t`` → ``DpuSet``；``DPU_FOREACH`` → ``DpuSet`` 迭代产出单 DPU 子集；
- ``dpu_error_t`` 的非零返回 → ``DpuError`` 异常（镜像不做错误码）；
- ``dpu_load`` 的 program：numpy 镜像下是一个 ``fn(dpu_id, mram)`` 的 Python callable
  （真硬件对应 kernel 二进制路径/字节流，镜像无法执行，装载后 launch 抛 DpuError）；
- rank 层：``dpu_alloc_ranks`` / ``dpu_get_nr_ranks`` / ``DpuSet.by_rank``
  （镜像 ``DPU_RANK_FOREACH``）现在有了——但只是 ``dpu_id -> rank_id`` 的
  只读分组标签，不影响任何寻址/迭代语义，扁平 dpu_id 仍是图编译器侧唯一的
  切分粒度（rank 不进 graph/spec_prop.py 的切分决策、不进 comm/ 的通信
  计划，这是范围边界,不是尚未做完）；
- 所有拷贝/launch 同步阻塞（C 版同步语义）；异步事件由问题 6 的 HAL（hal_numpy）提供。

容量：C 版 MRAM/WRAM 容量由硬件定，镜像由 ``dpu_alloc(..., mram_bytes=...,
wram_bytes=...)`` 显式给定，默认 MRAM 64MiB、WRAM 64KiB（UPMEM 规格）。WRAM
是每台 DPU 一块独立的真实字节数组（不是校验用的数字）——DPU 内多 tasklet
并发建模（backend/hal_numpy.py 的 HazardTracker）依赖它是真实内存，才能让
"tasklet 读写 WRAM 的哪个区间"这件事有具体地址可以记录、可以检测重叠。
"""

from __future__ import annotations

from typing import Callable, Iterator, Literal

import numpy as np

DpuXfer = Literal["to_dpu", "from_dpu"]
DPU_XFER_TO_DPU: DpuXfer = "to_dpu"
DPU_XFER_FROM_DPU: DpuXfer = "from_dpu"

DpuLaunchPolicy = Literal["async", "sync"]
DPU_ASYNCHRONOUS: DpuLaunchPolicy = "async"
DPU_SYNCHRONOUS: DpuLaunchPolicy = "sync"

DEFAULT_MRAM_BYTES = 64 * 2**20  # UPMEM MRAM 64MiB
DEFAULT_WRAM_BYTES = 64 * 2**10  # UPMEM WRAM 64KiB

# numpy 镜像下的 DPU kernel：fn(dpu_id, mram)，直接读写本 DPU 的 MRAM 字节数组
DpuKernel = Callable[[int, np.ndarray], None]


class DpuError(RuntimeError):
    """镜像 dpu_error_t 的非零返回：越界、未 prepare、集合基数不符等。"""


class _Dpu:
    """单台 DPU 的硬件状态：独立 MRAM/WRAM 地址空间 + 已装载程序（设备模型）。

    WRAM 与 MRAM 是两块独立的字节数组——真实硬件上 tasklet 只能直接算
    WRAM 里的数据，MRAM 要先 DMA 进 WRAM。这里两者都建成真实内存（不是
    只有 MRAM），供 backend/hal_numpy.py 的多 tasklet hazard 检测使用。
    """

    __slots__ = ("mram", "wram", "program")

    def __init__(self, mram_bytes: int, wram_bytes: int) -> None:
        self.mram = np.zeros(mram_bytes, dtype=np.uint8)
        self.wram = np.zeros(wram_bytes, dtype=np.uint8)
        self.program: DpuKernel | bytes | None = None


class _Machine:
    """整机：扁平编号的 DPU 阵列 + push_xfer 的 per-DPU host buffer 登记表。

    ``rank_of`` 是 dpu_id -> rank_id 的只读分组标签（镜像 pre-g-driver-api
    的 rank 概念），均匀按 ``dpus_per_rank`` 分组，不影响任何寻址/迭代
    语义——扁平 dpu_id 仍是全部 DMA/launch 原语的唯一寻址方式。
    """

    def __init__(self, nr_dpus: int, mram_bytes: int, wram_bytes: int,
                 dpus_per_rank: int | None = None) -> None:
        self.dpus = [_Dpu(mram_bytes, wram_bytes) for _ in range(nr_dpus)]
        self.xfer_buffers: dict[int, np.ndarray] = {}
        self.freed = False
        self.dpus_per_rank = dpus_per_rank or nr_dpus
        self.rank_of = {i: i // self.dpus_per_rank for i in range(nr_dpus)}


class DpuSet:
    """镜像 struct dpu_set_t：一个 DPU 集合。dpu_alloc 得全集；迭代/取子集得单 DPU 集。"""

    def __init__(self, machine: _Machine, dpu_ids: tuple[int, ...]) -> None:
        self._machine = machine
        self._ids = dpu_ids

    def __iter__(self) -> Iterator["DpuSet"]:  # DPU_FOREACH
        return (DpuSet(self._machine, (dpu_id,)) for dpu_id in self._ids)

    def by_rank(self) -> Iterator["DpuSet"]:  # DPU_RANK_FOREACH
        """按 rank 分组迭代：每次产出同一个 rank 内全部成员组成的子集。"""
        by_rank: dict[int, list[int]] = {}
        for dpu_id in self._ids:
            by_rank.setdefault(self._machine.rank_of[dpu_id], []).append(dpu_id)
        return (DpuSet(self._machine, tuple(ids)) for ids in by_rank.values())

    def dpu(self, dpu_id: int) -> "DpuSet":
        return self.subset((dpu_id,))

    def subset(self, dpu_ids: tuple[int, ...]) -> "DpuSet":
        for dpu_id in dpu_ids:
            if dpu_id not in self._ids:
                raise DpuError(f"dpu_id {dpu_id} 不在本 DpuSet 成员 {self._ids} 内")
        return DpuSet(self._machine, dpu_ids)

    @property
    def dpu_ids(self) -> tuple[int, ...]:
        return self._ids

    def _member(self) -> tuple[int, _Dpu]:
        if len(self._ids) != 1:
            raise DpuError(f"该操作要求单 DPU 集合，当前成员为 {self._ids}")
        dpu_id = self._ids[0]
        return dpu_id, self._machine.dpus[dpu_id]


def _check_live(dpu_set: DpuSet) -> None:
    if dpu_set._machine.freed:
        raise DpuError("DpuSet 已 dpu_free")


def _check_range(dpu: _Dpu, offset: int, length: int) -> None:
    if offset < 0 or length < 0 or offset + length > dpu.mram.size:
        raise DpuError(
            f"MRAM 越界: offset={offset} length={length} capacity={dpu.mram.size}"
        )


def _as_bytes(src: object, length: int) -> np.ndarray:
    blob = np.ascontiguousarray(np.asarray(src)).view(np.uint8).reshape(-1)
    if blob.size != length:
        raise DpuError(f"host 缓冲区字节数 {blob.size} 与 length={length} 不符")
    return blob


def _writable_c_bytes(buffer: np.ndarray) -> np.ndarray:
    """返回供 DPU→host 写入的原缓冲区字节视图，拒绝会生成临时副本的数组。"""
    if not isinstance(buffer, np.ndarray):
        raise DpuError("DPU→host 的 host buffer 必须是 numpy.ndarray")
    if not buffer.flags.c_contiguous or not buffer.flags.writeable:
        raise DpuError("DPU→host 的 host buffer 必须可写且 C-contiguous")
    return buffer.view(np.uint8).reshape(-1)


def _c_bytes(buffer: np.ndarray) -> np.ndarray:
    """返回原 C-contiguous 缓冲区的字节视图，不为 host→DPU 读取要求可写。"""
    if not isinstance(buffer, np.ndarray):
        raise DpuError("host buffer 必须是 numpy.ndarray")
    if not buffer.flags.c_contiguous:
        raise DpuError("host buffer 必须 C-contiguous")
    return buffer.view(np.uint8).reshape(-1)


# ---------------------------------------------------------------------------
# 设备管理（dpu.h：归属编排器层，镜像一并给出供问题 6 使用）
# ---------------------------------------------------------------------------


def dpu_alloc(
    nr_dpus: int, profile: str | None = None, *,
    mram_bytes: int = DEFAULT_MRAM_BYTES, wram_bytes: int = DEFAULT_WRAM_BYTES,
) -> DpuSet:
    """分配 nr_dpus 台 DPU，返回全机集合（单一 rank）。profile 为 C 版占位（镜像忽略）。"""
    if nr_dpus <= 0:
        raise DpuError(f"nr_dpus={nr_dpus} 必须为正")
    return DpuSet(_Machine(nr_dpus, mram_bytes, wram_bytes), tuple(range(nr_dpus)))


def dpu_alloc_ranks(
    nr_ranks: int, profile: str | None = None, *, dpus_per_rank: int,
    mram_bytes: int = DEFAULT_MRAM_BYTES, wram_bytes: int = DEFAULT_WRAM_BYTES,
) -> DpuSet:
    """镜像 pre-g-driver-api 的 ``dpu_alloc_ranks``：按 rank 批量分配 DPU。

    只是给扁平 dpu_id 打上 rank 分组标签（``_Machine.rank_of``），不改变
    任何寻址/迭代语义——`dpu_alloc` 分配的单 rank 集合与这里分配的多 rank
    集合，对 `dpu_copy_to`/`dpu_launch` 等原语完全一样地使用。
    """
    if nr_ranks <= 0:
        raise DpuError(f"nr_ranks={nr_ranks} 必须为正")
    if dpus_per_rank <= 0:
        raise DpuError(f"dpus_per_rank={dpus_per_rank} 必须为正")
    nr_dpus = nr_ranks * dpus_per_rank
    machine = _Machine(nr_dpus, mram_bytes, wram_bytes, dpus_per_rank)
    return DpuSet(machine, tuple(range(nr_dpus)))


def dpu_free(dpu_set: DpuSet) -> None:
    _check_live(dpu_set)
    dpu_set._machine.freed = True


def dpu_get_nr_dpus(dpu_set: DpuSet) -> int:
    _check_live(dpu_set)
    return len(dpu_set._ids)


def dpu_get_nr_ranks(dpu_set: DpuSet) -> int:
    _check_live(dpu_set)
    return len({dpu_set._machine.rank_of[i] for i in dpu_set._ids})


# ---------------------------------------------------------------------------
# DMA 三件套 + broadcast（dpu.h：归属通信库层，comm/lowering.py 的唯一底座）
# ---------------------------------------------------------------------------


def dpu_copy_to(dpu_set: DpuSet, offset: int, src: object, length: int) -> None:
    """host → DPU：同一份 host 数据写入集合内每台 DPU 的 MRAM[offset, offset+length)。"""
    _check_live(dpu_set)
    blob = _as_bytes(src, length)
    for dpu_id in dpu_set._ids:
        dpu = dpu_set._machine.dpus[dpu_id]
        _check_range(dpu, offset, length)
        dpu.mram[offset : offset + length] = blob


def dpu_copy_from(dpu_set: DpuSet, offset: int, dst: np.ndarray, length: int) -> None:
    """DPU → host：从单台 DPU 的 MRAM 读 length 字节，填入 dst 的前 length 字节。"""
    _check_live(dpu_set)
    _, dpu = dpu_set._member()
    _check_range(dpu, offset, length)
    view = _writable_c_bytes(dst)
    if view.size < length:
        raise DpuError(f"host 缓冲区字节数 {view.size} 小于 length={length}")
    view[:length] = dpu.mram[offset : offset + length]


def dpu_prepare_xfer(dpu_set: DpuSet, buffer: np.ndarray) -> None:
    """为单台 DPU 登记下一次 push_xfer 使用的 host buffer（DPU_FOREACH 内逐台调用）。"""
    _check_live(dpu_set)
    dpu_id, _ = dpu_set._member()
    _c_bytes(buffer)
    dpu_set._machine.xfer_buffers[dpu_id] = buffer


def dpu_push_xfer(dpu_set: DpuSet, xfer: DpuXfer, offset: int, length: int, flags: int = 0) -> None:
    """批量传输：对集合内每台 DPU，用其 prepare_xfer 登记的 buffer 做同 offset 同长 DMA。

    把"对 N 个 DPU 的 N 次同位 DMA"合并为一次调用（方案问题 3 二.(2)：缓解主机带宽
    瓶颈）；成员间 offset/length 必须一致是 C 版原语本身的约束。
    """
    _check_live(dpu_set)
    if xfer not in (DPU_XFER_TO_DPU, DPU_XFER_FROM_DPU):
        raise DpuError(f"未知 xfer 方向: {xfer!r}")
    for dpu_id in dpu_set._ids:
        dpu = dpu_set._machine.dpus[dpu_id]
        _check_range(dpu, offset, length)
        buffer = dpu_set._machine.xfer_buffers.get(dpu_id)
        if buffer is None:
            raise DpuError(f"DPU{dpu_id} 未 dpu_prepare_xfer")
        view = _writable_c_bytes(buffer) if xfer == DPU_XFER_FROM_DPU else _c_bytes(buffer)
        if view.size < length:
            raise DpuError(f"DPU{dpu_id} 登记的 buffer 字节数 {view.size} 小于 length={length}")
        if xfer == DPU_XFER_TO_DPU:
            dpu.mram[offset : offset + length] = view[:length]
        else:
            view[:length] = dpu.mram[offset : offset + length]


def dpu_broadcast_to(dpu_set: DpuSet, offset: int, src: object, length: int, flags: int = 0) -> None:
    """广播：一份 host 数据一次调用写入集合内每台 DPU 的同一 MRAM 偏移。"""
    dpu_copy_to(dpu_set, offset, src, length)


# ---------------------------------------------------------------------------
# kernel 装载 / 执行 / 同步（dpu.h：归属编排器层；镜像内同步执行）
# ---------------------------------------------------------------------------


def dpu_load(dpu_set: DpuSet, program: DpuKernel | bytes) -> None:
    """装载程序到集合内每台 DPU。numpy 镜像接受 callable；bytes 仅登记、不可执行。"""
    _check_live(dpu_set)
    for dpu_id in dpu_set._ids:
        dpu_set._machine.dpus[dpu_id].program = program


def dpu_launch(dpu_set: DpuSet, policy: DpuLaunchPolicy) -> None:
    """触发集合内每台 DPU 执行已装载程序。镜像内同步执行（异步由问题 6 HAL 提供）。"""
    _check_live(dpu_set)
    for dpu_id in dpu_set._ids:
        program = dpu_set._machine.dpus[dpu_id].program
        if program is None:
            raise DpuError(f"DPU{dpu_id} 未 dpu_load")
        if not callable(program):
            raise DpuError("numpy 镜像无法执行 kernel 二进制，需 Python callable")
        program(dpu_id, dpu_set._machine.dpus[dpu_id].mram)


def dpu_sync(dpu_set: DpuSet) -> None:
    """等待完成。镜像内 launch/DMA 均同步阻塞，sync 为空操作。"""
    _check_live(dpu_set)


def dpu_status(dpu_set: DpuSet) -> tuple[bool, bool]:
    """返回 (done, fault)。镜像内同步执行，恒为 (True, False)。"""
    _check_live(dpu_set)
    return True, False


def dpu_log_read(dpu_set: DpuSet, stream) -> None:
    """读 DPU 日志。镜像无 printf 通道，输出成员清单占位。"""
    _check_live(dpu_set)
    stream.write(f"DpuSet{list(dpu_set._ids)}: no logs (numpy mirror)\n")
