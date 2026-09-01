"""提供厂商 DPU SDK 的 NumPy 镜像，实现设备、DMA 和程序装载接口。"""

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

# NumPy 镜像中的 DPU 内核函数类型。
DpuKernel = Callable[[int, np.ndarray], None]


class DpuError(RuntimeError):
    """镜像 dpu_error_t 的非零返回：越界、未 prepare、集合基数不符等。"""


class _Dpu:
    """保存单台 DPU 的 MRAM、WRAM 和已装载程序。"""

    __slots__ = ("mram", "wram", "program")

    def __init__(self, mram_bytes: int, wram_bytes: int) -> None:
        self.mram = np.zeros(mram_bytes, dtype=np.uint8)
        self.wram = np.zeros(wram_bytes, dtype=np.uint8)
        self.program: DpuKernel | bytes | None = None


class _Machine:
    """保存 DPU 阵列、DMA 缓冲区和 rank 分组信息。"""

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
    """按 rank 批量分配 DPU，并返回全集。"""
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
    """使用各 DPU 已登记的缓冲区执行同偏移、同长度的批量 DMA。"""
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


def dpu_load(dpu_set: DpuSet, program: DpuKernel | bytes) -> None:
    """装载程序到集合内每台 DPU。numpy 镜像接受 callable；bytes 仅登记、不可执行。"""
    _check_live(dpu_set)
    for dpu_id in dpu_set._ids:
        dpu_set._machine.dpus[dpu_id].program = program


def dpu_launch(dpu_set: DpuSet, policy: DpuLaunchPolicy) -> None:
    """触发集合内每台 DPU 执行已装载程序，并同步返回。"""
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
