"""把统一 IR 的四维写成 pimir 的文本片段。

`elem_strides` 只决定 A 路 `#pim.placement` 的 `order`：`convert-triton-to-pim`
把它落进张量编码，`-pim-explicit-dma` 按它重算 `elem_stride`。B 路张量编码的
`order` 固定行主序 —— 那条路上没有 pass 读它。L1 / L2 / DDR 级的步幅仍归
编排器，不经这里。

布局编码的形状受 FlagTree 的 verifier 约束（`TaskletTiledEncodingAttr::verify`）：
四组数组同秩、取值全为正、`order` 是 `[0, rank)` 的排列。
"""

from __future__ import annotations

from contracts.mem_layout import check_elem_strides
from contracts.op_contract import DpuShard, PIMHardwareConfig


def tasklet_tiled(shape: tuple[int, ...], *, shard: DpuShard | None,
                  num_tasklets: int,
                  elem_strides: tuple[int, ...] = ()) -> str:
    """跨 DPU 切分决策 + 排布 → `#pim.tasklet_tiled<{...}>`；单 DPU 返回空串。

    - `dpusPerDevice` 只在被切的那一维记 DPU 数，其余维是 1 —— 它就是
      「这块张量第几维分给了几台 DPU」的编码。全 1 时不发编码：没有跨 DPU
      决策可记，文本与改动前逐字节相同。
    - `order` 由 `elem_strides` 推出：**步幅最小的维是最内层**，排在最前。
      空步幅按行主序紧密处理，与 FlagTree 默认 builder 的 `rank-1-i` 同序。
      原先这里写死行主序，排布字段进不来，Memory Layout 这一维就没有下发内容。
    - `taskletsPerDpu` 按形状夹取（见 `_tasklets_per_dpu`），与 FlagTree 两个
      builder 的算式同口径。

    只有 `shard` 档有编码可发。`replicate` 与 `partial` 每台 DPU 都持有完整形状，
    编码就是全 1，而 printer 省略全 1 —— 发与不发逐字节相同，所以返回空串。
    **这正是模块级 `#pim.placement` 存在的理由**：张量编码这一层分不出这两档与
    单 DPU 的区别（三者编码都是全 1），模块属性分得出，而它们的归约与容量口径
    并不相同。两个载体回答的是不同的问题，不是一份信息写两遍。
    """
    if not shape:
        raise ValueError("张量形状的秩必须为正")
    # 与统一 IR 的分片校验同一份判据：重叠的步幅在这里也要抛，不能照单下发。
    check_elem_strides(shape, elem_strides)
    if shard is None or shard.num_dpus <= 1 or shard.kind != "shard":
        return ""
    rank = len(shape)
    if not 0 <= shard.dim < rank:
        raise ValueError(
            f"切分维 {shard.dim} 超出秩 {rank}（{shard}）")

    order = _layout_order(rank, elem_strides)
    tasklets = _tasklets_per_dpu(shape, order, num_tasklets)
    dpus = [shard.num_dpus if dim == shard.dim else 1 for dim in range(rank)]

    return ("#pim.tasklet_tiled<{"
            f"sizePerTasklet = [{_ints([1] * rank)}], "
            f"taskletsPerDpu = [{_ints(tasklets)}], "
            f"dpusPerDevice = [{_ints(dpus)}], "
            f"order = [{_ints(order)}]}}>")


def _layout_order(rank: int, elem_strides: tuple[int, ...]) -> list[int]:
    """维序，最内层在前。空步幅 = 行主序紧密。

    步幅相等时按维号降序，与 FlagTree 默认 builder 的 `rank-1-i` 同序 ——
    长度为 1 的维会有相等的步幅，顺序不一致会让两条路的文本对不上。
    """
    if not elem_strides:
        return list(reversed(range(rank)))
    return sorted(range(rank), key=lambda dim: (elem_strides[dim], -dim))


def _tasklets_per_dpu(shape: tuple[int, ...], order: list[int],
                      num_tasklets: int) -> list[int]:
    """逐维分配 tasklet，算式与 FlagTree 的 builder 逐字对应。

    夹取是必须的：给长度为 1 的轴分 16 个 tasklet 是做不到的分配。FlagTree 的
    两个 builder 都按 `avail = shape[i] / sizePerTasklet[i]` 夹取，本仓的
    `sizePerTasklet` 恒为 1，所以可用长度就是这一维的长度。
    """
    tasklets = [1] * len(shape)
    remaining = num_tasklets
    for dim in order:
        if remaining <= 1:
            break
        avail = max(1, shape[dim])
        tasklets[dim] = max(1, min(remaining, avail))
        remaining //= tasklets[dim]
    if remaining > 1:
        # 整块比 tasklet 还少：余数落在最外维，多出来的 tasklet 空转。
        tasklets[order[-1]] *= remaining
    return tasklets


# 下发切分决策的模块属性名。值是一个 `#pim.placement`（FlagTree 侧
# `PlacementSpecAttr`，名字常量在它的 `Dialect.h`），**两仓之间的契约**。
#
# 原先这里是 `pim.shard-dim` / `pim.shard-dpus` 两个裸整数，与 FlagTree 读的
# `pim.placement` 是两套互不相识的载体：各自都有测试、各自都绿，但在一条真实
# tp2 链路上从未相遇 —— 实测下发的模块头里有 shard-dim 而没有 placement，
# 于是 `dpusPerDevice` 始终是空的。收口成一个之后那条链路才真的接上。
PLACEMENT_ATTR = "pim.placement"


def module_attributes(hardware: PIMHardwareConfig) -> tuple[str, ...]:
    """模块级硬件属性，对齐 A 路的富度。

    `pim.dma-align` 与下发算子编译契约时用的那份（`contracts/op_contract.py`
    的 `dma_align`）是同一个值，两处必须一致。
    """
    return (
        f'"pim.num-dpus" = {hardware.num_dpus} : i32',
        f'"pim.num-tasklets" = {hardware.num_tasklets} : i32',
        f'"pim.dma-align" = {hardware.dma_align} : i32',
    )


def placement_attribute(shard, *, rank: int,
                         elem_strides: tuple[int, ...] = (),
                         mram_offset: int = 0,
                         align_bytes: int = 0) -> tuple[str, ...]:
    """跨 DPU 切分决策的模块属性形式。没有切分时返回空。

    写成 `#pim.placement`，FlagTree 的 `convert-triton-to-pim` 读它、经
    placement 版 builder 落进每个张量编码的 `dpusPerDevice`。A 路的张量编码
    全部由 FlagTree 生成（图编译器碰不到），所以模块属性是唯一的交接点。

    `rank` 是**被标注张量**的秩，必须与 `shard.dim` 同一个坐标系。越界直接抛，
    与 `tasklet_tiled` 同一条纪律：FlagTree 的 placement 版 builder 对越界的维
    号静默跳过（`dpusPerDevice` 恒全 1），决策整条丢掉而无任何诊断 —— 这个口
    子必须在下发侧堵住。

    三档都发（`shard` / `replicate` / `partial`），不只发切分那一档。
    早先只发 shard，理由是「replicate / partial 在 PIMMLIR 侧编码都是全 1，
    与不发等价」—— 这个理由只在**张量编码**那一层成立。模块级的
    `#pim.placement` 能分辨三者，而它们的归约与容量口径并不相同：
    `partial` 每台持有一份全形状的局部和、要跨 DPU 归约才完整，归约时得有地方
    接收对端那一份；`replicate` 不欠归约。不发就让 PIMMLIR 分不清「复制」与
    「单 DPU」—— 那正是 `#pim.placement` 存在的理由（见 `PIMAttrDefs.td`
    Placement 段注释）。

    仍然**只在多 DPU 时发**：`num_dpus <= 1` 一个字不加，所以单卡口径下发的
    文本与改动前逐字节相同（§5.2 产物不变判据）。
    """
    if rank <= 0:
        raise ValueError(f"rank 必须为正，got {rank}")
    if shard is None or shard.num_dpus <= 1:
        return ()
    fields = [f"kind = {shard.kind}"]
    if shard.kind == "shard":
        # 秩校验只对 shard 有意义：另两档不指维号。越界直接抛，与
        # `tasklet_tiled` 同一条纪律 —— FlagTree 的 placement 版 builder 对越界
        # 维号静默跳过（`dpusPerDevice` 恒全 1），决策整条丢掉而无任何诊断。
        if not 0 <= shard.dim < rank:
            raise ValueError(
                f"切分维 {shard.dim} 超出被标注张量的秩 {rank}（{shard}）。"
                f"图张量的维号要先换算到 kernel 张量的坐标系"
                f"（见 contracts.op_contract.flatten_shard_dim）")
        fields.append(f"dim = {shard.dim}")
    fields.append(f"numDpus = {shard.num_dpus}")
    if shard.kind == "partial":
        # `partial` 必须写明怎么归约，否则 FlagTree 直接拒收这份 pimir：
        # `a partial placement must say how its pieces combine; set reduce to
        # sum or mean`（见 FlagTree 的 placement_negative.mlir）。
        # 早先这里按「reduce 没有读者」不下发 —— 那个判断对这版 pass 不成立，
        # 归约方式既参与校验，也决定归约暂存的开销。
        fields.append(f"reduce = {shard.reduce}")
    # 排布：步幅最小的维是最内层。行主序不写 —— 与 FlagTree 的默认序相同，
    # 写了反而让单卡口径的文本不再逐字节不变。
    order = _layout_order(rank, elem_strides)
    if order != list(reversed(range(rank))):
        fields.append(f"order = [{_ints(order)}]")
    # 地址与对齐：0 是默认值，不写才让单卡口径的文本保持逐字节不变。
    # `alignBytes` 由 `-pim-tile-to-budget` 读取：比模块级 `pim.dma-align` 更严
    # 时按它选分块，让每块缓冲都满足分片的起始对齐。
    if mram_offset:
        fields.append(f"mramOffset = {mram_offset}")
    if align_bytes:
        fields.append(f"alignBytes = {align_bytes}")
    return (f'{PLACEMENT_ATTR} = #pim.placement<{", ".join(fields)}>',)


def _ints(values: list[int]) -> str:
    return ", ".join(str(v) for v in values)
