"""行级路由内核（gather_rows / scatter_add）与运行期行数通道的单元测试。

exact_dispatch 的地基：镜像语义对齐稠密 where 路径（routing 先落 fp16、
专家升序累加）、行数来自 `hal.moe_rows` 缺了就抛、包络形状进编译缓存键。
"""

from __future__ import annotations

import ctypes
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

import runtime.kernels_pim as kernels_pim
from backend.hal_numpy import ExecutionContext, NumpyBackend, NumpyBackendConfig
from contracts.exec_plan import Access, Command
from contracts.op_semantics import GATHER_ROWS_KERNEL, SCATTER_ADD_KERNEL
from runtime import kernels as K
from runtime.kernels import DEFAULT_OPS_CONTEXT, register_all, reset_route_counts


def _backend() -> NumpyBackend:
    return NumpyBackend(NumpyBackendConfig(num_dpus=1, mram_bytes_per_dpu=65536))


def _submit_dispatch(backend, gate=77, ctx=None):
    """提交一个假的 dispatch 命令：行级命令带 gate 标签，submit 要求它先在
    事件表里（真实流程中 dispatch 就是先行命令）。dma_in 即可，不跑内核。
    已在表里就复用，同表多次提交行级命令不会撞 id。"""
    events = backend.events_table(ctx)
    if gate in events:
        return events[gate]
    return backend.submit(Command(
        id=gate, op="dma_in", dpu_id=0,
        payload={"offset": 65000, "data": np.zeros(1, np.float16)},
        reads=[], writes=[], waits=[]), ctx)


def _launch(backend, kernel_name, reads_data, arg_shapes, arg_dtypes,
            out_shape, dtype, *, gate=77, expert=3,
            cmd_id=0, waits=None, lane_key=0):
    """写张量、构造 launch 命令、submit/wait，返回 (结果, 写地址)。"""
    off = 0
    accesses = []
    for data, name in zip(reads_data, arg_dtypes):
        blob = np.ascontiguousarray(data, dtype=np.dtype(name))
        backend.write_local(0, off, blob)
        accesses.append((("dpu", 0), off, blob.nbytes))
        off += 512
    write_off = off
    write_nbytes = int(np.prod(out_shape)) * np.dtype(dtype).itemsize
    cmd = Command(
        id=cmd_id, op="launch", dpu_id=0,
        payload={"kernel": kernel_name, "node": "n",
                 "arg_kinds": ["tensor"] * len(reads_data),
                 "arg_shapes": arg_shapes, "arg_dtypes": arg_dtypes,
                 "dtype": dtype, "out_shape": out_shape,
                 "gate": gate, "expert": expert},
        reads=[Access(loc, o, n) for loc, o, n in accesses],
        writes=[Access(("dpu", 0), write_off, write_nbytes)],
        waits=waits or [], num_tasklets=4,
    )
    ctx = ExecutionContext(lane_key=lane_key) if lane_key else None
    _submit_dispatch(backend, gate, ctx)
    event = backend.submit(cmd, ctx)
    backend.wait(event)
    return (backend.read_local(0, write_off, out_shape, np.dtype(dtype)),
            write_off)


def _set_rows(backend, rows, *, gate=77, expert=3, lane_key=0):
    backend.moe_rows[(lane_key, gate, expert)] = rows


# ---------- scatter_add 镜像语义 ----------


def test_scatter_add_mirror_matches_manual_loop() -> None:
    """`acc[idx[i]] += w[i]*out[i]` 逐位等于手写循环（fp16 同宽）。"""
    rng = np.random.default_rng(3)
    acc = rng.standard_normal((6, 4)).astype(np.float16)
    out = rng.standard_normal((5, 4)).astype(np.float16)
    w = rng.random(5).astype(np.float16)
    idx = np.array([5, 1, 3, 1, 0], dtype=np.int64)  # 不同专家内才去重；镜像允许重复
    orig = acc.copy()
    got = kernels_pim.scatter_add(acc, out, w, idx, 3)
    want = acc.copy()
    for i in range(3):
        want[idx[i]] = want[idx[i]] + w[i] * out[i]
    assert np.array_equal(got, want)
    assert np.array_equal(acc, orig)  # 镜像不原地改输入


def test_scatter_add_mirror_matches_dense_where_accumulation() -> None:
    """镜像累加与稠密 where 路径（routing 落 fp16、专家升序）数值一致。

    未选中专家在稠密侧贡献恰为 +0.0，加与不加数值相同（±0.0 差异不参与
    判等，随机量级下也不会出现）。
    """
    rng = np.random.default_rng(5)
    n_exp, rows, h = 5, 1, 4
    out = rng.standard_normal((n_exp, rows, h))
    out /= max(np.abs(out).min(), 1e-6)          # 避开 fp16 下溢的 ±0.0 边缘
    out = out.astype(np.float16)
    out[1] = 0.0                                  # 未选中行：where 后恰为 0
    out[3] = 0.0
    w = rng.random(n_exp).astype(np.float16)
    w[1] = 0.0
    w[3] = 0.0
    # 稠密：acc = t0；acc = acc + t_e，e 升序（make_static_moe_forward 的顺序）
    dense = out[0] * w[0]
    for e in range(1, n_exp):
        dense = dense + out[e] * w[e]
    # exact：命中专家升序各散写一次
    idx = np.zeros(rows, dtype=np.int64)
    exact = np.zeros((rows, h), np.float16)
    for e in (0, 2, 4):
        exact = kernels_pim.scatter_add(exact, out[e], np.array([w[e]], np.float16),
                                        idx, rows)
    assert np.array_equal(dense.astype(np.float16), exact)


def test_scatter_add_mirror_rejects_rows_out_of_range() -> None:
    acc = np.zeros((4, 4), np.float16)
    out = np.zeros((3, 4), np.float16)
    w = np.ones(3, np.float16)
    idx = np.zeros(3, np.int64)
    with pytest.raises(ValueError, match="超出索引表长度"):
        kernels_pim.scatter_add(acc, out, w, idx, 4)
    with pytest.raises(ValueError, match="超出索引表长度"):
        kernels_pim.scatter_add(acc, out, w, idx, -1)


# ---------- gather_rows 内核 ----------


def test_gather_rows_kernel_prefix_and_clamped_tail() -> None:
    """前 rows 行 = table[idx[:rows]]；行数之外的索引钳到 0 行（哨兵也不读）。"""
    backend = _backend()
    register_all(backend)
    rng = np.random.default_rng(7)
    table = rng.standard_normal((8, 4)).astype(np.float16)
    idx = np.array([3, 1, 4, 0, 999, 999, 7, 2], dtype=np.int64)  # 尾部越界哨兵
    _set_rows(backend, 4)
    got, _ = _launch(backend, GATHER_ROWS_KERNEL, [table, idx],
                     [(8, 4), (8,)], ["float16", "int64"], (8, 4), "float16")
    assert np.array_equal(got[:4], table[idx[:4]])
    assert np.array_equal(got[4:], np.broadcast_to(table[0], (4, 4)))


def _ptr_view(ptr, shape, dtype):
    n = int(np.prod(shape)) * np.dtype(dtype).itemsize
    return np.frombuffer((ctypes.c_ubyte * n).from_address(ptr),
                         dtype=np.dtype(dtype)).reshape(shape)


def test_gather_rows_compiled_and_mirror_routes_agree() -> None:
    """编译路由与镜像路由产出逐字节相同的包络：注入伪编译内核对拍两条路。"""
    backend = _backend()
    register_all(backend)
    rng = np.random.default_rng(9)
    table = rng.standard_normal((8, 4)).astype(np.float16)
    idx = np.array([2, 0, 5, 2, 3, 6, 1, 4], dtype=np.int64)
    key = (DEFAULT_OPS_CONTEXT, "gather_rows", (8, 4), (8,), "int64")
    reset_route_counts()
    try:
        def fake(tp, ip, op):  # 模拟编出来的包络整块查表（实参是 c_void_p）
            table_view = _ptr_view(tp.value, (8, 4), np.float16)
            ids_view = _ptr_view(ip.value, (8,), np.int64)
            np.copyto(_ptr_view(op.value, (8, 4), np.float16), table_view[ids_view])
        K._COMPILED_KERNEL_CACHE[key] = fake
        _set_rows(backend, 3)
        hit, _ = _launch(backend, GATHER_ROWS_KERNEL, [table, idx],
                         [(8, 4), (8,)], ["float16", "int64"], (8, 4), "float16")
        assert K.route_counts().get(("gather_rows", "hit"), 0) >= 1

        K._COMPILED_KERNEL_CACHE[key] = None
        _set_rows(backend, 3)
        fell, _ = _launch(backend, GATHER_ROWS_KERNEL, [table, idx],
                          [(8, 4), (8,)], ["float16", "int64"], (8, 4), "float16",
                          cmd_id=1)
        assert K.route_counts().get(("gather_rows", "fallback"), 0) >= 1
        assert np.array_equal(hit, fell)
        assert np.array_equal(hit[:3], table[idx[:3]])
    finally:
        K._COMPILED_KERNEL_CACHE.pop(key, None)


def test_compiled_cache_key_stable_across_rows() -> None:
    """行数不进缓存键：同一包络形状跑不同行数，编译条目数不增长。"""
    def gather_rows_keys():
        return {k for k in K._COMPILED_KERNEL_CACHE
                if isinstance(k, tuple) and len(k) == 5 and k[1] == "gather_rows"}

    rng = np.random.default_rng(11)
    table = rng.standard_normal((8, 4)).astype(np.float16)
    idx = np.arange(8, dtype=np.int64)
    before = gather_rows_keys()
    for rows in (2, 5, 8):
        backend = _backend()  # 换后端模拟每步，moe_rows 各自重设
        register_all(backend)
        _set_rows(backend, rows)
        _launch(backend, GATHER_ROWS_KERNEL, [table, idx],
                [(8, 4), (8,)], ["float16", "int64"], (8, 4), "float16")
    # 行数只影响内核实参：三种行数至多新增一条包络键。
    assert len(gather_rows_keys() - before) <= 1


def test_missing_moe_rows_raises() -> None:
    """dispatch 没写行数就执行行级内核，直接抛不猜包络。"""
    backend = _backend()
    register_all(backend)
    table = np.zeros((4, 4), np.float16)
    idx = np.zeros(4, np.int64)
    with pytest.raises(ValueError, match="moe_rows 里没有"):
        _launch(backend, GATHER_ROWS_KERNEL, [table, idx],
                [(4, 4), (4,)], ["float16", "int64"], (4, 4), "float16")


def test_moe_rows_isolated_by_lane() -> None:
    """行数表按 (泳道, dispatch, 专家) 取键：别的泳道写了也读不到。"""
    backend = _backend()
    register_all(backend)
    table = np.arange(16, dtype=np.float16).reshape(4, 4)
    idx = np.array([1, 0, 2, 3], dtype=np.int64)
    _set_rows(backend, 2, lane_key=1)
    with pytest.raises(ValueError, match="moe_rows 里没有"):
        _launch(backend, GATHER_ROWS_KERNEL, [table, idx],
                [(4, 4), (4,)], ["float16", "int64"], (4, 4), "float16")
    got, _ = _launch(backend, GATHER_ROWS_KERNEL, [table, idx],
                     [(4, 4), (4,)], ["float16", "int64"], (4, 4), "float16",
                     lane_key=1)
    assert np.array_equal(got[:2], table[idx[:2]])


# ---------- scatter_add 内核与多专家写序 ----------


def test_scatter_add_kernel_accumulates_experts_in_waits_order() -> None:
    """两个专家先后散写同一累加器：结果等于专家升序的手写累加。

    两份专家数据用互不重叠的 MRAM 偏移，避免提交后覆盖读数据的竞态。
    """
    backend = _backend()
    register_all(backend)
    rng = np.random.default_rng(13)
    acc = np.zeros((6, 4), np.float16)
    out_e0 = rng.standard_normal((8, 4)).astype(np.float16)
    out_e2 = rng.standard_normal((8, 4)).astype(np.float16)
    w0 = rng.random(8).astype(np.float16)
    w2 = rng.random(8).astype(np.float16)
    idx_e0 = np.array([0, 3, 5, 0, 0, 0, 0, 0], dtype=np.int64)   # 尾部被钳
    idx_e2 = np.array([1, 4, 5, 0, 0, 0, 0, 0], dtype=np.int64)
    # acc@0  out_e0@512  w0@1024  idx_e0@1536  out_e2@2048  w2@2560  idx_e2@3072
    backend.write_local(0, 0, acc)
    for off, data in ((512, out_e0), (1024, w0), (1536, idx_e0),
                      (2048, out_e2), (2560, w2), (3072, idx_e2)):
        backend.write_local(0, off, np.ascontiguousarray(data))
    _set_rows(backend, 3, expert=0)
    _set_rows(backend, 3, expert=2)
    _submit_dispatch(backend)
    args = dict(arg_shapes=[(6, 4), (8, 4), (8,), (8,)],
                arg_dtypes=["float16", "float16", "float16", "int64"],
                out_shape=(6, 4), dtype="float16")

    def expert_cmd(cmd_id, expert, base, waits):
        return Command(
            id=cmd_id, op="launch", dpu_id=0,
            payload={"kernel": SCATTER_ADD_KERNEL, "node": "n",
                     "arg_kinds": ["tensor"] * 4, **args,
                     "gate": 77, "expert": expert},
            reads=[Access(("dpu", 0), 0, acc.nbytes),
                   Access(("dpu", 0), base, out_e0.nbytes),
                   Access(("dpu", 0), base + 512, w0.nbytes),
                   Access(("dpu", 0), base + 1024, idx_e0.nbytes)],
            writes=[Access(("dpu", 0), 0, acc.nbytes)],
            waits=waits, num_tasklets=4)

    backend.submit(expert_cmd(0, 0, 512, []))
    backend.submit(expert_cmd(1, 2, 2048, [0]))
    backend.wait(backend.events_table()[0])
    backend.wait(backend.events_table()[1])
    got = backend.read_local(0, 0, (6, 4), np.float16)
    want = acc.copy()
    for i in range(3):
        want[idx_e0[i]] += w0[i] * out_e0[i]
    for i in range(3):
        want[idx_e2[i]] += w2[i] * out_e2[i]
    assert np.array_equal(got, want.astype(np.float16))


def test_row_kernels_registered_under_contract_names() -> None:
    backend = _backend()
    register_all(backend)
    assert backend._kernels.lookup(GATHER_ROWS_KERNEL).__name__ == "gather_rows_kernel"
    assert backend._kernels.lookup(SCATTER_ADD_KERNEL).__name__ == "scatter_add_kernel"
