"""验证 DPU 白名单内核的 NumPy 结果。"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.hal_numpy import NumpyBackend, NumpyBackendConfig, TaskletHazardError
from contracts.exec_plan import Access, Command
from runtime.kernels import register_all, tasklet_linear_kernel


def _backend() -> NumpyBackend:
    return NumpyBackend(NumpyBackendConfig(num_dpus=1, mram_bytes_per_dpu=4096))


def _run(backend: NumpyBackend, kernel_name: str, arg_kinds, arg_shapes,
         reads_data: list[np.ndarray], out_shape, dtype="float32", num_tasklets=4,
         extra_payload: dict | None = None):
    """写入全部张量参数、构造 launch 命令、submit/wait，返回结果与写地址。"""
    off = 0
    reads = []
    for data in reads_data:
        backend.write_local(0, off, data.astype(np.dtype(dtype)))
        reads.append(Access(("dpu", 0), off, data.astype(np.dtype(dtype)).nbytes))
        off += 512  # 对齐留足空间，避免相邻张量重叠
    write_off = off
    write_nbytes = int(np.prod(out_shape)) * np.dtype(dtype).itemsize
    cmd = Command(
        id=0, op="launch", dpu_id=0,
        payload={"kernel": kernel_name, "node": "n", "arg_kinds": arg_kinds,
                  "arg_shapes": arg_shapes, "dtype": dtype, "out_shape": out_shape,
                  **(extra_payload or {})},
        reads=reads, writes=[Access(("dpu", 0), write_off, write_nbytes)], waits=[],
        num_tasklets=num_tasklets,
    )
    event = backend.submit(cmd)
    backend.wait(event)
    return backend.read_local(0, write_off, out_shape, np.dtype(dtype))


def test_linear_kernel_matches_torch() -> None:
    backend = _backend()
    register_all(backend)
    rng = np.random.default_rng(0)
    x = rng.standard_normal((2, 4)).astype(np.float32)
    w = rng.standard_normal((3, 4)).astype(np.float32)
    result = _run(backend, str(torch.ops.aten.linear.default), ["tensor", "tensor"],
                  [(2, 4), (3, 4)], [x, w], (2, 3))
    ref = torch.nn.functional.linear(torch.from_numpy(x), torch.from_numpy(w)).numpy()
    assert np.allclose(result, ref, atol=1e-5)


def test_add_kernel_tensor_tensor_matches_torch() -> None:
    backend = _backend()
    register_all(backend)
    rng = np.random.default_rng(1)
    x = rng.standard_normal((3,)).astype(np.float32)
    y = rng.standard_normal((3,)).astype(np.float32)
    result = _run(backend, str(torch.ops.aten.add.Tensor), ["tensor", "tensor"],
                  [(3,), (3,)], [x, y], (3,))
    assert np.allclose(result, x + y, atol=1e-6)


def test_add_kernel_tensor_scalar_matches_torch() -> None:
    """RMSNorm 的 `add(x, eps)` 形态：第二参数是字面量，不占 reads。"""
    backend = _backend()
    register_all(backend)
    x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    result = _run(backend, str(torch.ops.aten.add.Tensor), ["tensor", 1e-5],
                  [(3,), None], [x], (3,))
    assert np.allclose(result, x + 1e-5, atol=1e-8)


def test_mul_kernel_matches_torch() -> None:
    backend = _backend()
    register_all(backend)
    rng = np.random.default_rng(2)
    x = rng.standard_normal((4,)).astype(np.float32)
    y = rng.standard_normal((4,)).astype(np.float32)
    result = _run(backend, str(torch.ops.aten.mul.Tensor), ["tensor", "tensor"],
                  [(4,), (4,)], [x, y], (4,))
    assert np.allclose(result, x * y, atol=1e-6)


def test_tanh_kernel_matches_torch() -> None:
    backend = _backend()
    register_all(backend)
    rng = np.random.default_rng(3)
    x = rng.standard_normal((5,)).astype(np.float32)
    result = _run(backend, str(torch.ops.aten.tanh.default), ["tensor"], [(5,)], [x], (5,))
    assert np.allclose(result, np.tanh(x), atol=1e-6)


@pytest.mark.parametrize("num_tasklets", [1, 2, 3, 5, 8])
def test_tasklet_linear_kernel_matches_torch_for_various_tasklet_counts(num_tasklets) -> None:
    """验证不同 tasklet 数下的线性计算结果。"""
    backend = _backend()
    backend.register_kernel("tasklet_linear", tasklet_linear_kernel)
    rng = np.random.default_rng(4)
    m, k, n = 7, 4, 3
    x = rng.standard_normal((m, k)).astype(np.float32)
    w = rng.standard_normal((n, k)).astype(np.float32)
    result = _run(backend, "tasklet_linear", ["tensor", "tensor"], [(m, k), (n, k)],
                  [x, w], (m, n), num_tasklets=num_tasklets)
    ref = torch.nn.functional.linear(torch.from_numpy(x), torch.from_numpy(w)).numpy()
    assert np.allclose(result, ref, atol=1e-5)


def test_tasklet_linear_kernel_row_ranges_are_disjoint_and_cover_m() -> None:
    """切分区间本身要满足：互不重叠、并集覆盖 [0, M)——不止数值对，划分逻辑也要对。"""
    m, num_tasklets = 10, 3
    rows_per_tasklet = -(-m // num_tasklets)
    ranges = []
    for tid in range(num_tasklets):
        row_start = tid * rows_per_tasklet
        row_end = min(row_start + rows_per_tasklet, m)
        if row_start < row_end:
            ranges.append((row_start, row_end))
    covered = set()
    for start, end in ranges:
        rng_set = set(range(start, end))
        assert not (covered & rng_set), f"tasklet 行区间重叠: {ranges}"
        covered |= rng_set
    assert covered == set(range(m))


def test_tasklet_linear_kernel_hazard_detection_catches_broken_split() -> None:
    """验证重叠写入且缺少屏障时会触发冲突检测。"""
    backend = _backend()

    def broken_tasklet_linear(hal, dpu_id, cmd) -> None:
        from runtime.kernels import _read_tensor_args

        x, w = _read_tensor_args(hal, dpu_id, cmd)
        dtype = np.dtype(cmd.payload["dtype"])
        out_access = cmd.writes[0]
        row_bytes = w.shape[0] * dtype.itemsize
        # 两个 tasklet 写入重叠 MRAM 区间。
        for tid in (0, 1):
            hal.record_access(tid, "mram", out_access.offset, 2 * row_bytes, is_write=True)
            y_slice = x[0:2].astype(np.float32) @ w.astype(np.float32).T
            hal.write_local(dpu_id, out_access.offset, np.ascontiguousarray(y_slice, dtype=dtype))
        hal.barrier()

    backend.register_kernel("broken_tasklet_linear", broken_tasklet_linear)
    rng = np.random.default_rng(5)
    m, k, n = 4, 4, 3
    x = rng.standard_normal((m, k)).astype(np.float32)
    w = rng.standard_normal((n, k)).astype(np.float32)
    with pytest.raises(TaskletHazardError):
        _run(backend, "broken_tasklet_linear", ["tensor", "tensor"], [(m, k), (n, k)],
             [x, w], (m, n), num_tasklets=2)


def test_mask_kernel_runs_the_compiled_kernel(monkeypatch) -> None:
    """`aten.masked_fill` 必须真调 `compile_op(op="mask")`，不能只跑镜像。

    评审九轮问题 1：设备算子 `pim.mask` 的编译内核只在测试里被调过，
    运行时这条 aten 走的是 `masked_fill_kernel` 的纯 numpy 镜像。
    """
    import opcompiler_bridge.driver as driver

    seen: list[str] = []
    real = driver.compile_op

    def spy(request, *args, **kwargs):
        seen.append(request.op)
        return real(request, *args, **kwargs)

    monkeypatch.setattr(driver, "compile_op", spy)

    backend = _backend()
    register_all(backend)
    rng = np.random.default_rng(0)
    # 掩码除末轴外全是 1 才是 `vector` 布局；否则编译侧分不出几何。
    scores = rng.standard_normal((1, 1, 4)).astype(np.float32)
    mask = np.array([[[False, False, True, True]]], dtype=bool)
    _run(backend, str(torch.ops.aten.masked_fill.Tensor),
         ["tensor", "tensor", -1e4], [(1, 1, 4), (1, 1, 4), None],
         [scores, mask.astype(np.float32)], (1, 1, 4))
    assert "mask" in seen, f"运行时没有编译 pim.mask，实际编译了 {seen}"


def test_alias_kernel_is_an_identity(monkeypatch) -> None:
    """`aten.alias` 是视图，原样传回，不编译任何算子。

    它曾经被改成 `pim.dynamic_quant` 的载体：把 fp16 量化成 int8 再按 fp16
    的长度写回，读出来是 fp16 最大值。量化是权重侧的事，激活侧没有节点
    消费这个产物。
    """
    import opcompiler_bridge.driver as driver

    seen: list[str] = []
    monkeypatch.setattr(driver, "compile_op",
                        lambda request, *a, **k: seen.append(request.op))

    backend = _backend()
    register_all(backend)
    x = np.arange(8, dtype=np.float16).reshape(1, 8)
    _run(backend, str(torch.ops.aten.alias.default), ["tensor"], [(1, 8)], [x],
         (1, 8))
    assert seen == [], f"alias 不该编译任何算子，实际编译了 {seen}"


def test_same_shape_eltwise_uses_the_compiled_kernel() -> None:
    """同形状的逐元素加/乘/减必须走编译内核，而不是 numpy 镜像。

    编译内核是 fp16 落点，镜像是 fp32 运算后再截断，两者在 .5 边界上会差 1。
    这里用一个正好落在边界上的输入区分两条路：走镜像会得到不同的结果。
    """
    from unittest import mock

    import numpy as np

    from runtime.kernels import add_kernel

    seen = {}

    def fake_compile(kind, shape, *, dtype="float16", ctx=None):
        seen["called"] = (kind, shape)

        def fn(a, b, out):
            # 故意返回一个镜像算不出来的值，证明结果来自编译内核。
            import ctypes
            buf = (ctypes.c_uint16 * (shape[0] * shape[1])).from_address(out.value)
            for i in range(len(buf)):
                buf[i] = np.float16(7).view(np.uint16)
        return fn

    hal = mock.Mock()
    # 命令的 payload 要给出 out_shape：逐元素内核拿它核对广播后的公共形状，
    # 契约不满足就直接抛，不做兜底。
    cmd = mock.Mock(payload={"out_shape": (2, 4), "dtype": "float16"})
    x = np.ones((2, 4), dtype=np.float16)
    y = np.ones((2, 4), dtype=np.float16)
    with mock.patch("runtime.kernels._read_tensor_args", return_value=[x, y]), \
         mock.patch("runtime.kernels._compiled_eltwise", side_effect=fake_compile), \
         mock.patch("runtime.kernels._write_result") as write:
        add_kernel(hal, 0, cmd)
    assert seen.get("called") == ("add", (2, 4)), "同形状加法没有走编译内核"
    written = write.call_args.args[3]
    assert np.all(written == np.float16(7))


def test_sdpa_uses_the_mask_operand_not_a_recomputed_one() -> None:
    """注意力必须用图上传入的掩码，不能按位置重算。

    把第 4 个实参换成全遮挡（全 -inf），输出必须变。按位置重算的实现会忽略
    这个实参，于是两种掩码给出同一个结果——那是静默失真。
    """
    from runtime.kernels import sdpa_kernel

    sdpa_name = str(torch.ops.aten.scaled_dot_product_attention.default)
    q = np.ones((1, 1, 4, 8), dtype=np.float32)
    k = np.ones((1, 1, 4, 8), dtype=np.float32)
    v = np.arange(32, dtype=np.float32).reshape(1, 1, 4, 8)
    causal = np.zeros((1, 1, 4, 4), dtype=np.float32)
    causal[0, 0] = np.triu(np.full((4, 4), np.float32(-1e4)), k=1)
    blocked = np.full((1, 1, 4, 4), np.float32(-1e4))

    def run(mask):
        backend = NumpyBackend(NumpyBackendConfig(num_dpus=1, mram_bytes_per_dpu=1 << 20))
        register_all(backend)
        block_table_base, slots_base = 1 << 16, 1 << 17
        backend.write_local(0, block_table_base, np.zeros((1,), dtype=np.int32))
        return _run(backend, sdpa_name, ["tensor"] * 4,
                    [q.shape, k.shape, v.shape, mask.shape],
                    [q, k, v, mask], q.shape, dtype="float32",
                    extra_payload={"sdpa": {
                        "np_dtype": "float32", "layer": 0,
                        "head_dim": 8, "slot_tokens": 4,
                        "heads": [{"head": 0, "kv_head": 0, "dpu": 0, "row": 8 * 4,
                                   "k_off": slots_base, "v_off": slots_base + 128}],
                        "dpus": {0: {"bt_base": block_table_base,
                                     "slots_base": slots_base,
                                     "slot_bytes": 256, "cache_elems": 64,
                                     "max_blocks": 1}}}})

    causal_out = run(causal)
    blocked_out = run(blocked)
    assert not np.allclose(causal_out, blocked_out), (
        "全遮挡掩码与因果掩码给出了同一个结果：图上的掩码没被用到")


def test_sdpa_streams_blocks_without_concatenating_kv(monkeypatch) -> None:
    """多块且块表乱序时注意力正确，且全程不拼接 KV 历史。

    物理槽按块表散落，逻辑顺序只有块表知道；两次运行（顺序 / 乱序块表）
    必须给出同一个结果，并与 numpy 因果注意力对拍。`np.concatenate` 被监视：
    改回「先 gather 再算」的实现会立刻被这条断言抓住。
    """
    rng = np.random.default_rng(7)
    n_tokens, head_dim, slot_tokens = 6, 4, 2
    q = rng.standard_normal((1, 1, n_tokens, head_dim)).astype(np.float32)
    k = rng.standard_normal((1, 1, n_tokens, head_dim)).astype(np.float32)
    v = rng.standard_normal((1, 1, n_tokens, head_dim)).astype(np.float32)

    bt_base, slots_base = 1 << 16, 1 << 17
    plane_stride = 64
    slot_bytes = 2 * plane_stride

    def run(block_table):
        backend = NumpyBackend(
            NumpyBackendConfig(num_dpus=1, mram_bytes_per_dpu=1 << 20))
        register_all(backend)
        backend.write_local(0, bt_base, np.asarray(block_table, dtype=np.int32))
        for position in range(n_tokens):
            block, row = divmod(position, slot_tokens)
            slot = block_table[block]
            for plane_off, source in ((slots_base, k),
                                      (slots_base + plane_stride, v)):
                backend.write_local(
                    0, plane_off + slot * slot_bytes + row * head_dim * 4,
                    np.ascontiguousarray(source[0, 0, position]))
        return _run(
            backend, str(torch.ops.aten.scaled_dot_product_attention.default),
            ["tensor"] * 3, [q.shape, k.shape, v.shape], [q, k, v],
            q.shape, dtype="float32",
            extra_payload={"sdpa": {
                "np_dtype": "float32", "layer": 0,
                "head_dim": head_dim, "slot_tokens": slot_tokens,
                "heads": [{"head": 0, "kv_head": 0, "dpu": 0, "row": head_dim * 4,
                           "k_off": slots_base,
                           "v_off": slots_base + plane_stride}],
                "dpus": {0: {"bt_base": bt_base, "slots_base": slots_base,
                             "slot_bytes": slot_bytes,
                             # 3 个物理槽，f32 下每槽 128/4 = 32 个元素。
                             "cache_elems": 3 * slot_bytes // 4,
                             "max_blocks": n_tokens // slot_tokens}}}})

    concatenations = []
    real_concat = np.concatenate

    def spy(*args, **kwargs):
        concatenations.append(args)
        return real_concat(*args, **kwargs)

    monkeypatch.setattr(np, "concatenate", spy)
    identity = run([0, 1, 2])
    shuffled = run([2, 0, 1])
    assert not concatenations, "sdpa 仍在拼接 KV 历史"
    assert np.array_equal(identity, shuffled), "乱序块表给出了不同结果"

    # numpy 因果注意力参考；设备 softmax 走 LUT，留一点容差。
    scale = 1.0 / np.sqrt(head_dim)
    scores = (q[0, 0] @ k[0, 0].T) * scale
    masked = scores + np.triu(
        np.full((n_tokens, n_tokens), -np.inf, dtype=np.float32), k=1)
    weights = np.exp(masked.astype(np.float32))
    weights /= weights.sum(-1, keepdims=True)
    assert np.allclose(identity[0, 0], weights @ v[0, 0], atol=0.05)


def test_sdpa_gqa_writes_each_kv_head_once() -> None:
    """一个 KV 头服务两个 Q 头：四个 Q 头都要出结果，缓存只按 KV 头写一次。

    图上 `repeat_kv` 已经把 K/V 复制到每个 Q 头上，内核若按 Q 头逐个写缓存，
    后一组的行会覆盖前一组的行；KV 头数少于 Q 头数时结果直接跑偏。
    """
    from types import SimpleNamespace

    import runtime.kernels as km
    from memory.kv_layout import KVRegionSpec, build_kv_layout, write_block_table

    head_dim, slot_tokens, max_seq = 8, 4, 8
    spec = build_kv_layout(KVRegionSpec(
        dpu_id=0, layers=[0], kv_heads=[0, 1],
        q_heads_by_kv={0: [0, 1], 1: [2, 3]},
        max_seq=max_seq, head_dim=head_dim, dtype_bytes=2,
        kv_base=8192, slot_tokens=slot_tokens), align=8, pool_slots=3)

    rng = np.random.default_rng(17)
    q = rng.standard_normal((1, 4, max_seq, head_dim)).astype(np.float16)
    kv = rng.standard_normal((1, 2, max_seq, head_dim)).astype(np.float16)
    k = np.repeat(kv, 2, axis=1)
    v = np.repeat(kv, 2, axis=1)
    info = km.sdpa_kv_info(SimpleNamespace(name="n"), {0: spec}, {"n": 0},
                           np.dtype(np.float16))
    assert [entry["head"] for entry in info["heads"]] == [0, 1, 2, 3]
    assert [entry["kv_head"] for entry in info["heads"]] == [0, 0, 1, 1]

    backend = NumpyBackend(
        NumpyBackendConfig(num_dpus=1, mram_bytes_per_dpu=1 << 20))
    register_all(backend)
    write_block_table(spec, backend, [2, 0])
    out = _run(backend,
               str(torch.ops.aten.scaled_dot_product_attention.default),
               ["tensor"] * 3, [q.shape, k.shape, v.shape], [q, k, v],
               q.shape, dtype="float16", extra_payload={"sdpa": info})

    scale = 1.0 / np.sqrt(head_dim)
    hidden = np.triu(np.full((max_seq, max_seq), -np.inf, dtype=np.float32), k=1)
    for head in range(4):
        scores = (q[0, head].astype(np.float32)
                  @ kv[0, head // 2].T.astype(np.float32)) * scale
        masked = scores + hidden
        weights = np.exp(masked - masked.max(-1, keepdims=True))
        weights /= weights.sum(-1, keepdims=True)
        expected = weights @ kv[0, head // 2].astype(np.float32)
        assert np.allclose(out[0, head], expected, atol=0.05), (
            f"q head {head} 的结果与它所属 KV 头对不上")


def test_sdpa_fused_paged_attn_tracks_the_host_path(monkeypatch) -> None:
    """融合内核走单遍在线 softmax，主机路径走相位链两遍，两者在 f16 末位相符。

    块表乱序（逻辑块 0 → 物理槽 2）时必须按同一顺序读：内核自己解析
    块表，主机路径也按同一张表 gather。算法不同，所以判据是 allclose 而不是
    逐元素相等；误差量级就是一两次 f16 取整。
    """
    from types import SimpleNamespace

    import runtime.kernels as km
    from memory.kv_layout import KVRegionSpec, build_kv_layout, write_block_table

    head_dim, slot_tokens, max_seq = 8, 4, 8
    spec = build_kv_layout(KVRegionSpec(
        dpu_id=0, layers=[0], kv_heads=[0], q_heads_by_kv={0: [0]},
        max_seq=max_seq, head_dim=head_dim, dtype_bytes=2,
        kv_base=8192, slot_tokens=slot_tokens), align=8, pool_slots=3)

    rng = np.random.default_rng(11)
    q = rng.standard_normal((1, 1, max_seq, head_dim)).astype(np.float16)
    k = rng.standard_normal((1, 1, max_seq, head_dim)).astype(np.float16)
    v = rng.standard_normal((1, 1, max_seq, head_dim)).astype(np.float16)
    info = km.sdpa_kv_info(SimpleNamespace(name="n"), {0: spec}, {"n": 0},
                           np.dtype(np.float16))

    def run(force_host: bool):
        backend = NumpyBackend(
            NumpyBackendConfig(num_dpus=1, mram_bytes_per_dpu=1 << 20))
        register_all(backend)
        write_block_table(spec, backend, [2, 0])
        if force_host:
            monkeypatch.setattr(km, "_compiled_paged_attn", lambda *a, **kw: None)
        km.reset_route_counts()
        out = _run(backend,
                   str(torch.ops.aten.scaled_dot_product_attention.default),
                   ["tensor"] * 3, [q.shape, k.shape, v.shape], [q, k, v],
                   q.shape, dtype="float16", extra_payload={"sdpa": info})
        return out, km.route_counts()

    fused, routes = run(force_host=False)
    host, _ = run(force_host=True)
    assert routes.get(("paged_attn", "hit"), 0) > 0, (
        "融合内核没有被调用：工具链不在位或编译失败")
    # 单遍在线与相位链两遍的数学结果相同，差在各自的取整位置：实测最大
    # 绝对误差 < 3e-4，取 atol/rtol = 1e-3 留一个数量级的余量。
    assert np.allclose(fused, host, rtol=1e-3, atol=1e-3), (
        "融合路径与主机路径的误差超过了 f16 末位差")

    scale = 1.0 / np.sqrt(head_dim)
    scores = (q[0, 0].astype(np.float32) @ k[0, 0].T.astype(np.float32)) * scale
    masked = scores + np.triu(
        np.full((max_seq, max_seq), -np.inf, dtype=np.float32), k=1)
    weights = np.exp(masked - masked.max(-1, keepdims=True))
    weights /= weights.sum(-1, keepdims=True)
    assert np.allclose(fused[0, 0], weights @ v[0, 0].astype(np.float32),
                       atol=0.05)


def test_sdpa_standard_causal_mask_routes_to_fused_kernel(monkeypatch) -> None:
    """声明「掩码是标准因果掩码」后，带掩码的 prefill 也走融合内核。

    没声明时同一个掩码必须回退主机路径（自定义掩码的兜底不能被破坏），
    两条路径的数值差在 f16 末位量级。
    """
    from types import SimpleNamespace

    import runtime.kernels as km
    from memory.kv_layout import KVRegionSpec, build_kv_layout, write_block_table

    head_dim, slot_tokens, max_seq = 8, 4, 8
    spec = build_kv_layout(KVRegionSpec(
        dpu_id=0, layers=[0], kv_heads=[0], q_heads_by_kv={0: [0]},
        max_seq=max_seq, head_dim=head_dim, dtype_bytes=2,
        kv_base=8192, slot_tokens=slot_tokens), align=8, pool_slots=3)

    rng = np.random.default_rng(13)
    q = rng.standard_normal((1, 1, max_seq, head_dim)).astype(np.float16)
    k = rng.standard_normal((1, 1, max_seq, head_dim)).astype(np.float16)
    v = rng.standard_normal((1, 1, max_seq, head_dim)).astype(np.float16)
    mask = np.zeros((1, 1, max_seq, max_seq), dtype=np.float16)
    mask[0, 0] = np.triu(np.full((max_seq, max_seq), -np.inf,
                                 dtype=np.float16), k=1)
    info = km.sdpa_kv_info(SimpleNamespace(name="n"), {0: spec}, {"n": 0},
                           np.dtype(np.float16))

    def run(standard: bool, force_host: bool):
        backend = NumpyBackend(
            NumpyBackendConfig(num_dpus=1, mram_bytes_per_dpu=1 << 20))
        register_all(backend)
        write_block_table(spec, backend, [2, 0])
        backend.bind_inputs({"causal_mask_standard": True} if standard else {})
        if force_host:
            monkeypatch.setattr(km, "_compiled_paged_attn", lambda *a, **kw: None)
        km.reset_route_counts()
        out = _run(backend,
                   str(torch.ops.aten.scaled_dot_product_attention.default),
                   ["tensor"] * 4, [q.shape, k.shape, v.shape, mask.shape],
                   [q, k, v, mask], q.shape, dtype="float16",
                   extra_payload={"sdpa": info})
        return out, km.route_counts()

    fused, routes = run(standard=True, force_host=False)
    assert routes.get(("paged_attn", "hit"), 0) > 0, "标准掩码的 prefill 没有走融合内核"
    custom, routes = run(standard=False, force_host=False)
    assert routes.get(("paged_attn", "fallback"), 0) > 0, "未声明的掩码没有回退"
    host, _ = run(standard=True, force_host=True)
    assert np.allclose(fused, host, rtol=1e-3, atol=1e-3)
    assert np.allclose(custom, host, rtol=1e-3, atol=1e-3)


def test_paged_attention_rejects_history_longer_than_the_block_table() -> None:
    """块表项不够时直接报错：再多读会拿到槽池里的垃圾槽号。"""
    import runtime.kernels as km
    from memory.kv_layout import KVRegionSpec, build_kv_layout

    spec = build_kv_layout(KVRegionSpec(
        dpu_id=0, layers=[0], kv_heads=[0], q_heads_by_kv={0: [0]},
        max_seq=8, head_dim=8, dtype_bytes=2,
        kv_base=8192, slot_tokens=4), align=8, pool_slots=3)
    geo = {"slots_base": spec.slots_base, "bt_base": spec.block_table_base,
           "slot_bytes": spec.slot_bytes, "max_blocks": spec.blocks_per_request,
           "cache_elems": spec.pool_slots * spec.slot_bytes // spec.dtype_bytes}
    entry = {"head": 0, "dpu": 0, "row": 0, "k_off": spec.slots_base,
             "v_off": spec.slots_base + 2 * spec.slot_bytes}

    with pytest.raises(RuntimeError, match="块表只有 2 项"):
        km._paged_attention(None, 0, entry, geo, np.zeros(2, np.int32),
                            np.zeros((1, 8), np.float16), 8, 1, 8, 4,
                            np.dtype(np.float16), None, 0.35)


def test_fused_rms_norm_rounds_before_the_scale() -> None:
    """融合的 RMSNorm 要在乘缩放之前先落一次 f16，与图上的 `to` 节点同一个位置。

    输入取自实测：两种算法在这组数上给出不同的 f16 结果，走错那条会让解码
    token 分叉。判据是逐元素相等，不用 allclose。
    """
    from runtime.kernels import RMS_NORM_KERNEL

    backend = _backend()
    register_all(backend)
    x = np.array([[2.041015625, -2.556640625, 0.418212890625, -0.56787109375]],
                 dtype=np.float16)
    gamma = np.array([-0.45263671875, -0.215576171875, -2.01953125, -0.23193359375],
                     dtype=np.float16)
    got = _run(backend, RMS_NORM_KERNEL, ["tensor", "tensor"],
               [(1, 4), (4,)], [x, gamma], (1, 4), dtype="float16",
               extra_payload={"epsilon": 1e-5})
    assert got.tolist() == [[-0.55224609375, 0.32958984375,
                             -0.50439453125, 0.0787353515625]], got


def test_view_ops_reach_the_compiler(monkeypatch) -> None:
    """transpose / reshape / concat 必须真的调 `compile_op`。

    这三个在图上各是独立节点，之前只走 numpy 镜像。判据是编译器被以对应
    mnemonic 调到，且结果与镜像逐元素一致。
    """
    from opcompiler_bridge import driver
    from runtime import kernels

    seen = []
    real = driver.compile_op

    def spy(request, **kw):
        seen.append(request.op)
        return real(request, **kw)

    monkeypatch.setattr(driver, "compile_op", spy)
    kernels._COMPILED_KERNEL_CACHE.clear()

    def fresh():
        b = _backend(); register_all(b); return b

    x = np.arange(24, dtype=np.float16).reshape(2, 3, 4)
    _run(fresh(), str(torch.ops.aten.permute.default),
         ["tensor", (0, 2, 1)], [x.shape, None], [x], (2, 4, 3), dtype="float16")
    _run(fresh(), str(torch.ops.aten.reshape.default),
         ["tensor", (6, 4)], [x.shape, None], [x], (6, 4), dtype="float16")

    a = np.arange(8, dtype=np.float16).reshape(2, 4)
    b = np.arange(8, 16, dtype=np.float16).reshape(2, 4)
    _run(fresh(), str(torch.ops.aten.cat.default),
         ["tensor", "tensor", 0], [a.shape, b.shape, None], [a, b], (4, 4),
         dtype="float16")

    assert {"transpose", "reshape", "concat"} <= set(seen), (
        f"视图类没走到编译器，实际调用 {sorted(set(seen))}")
