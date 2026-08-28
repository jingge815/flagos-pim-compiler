"""算子编译器桥接（`opcompiler_bridge`）单测：`compiled_linear_kernel` 对拍
`linear_kernel`（现有纯 NumPy 实现）。

跟 `tests/test_kernels.py` 的 `test_linear_kernel_matches_torch` 用同一套
"直接构造 Command，不经 exec_plan_gen"判据，只是换成算子编译器产物。

**需要 GPU**（`compile_op` 第一次为某个 shape 编译时会在 GPU 上跑一次目标
kernel 拿 TTIR，见 `opcompiler_bridge/driver.py`）。CI/无 GPU 环境下用
`pytest.mark.skipif(not torch.cuda.is_available())` 跳过。不需要
`genesim_bridge.env.prepare_triton_env`——这台机器现在只有一份 triton
安装，默认自带 PIM pass 支持（见 `opcompiler_bridge/driver.py` 模块
docstring 的说明）。

K 必须 >= 16（`tl.dot` 的硬约束，见 `opcompiler_bridge/kernel_src.py`），跟
`test_kernels.py` 里 K=4 的手写 NumPy 对拍用例不是同一组数据。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.hal_numpy import NumpyBackend, NumpyBackendConfig
from contracts.exec_plan import Access, Command
from contracts.op_contract import PIMHardwareConfig
from runtime.kernels import compiled_linear_kernel, linear_kernel, register_all

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="opcompiler_bridge.driver 编译时需要在 GPU 上跑一次目标 kernel",
)


def _backend(mram_bytes: int = 1 << 20) -> NumpyBackend:
    return NumpyBackend(
        NumpyBackendConfig(num_dpus=1, mram_bytes_per_dpu=mram_bytes)
    )


_ALIGN = 4096


def _run(backend: NumpyBackend, kernel_name: str, arg_shapes,
         reads_data: list[np.ndarray], out_shape, dtype="float32",
         hardware: PIMHardwareConfig | None = None):
    """写入全部张量、构造 launch 命令、submit/wait，返回结果。

    张量之间按 `_ALIGN` 向上对齐排布（不是固定步长——真实 shape 下 weight 有
    好几 MB，固定 4096 会让相邻张量重叠）。
    """
    off = 0
    reads = []
    for data in reads_data:
        blob = data.astype(np.dtype(dtype))
        backend.write_local(0, off, blob)
        reads.append(Access(("dpu", 0), off, blob.nbytes))
        off += -(-blob.nbytes // _ALIGN) * _ALIGN
    write_off = off
    write_nbytes = int(np.prod(out_shape)) * np.dtype(dtype).itemsize
    if hardware is None:
        hardware = PIMHardwareConfig(
            num_dpus=1,
            num_tasklets=4,
            mram_bytes_per_dpu=backend.config.mram_bytes_per_dpu,
            wram_bytes_per_dpu=64 * 1024,
            # 真实 DMA 对齐要求是字节级的小数字，跟 `_ALIGN` 这个 MRAM 里
            # 张量摆放用的 4096B 页对齐是两个不同概念——用 `_ALIGN` 会导致
            # 所有 tile（几十到几百字节）都过不了 `pim-tile-to-budget` 的
            # DMA 对齐检查。这里取 FlagTree `kDefaultDmaAlign` 同样的 8
            # 字节，本文件里最小的测试 shape（M=2/K=16/N=4）也能整除。
            dma_align=8,
        )
    cmd = Command(
        id=0, op="launch", dpu_id=0,
        payload={"kernel": kernel_name, "node": "n",
                  "arg_kinds": ["tensor", "tensor"], "arg_shapes": arg_shapes,
                  "dtype": dtype, "out_shape": out_shape,
                  "hardware": hardware.to_payload()},
        reads=reads, writes=[Access(("dpu", 0), write_off, write_nbytes)], waits=[],
        num_tasklets=hardware.num_tasklets,
    )
    event = backend.submit(cmd)
    backend.wait(event)
    return backend.read_local(0, write_off, out_shape, np.dtype(dtype))


def test_compile_request_uses_explicit_hardware_budget(monkeypatch) -> None:
    from contracts.op_contract import OpCompileRequest
    import opcompiler_bridge.driver as driver

    seen = {}

    def fake_make_ttir(request):
        return "module {}"

    def fake_run(ttir, hardware):
        seen["hardware"] = hardware
        return (
            "module { func.func @k(%a: !emitc.ptr<f32>, %b: !emitc.ptr<f32>, "
            "%c: !emitc.ptr<f32>) { return } }"
        )

    monkeypatch.setattr(driver, "_make_ttir", fake_make_ttir)
    monkeypatch.setattr(driver, "_run_triton_opt", fake_run)
    monkeypatch.setattr(driver, "_translate_to_c", lambda _: "void k(float *a, float *b, float *c) {}")

    def fake_subprocess_run(cmd, *args, **kwargs):
        if cmd and cmd[0] == "gcc":
            Path(cmd[5]).write_bytes(b"")
        return type("P", (), {"returncode": 0, "stderr": ""})()

    monkeypatch.setattr(driver.subprocess, "run", fake_subprocess_run)

    hw = PIMHardwareConfig(1, 7, 1 << 20, 4096, 256)
    result = driver.compile_op(
        OpCompileRequest("linear", [(1, 16), (4, 16)], hw, "float32"),
        force=True,
    )

    assert result.symbol == "k"
    assert seen["hardware"] == hw


def test_compiled_linear_matches_handwritten_numpy() -> None:
    """编译产物与手写 `linear_kernel` 在相同随机输入下逐元素一致。"""
    rng = np.random.default_rng(0)
    M, K, N = 2, 16, 4
    x = rng.standard_normal((M, K)).astype(np.float32)
    w = rng.standard_normal((N, K)).astype(np.float32)

    hand_backend = _backend()
    register_all(hand_backend)
    hand_result = _run(
        hand_backend, str(torch.ops.aten.linear.default), [(M, K), (N, K)],
        [x, w], (M, N),
    )

    compiled_backend = _backend()
    register_all(compiled_backend, use_compiled_linear=True)
    compiled_result = _run(
        compiled_backend, str(torch.ops.aten.linear.default), [(M, K), (N, K)],
        [x, w], (M, N),
    )

    np.testing.assert_allclose(compiled_result, hand_result, atol=1e-4)


@pytest.mark.parametrize("num_tasklets", [1, 2, 4, 8])
def test_compiled_linear_matches_torch_across_tasklet_counts(num_tasklets) -> None:
    """算子编译器泛化后的 `pim-lower-to-emitc`（原 pim-lower-single-tasklet）
    按 num_tasklets 把 M 维静态切分——覆盖 1（退化回归）/2/4（整除）/8（大于
    M，尾部 tasklet 空转）四档，每档都要跟 torch 参考逐元素对齐，这是方案
    约束 4"必须显式跑通 num_tasklets>1"在算子编译器这一层的直接验证。
    """
    rng = np.random.default_rng(1)
    M, K, N = 4, 32, 8
    x = rng.standard_normal((M, K)).astype(np.float32)
    w = rng.standard_normal((N, K)).astype(np.float32)

    backend = _backend()
    register_all(backend, use_compiled_linear=True)
    result = _run(
        backend, str(torch.ops.aten.linear.default), [(M, K), (N, K)],
        [x, w], (M, N),
        hardware=PIMHardwareConfig(1, num_tasklets, 1 << 20, 64 * 1024, 64),
    )
    ref = torch.nn.functional.linear(torch.from_numpy(x), torch.from_numpy(w)).numpy()
    np.testing.assert_allclose(result, ref, atol=1e-4)


def test_compiled_linear_matches_torch() -> None:
    rng = np.random.default_rng(1)
    M, K, N = 4, 32, 8
    x = rng.standard_normal((M, K)).astype(np.float32)
    w = rng.standard_normal((N, K)).astype(np.float32)

    backend = _backend()
    register_all(backend, use_compiled_linear=True)
    result = _run(
        backend, str(torch.ops.aten.linear.default), [(M, K), (N, K)],
        [x, w], (M, N),
    )
    ref = torch.nn.functional.linear(torch.from_numpy(x), torch.from_numpy(w)).numpy()
    np.testing.assert_allclose(result, ref, atol=1e-4)


@pytest.mark.parametrize(
    "M,K,N",
    [
        (1, 4096, 512),   # decode 的 q/k/v_proj（8 卡切分后的本地分片）
        (1, 512, 4096),   # decode 的 o_proj（行切）
        (2, 16, 4),       # 小 shape，跑得快，覆盖同一条路径
    ],
)
def test_compiled_linear_float16_matches_handwritten_numpy(M, K, N) -> None:
    """**float16 存储**下编译产物与手写 `linear_kernel` 一致。

    这一组是真实 llama2-7b 的实际配置（模型是 fp16、MRAM 里 2 字节/元素），
    也是当初漏掉 dtype 契约时唯一能暴露问题的场景：f32 的用例全都过、只有
    真实模型端到端才发现生成文本分叉（详见 contracts/op_contract.py 的说明）。
    C 里没有可移植的 half 类型，编译产物用 `uint16_t` 存储 + 位转换 helper，
    所以这里比 f32 用例多一层需要验证的东西。

    判据是跟 `linear_kernel` 对齐而不是跟 torch 对齐：两者都"按 f16 读、
    用 f32 算、窄回 f16 写"，但归约顺序不同，f16 的有效位数只有 ~11 bit，
    逐元素完全相等不现实——取 2% 的相对容差，K=4096 时累积误差也在这个量级内。
    """
    rng = np.random.default_rng(3)
    x = rng.standard_normal((M, K)).astype(np.float16)
    w = rng.standard_normal((N, K)).astype(np.float16)
    # 真实 shape 的 weight 有几 MB，MRAM 要按实际大小给（外加对齐余量）
    mram = max(1 << 20, (x.nbytes + w.nbytes + M * N * 2) * 2 + 3 * _ALIGN)

    hand = _backend(mram)
    register_all(hand)
    hand_result = _run(
        hand, str(torch.ops.aten.linear.default), [(M, K), (N, K)],
        [x, w], (M, N), dtype="float16",
    )

    compiled = _backend(mram)
    register_all(compiled, use_compiled_linear=True)
    compiled_result = _run(
        compiled, str(torch.ops.aten.linear.default), [(M, K), (N, K)],
        [x, w], (M, N), dtype="float16",
    )

    np.testing.assert_allclose(
        compiled_result.astype(np.float32),
        hand_result.astype(np.float32),
        rtol=2e-2, atol=2e-2,
    )


def test_compiled_linear_real_llama_shape_with_tight_wram_triggers_tile_rewrite() -> None:
    """真实 llama2-7b o_proj shape（M=1/K=512/N=4096, float16）下强制触发
    `pim-tile-to-budget` 的分块改写。

    `kernel_src.py::pick_blocks(K=512, N=4096)` 选出的前端 tile 是
    BLOCK_N=512/BLOCK_K=32，三个 staging buffer 共 33856 字节。这里给一个
    塞不下这个 tile、但塞得下 N 缩到 128（tile_n=128 时共 8512 字节）的
    WRAM 预算（16384），逼真实 llama2-7b 尺度的 kernel 走改写路径而不是
    直接用前端已选的 tile——这跟 `test_compiled_linear_with_tight_wram_
    budget_triggers_tile_rewrite` 用小 shape 覆盖"改写机制本身对不对"是
    互补的：这条用真实模型尺度覆盖"改写在大 K（32）、真实 N-tile scf.for
    循环嵌套（不是前端全展开的直线代码）下还对不对"。
    """
    rng = np.random.default_rng(7)
    M, K, N = 1, 512, 4096
    x = rng.standard_normal((M, K)).astype(np.float16)
    w = rng.standard_normal((N, K)).astype(np.float16)
    mram = max(1 << 20, (x.nbytes + w.nbytes + M * N * 2) * 2 + 3 * _ALIGN)

    hand = _backend(mram)
    register_all(hand)
    hand_result = _run(
        hand, str(torch.ops.aten.linear.default), [(M, K), (N, K)],
        [x, w], (M, N), dtype="float16",
        hardware=PIMHardwareConfig(1, 4, mram, 65536, 64),
    )

    compiled = _backend(mram)
    register_all(compiled, use_compiled_linear=True)
    compiled_result = _run(
        compiled, str(torch.ops.aten.linear.default), [(M, K), (N, K)],
        [x, w], (M, N), dtype="float16",
        hardware=PIMHardwareConfig(1, 4, mram, 16384, 64),
    )

    np.testing.assert_allclose(
        compiled_result.astype(np.float32),
        hand_result.astype(np.float32),
        rtol=2e-2, atol=2e-2,
    )


def test_compiled_linear_falls_back_for_non_power_of_two_shapes() -> None:
    """真实 llama2-7b 的 MLP（intermediate_size=11008）/lm_head
    （vocab_size=32000）都不是 2 的幂，`kernel_src.py` 的整块 kernel 处理不了
    （`tl.arange` 要求 2 的幂）——`compiled_linear_kernel` 要静默退回
    `linear_kernel`，而不是报错。这里用一个非 2 的幂的 N（模拟这类 shape）
    验证 fallback 路径本身产出正确结果，且不会尝试触发 GPU 编译。
    """
    rng = np.random.default_rng(2)
    M, K, N = 2, 16, 11  # N=11 不是 2 的幂
    x = rng.standard_normal((M, K)).astype(np.float32)
    w = rng.standard_normal((N, K)).astype(np.float32)

    backend = _backend()
    register_all(backend, use_compiled_linear=True)
    result = _run(
        backend, str(torch.ops.aten.linear.default), [(M, K), (N, K)],
        [x, w], (M, N),
    )
    ref = torch.nn.functional.linear(torch.from_numpy(x), torch.from_numpy(w)).numpy()
    np.testing.assert_allclose(result, ref, atol=1e-4)


def test_compiled_linear_with_tight_wram_budget_triggers_tile_rewrite() -> None:
    """真正触发 FlagTree `pim-tile-to-budget` 的分块改写路径（不止校验）。

    `kernel_src.py::pick_blocks(K=32, N=8)` 选出的前端 tile 是 M=4/N=8/K=32，
    三个 staging buffer 共 1664 字节（512+1024+128）。给一个塞不下这个 tile
    的小 WRAM 预算（512 字节），逼 `pim-tile-to-budget` 在 IR 层面重新选一个
    更小的、满足预算的 2 的幂 tile（实测会搜到 M=4/N=8/K=8，xb=128/wb=256/
    ob=128 共 512，三者都整除 dma_align=64）并改写循环——旧代码在这个预算下
    会直接编译期报错(`_run_triton_opt` 抛 `RuntimeError`)，这是本次改动前
    唯一的行为。这里断言的是数值结果，不是 IR 结构：分块改写后的编译产物
    必须跟 torch 参考一致，证明改写出的 IR 在
    pim-explicit-dma/pim-lower-to-emitc/C/.so 全链路下数学上等价于不分块。
    """
    rng = np.random.default_rng(4)
    M, K, N = 4, 32, 8
    x = rng.standard_normal((M, K)).astype(np.float32)
    w = rng.standard_normal((N, K)).astype(np.float32)

    backend = _backend()
    register_all(backend, use_compiled_linear=True)
    result = _run(
        backend, str(torch.ops.aten.linear.default), [(M, K), (N, K)],
        [x, w], (M, N),
        hardware=PIMHardwareConfig(1, 4, 1 << 20, 512, 64),
    )
    ref = torch.nn.functional.linear(torch.from_numpy(x), torch.from_numpy(w)).numpy()
    np.testing.assert_allclose(result, ref, atol=1e-4)
