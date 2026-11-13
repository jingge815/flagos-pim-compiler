"""算子编译参数贯通：tasklet 同源、形状提示与 B 路硬件。"""

import dataclasses
from pathlib import Path

import pytest

from contracts.op_contract import (
    DEFAULT_HARDWARE_CONFIG,
    OpCompileRequest,
    PIMHardwareConfig,
    PimOptimizations,
)


def _hw(**changes):
    changes.setdefault("num_dpus", 1)
    return dataclasses.replace(DEFAULT_HARDWARE_CONFIG, **changes)


def test_request_tasklets_default_to_hardware():
    hw = _hw(num_tasklets=8)
    request = OpCompileRequest(
        op="linear", arg_shapes=[(2, 64), (16, 64)], hardware=hw)
    assert request.num_tasklets is None
    assert request.resolved_num_tasklets() == 8


def test_request_explicit_tasklets_must_match_hardware():
    hw = _hw(num_tasklets=8)
    with pytest.raises(ValueError, match="tasklet"):
        OpCompileRequest(
            op="linear", arg_shapes=[(2, 64), (16, 64)], hardware=hw,
            num_tasklets=4)
    request = OpCompileRequest(
        op="linear", arg_shapes=[(2, 64), (16, 64)], hardware=hw,
        num_tasklets=8)
    assert request.resolved_num_tasklets() == 8


def test_a_path_full_shape_options_only_touch_the_tile_pass():
    from opcompiler_bridge.pipeline import a_path_passes

    hw = _hw(num_tasklets=4)
    plain = a_path_passes(hw, target="pim:v1")
    shaped = a_path_passes(hw, target="pim:v1", full_m=2, full_n=4, full_k=16)
    assert plain[0] == shaped[0], "convert 的参数不能跟着形状提示变"
    assert "-pim-tile-to-budget" in plain
    assert shaped[-2] == "-pim-tile-to-budget=full-m=2 full-n=4 full-k=16"
    # 全为 -1 时保留原字符串。
    assert a_path_passes(hw, target="pim:v1", full_m=-1, full_n=-1,
                         full_k=-1) == plain


def test_a_path_compile_writes_request_full_shapes():
    """真实 A 路编译把请求形状写进 pim.tile-*。"""
    from opcompiler_bridge.driver import compile_op

    hw = _hw(num_tasklets=4)
    result = compile_op(OpCompileRequest(
        op="linear", arg_shapes=[(2, 16), (4, 16)], hardware=hw,
        dtype="float16", num_tasklets=4), force=True)
    text = result.pimir or ""
    assert '"pim.tile-m" = 2' in text, text[:800]
    assert '"pim.tile-n" = 4' in text, text[:800]
    assert '"pim.tile-k" = 16' in text, text[:800]


def test_bpath_module_uses_request_hardware_budget():
    from opcompiler_bridge.driver import _make_oplevel_mlir

    small = _hw(num_tasklets=4, wram_bytes_per_dpu=4096, mram_bytes_per_dpu=1 << 20)
    large = _hw(num_tasklets=4, wram_bytes_per_dpu=8192, mram_bytes_per_dpu=1 << 21)
    request_small = OpCompileRequest(
        op="eltwise", arg_shapes=[(1, 16), (1, 16)], hardware=small,
        kind="add")
    request_large = dataclasses.replace(request_small, hardware=large)
    text_small = _make_oplevel_mlir(request_small)
    text_large = _make_oplevel_mlir(request_large)
    assert '"pim.wram-bytes" = 4096' in text_small
    assert '"pim.wram-bytes" = 8192' in text_large
    assert '"pim.mram-bytes" = 1048576' in text_small


def test_bpath_pimir_cache_key_includes_hardware(tmp_path, monkeypatch):
    from genesim_bridge import placement_export
    from opcompiler_bridge import driver

    calls = []

    def fake_run(text, hardware=None):
        calls.append(hardware)
        return text, text

    monkeypatch.setattr(driver, "_run_oplevel_triton_opt", fake_run)
    monkeypatch.setattr(driver, "_CACHE_DIR", tmp_path)
    small = _hw(num_tasklets=4, wram_bytes_per_dpu=4096)
    large = _hw(num_tasklets=4, wram_bytes_per_dpu=8192)
    first = placement_export._bpath_pimir("eltwise", [(1, 16)], [(1, 16)], small)
    second = placement_export._bpath_pimir("eltwise", [(1, 16)], [(1, 16)], large)
    assert first != second, "不同硬件不能复用同一份 B 路缓存"
    assert calls[0] == small and calls[1] == large


def test_cache_key_changes_when_tasklets_change():
    from opcompiler_bridge.driver import _cache_key

    request = OpCompileRequest(
        op="linear", arg_shapes=[(2, 64), (16, 64)], hardware=_hw(num_tasklets=4),
        num_tasklets=4)
    other = dataclasses.replace(
        request, hardware=_hw(num_tasklets=8), num_tasklets=8)
    assert _cache_key(request) != _cache_key(other)


# ---- 编译优化开关：A/B 两路按配置剔除对应 pass ----


def test_a_path_optimization_switches_remove_passes():
    from opcompiler_bridge.pipeline import a_path_passes

    hw = _hw(
        num_tasklets=4,
        optimizations=PimOptimizations(
            constant_fold=False, cse=False, dce=False, loop_tiling=False))
    passes = a_path_passes(hw, target="pim:v1")

    assert "-pim-constant-fold" not in passes
    assert "-pim-cse" not in passes
    assert "-cse" not in passes
    assert "-pim-dce" not in passes
    assert "-pim-tile-to-budget" not in passes
    # 融合与显式 DMA 是目标格式硬要求，开关不能把它们关掉。
    assert "-pim-fuse-activation" in passes
    assert "-pim-explicit-dma" in passes


def test_a_path_optimization_switches_keep_the_original_order():
    from opcompiler_bridge.pipeline import a_path_passes

    hw = _hw(
        num_tasklets=4,
        optimizations=PimOptimizations(
            constant_fold=False, cse=True, dce=False, loop_tiling=True))
    passes = a_path_passes(hw, target="pim:v1")

    assert passes.index("-pim-cse") < passes.index("-pim-fuse-activation")
    assert passes.index("-pim-fuse-activation") < passes.index(
        "-pim-tile-to-budget")
    assert "-pim-constant-fold" not in passes
    assert "-pim-dce" not in passes


def test_b_path_optimization_switches_remove_passes():
    from opcompiler_bridge.pipeline import b_path_passes

    passes = b_path_passes(PimOptimizations(
        constant_fold=False, cse=False, dce=False, loop_tiling=False))

    assert "-pim-constant-fold" not in passes
    assert "-pim-cse" not in passes
    assert "-pim-dce" not in passes
    assert "-pim-tile-to-budget" not in passes
    # 融合、相位展开与契约校验不能被开关关掉。
    assert "-pim-fuse-activation" in passes
    assert "-pim-expand-phases" in passes
    assert "-pim-verify-gml-contract" in passes


def test_hardware_optimizations_enter_the_operator_cache_key():
    """优化开关属于编译请求，缓存键必须随它失效。"""
    from opcompiler_bridge.driver import _cache_key

    request = OpCompileRequest(
        op="linear", arg_shapes=[(2, 64), (16, 64)], hardware=_hw(num_tasklets=4),
        num_tasklets=4)
    changed = dataclasses.replace(
        request,
        hardware=dataclasses.replace(
            request.hardware,
            optimizations=PimOptimizations(cse=False, loop_tiling=False)))
    assert _cache_key(request) != _cache_key(changed)


# ---- B 路产物必须携带 DPU/tasklet，调优才知道这些参数真的进了 B 路 ----


def test_bpath_pimir_carries_dpu_and_tasklet_attributes(tmp_path, monkeypatch):
    from genesim_bridge import placement_export
    from opcompiler_bridge import driver

    monkeypatch.setattr(
        driver, "_run_oplevel_triton_opt",
        lambda text, hardware=None: (text, text))
    monkeypatch.setattr(driver, "_CACHE_DIR", tmp_path)

    hw4 = _hw(num_dpus=8, num_tasklets=4)
    text4 = Path(placement_export._bpath_pimir(
        "eltwise", [(1, 16)], [(1, 16)], hw4)).read_text()
    hw8 = dataclasses.replace(hw4, num_tasklets=8)
    text8 = Path(placement_export._bpath_pimir(
        "eltwise", [(1, 16)], [(1, 16)], hw8)).read_text()

    assert '"pim.num-dpus" = 8' in text4
    assert '"pim.num-tasklets" = 4' in text4
    assert '"pim.num-tasklets" = 8' in text8
    assert text4 != text8, "B 路 PIMIR 必须随 tasklet 变化"


def test_single_dpu_bpath_pimir_keeps_the_old_text(tmp_path, monkeypatch):
    """单 DPU 不写 DPU/tasklet 模块属性，旧口径逐字节不变。"""
    from genesim_bridge import placement_export
    from opcompiler_bridge import driver

    monkeypatch.setattr(
        driver, "_run_oplevel_triton_opt",
        lambda text, hardware=None: (text, text))
    monkeypatch.setattr(driver, "_CACHE_DIR", tmp_path)

    text = Path(placement_export._bpath_pimir(
        "eltwise", [(1, 16)], [(1, 16)],
        _hw(num_dpus=1, num_tasklets=4))).read_text()
    assert "pim.num-dpus" not in text
    assert "pim.num-tasklets" not in text


def test_bpath_matmul_tasklet_split_reaches_emitc():
    """B 路矩阵乘降级必须按 tasklet 切 M，而不是只写模块属性。"""
    from genesim_bridge.placement_export import _bpath_pimir
    from opcompiler_bridge.driver import _run_passes, _triton_opt

    def emitc(tasklets: int) -> str:
        text = Path(_bpath_pimir(
            "matmul", [(4, 8), (8, 4)], [(4, 4)],
            _hw(num_dpus=8, num_tasklets=tasklets),
            force_compile=True)).read_text()
        return _run_passes(
            _triton_opt(), text, ["-pim-lower-to-emitc"], "emitc")

    single = emitc(1)
    double = emitc(2)
    assert double.count("emitc.for") > single.count("emitc.for"), (
        "tasklet=2 应比 tasklet=1 多一个 M 循环块")


# ---- 评审 round09：编译目标只能来自请求级编译契约 ----


def test_profile_hardware_carries_the_fixed_compiler_target():
    """正式链路的硬件契约从画像派生，目标不能另走全局环境。"""
    from contracts.hardware_profile import load_profile

    profile = load_profile(
        Path(__file__).resolve().parent.parent
        / "contracts" / "profiles" / "ceva_64core_4e.json")
    hardware = profile.to_pim_hardware_config(8, 4)
    assert profile.compiler_target == "pim:v1"
    assert hardware.compiler_target == profile.compiler_target


def test_a_path_target_comes_from_the_request_contract(monkeypatch):
    """环境里的 FLAGTREE_PIM_TARGET 不能改变正式 A 路的下发目标。"""
    from opcompiler_bridge import driver

    monkeypatch.setenv("FLAGTREE_PIM_TARGET", "pim:other")
    hardware = _hw(num_tasklets=4)
    calls = []
    original = driver._run_passes

    def capture(tool, text, passes, stage):
        calls.append(list(passes))
        return text

    monkeypatch.setattr(driver, "_run_passes", capture)
    try:
        driver._run_triton_opt("module {}", hardware)
    finally:
        monkeypatch.setattr(driver, "_run_passes", original)
    assert calls[0][0].startswith("-convert-triton-to-pim=target=pim:v1")
    assert "pim:other" not in calls[0][0]


def test_b_path_module_target_comes_from_the_request_contract():
    """B 路的 pim.target 属性也只能取请求契约里的目标。"""
    from opcompiler_bridge.driver import _module_text

    request = OpCompileRequest(
        op="eltwise", arg_shapes=[(1, 16), (1, 16)],
        hardware=_hw(num_tasklets=4, compiler_target="pim:v2"), kind="add")
    text = _module_text("func.func @kernel() {\n  return\n}", request)
    assert 'pim.target = "pim:v2"' in text


def test_real_compile_target_is_stable_under_environment_override(monkeypatch):
    """真实 PIM MLIR 在相反环境下仍由请求契约决定，内容逐字节相同。"""
    from opcompiler_bridge.driver import _make_ttir, _run_triton_opt

    request = OpCompileRequest(
        op="linear", arg_shapes=[(2, 16), (4, 16)],
        hardware=_hw(num_tasklets=1, mram_bytes_per_dpu=1 << 20),
        dtype="float16", num_tasklets=1)
    ttir = _make_ttir(request)
    monkeypatch.delenv("FLAGTREE_PIM_TARGET", raising=False)
    baseline, _ = _run_triton_opt(ttir, request.hardware)
    monkeypatch.setenv("FLAGTREE_PIM_TARGET", "pim:other")
    overridden, _ = _run_triton_opt(ttir, request.hardware)
    assert overridden == baseline
    assert "pim:other" not in overridden


def test_compile_cache_key_includes_the_compiler_target():
    """缓存身份只从实际生效的请求契约派生。"""
    from opcompiler_bridge.driver import _cache_key

    request = OpCompileRequest(
        op="linear", arg_shapes=[(2, 64), (16, 64)],
        hardware=_hw(num_tasklets=4), num_tasklets=4)
    other = dataclasses.replace(
        request,
        hardware=dataclasses.replace(request.hardware,
                                     compiler_target="pim:v2"))
    assert _cache_key(request) != _cache_key(other)


# ---- 评审 round01：工具链内容身份 ----


def test_compiler_fingerprint_includes_gcc_content(tmp_path, monkeypatch):
    from opcompiler_bridge import driver

    fake = tmp_path / "gcc"
    fake.write_text("gcc-one")
    monkeypatch.setattr(driver, "_gcc", lambda: fake)
    driver._tool_content_digest.cache_clear()
    first = driver._tool_content_digest()
    fake.write_text("gcc-two")
    driver._tool_content_digest.cache_clear()
    assert driver._tool_content_digest() != first


def test_compiler_identity_declares_source_and_every_tool(monkeypatch):
    from opcompiler_bridge import driver

    identity = driver.compiler_identity()
    assert len(identity["graph_compiler"]["source_sha256"]) == 64
    assert identity["graph_compiler"]["source_dirs"]
    for tool in ("triton_opt", "libtriton", "mlir_translate", "gcc"):
        entry = identity["tools"].get(tool)
        assert entry is not None, tool
        assert "path" in entry
        assert "sha256" in entry


@pytest.mark.parametrize("tool_name", [
    "_triton_opt", "_loaded_libtriton", "_mlir_translate", "_gcc"])
def test_each_compiler_tool_content_changes_the_fingerprint(
        tmp_path, monkeypatch, tool_name):
    from opcompiler_bridge import driver

    fake = tmp_path / tool_name
    fake.write_bytes(b"tool-one")
    monkeypatch.setattr(driver, tool_name, lambda: fake)
    driver._tool_content_digest.cache_clear()
    first = driver._tool_content_digest()
    fake.write_bytes(b"tool-two")
    driver._tool_content_digest.cache_clear()
    assert driver._tool_content_digest() != first, tool_name
