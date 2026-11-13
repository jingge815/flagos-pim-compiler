import pytest

from contracts.op_contract import (
    OpCompileRequest,
    PIMHardwareConfig,
    PimOptimizations,
)


def test_hardware_config_round_trips_through_payload():
    cfg = PIMHardwareConfig(
        num_dpus=8,
        num_tasklets=4,
        mram_bytes_per_dpu=4 * 2**30,
        wram_bytes_per_dpu=65536,
        dma_align=1024,
    )

    assert PIMHardwareConfig.from_payload(cfg.to_payload()) == cfg


def test_hardware_config_carries_the_compiler_target_in_its_payload():
    """编译目标属于请求级契约，缓存键与下发文本都从它派生。"""
    cfg = PIMHardwareConfig(
        num_dpus=8,
        num_tasklets=4,
        mram_bytes_per_dpu=4 * 2**30,
        wram_bytes_per_dpu=65536,
        dma_align=1024,
        compiler_target="pim:v2",
    )

    assert cfg.to_payload()["compiler_target"] == "pim:v2"
    assert PIMHardwareConfig.from_payload(cfg.to_payload()) == cfg
    # 旧 payload 没有该键时按固定目标读回，旧产物仍可执行。
    legacy = cfg.to_payload()
    legacy.pop("compiler_target")
    assert PIMHardwareConfig.from_payload(legacy).compiler_target == "pim:v1"


@pytest.mark.parametrize(
    "field,value",
    [
        ("num_dpus", 0),
        ("num_tasklets", 0),
        ("mram_bytes_per_dpu", 0),
        ("wram_bytes_per_dpu", 0),
        ("dma_align", 0),
        ("num_dpus", 3),
    ],
)
def test_hardware_config_rejects_invalid_values(field, value):
    kwargs = dict(
        num_dpus=8,
        num_tasklets=4,
        mram_bytes_per_dpu=4 * 2**30,
        wram_bytes_per_dpu=65536,
        dma_align=1024,
    )
    kwargs[field] = value

    with pytest.raises(ValueError, match=field):
        PIMHardwareConfig(**kwargs)


def test_op_compile_request_requires_hardware_config():
    with pytest.raises(TypeError, match="hardware"):
        OpCompileRequest(op="linear", arg_shapes=[(1, 16), (4, 16)])


def test_optimizations_default_to_all_enabled_and_round_trip():
    """默认保持旧行为：五类优化中可开关的四类全部打开。"""
    opt = PimOptimizations()
    assert opt.to_payload() == {
        "constant_fold": True,
        "cse": True,
        "dce": True,
        "loop_tiling": True,
    }
    assert PimOptimizations.from_payload(opt.to_payload()) == opt


def test_optimizations_reject_unknown_keys_and_non_bool_values():
    with pytest.raises(ValueError, match="未知"):
        PimOptimizations.from_payload({"cse": True, "mystery": True})
    with pytest.raises(ValueError, match="cse"):
        PimOptimizations(cse="yes")  # type: ignore[arg-type]


def test_hardware_payload_carries_optimizations_and_legacy_defaults():
    """优化开关属于编译请求；缓存键与侧车都从硬件 payload 取。"""
    cfg = PIMHardwareConfig(
        num_dpus=8,
        num_tasklets=4,
        mram_bytes_per_dpu=4 * 2**30,
        wram_bytes_per_dpu=65536,
        dma_align=1024,
        optimizations=PimOptimizations(cse=False, loop_tiling=False),
    )
    assert cfg.to_payload()["optimizations"] == {
        "constant_fold": True,
        "cse": False,
        "dce": True,
        "loop_tiling": False,
    }
    assert PIMHardwareConfig.from_payload(cfg.to_payload()) == cfg

    # 旧 payload 没有优化块时按全开读回，旧产物行为不变。
    legacy = cfg.to_payload()
    legacy.pop("optimizations")
    assert PIMHardwareConfig.from_payload(legacy).optimizations == PimOptimizations()
