import pytest

from contracts.op_contract import OpCompileRequest, PIMHardwareConfig


def test_hardware_config_round_trips_through_payload():
    cfg = PIMHardwareConfig(
        num_dpus=8,
        num_tasklets=4,
        mram_bytes_per_dpu=4 * 2**30,
        wram_bytes_per_dpu=65536,
        dma_align=1024,
    )

    assert PIMHardwareConfig.from_payload(cfg.to_payload()) == cfg


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
