import threading

import numpy as np
import pytest
import torch

from backend import VendorBackend
from backend.hal_numpy import NumpyBackend, NumpyBackendConfig
from contracts.exec_plan import Access, Command, ExecutionPlan
from contracts.pim_tensor_spec import Placement, PIMTensorSpec, TensorShardDetail


def test_runtime_contracts_have_the_expected_fields() -> None:
    shard = TensorShardDetail(
        dpu_id=0,
        shard_dim=1,
        start_idx=0,
        end_idx=4,
        local_shape=(2, 2),
    )
    spec = PIMTensorSpec(
        device="dpu",
        placement=Placement(kind="Shard", dim=1),
        residency="pinned",
        pinned_dpu_id=0,
        shard_map={0: shard},
        reduce_type=None,
    )
    plan = ExecutionPlan(
        commands=[
            Command(
                id=0,
                op="launch",
                dpu_id=0,
                payload={"kernel": "noop"},
                reads=[],
                writes=[Access(("dpu", 0), 0, 16)],
                waits=[],
            )
        ]
    )
    assert spec.shard_map[0].local_shape == (2, 2)
    assert plan.commands[0].writes[0].offset == 0


def test_vendor_backend_stub_is_available_from_package() -> None:
    with pytest.raises(NotImplementedError, match="Vendor backend is not wired yet"):
        VendorBackend()


def test_dpu_isolation_and_copy_round_trip() -> None:
    backend = NumpyBackend(NumpyBackendConfig(num_dpus=2, mram_bytes_per_dpu=64))
    payload = np.arange(8, dtype=np.int32)
    backend.copy_to_dpu(0, 0, payload)
    assert np.array_equal(backend.copy_from_dpu(0, 0, payload.shape, payload.dtype), payload)
    assert np.array_equal(
        backend.copy_from_dpu(1, 0, payload.shape, payload.dtype),
        np.zeros_like(payload),
    )


def test_submit_wait_query_and_kernel_stub() -> None:
    backend = NumpyBackend(NumpyBackendConfig(num_dpus=1, mram_bytes_per_dpu=64))
    gate = threading.Event()

    def fill(hal, dpu_id, cmd) -> None:
        gate.wait()
        hal.write_local(dpu_id, cmd.payload["offset"], np.asarray(cmd.payload["value"], dtype=np.int32))

    backend.register_kernel("fill", fill)
    event = backend.submit(
        Command(
            id=1,
            op="launch",
            dpu_id=0,
            payload={"kernel": "fill", "offset": 0, "value": [1, 2, 3, 4]},
            reads=[],
            writes=[Access(("dpu", 0), 0, 16)],
            waits=[],
        )
    )
    assert not backend.query(event)
    gate.set()
    backend.wait(event)
    assert backend.query(event)


def test_submit_serializes_commands_on_the_same_dpu() -> None:
    backend = NumpyBackend(NumpyBackendConfig(num_dpus=1, mram_bytes_per_dpu=64))
    first_started = threading.Event()
    allow_first_to_finish = threading.Event()

    def write_value(hal, dpu_id, cmd) -> None:
        if cmd.payload["value"] == [1]:
            first_started.set()
            allow_first_to_finish.wait()
        hal.write_local(dpu_id, cmd.payload["offset"], np.asarray(cmd.payload["value"], dtype=np.int32))

    backend.register_kernel("write_value", write_value)
    first = backend.submit(
        Command(1, "launch", 0, {"kernel": "write_value", "offset": 0, "value": [1]})
    )
    assert first_started.wait(timeout=1.0)
    second = backend.submit(
        Command(2, "launch", 0, {"kernel": "write_value", "offset": 0, "value": [2]})
    )
    allow_first_to_finish.set()

    backend.wait(second)
    assert backend.query(first)
    assert backend.read_local(0, 0, (1,), np.int32)[0] == 2


def test_submit_dma_and_host_paths_return_results_and_propagate_errors() -> None:
    backend = NumpyBackend(NumpyBackendConfig(num_dpus=1, mram_bytes_per_dpu=32))
    payload = np.arange(4, dtype=np.int32)

    backend.wait(
        backend.submit(Command(1, "dma_in", 0, {"offset": 0, "data": payload}))
    )
    out = backend.submit(Command(2, "dma_out", 0, {"offset": 0, "shape": payload.shape, "dtype": payload.dtype}))
    assert np.array_equal(backend.wait(out), payload)

    host = backend.submit(Command(3, "host_op", None, {"fn": lambda cmd: cmd.id + 10}))
    assert backend.wait(host) == 13

    def fail_host(cmd) -> None:
        raise RuntimeError(f"host failure in cmd {cmd.id}")

    failed_host = backend.submit(Command(4, "host_op", None, {"fn": fail_host}))
    with pytest.raises(RuntimeError, match="host failure in cmd 4"):
        backend.wait(failed_host)

    bad = backend.submit(Command(5, "dma_in", 0, {"offset": 24, "data": payload}))
    with pytest.raises(ValueError, match="invalid MRAM range"):
        backend.wait(bad)


def test_dma_in_snapshots_host_payload_at_submit_time() -> None:
    backend = NumpyBackend(NumpyBackendConfig(num_dpus=1, mram_bytes_per_dpu=32))
    launch_started = threading.Event()
    release_launch = threading.Event()

    def block_stream(hal, dpu_id, cmd) -> None:
        launch_started.set()
        release_launch.wait()

    backend.register_kernel("block_stream", block_stream)
    launch = backend.submit(Command(1, "launch", 0, {"kernel": "block_stream"}))
    assert launch_started.wait(timeout=1.0)
    payload = np.asarray([1, 2, 3, 4], dtype=np.int32)
    dma = backend.submit(Command(2, "dma_in", 0, {"offset": 0, "data": payload}))
    payload[:] = 99
    release_launch.set()

    backend.wait(dma)
    assert np.array_equal(backend.read_local(0, 0, (4,), np.int32), np.asarray([1, 2, 3, 4], dtype=np.int32))
    assert backend.query(launch)


def test_command_waits_are_honored_before_execution() -> None:
    backend = NumpyBackend(NumpyBackendConfig(num_dpus=2, mram_bytes_per_dpu=32))
    first_started = threading.Event()
    release_first = threading.Event()
    second_ran = threading.Event()

    def write_one(hal, dpu_id, cmd) -> None:
        first_started.set()
        release_first.wait()
        hal.write_local(dpu_id, cmd.payload["offset"], np.asarray(cmd.payload["value"], dtype=np.int32))

    backend.register_kernel("write_one", write_one)
    first = backend.submit(
        Command(
            1,
            "launch",
            0,
            {"kernel": "write_one", "offset": 0, "value": [1]},
        )
    )
    assert first_started.wait(timeout=1.0)

    def write_two(hal, dpu_id, cmd) -> None:
        second_ran.set()
        hal.write_local(dpu_id, cmd.payload["offset"], np.asarray(cmd.payload["value"], dtype=np.int32))

    backend.register_kernel("write_two", write_two)
    second = backend.submit(
        Command(
            2,
            "launch",
            1,
            {"kernel": "write_two", "offset": 0, "value": [2]},
            waits=[1],
        )
    )
    assert not second_ran.wait(timeout=0.05)
    release_first.set()
    backend.wait(second)
    assert backend.read_local(1, 0, (1,), np.int32)[0] == 2
    assert backend.query(first)


def test_same_dpu_dma_waits_for_prior_launch_without_explicit_waits() -> None:
    backend = NumpyBackend(NumpyBackendConfig(num_dpus=1, mram_bytes_per_dpu=32))
    launch_started = threading.Event()
    release_launch = threading.Event()

    def delayed_fill(hal, dpu_id, cmd) -> None:
        launch_started.set()
        release_launch.wait()
        hal.write_local(dpu_id, 0, np.asarray([7], dtype=np.int32))

    backend.register_kernel("delayed_fill", delayed_fill)
    launch = backend.submit(Command(1, "launch", 0, {"kernel": "delayed_fill"}))
    assert launch_started.wait(timeout=1.0)
    dma = backend.submit(Command(2, "dma_out", 0, {"offset": 0, "shape": (1,), "dtype": np.int32}))
    assert not backend.query(dma)
    release_launch.set()

    assert np.array_equal(backend.wait(dma), np.asarray([7], dtype=np.int32))
    assert backend.query(launch)


def test_device_to_device_direct_copy_has_no_phase1_api() -> None:
    backend = NumpyBackend(NumpyBackendConfig(num_dpus=2, mram_bytes_per_dpu=32))
    assert backend.config.allow_device_to_device is False
    assert not hasattr(backend, "copy_dpu_to_dpu")
    with pytest.raises(ValueError, match="device-to-device copies are not supported"):
        NumpyBackend(NumpyBackendConfig(num_dpus=2, mram_bytes_per_dpu=32, allow_device_to_device=True))


def test_read_local_returns_a_copy_and_push_xfer_fans_out() -> None:
    backend = NumpyBackend(NumpyBackendConfig(num_dpus=2, mram_bytes_per_dpu=64))
    payload = np.arange(4, dtype=np.float32)
    backend.push_xfer([(0, 0, payload), (1, 16, payload)])
    host_copy = backend.read_local(0, 0, payload.shape, payload.dtype)
    host_copy[0] = -1
    assert backend.read_local(0, 0, payload.shape, payload.dtype)[0] != -1
    assert np.array_equal(backend.read_local(1, 16, payload.shape, payload.dtype), payload)


def test_llama2_smoke_payload_stays_model_agnostic() -> None:
    hidden_size = 4096
    sequence_length = 128
    activation = np.arange(sequence_length * hidden_size, dtype=np.float16).reshape(
        1,
        sequence_length,
        hidden_size,
    )
    backend = NumpyBackend(NumpyBackendConfig(num_dpus=1, mram_bytes_per_dpu=activation.nbytes * 2))

    backend.copy_to_dpu(0, 0, activation)

    assert np.array_equal(backend.copy_from_dpu(0, 0, activation.shape, activation.dtype), activation)
