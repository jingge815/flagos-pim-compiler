"""serve_pim 的 `--pim-*` 参数翻译与启动预检的单元测试。"""

from __future__ import annotations

import argparse
import os
from types import SimpleNamespace

import pytest

from scripts.serve_pim import PIM_FLAGS, add_pim_args, export_env, preflight

MAX_SEQ = 16
BLOCK_SIZE = 16


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch) -> None:
    """清掉 PIM 环境变量，避免宿主的设置影响断言。

    `export_env` 直接写 `os.environ`，monkeypatch 撤销不了，所以用例结束后
    再手动清一遍，防止污染后续用例（例如 `PIM_NUM_DPUS=7` 会让编译失败）。
    """
    names = [env for _flag, _dest, env, _help in PIM_FLAGS]
    names.append("PIM_SYS_RESERVE_BYTES")
    for name in names:
        monkeypatch.delenv(name, raising=False)
    yield
    for name in names:
        os.environ.pop(name, None)


def _config(**overrides) -> SimpleNamespace:
    """构造一份能通过预检的已解析 vLLM 配置，按需覆盖单项。"""
    values = {
        "scheduler_config": SimpleNamespace(
            max_num_seqs=2, enable_chunked_prefill=True,
            max_num_batched_tokens=MAX_SEQ),
        "model_config": SimpleNamespace(
            max_model_len=MAX_SEQ,
            hf_config=SimpleNamespace(
                hidden_size=64, num_attention_heads=4, num_key_value_heads=4,
                num_hidden_layers=4),
        ),
        "cache_config": SimpleNamespace(block_size=BLOCK_SIZE),
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_supported_config_passes() -> None:
    """默认配置通过预检。"""
    preflight(_config())


def test_pool_too_small_reports_required_slots(monkeypatch) -> None:
    """槽池装不下一个满长请求时退出，并给出建议槽数。"""
    monkeypatch.setenv("PIM_KV_POOL_SLOTS", "0")
    with pytest.raises(SystemExit) as excinfo:
        preflight(_config())
    message = str(excinfo.value)
    assert "PIM_KV_POOL_SLOTS" in message
    # 满长请求 == 1 块，默认池 == 2 × 1 + 1 块。
    assert "--pim-kv-pool-slots 3" in message


def test_chunked_prefill_budget_advice(monkeypatch) -> None:
    """单个分块超过 prefill 图长度时，给出两种修法。"""
    monkeypatch.setenv("PIM_PREFILL_SEQ_LEN", str(MAX_SEQ))
    budget = MAX_SEQ + 1
    config = _config(scheduler_config=SimpleNamespace(
        max_num_seqs=2, enable_chunked_prefill=True,
        max_num_batched_tokens=budget))
    with pytest.raises(SystemExit) as excinfo:
        preflight(config)
    message = str(excinfo.value)
    assert f"--max-num-batched-tokens {MAX_SEQ}" in message
    assert f"--pim-prefill-seq-len 调到 {budget}" in message


def test_disabled_chunked_prefill_skips_budget_check(monkeypatch) -> None:
    """关掉分块 prefill 时不校验预算，一次要跑完整段 prompt。"""
    monkeypatch.setenv("PIM_PREFILL_SEQ_LEN", str(MAX_SEQ))
    config = _config(scheduler_config=SimpleNamespace(
        max_num_seqs=2, enable_chunked_prefill=False,
        max_num_batched_tokens=MAX_SEQ + 1))
    preflight(config)


def test_prefill_len_beyond_max_model_len(monkeypatch) -> None:
    """prefill 图长度超过 max_model_len 时报错。"""
    monkeypatch.setenv("PIM_PREFILL_SEQ_LEN", str(MAX_SEQ * 2))
    with pytest.raises(SystemExit, match="PIM_PREFILL_SEQ_LEN"):
        preflight(_config())


def test_mram_budget_exceeded(monkeypatch) -> None:
    """显式给的槽池超过 MRAM 预算时报错。"""
    monkeypatch.setenv("PIM_KV_POOL_SLOTS", "8")
    monkeypatch.setenv("PIM_MRAM_BYTES", "1024")
    monkeypatch.setenv("PIM_SYS_RESERVE_BYTES", "0")
    with pytest.raises(SystemExit, match="MRAM"):
        preflight(_config())


def test_mram_budget_skipped_for_auto_pool(monkeypatch) -> None:
    """不设槽池时预检不算 KV 字节，由编译期按 MRAM 自动定容。"""
    monkeypatch.setenv("PIM_MRAM_BYTES", "1024")
    monkeypatch.setenv("PIM_SYS_RESERVE_BYTES", "0")
    preflight(_config())


def test_problems_reported_together(monkeypatch) -> None:
    """多个问题一次性列全，不用改一处跑一次。"""
    monkeypatch.setenv("PIM_KV_POOL_SLOTS", "0")
    monkeypatch.setenv("PIM_PREFILL_SEQ_LEN", str(MAX_SEQ))
    config = _config(scheduler_config=SimpleNamespace(
        max_num_seqs=2, enable_chunked_prefill=True,
        max_num_batched_tokens=MAX_SEQ + 1))
    with pytest.raises(SystemExit) as excinfo:
        preflight(config)
    message = str(excinfo.value)
    assert "PIM_KV_POOL_SLOTS" in message
    assert "max_num_batched_tokens" in message


def test_export_env_writes_all_flags(monkeypatch) -> None:
    """`--pim-*` 逐项写入对应环境变量。"""
    args = SimpleNamespace(**{dest: 7 for _flag, dest, _env, _help in PIM_FLAGS})
    export_env(args)
    for _flag, _dest, env, _help in PIM_FLAGS:
        assert os.environ[env] == "7"


def test_export_env_leaves_unset_flags_alone(monkeypatch) -> None:
    """没写的选项不动环境变量。"""
    monkeypatch.setenv("PIM_NUM_DPUS", "8")
    args = SimpleNamespace(**{dest: None for _flag, dest, _env, _help in PIM_FLAGS})
    export_env(args)
    assert os.environ["PIM_NUM_DPUS"] == "8"


def test_parser_accepts_pim_flags() -> None:
    """参数解析器能识别 `--pim-*`，默认值为 None。"""
    parser = argparse.ArgumentParser()
    add_pim_args(parser)
    args = parser.parse_args(["--pim-num-dpus", "8", "--pim-mram-bytes", "1024"])
    assert args.pim_num_dpus == 8
    assert args.pim_mram_bytes == 1024
    assert args.pim_kv_pool_slots is None


def _layout(config, *, pool_slots: int, parallel_slots: int,
            align: int | None = None):
    """按真实 `build_kv_layout` 计算同一 KV 几何的字节数。"""
    from memory.kv_layout import KVRegionSpec, build_kv_layout
    from memory.mem_planner import default_hw_budget

    hf = config.model_config.hf_config
    head_dim = getattr(hf, "head_dim", None) or (
        hf.hidden_size // hf.num_attention_heads)
    num_dpus = int(os.environ.get("PIM_NUM_DPUS", "4"))
    kv_heads = -(-hf.num_key_value_heads // num_dpus)
    spec = KVRegionSpec(
        dpu_id=0,
        layers=list(range(hf.num_hidden_layers)),
        kv_heads=list(range(kv_heads)),
        q_heads_by_kv={head: [] for head in range(kv_heads)},
        max_seq=config.model_config.max_model_len,
        head_dim=head_dim,
        dtype_bytes=2,
        kv_base=0,
        slot_tokens=config.cache_config.block_size,
    )
    return build_kv_layout(
        spec,
        default_hw_budget().align if align is None else align,
        pool_slots=pool_slots,
        block_table_slots=parallel_slots,
    ).kv_allocated_bytes


def test_kv_estimate_reuses_the_real_layout() -> None:
    """启动预检的 KV 估算必须与 `build_kv_layout` 同一个公式。"""
    from scripts.serve_pim import _kv_bytes_per_dpu

    config = _config()
    assert _kv_bytes_per_dpu(config.model_config, BLOCK_SIZE, 3, 2) == _layout(
        config, pool_slots=3, parallel_slots=2)


def test_preflight_accepts_a_layout_that_fits_under_the_contract_alignment(
        monkeypatch) -> None:
    """旧预检按 256 字节对齐会拒掉实际装得下的配置；契约对齐必须放行。"""
    from contracts.op_contract import DPU_SYS_RESERVE_BYTES

    config = _config()
    needed = _layout(config, pool_slots=3, parallel_slots=2)
    old = _layout(config, pool_slots=3, parallel_slots=2, align=256)
    assert old > needed, "探针前提：256 字节对齐确实会算出更大的字节数"
    monkeypatch.setenv("PIM_KV_POOL_SLOTS", "3")
    monkeypatch.setenv("PIM_MRAM_BYTES", str(DPU_SYS_RESERVE_BYTES + needed))
    preflight(config)


def test_preflight_rejects_one_byte_over_the_budget(monkeypatch) -> None:
    """真正超过预算一个字节时仍必须拒绝，不能把判据放宽成不算。"""
    from contracts.op_contract import DPU_SYS_RESERVE_BYTES

    config = _config()
    needed = _layout(config, pool_slots=3, parallel_slots=2)
    monkeypatch.setenv("PIM_KV_POOL_SLOTS", "3")
    monkeypatch.setenv(
        "PIM_MRAM_BYTES", str(DPU_SYS_RESERVE_BYTES + needed - 1))
    with pytest.raises(SystemExit, match="MRAM"):
        preflight(config)


def test_serve_pim_has_no_independent_alignment_or_old_budget() -> None:
    """启动入口不能再写 256 对齐或示例性的 4 GiB 预算。"""
    from pathlib import Path

    import scripts.serve_pim as serve_pim

    source = Path(serve_pim.__file__).read_text()
    assert "ALIGN = 256" not in source
    assert "4294967296" not in source
