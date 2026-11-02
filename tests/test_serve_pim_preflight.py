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
