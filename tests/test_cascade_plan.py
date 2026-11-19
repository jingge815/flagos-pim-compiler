"""cascade/plan.py 的蓝图与控制块镜像测试（设计 4.1/4.6/4.7、7.1）。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from cascade.plan import (ControlBlock, CascadeBlueprint, from_bundle,
                          plan_text)

PROMPT_HIDDEN = 1024 * 4096 * 2   # 8MiB
DECODE_HIDDEN = 4096 * 2          # 8KiB
LOGITS = 32000 * 2                # 全宽 fp16
SHARD_LOGITS = 16000 * 2          # 列切一半


def _write_ini(path: Path, sections: list[dict]) -> None:
    lines = []
    for i, kv in enumerate(sections):
        lines.append(f"[{i}]")
        lines += [f"{k}={v}" for k, v in kv.items()]
    path.write_text("\n".join(lines) + "\n")


def _core_maps(run_dir: Path, core: int, *, layers: str, logits: int,
               tp2: bool = False) -> None:
    """写一个核的 ddr 产物：inputs/outputs/control 三份 ini。"""
    d = run_dir / f"ddr_core{core}"
    d.mkdir(parents=True)
    inputs = [{"input_file": f"/x/h{core}.bin", "input_name": "hidden_states",
               "address": hex(0x100000000 + core * (1 << 33) + 0x2000),
               "size": PROMPT_HIDDEN}]
    if not tp2 or core == 0 or True:  # 两策略下首核/两核都要 embedding_row
        inputs.append({"input_file": f"/x/row{core}.bin",
                       "input_name": "embedding_row",
                       "address": hex(0x100000000 + core * (1 << 33) + 0x5000),
                       "size": DECODE_HIDDEN})
    _write_ini(d / "inputs_map.ini", inputs)
    outputs = [{"output_name": "output",
                "address": hex(0x100000000 + core * (1 << 33) + 0x10000),
                "size": PROMPT_HIDDEN}]
    for i in range(16 if not tp2 else 32):
        outputs.append({"output_name": f"key_cache_{i}",
                        "address": hex(0x2000000000 + core * 4096 + i * 16),
                        "size": 1024})
    outputs.append({"output_name": "lm_output", "address": "0x3000000000",
                    "size": logits})
    _write_ini(d / "outputs_map.ini", outputs)
    (d / "control_map.ini").write_text(
        f"[control]\ncore_id={core}\naddress={hex(0x100000000 + core * (1 << 33) + 0x1000)}\n")
    assert layers


def _pp2_bundle(tmp_path: Path) -> Path:
    bundle = tmp_path / "bundle.json"
    bundle.write_text(json.dumps({
        "schema_version": 1, "complete": True, "graphs": [],
        "cascade": {"strategy": "pp2", "num_cores": 2,
                    "core_layers": [[0, 16], [16, 32]], "lm_head_core": 1}}))
    _core_maps(tmp_path, 0, layers="0,16", logits=LOGITS)
    _core_maps(tmp_path, 1, layers="16,32", logits=LOGITS)
    return bundle


def _tp2_bundle(tmp_path: Path, layers: int = 2) -> Path:
    bundle = tmp_path / "bundle.json"
    bundle.write_text(json.dumps({
        "schema_version": 1, "complete": True, "graphs": [],
        "cascade": {"strategy": "tp2", "num_cores": 2, "tp_width": 2,
                    "core_layers": [[0, layers], [0, layers]],
                    "lm_head_core": None}}))
    _core_maps(tmp_path, 0, layers="all", logits=SHARD_LOGITS, tp2=True)
    _core_maps(tmp_path, 1, layers="all", logits=SHARD_LOGITS, tp2=True)
    return bundle


# ---- ControlBlock 镜像 ----

def test_control_block_layout_roundtrip() -> None:
    block = ControlBlock(cmd=2, layer_lo=16, layer_hi=32,
                         in_addr=0x100000000 + 0x5000, out_addr=0x2_0000_1000,
                         line_idx=7)
    raw = block.pack()
    assert len(raw) == 64
    assert ControlBlock.unpack(raw) == block
    # 设计 4.1 的字段偏移：in_addr@16、out_addr@24 都是 64 位
    import struct
    assert struct.unpack_from("<Q", raw, 16)[0] == block.in_addr
    assert struct.unpack_from("<Q", raw, 24)[0] == block.out_addr
    assert struct.unpack_from("<I", raw, 0)[0] == 0      # go 缺省 0
    assert struct.unpack_from("<I", raw, 36)[0] == 0     # done 缺省 0


def test_control_block_go_done_roundtrip() -> None:
    block = ControlBlock(cmd=3).with_go()
    assert block.go == 1
    assert ControlBlock.unpack(block.pack()).go == 1
    done = ControlBlock.unpack(
        block.pack()[:36] + (1).to_bytes(4, "little") + block.pack()[40:])
    assert done.done == 1


# ---- PP2 蓝图 ----

def test_from_bundle_pp2_segments_and_transfers(tmp_path) -> None:
    plan: CascadeBlueprint = from_bundle(_pp2_bundle(tmp_path), tmp_path, steps=3)
    assert plan.strategy == "pp2" and plan.num_cores == 2
    assert plan.control_addrs == {0: 0x100000000 + 0x1000,
                                  1: 0x100000000 + (1 << 33) + 0x1000}
    # 段：prompt 3 段（c0、c1、head）+ 每步 3 段（c0、c1、head）
    assert len(plan.segments) == 3 + 3 * 3
    prompt_c0 = plan.segments[0]
    assert prompt_c0.seg_id == "prompt:c0" and prompt_c0.core_id == 0
    assert prompt_c0.kind == "prompt_layers"
    assert (prompt_c0.layer_lo, prompt_c0.layer_hi) == (0, 16)
    assert prompt_c0.in_addrs["hidden"] == 0x100000000 + 0x2000
    head = [s for s in plan.segments if s.kind == "lm_head"][0]
    assert head.core_id == 1 and "lm_output" in head.out_addrs
    decode_c0 = [s for s in plan.segments if s.seg_id == "decode:1:c0"][0]
    assert decode_c0.in_addrs["hidden"] == 0x100000000 + 0x5000  # embedding_row
    # 搬运：prompt 相 3 条（hidden 8MiB + token + embedding）+ 每步 3 条
    assert len(plan.transfers) == 3 + 3 * 3
    handoffs = [t for t in plan.transfers if t.kind == "hidden_handoff"]
    prompt_xfer = handoffs[0]
    assert prompt_xfer.nbytes == PROMPT_HIDDEN
    assert prompt_xfer.after_seg == "prompt:c0"
    assert (prompt_xfer.src_core, prompt_xfer.dst_core) == (0, 1)
    assert handoffs[1].nbytes == DECODE_HIDDEN and handoffs[1].after_seg == "decode:0:c0"
    emb = [t for t in plan.transfers if t.kind == "embedding"]
    assert len(emb) == 4 and all(t.nbytes == DECODE_HIDDEN and t.dst_core == 0
                                 for t in emb)
    tokens = [t for t in plan.transfers if t.kind == "token"]
    assert len(tokens) == 4 and all(t.nbytes == 2 and t.src_core == 1
                                    and t.dst_core is None for t in tokens)


def test_from_bundle_pp2_token_after_head(tmp_path) -> None:
    plan = from_bundle(_pp2_bundle(tmp_path), tmp_path, steps=1)
    token = [t for t in plan.transfers if t.kind == "token"][0]
    assert token.after_seg.startswith("head")


def test_plan_text_printable(tmp_path) -> None:
    plan = from_bundle(_pp2_bundle(tmp_path), tmp_path, steps=1)
    text = plan_text(plan)
    assert "prompt:c0" in text and "hidden_handoff" in text
    assert "8388608B" in text


def test_from_bundle_rejects_single_core_bundle(tmp_path) -> None:
    bundle = tmp_path / "bundle.json"
    bundle.write_text(json.dumps({"schema_version": 1, "graphs": []}))
    with pytest.raises(ValueError, match="cascade"):
        from_bundle(bundle, tmp_path, steps=1)


# ---- TP2 蓝图 ----

def test_from_bundle_tp2_segments_and_transfers(tmp_path) -> None:
    plan = from_bundle(_tp2_bundle(tmp_path, layers=2), tmp_path, steps=1)
    assert plan.strategy == "tp2"
    # prompt 相：每层 attn×2 + mlp×2 + head_shard×2，decode 步同构
    attn = [s for s in plan.segments if s.kind == "attn_shard"]
    mlp = [s for s in plan.segments if s.kind == "mlp_shard"]
    heads = [s for s in plan.segments if s.kind == "head_shard"]
    assert len(attn) == 2 * 2 * 2 and len(mlp) == 2 * 2 * 2 and len(heads) == 2 * 2
    a0 = [s for s in attn if s.seg_id == "prompt:L0:attn:c0"][0]
    assert a0.layer_lo == 0 and a0.layer_hi == 1
    assert a0.in_addrs["hidden"] == 0x100000000 + 0x2000
    # 归约：每层 2 次（attn、mlp）×（prompt + decode 步）
    ar = [t for t in plan.transfers if t.kind == "all_reduce"]
    assert len(ar) == 2 * 2 * 2
    prompt_ar = [t for t in ar if t.after_seg.startswith("prompt:L0:attn")]
    assert prompt_ar[0].nbytes == 4 * PROMPT_HIDDEN   # 读两核 + 写回两核
    decode_ar = [t for t in ar if t.after_seg.startswith("decode:0:L0:mlp")]
    assert decode_ar[0].nbytes == 4 * DECODE_HIDDEN
    gather = [t for t in plan.transfers if t.kind == "logits_gather"]
    assert len(gather) == 2 * 2 and all(t.nbytes == SHARD_LOGITS for t in gather)
    emb = [t for t in plan.transfers if t.kind == "embedding"]
    assert len(emb) == 4 and {t.dst_core for t in emb} == {0, 1}
