"""CEVA 窗口变体派生与 exact 防线（M4b 首片，docs/m0-ceva-probe-20261118.md）。

全部离线：小契约与小图目录，不加载模型、不启动厂商工具。窗口平移语义与
runtime/materialize.py 的 WindowSpec 对齐：变体之间只有地址摘要不同，文件逐字节共享。
"""

from __future__ import annotations

import subprocess
import sys
import types
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent

# 三节点链：入口缓冲 -> Gemm -> 出口缓冲；满足 gml_structure_check 五条规则。
_MINI_GML = '''graph [
  directed 1
  relay2gml_version "26.10.1"
  node [
    id 1
    node_id 1
    label "nprm_0_i0"
    name "nprm_0_i0"
    is_buffer 1
    output_buffer "input_buffer_3.bin"
  ]
  node [
    id 2
    node_id 2
    label "output"
    name "output"
    is_buffer 1
    input_buffer "input_buffer_2.bin"
  ]
  node [
    id 3
    node_id 3
    label "self_attn_q_proj_MatMul_qidx0_params_3"
    name "self_attn_q_proj_MatMul_qidx0_params_3"
    op_type "Gemm"
    input_buffer "input_buffer_3.bin"
    output_buffer "input_buffer_2.bin"
  ]
  edge [
    source 1
    target 3
    dims "1x1x4096"
  ]
  edge [
    source 3
    target 2
    dims "1x1x4096"
  ]
]
'''


def _decode_spec():
    """最小 decode 契约：hidden 进/出各 [1,1,4096] fp16，KV 出口 int8。"""
    from contracts.ceva import CevaGraphSpec, CevaIoBinding
    from contracts.compile_slots import CompileSlots

    io = (
        CevaIoBinding(direction="input", index=0, name="nprm_0_i0", label=None,
                      shape=(1, 1, 4096), dtype="float16", sf=1.0),
        CevaIoBinding(direction="input", index=1, name="nprm_0_i1", label=None,
                      shape=(1, 32, 1024, 128), dtype="int8", sf=1.0),
        CevaIoBinding(direction="output", index=0,
                      name="tvmgen_default_nprm_main_0_output_0", label="output",
                      shape=(1, 1, 4096), dtype="float16", sf=1.0),
        CevaIoBinding(direction="output", index=1,
                      name="tvmgen_default_nprm_main_0_output_1",
                      label="key_cache_out", shape=(1, 32, 1024, 128),
                      dtype="int8", sf=1.0),
    )
    return CevaGraphSpec(stage="decode", layer_id=0, query_tokens=1,
                         slots=CompileSlots(), version="26.10.1", io=io,
                         runtime_dir=Path("."))


def _write_mini_graph(bundle: Path) -> Path:
    """在包里落一张最小 decode 子图（GML + IO_info + 两个 bin）。"""
    directory = (Path("decode") / "model_layers_0_decode_v2" / "parser_output"
                 / "tvmgen_default_nprm_main_0" / "runtime_files")
    rf = bundle / directory
    rf.mkdir(parents=True)
    (rf / "relay2gml_graph.gml").write_text(_MINI_GML)
    (rf / "IO_info.txt").write_text("{}")
    (rf / "input_buffer_3.bin").write_bytes(b"\x01" * 16)
    (rf / "input_buffer_2.bin").write_bytes(b"\x02" * 8)
    return directory


# ---- exact 防线 ----


def test_exact_dispatch_rejected_at_boundary() -> None:
    """exact_dispatch 直接抛错并引用 M0 结论；dense/dynamic/未指定都放行。"""
    from gml_bridge.ceva import require_supported_moe_execution

    with pytest.raises(ValueError, match="docs/m0-ceva-probe-20261118.md"):
        require_supported_moe_execution("exact_dispatch")
    with pytest.raises(ValueError, match="dense_sim / dynamic_dispatch"):
        require_supported_moe_execution("exact_dispatch")
    for allowed in (None, "dense_sim", "dynamic_dispatch"):
        require_supported_moe_execution(allowed)


def test_bundle_rejects_exact_before_any_work(tmp_path) -> None:
    """export_ceva_bundle 把 exact 拦在第一句：不碰模型、不落任何文件。"""
    from gml_bridge.ceva import export_ceva_bundle

    with pytest.raises(ValueError, match="exact_dispatch"):
        export_ceva_bundle(None, tmp_path / "ref", tmp_path / "out", [],
                           moe_execution="exact_dispatch")
    assert not (tmp_path / "out").exists()


# ---- 窗口几何 ----


def test_window_geometry_from_decode_contract() -> None:
    """步长 = hidden 进+出两块边界激活（8KiB×2），4K 对齐；CEVA 无块表。"""
    from gml_bridge.ceva import window_geometry

    spec = _decode_spec()
    geometry = window_geometry(spec, pool=2)
    assert (geometry.pool, geometry.bt_stride) == (2, 0)
    assert geometry.act_stride == 16384
    with pytest.raises(ValueError, match="pool"):
        window_geometry(spec, pool=0)


# ---- 变体派生 ----


def test_derive_window_variants_shift_only(tmp_path) -> None:
    """变体 = 窗口平移：文件逐字节共享，记录只有目录与两个平移量不同。"""
    from gml_bridge.ceva import (CevaWindowGeometry, GML_NAME,
                                 derive_window_variants, parse_vendor_graph)

    directory = _write_mini_graph(tmp_path)
    geometry = CevaWindowGeometry(pool=3, act_stride=16384, bt_stride=0)
    records = derive_window_variants(tmp_path, directory, geometry)

    assert [r["window"] for r in records] == [0, 1, 2]
    assert records[0]["directory"] == str(directory)
    assert records[1]["directory"].endswith(
        "model_layers_0_decode_v2_w1/parser_output/"
        "tvmgen_default_nprm_main_0/runtime_files")
    assert [r["act_shift"] for r in records] == [0, 16384, 32768]
    assert {r["bt_shift"] for r in records} == {0}
    assert len({r["gml_sha256"] for r in records}) == 1

    # 每个变体目录与窗口 0 同名同字节；GML 解析结果也逐节点一致。
    base = tmp_path / directory
    base_files = {p.name: p.read_bytes() for p in sorted(base.iterdir())}
    for record in records[1:]:
        variant = tmp_path / record["directory"]
        assert {p.name: p.read_bytes() for p in sorted(variant.iterdir())} == base_files
        ours = parse_vendor_graph(variant)
        assert ours.nodes.keys() == parse_vendor_graph(base).nodes.keys()
        assert ours.edges == parse_vendor_graph(base).edges

    # 结构校验脚本对窗口 0 与派生变体都要通过（字节相同，逐个跑一遍确认）。
    for record in records:
        gml = tmp_path / record["directory"] / GML_NAME
        proc = subprocess.run(
            [sys.executable, str(REPO_ROOT / "scripts" / "gml_structure_check.py"),
             str(gml)], capture_output=True, text=True, timeout=120)
        assert proc.returncode == 0, proc.stdout + proc.stderr


def test_derive_window_variants_rerun_overwrites(tmp_path) -> None:
    """重跑派生会覆盖旧变体目录，不残留上一次的字节。"""
    from gml_bridge.ceva import CevaWindowGeometry, derive_window_variants

    directory = _write_mini_graph(tmp_path)
    geometry = CevaWindowGeometry(pool=2, act_stride=16384, bt_stride=0)
    derive_window_variants(tmp_path, directory, geometry)
    stale = tmp_path / str(directory).replace("_v2", "_v2_w1") / "stale.bin"
    stale.write_bytes(b"old")
    derive_window_variants(tmp_path, directory, geometry)
    assert not stale.exists()


# ---- bundle 接线与 CLI ----


def test_bundle_records_windows(tmp_path, monkeypatch) -> None:
    """export_ceva_bundle 写 windows 字段：池深、步长与每个变体的地址摘要。"""
    import json

    import torch

    from gml_bridge import ceva
    from quant.calibration import LlamaCalibration

    monkeypatch.setattr(ceva, "load_ceva_spec",
                        lambda *a, **k: _decode_spec())
    monkeypatch.setattr(
        ceva, "export_ceva_graph",
        lambda model, spec, out_dir, **k: _write_mini_graph_spec(spec, out_dir))
    # 参考目录给出本层子图，让 bundle_complete 能算。
    reference = tmp_path / "ref"
    vendor = ceva.graph_dir(reference, "decode", 0) / ceva.RUNTIME_DIR
    vendor.mkdir(parents=True)
    (vendor / ceva.GML_NAME).write_text(_MINI_GML)

    out = tmp_path / "bundle"
    model = torch.nn.Module()
    model.config = types.SimpleNamespace(hidden_size=64, num_attention_heads=4,
                                         intermediate_size=128,
                                         num_hidden_layers=1)
    manifest_path = ceva.export_ceva_bundle(
        model, reference, out, [("decode", 0)],
        use_opcompiler=False, calibration=LlamaCalibration(kv=[(0.1, 0.2)]),
        moe_execution="dynamic_dispatch", window_pool=2, log=lambda *_: None)
    manifest = json.loads(manifest_path.read_text())
    assert manifest["moe_execution"] == "dynamic_dispatch"
    entry = manifest["graphs"][0]
    windows = entry["windows"]
    assert (windows["pool"], windows["act_stride"], windows["bt_stride"]) == (2, 16384, 0)
    assert [(v["window"], v["act_shift"]) for v in windows["variants"]] \
        == [(0, 0), (1, 16384)]
    variant_dir = out / windows["variants"][1]["directory"]
    assert (variant_dir / ceva.GML_NAME).is_file()


def _write_mini_graph_spec(spec, out_dir) -> dict:
    """假导出：不跑 FX/量化，直接落最小图，返回一个最小清单条目。"""
    rf = Path(out_dir)
    rf.mkdir(parents=True, exist_ok=True)
    (rf / "relay2gml_graph.gml").write_text(_MINI_GML)
    (rf / "IO_info.txt").write_text("{}")
    (rf / "input_buffer_3.bin").write_bytes(b"\x01" * 16)
    (rf / "input_buffer_2.bin").write_bytes(b"\x02" * 8)
    files = [{"path": p.name, "size_bytes": p.stat().st_size,
              "sha256": "", "role": "generated", "source": None}
             for p in sorted(rf.iterdir())]
    return {"stage": spec.stage, "layer_id": spec.layer_id, "files": files}


def test_cli_window_and_moe_flags() -> None:
    """--moe-execution/--window-pool 可解析，默认不派生；只用于 CEVA 模式。"""
    from scripts.export_gml import build_parser

    args = build_parser().parse_args(
        ["--ceva-reference", "/r", "--moe-execution", "dynamic_dispatch",
         "--window-pool", "3"])
    assert (args.moe_execution, args.window_pool) == ("dynamic_dispatch", 3)
    defaults = build_parser().parse_args([])
    assert defaults.moe_execution is None and defaults.window_pool == 1


def test_cli_rejects_moe_flags_without_ceva(monkeypatch, capsys) -> None:
    """非 CEVA 模式给 --moe-execution 直接报参数错误，不静默接受。"""
    from scripts import export_gml

    monkeypatch.setattr(sys, "argv",
                        ["export_gml.py", "--moe-execution", "dense_sim"])
    with pytest.raises(SystemExit) as exc:
        export_gml.main()
    assert exc.value.code == 2 and "--ceva-reference" in capsys.readouterr().err


def test_cli_ceva_exact_fails_before_model_load(monkeypatch) -> None:
    """CEVA 模式 + exact_dispatch 在加载模型前就抛 M0 结论，不白载权重。"""
    from scripts import export_gml

    def boom(*a, **k):
        raise AssertionError("不应加载模型")

    monkeypatch.setattr(export_gml, "_load_model", boom)
    monkeypatch.setattr(
        sys, "argv", ["export_gml.py", "--ceva-reference", "/r",
                      "--use-opcompiler", "--moe-execution", "exact_dispatch"])
    with pytest.raises(ValueError, match="docs/m0-ceva-probe-20261118.md"):
        export_gml.main()
