"""genesim 端到端仿真的慢速冒烟。

仿真本身要十几分钟，不进快速回归（`-m "not slow"` 排除）。这里真的调用
流水线的仿真步骤，而不是只读上一次的产物——只读产物的话，仿真彻底坏掉时
只要六小时内有人手工跑过一次，测试照样是绿的。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from genesim_bridge.paths import genesim_root
from scripts.run_full_pipeline import limited_config, step_e_simulate

# 与流水线默认配置一致：tp2_pp4，按算子编译器实测的分块算成本。
_CONFIG = "sim_llama2_7b_pp_tp2pp4_globalcost.yaml"


@pytest.mark.slow
def test_simulation_produces_a_summary(tmp_path: Path) -> None:
    """跑一次仿真，核对产物里的请求数与输出 token。

    `step_e_simulate` 每次使用本次运行号下的独立空 trace 目录，所以这次
    运行的每个算子都重新走过 trace 生成；它内部还会核对 pim mlir 的来源
    （见流水线的 `verify_trace_provenance`），模板产物不会蒙混过关。
    """
    genesim = genesim_root()
    summary = step_e_simulate(
        genesim, _CONFIG, tmp_path, label="pytest",
        results_dir=tmp_path / "results",
    )

    summary_path = tmp_path / "results" / "summary.json"
    assert summary_path.is_file(), f"仿真没产出 {summary_path}"
    assert json.loads(summary_path.read_text()) == summary

    assert summary["completed_requests"] > 0, summary["completed_requests"]
    assert summary["processed_tokens"] > 0, summary["processed_tokens"]
    assert summary["total_time_s"] > 0
    assert summary["throughput_tokens_per_s"] > 0
    # processed_tokens 是 prompt + generated，只处理了 prompt 也会 > 0。
    # 需求 P1 要的是输出 token，钉输出侧的吞吐。
    assert summary["throughput_output_tokens_per_s"] > 0, summary


def test_default_request_count_follows_the_config() -> None:
    """不带参数跑时，仿真请求数跟随配置，不另设默认上限。

    评审 r2 问题 2：`--max-requests` 默认 1 把端到端仿真的默认覆盖从配置里的
    10 个请求降到 1 个，而文档记录的数字是 10 个请求。默认应跟随配置，
    需要收短时显式传参。
    """
    import sys

    from scripts.run_full_pipeline import parse_args

    saved = sys.argv
    sys.argv = ["run_full_pipeline.py"]
    try:
        args = parse_args()
    finally:
        sys.argv = saved
    assert args.max_requests is None, args.max_requests


def test_limited_config_caps_the_request_count(tmp_path: Path) -> None:
    """默认只仿 1 个请求，覆盖 prefill 与 decode 即可。

    10 个请求共 8084 个 token，仿真要二十多分钟。第 1 个请求是 726 个输入
    token、69 个输出 token，两种阶段都有。原配置不能改，genesim 仓的 yaml
    还要能复现完整数字。
    """
    genesim = genesim_root()
    source = genesim / "conf" / _CONFIG
    out = limited_config(source, tmp_path / "conf", max_requests=1)
    import yaml
    loaded = yaml.safe_load(out.read_text())
    assert loaded["trace"]["num_requests"] == 1
    assert yaml.safe_load(source.read_text())["trace"]["num_requests"] == 10


def _write_fake_sim_setup(tmp_path: Path, *, with_pimir: bool) -> tuple[Path, Path]:
    """搭一个最小的「仿真 IR + sidecar + 配置」三件套，返回 (假 genesim 根, 配置路径)。"""
    genesim = tmp_path / "genesim"
    (genesim / "models").mkdir(parents=True)
    ir = {
        "operators": [
            {"op_id": 0, "op_type": "MODEL_INPUT"},
            {"op_id": 1, "op_type": "GEMM"},
            {"op_id": 2, "op_type": "ROPE"},
            {"op_id": 3, "op_type": "MODEL_OUTPUT"},
        ]
    }
    (genesim / "models" / "sim.ir").write_text(json.dumps(ir))
    sidecar = {"operators": {"1": {"pimir_path": "/x/1.pimir.mlir"}}}
    if with_pimir:
        sidecar["operators"]["2"] = {"pimir_path": "/x/2.pimir.mlir"}
    (genesim / "models" / "placement.json").write_text(json.dumps(sidecar))
    config = tmp_path / "sim.yaml"
    config.write_text(
        'model:\n  ir_path: "models/sim.ir"\n'
        'scheduler:\n  compiler_placement_file: "models/placement.json"\n'
    )
    return genesim, config


def test_verify_sim_ir_matches_sidecar_accepts_matching_pair(tmp_path: Path) -> None:
    """IR 与 sidecar 同套编号时放行（入口/出口算子本就不在 sidecar 里）。"""
    from scripts.run_full_pipeline import verify_sim_ir_matches_sidecar

    genesim, config = _write_fake_sim_setup(tmp_path, with_pimir=True)
    verify_sim_ir_matches_sidecar(genesim, config)


def test_verify_sim_ir_matches_sidecar_rejects_stale_ir(tmp_path: Path) -> None:
    """注意力算子查不到 pimir_path 时必须在仿真启动前报出来。"""
    from scripts.run_full_pipeline import StepFailed, verify_sim_ir_matches_sidecar

    genesim, config = _write_fake_sim_setup(tmp_path, with_pimir=False)
    with pytest.raises(StepFailed, match="ROPE"):
        verify_sim_ir_matches_sidecar(genesim, config)


def test_verify_sim_ir_matches_sidecar_skips_without_sidecar(tmp_path: Path) -> None:
    """配置里没有 compiler_placement_file 时没有可对拍对象，直接返回。"""
    from scripts.run_full_pipeline import verify_sim_ir_matches_sidecar

    config = tmp_path / "sim.yaml"
    config.write_text('model:\n  ir_path: "models/sim.ir"\n')
    verify_sim_ir_matches_sidecar(tmp_path, config)


def _write_trace(
    traces: Path, op_id: int, op_type: str,
    source: str | None, *, parent: int = -1,
    pimir_sha256: str | None = None, num_instructions: int = 0,
    expert_id: int | None = None,
    pimir: dict | None = None,
    input_shapes: list | None = None,
    output_shapes: list | None = None,
) -> None:
    """按 GeneSim `save_trace_file` 的布局写一条最小 trace。"""
    import struct

    traces.mkdir(parents=True, exist_ok=True)
    signature = {} if source is None else {"trace_source": source}
    if pimir_sha256 is not None:
        signature["pimir_sha256"] = pimir_sha256
    meta = {
        "op_id": op_id,
        "op_type": op_type,
        "num_instructions": num_instructions,
        "compile_signature": signature,
    }
    if expert_id is not None:
        meta["expert_id"] = expert_id
    if parent >= 0:
        meta["parent_op_id"] = parent
    if pimir is not None:
        meta["pimir"] = pimir
    if input_shapes is not None:
        meta["input_shapes"] = input_shapes
    if output_shapes is not None:
        meta["output_shapes"] = output_shapes
    body = json.dumps(meta).encode()
    header = b"PIMT" + struct.pack("<I", 1) + struct.pack("<Q", 0) \
        + struct.pack("<I", len(body))
    (traces / f"op_{op_id}_{op_type}.pim_trace").write_bytes(header + body)


def test_trace_without_a_source_field_is_rejected(tmp_path: Path) -> None:
    """无 trace_source 的 trace 以 unknown 计数，必须拒绝而不是放行。"""
    from scripts.run_full_pipeline import StepFailed, verify_trace_provenance

    genesim, config = _trace_provenance_setup(tmp_path)
    _write_trace(genesim / "pim_traces", 7, "ROPE", None)
    with pytest.raises(StepFailed, match="unknown"):
        verify_trace_provenance(genesim, config)


def _trace_provenance_setup(tmp_path: Path) -> tuple[Path, Path]:
    """搭一份「仿真 IR + sidecar + 配置」，返回 (genesim 根, 配置路径)。"""
    genesim = tmp_path / "genesim"
    models = genesim / "models"
    models.mkdir(parents=True)
    (models / "sim.ir").write_text(json.dumps({
        "operators": [
            {"op_id": 1, "op_type": "GEMM"},
            {"op_id": 2, "op_type": "ROPE"},
        ],
        "dependencies": [],
        "subgraphs": [[1, 2]],
    }))
    (models / "placement.json").write_text(json.dumps({
        "operators": {
            "1": {"pimir_path": "/x/1.pimir.mlir"},
            "2": {"pimir_path": "/x/2.pimir.mlir"},
        },
    }))
    config = tmp_path / "sim.yaml"
    config.write_text(
        'model:\n  ir_path: "models/sim.ir"\n'
        'scheduler:\n  compiler_placement_file: "models/placement.json"\n'
    )
    return genesim, config


def test_trace_provenance_rejects_a_missing_required_op(tmp_path: Path) -> None:
    """源 IR 要求的算子没有 pimir trace 时必须失败，不能只看来源计数。"""
    from scripts.run_full_pipeline import StepFailed, verify_trace_provenance

    genesim, config = _trace_provenance_setup(tmp_path)
    _write_trace(genesim / "pim_traces", 1, "GEMM", "pimir")
    with pytest.raises(StepFailed, match="没有 pimir 来源"):
        verify_trace_provenance(genesim, config)


def test_trace_provenance_accepts_expanded_shards_as_parent_coverage(
    tmp_path: Path,
) -> None:
    """TP 展开出的分片用 parent_op_id 回指源算子，不能误报缺算子。"""
    from scripts.run_full_pipeline import verify_trace_provenance

    genesim, config = _trace_provenance_setup(tmp_path)
    _write_trace(genesim / "pim_traces", 100, "GEMM", "pimir", parent=1)
    _write_trace(genesim / "pim_traces", 101, "ROPE", "pimir", parent=2)
    verify_trace_provenance(genesim, config)


def test_audit_sidecar_coverage_rejects_missing_device_op(tmp_path: Path) -> None:
    """覆盖集合由源 IR 定义：缺条目的设备算子不能因落点不同而消失。"""
    from scripts.run_full_pipeline import StepFailed, audit_sidecar_coverage

    ir = {
        "operators": [
            {"op_id": 0, "op_type": "MODEL_INPUT"},
            {"op_id": 1, "op_type": "GEMM", "semantic_role": "q_proj",
             "expert_id": -1},
            {"op_id": 2, "op_type": "ROPE", "device_hint": "pim"},
            {"op_id": 3, "op_type": "MODEL_OUTPUT"},
        ],
        "dependencies": [], "subgraphs": [[1, 2]],
    }
    sidecar = {"operators": {
        "1": {"op_type": "GEMM", "semantic_role": "q_proj",
              "pimir_path": "/x/1.pimir.mlir"},
    }}
    with pytest.raises(StepFailed, match="ROPE"):
        audit_sidecar_coverage(ir, sidecar)


def test_audit_sidecar_coverage_rejects_an_unknown_exemption() -> None:
    """未知算子不能借 pimir_exempt_ops 绕过覆盖校验。"""
    from scripts.run_full_pipeline import StepFailed, audit_sidecar_coverage

    ir = {
        "operators": [
            {"op_id": 0, "op_type": "GEMM", "expert_id": -1},
            {"op_id": 1, "op_type": "BRAND_NEW_OP", "device_hint": "pim"},
        ],
        "dependencies": [], "subgraphs": [[0, 1]],
    }
    sidecar = {
        "operators": {
            "0": {"op_type": "GEMM", "pimir_path": "/x/0.pimir.mlir"},
        },
        "pimir_exempt_ops": {
            "1": {"op_type": "BRAND_NEW_OP", "reason": "未知算子"},
        },
    }
    with pytest.raises(StepFailed, match="豁免"):
        audit_sidecar_coverage(ir, sidecar)


def test_audit_sidecar_coverage_accepts_legitimate_exemptions() -> None:
    """模型边界、host 执行与非 DPU 稠密 GEMM 是合法豁免。"""
    from scripts.run_full_pipeline import audit_sidecar_coverage

    ir = {
        "operators": [
            {"op_id": 0, "op_type": "MODEL_INPUT"},
            {"op_id": 1, "op_type": "MOE_ROUTE", "device_hint": "cpu"},
            {"op_id": 2, "op_type": "GEMM", "semantic_role": "gate_proj",
             "expert_id": -1, "device_hint": "cpu"},
            {"op_id": 3, "op_type": "ROPE", "device_hint": "pim"},
            {"op_id": 4, "op_type": "MODEL_OUTPUT"},
        ],
        "dependencies": [], "subgraphs": [[1, 2, 3]],
    }
    sidecar = {
        "operators": {
            "3": {"op_type": "ROPE", "pimir_path": "/x/3.pimir.mlir"},
        },
        # host GEMM 的落点证据单独存 `host_gemms`，不混进设备条目。
        "host_gemms": {
            "2": {"op_type": "GEMM", "device_hint": "cpu",
                  "semantic_role": "gate_proj",
                  "weight": "model.layers.0.mlp.gate_proj.weight"},
        },
        "pimir_exempt_ops": {
            "1": {"op_type": "MOE_ROUTE", "device_hint": "cpu",
                  "reason": "host"},
            "2": {"op_type": "GEMM", "device_hint": "cpu",
                  "semantic_role": "gate_proj", "expert_id": -1,
                  "reason": "非 DPU GEMM"},
        },
    }
    audit_sidecar_coverage(ir, sidecar)


def _expert_gemm_ir(placements):
    """构造每层两个专家 GEMM 的源 IR；placements 给 (层, 角色, 专家)。"""
    operators, subgraphs = [], {}
    for op_id, (layer, role, expert) in enumerate(placements):
        operators.append({
            "op_id": op_id, "op_type": "GEMM", "semantic_role": role,
            "expert_id": expert,
        })
        subgraphs.setdefault(layer, []).append(op_id)
    return {
        "operators": operators, "dependencies": [],
        "subgraphs": [subgraphs[layer] for layer in sorted(subgraphs)],
        "moe_num_experts": 2,
    }


def _expert_entries(placements, weights):
    """按 op_id 生成 sidecar 条目；weights 给每条的权重路径。"""
    return {
        str(op_id): {
            "op_type": "GEMM", "semantic_role": role, "expert_id": expert,
            "weight": weight, "pimir_path": f"/x/{op_id}.pimir.mlir",
        }
        for op_id, ((_, role, expert), weight) in enumerate(
            zip(placements, weights))
    }


def test_audit_sidecar_coverage_rejects_swapped_expert_identity() -> None:
    """同层两个专家的专家号互换：集合相同，逐 op_id 身份必须抓出来。"""
    from scripts.run_full_pipeline import StepFailed, audit_sidecar_coverage

    ir = _expert_gemm_ir([(0, "gate_proj", 0), (0, "gate_proj", 1)])
    entries = _expert_entries(
        [(0, "gate_proj", 0), (0, "gate_proj", 1)],
        ["model.layers.0.mlp.experts.1.gate_proj.weight",
         "model.layers.0.mlp.experts.0.gate_proj.weight"])
    with pytest.raises(StepFailed, match="身份"):
        audit_sidecar_coverage(ir, {"operators": entries})


def test_audit_sidecar_coverage_rejects_a_cross_layer_expert_swap() -> None:
    """跨层互换专家权重：专家号集合不变，权重路径的层号必须逐条核对。"""
    from scripts.run_full_pipeline import StepFailed, audit_sidecar_coverage

    ir = _expert_gemm_ir([(0, "gate_proj", 0), (1, "gate_proj", 0)])
    entries = _expert_entries(
        [(0, "gate_proj", 0), (1, "gate_proj", 0)],
        ["model.layers.1.mlp.experts.0.gate_proj.weight",
         "model.layers.0.mlp.experts.0.gate_proj.weight"])
    with pytest.raises(StepFailed, match="身份"):
        audit_sidecar_coverage(ir, {"operators": entries})


def test_audit_sidecar_coverage_accepts_matching_expert_weights() -> None:
    """合法产物逐条身份一致时放行，不能把真实专家路径误判成异常。"""
    from scripts.run_full_pipeline import audit_sidecar_coverage

    placements = [(0, "gate_proj", 0), (0, "gate_proj", 1)]
    ir = _expert_gemm_ir(placements)
    entries = _expert_entries(placements, [
        "model.layers.0.mlp.experts.0.gate_proj.weight",
        "model.layers.0.mlp.experts.1.gate_proj.weight"])
    audit_sidecar_coverage(ir, {"operators": entries})


def _router_exempt_ir(device_hint: str):
    return {
        "operators": [{
            "op_id": 0, "op_type": "GEMM", "semantic_role": "router",
            "expert_id": -1, "device_hint": device_hint,
        }],
        "dependencies": [], "subgraphs": [[0]],
    }


def _router_exempt_entry(device_hint: str):
    return {"0": {
        "op_type": "GEMM", "device_hint": device_hint,
        "semantic_role": "router", "expert_id": -1,
        "weight": "model.layers.0.mlp.gate.weight",
    }}


@pytest.mark.parametrize("role", ["router", "q_proj", "lm_head"])
def test_audit_rejects_exempting_any_device_dense_gemm(role) -> None:
    """router/q_proj/lm_head 都落在 DPU 上，删 pimir 后不能自动豁免。"""
    from scripts.run_full_pipeline import StepFailed, audit_sidecar_coverage

    ir = {
        "operators": [{
            "op_id": 0, "op_type": "GEMM", "semantic_role": role,
            "expert_id": -1, "device_hint": "pim",
        }],
        "dependencies": [], "subgraphs": [[0]],
    }
    sidecar = {
        "operators": {},
        "pimir_exempt_ops": {"0": {
            "op_type": "GEMM", "device_hint": "cpu",
            "semantic_role": role, "expert_id": -1,
            "weight": f"model.layers.0.{role}.weight",
            "reason": "GEMM 权重不在 DPU 上",
        }},
    }
    with pytest.raises(StepFailed, match="豁免"):
        audit_sidecar_coverage(ir, sidecar)


def test_audit_sidecar_coverage_rejects_an_exempted_router_gemm() -> None:
    """设备 router 缺 pimir 不能按「非 DPU GEMM」豁免：落点证据是 pim。"""
    from scripts.run_full_pipeline import StepFailed, audit_sidecar_coverage

    with pytest.raises(StepFailed, match="豁免"):
        audit_sidecar_coverage(
            _router_exempt_ir("pim"),
            {"operators": {}, "pimir_exempt_ops": _router_exempt_entry("cpu")})


def test_audit_sidecar_coverage_rejects_a_forged_host_evidence() -> None:
    """豁免条目自报 host，但源 IR 的落点是 pim 时必须失败。"""
    from scripts.run_full_pipeline import StepFailed, audit_sidecar_coverage

    with pytest.raises(StepFailed, match="豁免"):
        audit_sidecar_coverage(
            _router_exempt_ir("pim"),
            {"operators": {}, "pimir_exempt_ops": _router_exempt_entry("pim")})


def test_audit_sidecar_coverage_accepts_a_host_dense_gemm_exemption() -> None:
    """真正落在 host 的稠密 GEMM 有落点证据，仍属合法豁免。"""
    from scripts.run_full_pipeline import audit_sidecar_coverage

    ir = {
        "operators": [{
            "op_id": 0, "op_type": "GEMM", "semantic_role": "gate_proj",
            "expert_id": -1, "device_hint": "cpu",
        }],
        "dependencies": [], "subgraphs": [[0]],
    }
    sidecar = {
        "operators": {},
        "host_gemms": {"0": {
            "op_type": "GEMM", "device_hint": "cpu",
            "semantic_role": "gate_proj",
            "weight": "model.layers.0.mlp.gate_proj.weight",
        }},
        "pimir_exempt_ops": {"0": {
            "op_type": "GEMM", "device_hint": "cpu",
            "semantic_role": "gate_proj", "expert_id": -1,
            "reason": "GEMM 权重不在 DPU 上，不产生 pimir",
        }},
    }
    audit_sidecar_coverage(ir, sidecar)


def test_audit_rejects_an_exempt_op_with_a_pim_placement_entry() -> None:
    """豁免条目说 host、sidecar 放置条目却说 PIM：两处矛盾必须失败。"""
    from scripts.run_full_pipeline import StepFailed, audit_sidecar_coverage

    ir = {
        "operators": [{
            "op_id": 0, "op_type": "MOE_ROUTE", "device_hint": "cpu",
        }],
        "dependencies": [], "subgraphs": [[0]],
    }
    sidecar = {
        "operators": {"0": {
            "op_type": "MOE_ROUTE", "device_hint": "pim",
            "shards": [{"dpu_id": 0, "local_in_features": 8,
                        "local_out_features": 8}],
            "pimir_path": "/x/0.pimir.mlir",
        }},
        "pimir_exempt_ops": {"0": {
            "op_type": "MOE_ROUTE", "device_hint": "cpu",
            "semantic_role": "", "reason": "host 执行",
        }},
    }
    with pytest.raises(StepFailed, match="豁免"):
        audit_sidecar_coverage(ir, sidecar)


def test_audit_rejects_a_boundary_with_a_pim_placement_entry() -> None:
    """模型边界同样不能一边豁免、一边被钉到设备上执行。"""
    from scripts.run_full_pipeline import StepFailed, audit_sidecar_coverage

    ir = {
        "operators": [{"op_id": 0, "op_type": "MODEL_INPUT"}],
        "dependencies": [], "subgraphs": [],
    }
    sidecar = {
        "operators": {"0": {
            "op_type": "MODEL_INPUT", "device_hint": "pim",
            "pimir_path": "/x/0.pimir.mlir",
        }},
        "pimir_exempt_ops": {"0": {
            "op_type": "MODEL_INPUT", "device_hint": "auto",
            "reason": "模型入口/出口，没有设备算子",
        }},
    }
    with pytest.raises(StepFailed, match="豁免"):
        audit_sidecar_coverage(ir, sidecar)


def _host_gemm_case(host_gemms=None):
    ir = {
        "operators": [{
            "op_id": 0, "op_type": "GEMM", "semantic_role": "gate_proj",
            "expert_id": -1, "device_hint": "cpu",
        }],
        "dependencies": [], "subgraphs": [[0]],
    }
    sidecar = {
        "operators": {},
        "pimir_exempt_ops": {"0": {
            "op_type": "GEMM", "device_hint": "cpu",
            "semantic_role": "gate_proj", "expert_id": -1,
            "weight": "model.layers.0.mlp.gate_proj.weight",
            "reason": "GEMM 权重不在 DPU 上",
        }},
    }
    if host_gemms is not None:
        sidecar["host_gemms"] = host_gemms
    return ir, sidecar


def test_audit_rejects_a_host_gemm_exemption_without_evidence() -> None:
    """host GEMM 豁免必须带 `host_gemms` 落点证据，不能手写一条豁免。"""
    from scripts.run_full_pipeline import StepFailed, audit_sidecar_coverage

    ir, sidecar = _host_gemm_case()
    with pytest.raises(StepFailed, match="豁免|host_gemms"):
        audit_sidecar_coverage(ir, sidecar)


def test_audit_rejects_a_host_gemm_exemption_with_a_mismatched_role() -> None:
    """`host_gemms` 的角色或权重身份与源 IR 不符时，豁免同样不成立。"""
    from scripts.run_full_pipeline import StepFailed, audit_sidecar_coverage

    for evidence in (
        {"op_type": "GEMM", "device_hint": "cpu",
         "semantic_role": "up_proj",
         "weight": "model.layers.0.mlp.gate_proj.weight"},
        {"op_type": "GEMM", "device_hint": "cpu",
         "semantic_role": "gate_proj"},
    ):
        ir, sidecar = _host_gemm_case({"0": evidence})
        with pytest.raises(StepFailed, match="豁免|host_gemms"):
            audit_sidecar_coverage(ir, sidecar)


def test_audit_rejects_a_host_gemm_exemption_with_a_forged_weight() -> None:
    """豁免条目里的权重必须与 `host_gemms` 的落点证据一致。"""
    from scripts.run_full_pipeline import StepFailed, audit_sidecar_coverage

    ir, sidecar = _host_gemm_case({
        "0": {"op_type": "GEMM", "device_hint": "cpu",
              "semantic_role": "gate_proj",
              "weight": "model.layers.0.mlp.gate_proj.weight"},
    })
    sidecar["pimir_exempt_ops"]["0"]["weight"] = "model.layers.0.mlp.up_proj.weight"
    with pytest.raises(StepFailed, match="豁免|host_gemms"):
        audit_sidecar_coverage(ir, sidecar)


def test_trace_provenance_rejects_an_exempted_router_gemm(tmp_path: Path) -> None:
    """trace 验收同样要核对豁免：设备 router 不能靠豁免从应覆盖集合消失。"""
    from scripts.run_full_pipeline import StepFailed, verify_trace_provenance

    genesim = tmp_path / "genesim"
    models = genesim / "models"
    models.mkdir(parents=True)
    (models / "sim.ir").write_text(json.dumps({
        "operators": [
            {"op_id": 1, "op_type": "GEMM", "semantic_role": "router",
             "expert_id": -1, "device_hint": "pim"},
        ],
        "dependencies": [], "subgraphs": [[1]],
    }))
    (models / "placement.json").write_text(json.dumps({
        "operators": {},
        "pimir_exempt_ops": {"1": {
            "op_type": "GEMM", "device_hint": "cpu",
            "semantic_role": "router", "expert_id": -1,
            "reason": "GEMM 权重不在 DPU 上",
        }},
    }))
    config = tmp_path / "sim.yaml"
    config.write_text(
        'model:\n  ir_path: "models/sim.ir"\n'
        'scheduler:\n  compiler_placement_file: "models/placement.json"\n'
    )
    # 先给一条合法 trace 走到豁免校验；旧实现会把它从应覆盖集合里排除后放行。
    _write_trace(genesim / "pim_traces", 1, "GEMM", "pimir")
    with pytest.raises(StepFailed, match="豁免"):
        verify_trace_provenance(genesim, config)


def test_sidecar_freshness_rejects_swapped_expert_identity(
    tmp_path: Path,
) -> None:
    """哈希只证明文件没变；仿真前复核也要逐 op_id 查专家身份。"""
    import hashlib

    from scripts.run_full_pipeline import StepFailed, verify_sidecar_freshness

    genesim = tmp_path / "genesim"
    models = genesim / "models"
    models.mkdir(parents=True)
    pimir = tmp_path / "kernel.pimir.mlir"
    pimir.write_text("// same kernel\n")
    digest = hashlib.sha256(pimir.read_text().encode()).hexdigest()
    (models / "sim.ir").write_text(json.dumps({
        "operators": [
            {"op_id": 1, "op_type": "GEMM", "semantic_role": "gate_proj",
             "expert_id": 0},
            {"op_id": 2, "op_type": "GEMM", "semantic_role": "gate_proj",
             "expert_id": 1},
        ],
        "dependencies": [], "subgraphs": [[1, 2]],
        "moe_num_experts": 2, "moe_top_k": 1, "moe_intermediate_size": 16,
        "moe_execution": "dynamic_dispatch",
    }))
    (models / "placement.json").write_text(json.dumps({
        "moe_config": {"num_experts": 2, "top_k": 1,
                       "intermediate_size": 16,
                       "execution": "dynamic_dispatch"},
        "operators": {
            "1": {"op_type": "GEMM", "semantic_role": "gate_proj",
                  "expert_id": 1,
                  "weight": "model.layers.0.mlp.experts.1.gate_proj.weight",
                  "pimir_path": str(pimir), "pimir_sha256": digest},
            "2": {"op_type": "GEMM", "semantic_role": "gate_proj",
                  "expert_id": 0,
                  "weight": "model.layers.0.mlp.experts.0.gate_proj.weight",
                  "pimir_path": str(pimir), "pimir_sha256": digest},
        },
    }))
    config = tmp_path / "sim.yaml"
    config.write_text(
        'model:\n  ir_path: "models/sim.ir"\n'
        'scheduler:\n  compiler_placement_file: "models/placement.json"\n'
    )
    with pytest.raises(StepFailed, match="身份"):
        verify_sidecar_freshness(genesim, config)


def test_trace_provenance_rejects_swapped_expert_identity(
    tmp_path: Path,
) -> None:
    """trace 验收也要逐 op_id 查专家身份，不能只看来源与应覆盖集合。"""
    from scripts.run_full_pipeline import StepFailed, verify_trace_provenance

    genesim = tmp_path / "genesim"
    models = genesim / "models"
    models.mkdir(parents=True)
    (models / "sim.ir").write_text(json.dumps({
        "operators": [
            {"op_id": 1, "op_type": "GEMM", "semantic_role": "gate_proj",
             "expert_id": 0},
            {"op_id": 2, "op_type": "GEMM", "semantic_role": "gate_proj",
             "expert_id": 1},
        ],
        "dependencies": [], "subgraphs": [[1, 2]],
        "moe_num_experts": 2, "moe_top_k": 1, "moe_intermediate_size": 16,
        "moe_execution": "dynamic_dispatch",
    }))
    (models / "placement.json").write_text(json.dumps({
        "moe_config": {"num_experts": 2, "top_k": 1,
                       "intermediate_size": 16,
                       "execution": "dynamic_dispatch"},
        "operators": {
            "1": {"op_type": "GEMM", "semantic_role": "gate_proj",
                  "expert_id": 1,
                  "weight": "model.layers.0.mlp.experts.1.gate_proj.weight",
                  "pimir_path": "/x/1.pimir.mlir"},
            "2": {"op_type": "GEMM", "semantic_role": "gate_proj",
                  "expert_id": 0,
                  "weight": "model.layers.0.mlp.experts.0.gate_proj.weight",
                  "pimir_path": "/x/2.pimir.mlir"},
        },
    }))
    config = tmp_path / "sim.yaml"
    config.write_text(
        'model:\n  ir_path: "models/sim.ir"\n'
        'scheduler:\n  compiler_placement_file: "models/placement.json"\n'
    )
    _write_trace(genesim / "pim_traces", 1, "GEMM", "pimir")
    _write_trace(genesim / "pim_traces", 2, "GEMM", "pimir")
    with pytest.raises(StepFailed, match="身份"):
        verify_trace_provenance(genesim, config)


def test_audit_sidecar_coverage_requires_every_moe_expert_role() -> None:
    """MoE 必须按 (层, 角色, 专家号) 全覆盖，不能只取专家号并集。"""
    from scripts.run_full_pipeline import StepFailed, audit_sidecar_coverage

    operators = []
    entries = {}
    for expert in range(2):
        for role in ("gate_proj", "up_proj", "down_proj"):
            op_id = len(operators)
            operators.append({
                "op_id": op_id, "op_type": "GEMM",
                "semantic_role": role, "expert_id": expert,
            })
            if (expert, role) != (1, "up_proj"):
                entries[str(op_id)] = {
                    "op_type": "GEMM", "semantic_role": role,
                    "expert_id": expert,
                    "weight": (
                        f"model.layers.0.mlp.experts.{expert}.{role}.weight"),
                    "pimir_path": f"/x/{op_id}.pimir.mlir",
                }
    ir = {
        "operators": operators,
        "dependencies": [],
        "subgraphs": [[op["op_id"] for op in operators]],
        "moe_num_experts": 2,
    }
    with pytest.raises(StepFailed, match="身份"):
        audit_sidecar_coverage(ir, {"operators": entries})


def test_skip_simulation_runs_the_coverage_audit(tmp_path, monkeypatch) -> None:
    """--skip-simulation 也必须做覆盖校验：跳过仿真后没有 trace 可查。"""
    from types import SimpleNamespace

    import scripts.run_full_pipeline as pipeline

    genesim = tmp_path / "genesim"
    models = genesim / "models"
    models.mkdir(parents=True)
    refined = models / "m_pimir.ir"
    refined.write_text(json.dumps({
        "operators": [{"op_id": 0, "op_type": "ROPE"}],
        "dependencies": [], "subgraphs": [[0]],
    }))
    sidecar = models / "m_placement.json"
    sidecar.write_text(json.dumps({"operators": {}}))

    args = SimpleNamespace(
        backend="genesim", run_name=None, token_check="warn",
        decode_steps=1, num_stages=2, num_dpus=4,
        model_dir=str(tmp_path / "model"), moe_execution=None,
        max_requests=1, skip_simulation=True, ab_compare=False,
        log_dir=str(tmp_path / "logs"), no_pu_mapping=True,
    )
    monkeypatch.setattr(pipeline, "parse_args", lambda: args)
    monkeypatch.setattr(pipeline, "genesim_root", lambda: genesim)
    monkeypatch.setattr(
        pipeline, "llama2_7b_model_dir", lambda: tmp_path / "model")
    monkeypatch.setattr(
        pipeline, "architecture_of",
        lambda path: ("llama", SimpleNamespace(
            artifact="m", required_roles={"q_proj"}, sim_config=None)))
    monkeypatch.setattr(pipeline, "_resolve_moe_execution", lambda *a: "")
    monkeypatch.setattr(
        pipeline, "step_a_model_ir", lambda *a, **k: models / "m.ir")
    monkeypatch.setattr(
        pipeline, "step_bcd_export", lambda *a, **k: sidecar)
    monkeypatch.setattr(
        pipeline, "step_a2_refine_pimir_ir", lambda *a, **k: refined)

    assert pipeline.main() == 1


def test_ir_op_ids_reads_the_sequence_in_order(tmp_path: Path) -> None:
    """编号序列是精化 IR 与图骨架「同套」的判据，顺序不同就算不同。"""
    from scripts.run_full_pipeline import _ir_op_ids

    a = tmp_path / "a.ir"
    b = tmp_path / "b.ir"
    a.write_text(json.dumps({"operators": [{"op_id": 0}, {"op_id": 1}, {"op_id": 2}]}))
    b.write_text(json.dumps({"operators": [{"op_id": 0}, {"op_id": 2}, {"op_id": 1}]}))
    assert _ir_op_ids(a) == [0, 1, 2]
    assert _ir_op_ids(a) != _ir_op_ids(b)


def _fused_activation_case(*, absorbed=(2,), manifest=True,
                           roles=("gate_proj",), extra_consumer=False):
    """一份「gate GEMM → SILU → down GEMM」的最小 IR 与 sidecar。

    `extra_consumer` 给主 GEMM 再连一个读者，模拟「融合会把主算子结果改成
    激活后的值、另一个读者却要激活前的值」的非法折叠。
    """
    ir = {
        "operators": [
            {"op_id": 0, "op_type": "MODEL_INPUT"},
            {"op_id": 1, "op_type": "GEMM", "semantic_role": "gate_proj",
             "expert_id": 0},
            {"op_id": 2, "op_type": "SILU"},
            {"op_id": 3, "op_type": "GEMM", "semantic_role": "down_proj",
             "expert_id": 0},
            {"op_id": 4, "op_type": "MODEL_OUTPUT"},
        ],
        "dependencies": [
            {"src_op_id": 1, "dst_op_id": 2},
            {"src_op_id": 2, "dst_op_id": 3},
        ],
        "subgraphs": [[1, 2, 3]],
    }
    if extra_consumer:
        ir["dependencies"].append({"src_op_id": 1, "dst_op_id": 3})
    sidecar = {
        "operators": {
            "1": {"op_type": "GEMM", "semantic_role": "gate_proj",
                  "expert_id": 0,
                  "weight": "model.layers.0.mlp.experts.0.gate_proj.weight",
                  "pimir_path": "/x/1.pimir.mlir"},
            # B+C+D 阶段给被折的激活也编了产物；它只是不生成 trace。
            "2": {"op_type": "SILU", "pimir_path": "/x/2.pimir.mlir"},
            "3": {"op_type": "GEMM", "semantic_role": "down_proj",
                  "expert_id": 0,
                  "weight": "model.layers.0.mlp.experts.0.down_proj.weight",
                  "pimir_path": "/x/3.pimir.mlir"},
        },
        "absorbed_ops": list(absorbed),
    }
    if manifest:
        sidecar["fused_activations"] = [
            {"layer": 0, "role": role, "expert": 0, "activation": "silu"}
            for role in roles
        ]
    return ir, sidecar


def test_audit_accepts_a_manifest_backed_fused_activation() -> None:
    """融合清单能唯一证明的吸收是合法的：主算子有 pimir，激活是它的唯一消费者。"""
    from scripts.run_full_pipeline import audit_sidecar_coverage

    ir, sidecar = _fused_activation_case()
    audit_sidecar_coverage(ir, sidecar)


def test_audit_rejects_absorbed_ops_without_a_fusion_manifest() -> None:
    """absorbed_ops 不能自报：没有融合清单就无法证明这个算子真的被折过。"""
    from scripts.run_full_pipeline import StepFailed, audit_sidecar_coverage

    ir, sidecar = _fused_activation_case(manifest=False)
    with pytest.raises(StepFailed, match="fused_activations"):
        audit_sidecar_coverage(ir, sidecar)


def test_audit_rejects_absorbing_a_non_activation_op() -> None:
    """只允许激活被吸收：GEMM/其它设备算子伪标 absorbed 必须失败。"""
    from scripts.run_full_pipeline import StepFailed, audit_sidecar_coverage

    ir, sidecar = _fused_activation_case(absorbed=(3,))
    with pytest.raises(StepFailed, match="SILU|吸收|激活"):
        audit_sidecar_coverage(ir, sidecar)


def test_audit_rejects_absorption_into_a_role_that_was_not_fused() -> None:
    """吸收必须与融合清单的 (层, 角色, 专家) 对上，不能挂到别的投影上。"""
    from scripts.run_full_pipeline import StepFailed, audit_sidecar_coverage

    ir, sidecar = _fused_activation_case(roles=("up_proj",))
    with pytest.raises(StepFailed, match="融合清单|吸收"):
        audit_sidecar_coverage(ir, sidecar)


def test_audit_rejects_absorbing_an_activation_with_another_consumer() -> None:
    """主算子还有别的读者时不能折激活：那个读者要的是激活前的值。"""
    from scripts.run_full_pipeline import StepFailed, audit_sidecar_coverage

    ir, sidecar = _fused_activation_case(extra_consumer=True)
    with pytest.raises(StepFailed, match="消费者|唯一"):
        audit_sidecar_coverage(ir, sidecar)


def test_audit_accepts_a_manifest_without_absorbed_ids_during_export() -> None:
    """B+C+D 阶段只产融合清单，吸收编号由 A2 回写，此时允许清单单独存在。"""
    from scripts.run_full_pipeline import audit_sidecar_coverage

    ir, sidecar = _fused_activation_case(absorbed=())
    audit_sidecar_coverage(ir, sidecar, require_absorbed_closure=False)


def test_audit_rejects_a_refined_sidecar_without_absorbed_ids() -> None:
    """A2 之后清单与吸收编号必须双向闭合：删掉吸收编号不能照样放行。"""
    from scripts.run_full_pipeline import StepFailed, validate_sidecar_contract

    ir, sidecar = _fused_activation_case(absorbed=())
    with pytest.raises(StepFailed, match="吸收|清单|闭合"):
        validate_sidecar_contract(ir, sidecar)


def _capacity_sidecar(**overrides):
    """一份自报带产物的 sidecar 骨架，容量契约取当前默认值。"""
    from memory.mem_planner import default_hw_budget

    hw = default_hw_budget()
    budget = {
        "mram_bytes_per_dpu": hw.mram_bytes,
        "sys_reserve_bytes": hw.sys_reserve_bytes,
        "align": hw.align,
    }
    budget.update(overrides.pop("budget", {}))
    body = {
        "operators": {
            "1": {"op_type": "GEMM", "semantic_role": "q_proj",
                  "expert_id": -1,
                  "shards": [{"dpu_id": 0, "local_in_features": 64,
                              "local_out_features": 64}],
                  "kernel_tile_n": 64,
                  "pimir_path": "/x/1.pimir.mlir"},
        },
        "requires_pimir": True,
        "hw_budget": budget,
    }
    body.update(overrides)
    return body


def _capacity_ir():
    return {
        "operators": [
            {"op_id": 0, "op_type": "MODEL_INPUT"},
            {"op_id": 1, "op_type": "GEMM", "semantic_role": "q_proj",
             "expert_id": -1},
            {"op_id": 2, "op_type": "MODEL_OUTPUT"},
        ],
        "dependencies": [],
        "subgraphs": [[1]],
    }


def test_audit_accepts_the_contract_hw_budget(tmp_path: Path) -> None:
    import hashlib

    from scripts.run_full_pipeline import audit_sidecar_coverage

    pimir = tmp_path / "op.pimir.mlir"
    _write_gemm_pimir(pimir, 64, 64, tile_n=64)
    sidecar = _capacity_sidecar()
    sidecar["operators"]["1"]["pimir_path"] = str(pimir)
    sidecar["operators"]["1"]["pimir_sha256"] = hashlib.sha256(
        pimir.read_bytes()).hexdigest()
    audit_sidecar_coverage(_capacity_ir(), sidecar)


@pytest.mark.parametrize("field", [
    "mram_bytes_per_dpu", "sys_reserve_bytes", "align"])
def test_audit_rejects_a_strict_sidecar_with_a_missing_budget_field(field) -> None:
    """总容量、系统预留、对齐三者缺一不可：任一缺失都让两侧预算口径分裂。"""
    from scripts.run_full_pipeline import StepFailed, audit_sidecar_coverage

    sidecar = _capacity_sidecar()
    sidecar["hw_budget"].pop(field)
    with pytest.raises(StepFailed, match=field):
        audit_sidecar_coverage(_capacity_ir(), sidecar)


def test_audit_rejects_a_strict_sidecar_without_hw_budget() -> None:
    from scripts.run_full_pipeline import StepFailed, audit_sidecar_coverage

    sidecar = _capacity_sidecar()
    sidecar.pop("hw_budget")
    with pytest.raises(StepFailed, match="hw_budget"):
        audit_sidecar_coverage(_capacity_ir(), sidecar)


def test_audit_rejects_zero_reserve_and_unit_align() -> None:
    """把系统预留写成 0、对齐写成 1 会让可用预算虚高，必须拒绝。"""
    from scripts.run_full_pipeline import StepFailed, audit_sidecar_coverage

    for overrides in ({"sys_reserve_bytes": 0}, {"align": 1}):
        with pytest.raises(StepFailed):
            audit_sidecar_coverage(
                _capacity_ir(),
                _capacity_sidecar(budget=dict(overrides)))


def test_audit_accepts_a_legacy_sidecar_without_a_budget() -> None:
    """明确标识的旧式非严格产物走兼容分支；严格产物不允许。"""
    from scripts.run_full_pipeline import audit_sidecar_coverage

    sidecar = _capacity_sidecar()
    sidecar.pop("requires_pimir")
    sidecar.pop("hw_budget")
    audit_sidecar_coverage(_capacity_ir(), sidecar)


def test_audit_rejects_an_absorbed_id_outside_the_source_ir() -> None:
    """被吸收编号必须存在于源 IR：越界/多余编号不能借吸收逃避覆盖。"""
    from scripts.run_full_pipeline import StepFailed, audit_sidecar_coverage

    ir, sidecar = _fused_activation_case(absorbed=(2, 99))
    with pytest.raises(StepFailed, match="op99"):
        audit_sidecar_coverage(ir, sidecar)


# ── round08 问题1：仿真必须消费本次运行的放置产物 ──────────────────────


def _placement_case(tmp_path: Path, *, tp_width: int, num_stages: int,
                    num_dpus: int = 4, num_layers: int = 4,
                    width: int = 768, local_out: int | None = None):
    """构造「IR + 与该策略匹配的 sidecar + PartitionPlan」三件套。"""
    import hashlib

    from contracts.partition_plan import PartitionPlan

    ops = [{"op_id": 0, "op_type": "MODEL_INPUT"}]
    subgraphs = []
    for _layer in range(num_layers):
        op_id = len(ops)
        ops.append({
            "op_id": op_id, "op_type": "GEMM", "semantic_role": "gate_proj",
            "input_shapes": [[1, width]], "output_shapes": [[1, width]],
        })
        subgraphs.append([op_id])
    ops.append({"op_id": len(ops), "op_type": "MODEL_OUTPUT"})
    ir = {"operators": ops, "dependencies": [], "subgraphs": subgraphs,
          "num_layers": num_layers}
    pimir = tmp_path / "cache.pimir.mlir"
    pimir.write_text("module {}\n")
    digest = hashlib.sha256(pimir.read_bytes()).hexdigest()
    stage_width = num_dpus // num_stages
    per_stage = num_layers // num_stages
    entries = {}
    for layer, op in enumerate(ops[1:-1]):
        stage = layer // per_stage
        stage_dpus = list(range(stage * stage_width, (stage + 1) * stage_width))
        entries[str(op["op_id"])] = {
            "op_type": "GEMM", "semantic_role": "gate_proj", "expert_id": -1,
            "shards": [{
                "dpu_id": dpu, "local_in_features": width,
                "local_out_features": (width // tp_width
                                       if local_out is None else local_out),
            } for dpu in stage_dpus[:tp_width]],
            "pimir_path": str(pimir), "pimir_sha256": digest,
        }
    sidecar = {
        "operators": entries, "requires_pimir": False,
        "dpu_to_cluster": [[0, dpu] for dpu in range(num_dpus)],
    }
    plan = PartitionPlan(
        num_dpus=num_dpus, num_stages=num_stages,
        dpu_to_cluster=tuple((0, dpu) for dpu in range(num_dpus)),
        source=f"fixed_tp{tp_width}_pp{num_stages}",
    )
    return ir, sidecar, plan, pimir


def test_placement_check_accepts_the_matching_strategy(tmp_path: Path) -> None:
    from scripts.run_full_pipeline import _assert_placement_matches_run

    ir, sidecar, plan, _ = _placement_case(tmp_path, tp_width=1, num_stages=4)
    _assert_placement_matches_run(sidecar, ir, plan=plan)


def test_placement_check_rejects_a_sidecar_from_another_strategy(
        tmp_path: Path) -> None:
    """tp4_pp2 的 sidecar 不能当成 tp1_pp4 的产物：分片数与 DPU 归属都不对。"""
    from scripts.run_full_pipeline import (
        StepFailed, _assert_placement_matches_run)

    ir, _sidecar, plan, _ = _placement_case(tmp_path, tp_width=1, num_stages=4)
    _, stale, _stale_plan, _ = _placement_case(
        tmp_path, tp_width=4, num_stages=1)
    with pytest.raises(StepFailed, match="分片数|tp"):
        _assert_placement_matches_run(stale, ir, plan=plan)


def test_placement_check_rejects_local_shapes_from_another_tp_width(
        tmp_path: Path) -> None:
    """每片本地宽度必须等于全局宽度/tp：op152 的 192 不能冒充 768。"""
    from scripts.run_full_pipeline import (
        StepFailed, _assert_placement_matches_run)

    ir, sidecar, plan, _ = _placement_case(
        tmp_path, tp_width=1, num_stages=4, local_out=192)
    with pytest.raises(StepFailed, match="本地|local"):
        _assert_placement_matches_run(sidecar, ir, plan=plan)


def test_placement_check_rejects_a_wrong_stage_mapping(tmp_path: Path) -> None:
    """分片数对、但层落在别的流水段时同样不能放行。"""
    from scripts.run_full_pipeline import (
        StepFailed, _assert_placement_matches_run)

    ir, sidecar, plan, _ = _placement_case(tmp_path, tp_width=2, num_stages=2)
    # 把第 0 层的两个分片挪到 stage1 的 DPU 上。
    first = sidecar["operators"]["1"]["shards"]
    sidecar["operators"]["1"]["shards"] = [
        {**shard, "dpu_id": dpu}
        for shard, dpu in zip(first, (2, 3))]
    with pytest.raises(StepFailed, match="stage|流水段|DPU"):
        _assert_placement_matches_run(sidecar, ir, plan=plan)


def test_prepare_simulation_config_binds_the_run_placement(tmp_path: Path) -> None:
    """临时配置显式改写并复核放置、IR、trace 与 results 路径。"""
    import yaml

    from scripts.run_full_pipeline import prepare_simulation_config

    source = tmp_path / "conf" / "sim.yaml"
    source.parent.mkdir()
    source.write_text(
        'model:\n  ir_path: "models/sim.ir"\n'
        'trace:\n  num_requests: 10\n'
        'scheduler:\n  compiler_placement_file: "models/old.json"\n')
    fresh = tmp_path / "models" / "fresh.json"
    fresh.parent.mkdir()
    fresh.write_text("{}")
    refined = tmp_path / "models" / "run_pimir.ir"
    refined.write_text("{}")
    traces = tmp_path / "run" / "pim_traces_pimir"
    results = tmp_path / "run" / "results_pimir"

    out = prepare_simulation_config(
        source, tmp_path / "logs", max_requests=1, placement=fresh,
        ir_path=refined, traces_dir=traces, results_dir=results,
        output_name="sim_pimir.yaml")
    loaded = yaml.safe_load(out.read_text())
    assert out.name == "sim_pimir.yaml"
    assert loaded["trace"]["num_requests"] == 1
    assert Path(loaded["scheduler"]["compiler_placement_file"]).resolve() == (
        fresh.resolve())
    assert Path(loaded["model"]["ir_path"]).resolve() == refined.resolve()
    assert Path(loaded["output"]["pim_traces_dir"]).resolve() == traces.resolve()
    assert Path(loaded["output"]["results_dir"]).resolve() == results.resolve()

    kept = prepare_simulation_config(source, tmp_path / "logs2")
    assert yaml.safe_load(kept.read_text())[
        "scheduler"]["compiler_placement_file"] == "models/old.json"


def test_run_directory_is_derived_from_artifact_and_strategy(tmp_path: Path) -> None:
    """不同架构/策略/语义/运行号的目录不能同名，否则会互删产物。"""
    from scripts.run_full_pipeline import run_directory

    a = run_directory(tmp_path, "qwen3_moe", 8, 2)
    b = run_directory(tmp_path, "qwen3_moe", 4, 4)
    assert a != b
    assert a.parent == b.parent == tmp_path
    assert "tp4_pp2" in a.name and "tp1_pp4" in b.name
    dynamic = run_directory(
        tmp_path, "qwen3_moe", 8, 2,
        moe_execution="dynamic_dispatch", run_id="run-a")
    dense = run_directory(
        tmp_path, "qwen3_moe", 8, 2,
        moe_execution="dense_sim", run_id="run-b")
    assert len({dynamic, dense}) == 2
    assert "dynamic_dispatch" in dynamic.name and "run-a" in dynamic.name


def test_simulation_step_never_deletes_shared_outputs(
        tmp_path: Path, monkeypatch) -> None:
    """E 阶段只写本次运行目录，另一任务的共享 trace/summary 必须保留。"""
    import yaml

    import scripts.run_full_pipeline as pipeline

    genesim, plan_path = _step_e_case(tmp_path)
    shared_traces = genesim / "pim_traces"
    shared_traces.mkdir()
    sentinel = shared_traces / "op_999_SENTINEL.pim_trace"
    sentinel.write_bytes(b"another-task")
    shared_results = genesim / "results"
    shared_results.mkdir()
    shared_summary = shared_results / "summary.json"
    shared_summary.write_text('{"total_time_s": 999.0}')

    monkeypatch.setattr(
        pipeline, "_genesim_sim_command",
        lambda _g, c: ["true", str(c)])
    monkeypatch.setattr(
        pipeline, "_run",
        lambda cmd, **k: _fake_sim_run(genesim, Path(cmd[1])))
    log_dir = tmp_path / "logs"
    summary = pipeline.step_e_simulate(
        genesim, "sim.yaml", log_dir, label="pimir",
        placement=genesim / "models" / "qwen3_moe_tp1_pp4_placement.json",
        plan_path=plan_path, num_dpus=4, num_stages=4, max_requests=1)

    assert summary["total_time_s"] == 1.0
    assert sentinel.read_bytes() == b"another-task"
    assert json.loads(shared_summary.read_text())["total_time_s"] == 999.0
    prepared = yaml.safe_load((log_dir / "sim_pimir.yaml").read_text())
    traces = Path(prepared["output"]["pim_traces_dir"])
    results = Path(prepared["output"]["results_dir"])
    assert traces.resolve() == (log_dir / "pim_traces_pimir").resolve()
    assert traces.resolve() != shared_traces.resolve()
    assert results.resolve() == (log_dir / "results_pimir").resolve()
    assert (results / "summary.json").is_file()
    assert any(traces.glob("op_*.pim_trace"))


def test_interleaved_runs_keep_their_own_traces_and_summaries(
        tmp_path: Path, monkeypatch) -> None:
    """同一 GeneSim 根下两次语义交错执行，各自产物与临时配置都不互相覆盖。"""
    import yaml

    import scripts.run_full_pipeline as pipeline

    genesim, plan_path = _step_e_case(tmp_path)
    shared_traces = genesim / "pim_traces"
    shared_traces.mkdir()
    sentinel = shared_traces / "op_999_SENTINEL.pim_trace"
    sentinel.write_bytes(b"another-task")

    monkeypatch.setattr(
        pipeline, "_genesim_sim_command",
        lambda _g, c: ["true", str(c)])
    monkeypatch.setattr(
        pipeline, "_run",
        lambda cmd, **k: _fake_sim_run(genesim, Path(cmd[1])))
    log_dir = tmp_path / "logs"
    placement = genesim / "models" / "qwen3_moe_tp1_pp4_placement.json"
    first = pipeline.step_e_simulate(
        genesim, "sim.yaml", log_dir, label="pimir",
        placement=placement, plan_path=plan_path,
        num_dpus=4, num_stages=4, max_requests=1,
        ir_path=genesim / "models" / "sim.ir")
    second = pipeline.step_e_simulate(
        genesim, "sim.yaml", log_dir, label="global",
        placement=placement, plan_path=plan_path,
        num_dpus=4, num_stages=4, max_requests=1,
        ir_path=genesim / "models" / "sim.ir")

    assert first["total_time_s"] == 1.0
    assert second["total_time_s"] == 2.0
    assert sentinel.read_bytes() == b"another-task"
    for label, expected in (("pimir", 1.0), ("global", 2.0)):
        prepared = yaml.safe_load(
            (log_dir / f"sim_{label}.yaml").read_text())
        traces = Path(prepared["output"]["pim_traces_dir"])
        results = Path(prepared["output"]["results_dir"])
        assert traces.resolve() == (
            log_dir / f"pim_traces_{label}").resolve()
        assert any(traces.glob("op_*.pim_trace"))
        assert json.loads(
            (results / "summary.json").read_text())["total_time_s"] == expected
        assert Path(prepared["model"]["ir_path"]).is_file()


def test_trace_provenance_reads_the_configured_trace_dir(tmp_path: Path) -> None:
    """trace 验收只看本次配置的输出目录，共享目录里的旧 trace 不参与。"""
    from scripts.run_full_pipeline import verify_trace_provenance

    genesim, config = _trace_provenance_setup(tmp_path)
    shared = genesim / "pim_traces"
    _write_trace(shared, 1, "GEMM", None)  # 共享目录里的 unknown 旧 trace
    run_traces = genesim / "run" / "pim_traces_pimir"
    _write_trace(run_traces, 1, "GEMM", "pimir")
    _write_trace(run_traces, 2, "ROPE", "pimir")
    import yaml

    loaded = yaml.safe_load(config.read_text())
    loaded["output"] = {"pim_traces_dir": str(run_traces)}
    config.write_text(yaml.safe_dump(loaded))
    verify_trace_provenance(genesim, config)


def _fake_sim_run(genesim: Path, config_path: Path):
    """模拟一次仿真：按临时配置里的输出目录写 trace 与 summary。"""
    import struct

    import yaml

    loaded = yaml.safe_load(Path(config_path).read_text())
    traces = Path(loaded["output"]["pim_traces_dir"])
    results = Path(loaded["output"]["results_dir"])
    ir_path = Path(loaded["model"]["ir_path"])
    if not ir_path.is_absolute():
        ir_path = genesim / ir_path
    sidecar_path = Path(loaded["scheduler"]["compiler_placement_file"])
    sidecar = json.loads(sidecar_path.read_text())
    absorbed = {int(op_id) for op_id in sidecar.get("absorbed_ops") or []}
    traces.mkdir(parents=True, exist_ok=True)
    for op in json.loads(ir_path.read_text())["operators"]:
        if op["op_type"] in ("MODEL_INPUT", "MODEL_OUTPUT"):
            continue
        entry = sidecar.get("operators", {}).get(str(op["op_id"]), {})
        source = "absorbed" if op["op_id"] in absorbed else "pimir"
        signature = {"trace_source": source}
        if entry.get("pimir_sha256"):
            signature["pimir_sha256"] = entry["pimir_sha256"]
        meta = {"op_id": op["op_id"], "op_type": op["op_type"],
                "num_instructions": 0 if source == "absorbed" else 1,
                "trace_source": source,
                "compile_signature": signature}
        body = json.dumps(meta).encode()
        header = b"PIMT" + struct.pack("<I", 1) + struct.pack("<Q", 0) \
            + struct.pack("<I", len(body))
        (traces / f"op_{op['op_id']}_{op['op_type']}.pim_trace").write_bytes(
            header + body)
    results.mkdir(parents=True, exist_ok=True)
    # 不同 label 的运行写出可区分的 summary，交错执行时不会互相覆盖。
    label = results.name
    (results / "summary.json").write_text(json.dumps({
        "total_time_s": 1.0 if label.endswith("pimir") else 2.0,
        "throughput_tokens_per_s": 1.0,
    }))


def _step_e_case(tmp_path: Path) -> tuple[Path, Path]:
    """仿真配置模板指向一份不存在的旧 sidecar，证明 E 不再依赖它。"""
    genesim = tmp_path / "genesim"
    (genesim / "conf").mkdir(parents=True)
    (genesim / "models").mkdir()
    ir, sidecar, plan, _ = _placement_case(
        tmp_path, tp_width=1, num_stages=4)
    (genesim / "models" / "sim.ir").write_text(json.dumps(ir))
    (genesim / "models" / "qwen3_moe_tp1_pp4_placement.json").write_text(json.dumps(sidecar))
    (genesim / "conf" / "sim.yaml").write_text(
        'model:\n  ir_path: "models/sim.ir"\n'
        'trace:\n  num_requests: 10\n'
        'scheduler:\n  compiler_placement_file: "models/old_tp4_pp2.json"\n')
    plan_path = tmp_path / "plan.json"
    plan.write(plan_path)
    return genesim, plan_path


def test_simulation_step_consumes_the_current_placement(
        tmp_path: Path, monkeypatch) -> None:
    """E 阶段必须读取 BCD 返回的 sidecar，即使模板里写的是另一份。"""
    import yaml

    import scripts.run_full_pipeline as pipeline

    genesim, plan_path = _step_e_case(tmp_path)
    monkeypatch.setattr(
        pipeline, "_genesim_sim_command",
        lambda _g, c: ["true", str(c)])
    monkeypatch.setattr(
        pipeline, "_run",
        lambda cmd, **k: _fake_sim_run(genesim, Path(cmd[1])))
    log_dir = tmp_path / "logs"
    results = log_dir / "results_pimir"

    summary = pipeline.step_e_simulate(
        genesim, "sim.yaml", log_dir, label="pimir",
        placement=genesim / "models" / "qwen3_moe_tp1_pp4_placement.json",
        plan_path=plan_path,
        num_dpus=4, num_stages=4, results_dir=results, max_requests=1)

    assert summary["total_time_s"] == 1.0
    prepared = yaml.safe_load((log_dir / "sim_pimir.yaml").read_text())
    assert Path(prepared["scheduler"]["compiler_placement_file"]).resolve() == (
        (genesim / "models" / "qwen3_moe_tp1_pp4_placement.json").resolve())
    assert prepared["trace"]["num_requests"] == 1
    assert (results / "summary.json").is_file()
    assert Path(prepared["output"]["pim_traces_dir"]).resolve() == (
        log_dir / "pim_traces_pimir").resolve()
    assert Path(prepared["output"]["results_dir"]).resolve() == results.resolve()


def test_simulation_step_rejects_a_sidecar_from_another_strategy(
        tmp_path: Path, monkeypatch) -> None:
    """传错 sidecar 时在仿真启动前失败，不能跑出一份别的切分的数字。"""
    import scripts.run_full_pipeline as pipeline

    genesim, plan_path = _step_e_case(tmp_path)
    _, stale, _, _ = _placement_case(tmp_path, tp_width=4, num_stages=1)
    stale_path = genesim / "models" / "stale_tp4_pp2.json"
    stale_path.write_text(json.dumps(stale))
    monkeypatch.setattr(
        pipeline, "_genesim_sim_command", lambda *a, **k: ["true"])
    monkeypatch.setattr(
        pipeline, "_run",
        lambda *a, **k: pytest.fail("sidecar 不匹配时不该启动仿真"))

    with pytest.raises(pipeline.StepFailed, match="分片数|tp"):
        pipeline.step_e_simulate(
            genesim, "sim.yaml", tmp_path / "logs", label="pimir",
            placement=stale_path, plan_path=plan_path,
            num_dpus=4, num_stages=4)


# ── round08 问题4：导出、skip-simulation、freshness、仿真 IR 与 trace
#    共用同一份完整产物契约 ───────────────────────────────────────────


def _write_gemm_pimir(path: Path, local_in: int, local_out: int, *,
                      tile_n: int, tile_k: int = 16,
                      activation: str | None = None) -> str:
    """写一份与真实 A 路产物同形的分块 GEMM pimir，返回文本。

    模块属性与 `tt.dot` 的三维、K/N 循环嵌套都与 `_measure_kernel_tile_n`
    编出来的 linear kernel 一致；融合激活写在 `tt.dot` 的 `activation` 属性上。
    """
    act = (f' {{activation = #pim.act_spec<kind = {activation}>}}'
           if activation else "")
    dot = (f"      %0 = tt.dot %a, %b, %c{act} : "
           f"tensor<1x{tile_k}xf16> * tensor<{tile_k}x{tile_n}xf16> -> "
           f"tensor<1x{tile_n}xf32>\n")
    body = dot
    if local_in > tile_k:
        body = (
            f"      scf.for %arg4 = %c0_i32 to %c{local_in}_i32 "
            f"step %c{tile_k}_i32 : i32 {{\n"
            + "".join("  " + line + "\n"
                      for line in body.rstrip("\n").splitlines())
            + "      }\n")
    if local_out > tile_n:
        body = (
            f"      scf.for %arg3 = %c0_i32 to %c{local_out}_i32 "
            f"step %c{tile_n}_i32 : i32 {{\n"
            + "".join("  " + line + "\n"
                      for line in body.rstrip("\n").splitlines())
            + "      }\n")
    text = (
        "module attributes {\n"
        f'  "pim.tile-k" = {tile_k} : i64,\n'
        f'  "pim.tile-n" = {tile_n} : i64\n'
        "} {\n"
        "  tt.func public @linear_kernel(%a: !tt.ptr<f16>, %b: !tt.ptr<f16>, "
        "%c: !tt.ptr<f16>) {\n"
        "    %c0_i32 = arith.constant 0 : i32\n"
        f"    %c{local_in}_i32 = arith.constant {local_in} : i32\n"
        f"    %c{local_out}_i32 = arith.constant {local_out} : i32\n"
        f"    %c{tile_k}_i32 = arith.constant {tile_k} : i32\n"
        f"    %c{tile_n}_i32 = arith.constant {tile_n} : i32\n"
        f"{body}"
        "    tt.return\n"
        "  }\n"
        "}\n")
    path.write_text(text)
    return text


def _contract_case(tmp_path: Path, *, with_fingerprint: bool = False):
    """一份完整的 MoE 产物契约夹具：IR + sidecar + 真实 pimir 文件。

    A 路 GEMM 与 B 路算子各自引用同形的产物：GEMM 是 `tt.dot` 分块 kernel，
    B 路是带 `pim.traceTokens` 的单相算 kernel。两类产物不能用同一份——
    那正是 round19 指出的 A 路语义缺口。
    """
    import hashlib

    from genesim_bridge.ir_fingerprint import fingerprint_ir
    from memory.mem_planner import default_hw_budget

    pimir = tmp_path / "op.pimir.mlir"
    pimir.write_text(
        "module {\n"
        "  tt.func @op(%arg0: tensor<1x16xf16>) "
        "attributes {pim.traceTokens = 1 : i64} {\n"
        "    %0 = pim.lut %arg0 {kind = #pim.activation<silu>} : "
        "tensor<1x16xf16> -> tensor<1x16xf16>\n"
        "    tt.return\n"
        "  }\n"
        "}\n")
    digest = hashlib.sha256(pimir.read_bytes()).hexdigest()
    gemm_pimirs: dict[tuple[int, int], tuple[Path, str]] = {}

    def gemm_pimir(local_in: int, local_out: int) -> tuple[Path, str]:
        key = (local_in, local_out)
        if key not in gemm_pimirs:
            path = tmp_path / f"gemm_{local_in}_{local_out}.pimir.mlir"
            _write_gemm_pimir(
                path, local_in, local_out, tile_n=min(16, local_out))
            gemm_pimirs[key] = (
                path, hashlib.sha256(path.read_bytes()).hexdigest())
        return gemm_pimirs[key]

    ops = [
        {"op_id": 0, "op_type": "MODEL_INPUT"},
        {"op_id": 1, "op_type": "GEMM", "semantic_role": "router",
         "expert_id": -1, "device_hint": "pim",
         "input_shapes": [[1, 64]], "output_shapes": [[1, 2]]},
        {"op_id": 2, "op_type": "GEMM", "semantic_role": "gate_proj",
         "expert_id": 0, "device_hint": "pim",
         "input_shapes": [[1, 64]], "output_shapes": [[1, 16]]},
        {"op_id": 3, "op_type": "SILU", "expert_id": 0, "device_hint": "pim",
         "input_shapes": [[1, 16]], "output_shapes": [[1, 16]]},
        {"op_id": 4, "op_type": "GEMM", "semantic_role": "down_proj",
         "expert_id": 0, "device_hint": "pim",
         "input_shapes": [[1, 16]], "output_shapes": [[1, 64]]},
        {"op_id": 5, "op_type": "MODEL_OUTPUT"},
    ]
    ir = {
        "operators": ops,
        "dependencies": [
            {"src_op_id": 1, "dst_op_id": 2},
            {"src_op_id": 2, "dst_op_id": 3},
            {"src_op_id": 3, "dst_op_id": 4},
        ],
        "subgraphs": [[1, 2, 3, 4]],
        "num_layers": 1,
        "moe_num_experts": 1,
        "moe_top_k": 1,
        "moe_intermediate_size": 16,
        "moe_execution": "dynamic_dispatch",
    }
    if with_fingerprint:
        ir["model_fingerprint"] = fingerprint_ir(ir)

    def entry(**extra):
        body = {"device_hint": "pim", "pimir_path": str(pimir),
                "pimir_sha256": digest}
        body.update(extra)
        return body

    def gemm_entry(local_in: int, local_out: int, **extra):
        """单分片 GEMM：仍带 `shards`，与真实导出一样与 B 路条目区分开。"""
        path, path_digest = gemm_pimir(local_in, local_out)
        body = {
            "device_hint": "pim", "op_type": "GEMM",
            "shards": [{
                "dpu_id": 0,
                "local_in_features": local_in,
                "local_out_features": local_out,
            }],
            "kernel_tile_n": min(16, local_out),
            "pimir_path": str(path), "pimir_sha256": path_digest,
        }
        body.update(extra)
        return body

    hw = default_hw_budget()
    sidecar = {
        "requires_pimir": True,
        "hw_budget": {
            "mram_bytes_per_dpu": hw.mram_bytes,
            "sys_reserve_bytes": hw.sys_reserve_bytes,
            "align": hw.align,
        },
        "moe_config": {
            "num_experts": 1, "top_k": 1, "intermediate_size": 16,
            "execution": "dynamic_dispatch",
        },
        "operators": {
            "1": gemm_entry(64, 2, semantic_role="router", expert_id=-1),
            "2": gemm_entry(64, 16, semantic_role="gate_proj", expert_id=0,
                            weight="model.layers.0.mlp.experts.0.gate_proj.weight"),
            "3": entry(op_type="SILU", expert_id=0, trace_tokens=1,
                       local_input_shapes=[[1, 16]],
                       local_output_shapes=[[1, 16]]),
            "4": gemm_entry(16, 64, semantic_role="down_proj", expert_id=0,
                            weight="model.layers.0.mlp.experts.0.down_proj.weight"),
        },
    }
    if with_fingerprint:
        sidecar["ir_fingerprint"] = ir["model_fingerprint"]
    return ir, sidecar


def test_full_contract_accepts_a_complete_sidecar(tmp_path: Path) -> None:
    from scripts.run_full_pipeline import validate_sidecar_contract

    ir, sidecar = _contract_case(tmp_path)
    validate_sidecar_contract(ir, sidecar)


@pytest.mark.parametrize("case", [
    "missing_moe_config", "wrong_execution", "missing_budget",
    "forged_host_exemption", "illegal_absorption", "expert_mismatch",
    "wrong_fingerprint", "missing_fingerprint",
])
def test_full_contract_rejects_each_incomplete_field(
        tmp_path: Path, case: str) -> None:
    """MoE 元数据、容量、豁免、吸收、专家身份与指纹缺一项都不放行。"""
    from scripts.run_full_pipeline import (
        StepFailed, validate_sidecar_contract)

    ir, sidecar = _contract_case(
        tmp_path, with_fingerprint=case in (
            "wrong_fingerprint", "missing_fingerprint"))
    if case == "missing_moe_config":
        sidecar.pop("moe_config")
        match = "moe_config"
    elif case == "wrong_execution":
        sidecar["moe_config"]["execution"] = "dense_sim"
        match = "moe_config|execution"
    elif case == "missing_budget":
        sidecar.pop("hw_budget")
        match = "hw_budget"
    elif case == "forged_host_exemption":
        sidecar["operators"].pop("1")
        sidecar["pimir_exempt_ops"] = {
            "1": {"op_type": "GEMM", "semantic_role": "router",
                  "device_hint": "cpu"}}
        match = "豁免"
    elif case == "illegal_absorption":
        sidecar["absorbed_ops"] = [1]
        match = "吸收|absorbed"
    elif case == "expert_mismatch":
        sidecar["operators"]["2"]["expert_id"] = 1
        match = "身份"
    elif case == "wrong_fingerprint":
        sidecar["ir_fingerprint"] = "0" * 64
        match = "指纹|fingerprint"
    else:
        sidecar.pop("ir_fingerprint")
        match = "指纹|fingerprint"
    with pytest.raises(StepFailed, match=match):
        validate_sidecar_contract(ir, sidecar)


def _contract_genesim(tmp_path: Path, *, with_fingerprint: bool = False):
    """把契约夹具落成 genesim 目录 + 配置，供三个独立入口核对。"""
    genesim = tmp_path / "genesim"
    (genesim / "models").mkdir(parents=True)
    (genesim / "conf").mkdir()
    ir, sidecar = _contract_case(
        tmp_path, with_fingerprint=with_fingerprint)
    (genesim / "models" / "sim.ir").write_text(json.dumps(ir))
    (genesim / "models" / "sc.json").write_text(json.dumps(sidecar))
    config = genesim / "conf" / "sim.yaml"
    config.write_text(
        'model:\n  ir_path: "models/sim.ir"\n'
        'scheduler:\n  compiler_placement_file: "models/sc.json"\n')
    return genesim, ir, sidecar


@pytest.mark.parametrize("case,match", [
    ("missing_moe_config", "moe_config"),
    ("missing_budget", "hw_budget"),
    ("forged_host_exemption", "豁免"),
    ("illegal_absorption", "吸收|absorbed"),
])
def test_freshness_uses_the_full_contract(
        tmp_path: Path, case: str, match: str) -> None:
    """仿真前复核过去只核哈希与专家身份，缺 MoE/容量/豁免/吸收仍会放行。"""
    from scripts.run_full_pipeline import StepFailed, verify_sidecar_freshness

    genesim, _ir, sidecar = _contract_genesim(tmp_path)
    if case == "missing_moe_config":
        sidecar.pop("moe_config")
    elif case == "missing_budget":
        sidecar.pop("hw_budget")
    elif case == "forged_host_exemption":
        sidecar["operators"].pop("1")
        sidecar["pimir_exempt_ops"] = {
            "1": {"op_type": "GEMM", "semantic_role": "router",
                  "device_hint": "cpu"}}
    else:
        sidecar["absorbed_ops"] = [1]
    (genesim / "models" / "sc.json").write_text(json.dumps(sidecar))
    with pytest.raises(StepFailed, match=match):
        verify_sidecar_freshness(genesim, genesim / "conf" / "sim.yaml")


def test_freshness_accepts_the_complete_contract(tmp_path: Path) -> None:
    from scripts.run_full_pipeline import verify_sidecar_freshness

    genesim, _ir, _sidecar = _contract_genesim(tmp_path)
    verify_sidecar_freshness(genesim, genesim / "conf" / "sim.yaml")


def test_sim_ir_check_uses_the_full_contract(tmp_path: Path) -> None:
    from scripts.run_full_pipeline import StepFailed, verify_sim_ir_matches_sidecar

    genesim, _ir, sidecar = _contract_genesim(tmp_path)
    sidecar.pop("moe_config")
    (genesim / "models" / "sc.json").write_text(json.dumps(sidecar))
    with pytest.raises(StepFailed, match="moe_config"):
        verify_sim_ir_matches_sidecar(genesim, genesim / "conf" / "sim.yaml")


def test_trace_provenance_uses_the_full_contract(tmp_path: Path) -> None:
    """trace 验收也要核对 MoE 元数据等字段，不能只看来源与集合覆盖。"""
    import struct

    from scripts.run_full_pipeline import StepFailed, verify_trace_provenance

    genesim, _ir, sidecar = _contract_genesim(tmp_path)
    sidecar.pop("moe_config")
    (genesim / "models" / "sc.json").write_text(json.dumps(sidecar))
    traces = genesim / "pim_traces"
    traces.mkdir()
    for op_id, op_type in ((1, "GEMM"), (2, "GEMM"), (3, "SILU"),
                           (4, "GEMM")):
        meta = {"op_id": op_id, "op_type": op_type, "trace_source": "pimir"}
        body = json.dumps(meta).encode()
        header = b"PIMT" + struct.pack("<I", 1) + struct.pack("<Q", 0) \
            + struct.pack("<I", len(body))
        (traces / f"op_{op_id}_{op_type}.pim_trace").write_bytes(header + body)
    with pytest.raises(StepFailed, match="moe_config"):
        verify_trace_provenance(genesim, genesim / "conf" / "sim.yaml")


# ── round12 问题1：严格性上下文必须贯穿编译器复核入口 ────────────────


def _strict_contract_genesim(tmp_path: Path, *, require: bool = True):
    """契约夹具 + 显式严格/非严格仿真配置。"""
    genesim, ir, sidecar = _contract_genesim(tmp_path)
    config = genesim / "conf" / "sim.yaml"
    flag = "true" if require else "false"
    config.write_text(
        'model:\n  ir_path: "models/sim.ir"\n'
        'scheduler:\n  compiler_placement_file: "models/sc.json"\n'
        f'  require_compiler_pimir: {flag}\n')
    _write_pimir_traces(genesim)
    return genesim, ir, sidecar, config


def _run_post_refine_entry(entry: str, genesim: Path, config: Path) -> None:
    from scripts.run_full_pipeline import (
        verify_sidecar_freshness, verify_sim_ir_matches_sidecar,
        verify_trace_provenance)
    {
        "freshness": lambda: verify_sidecar_freshness(genesim, config),
        "sim_ir": lambda: verify_sim_ir_matches_sidecar(genesim, config),
        "trace": lambda: verify_trace_provenance(genesim, config),
    }[entry]()


@pytest.mark.parametrize("entry", ["freshness", "sim_ir", "trace"])
def test_strict_config_requires_the_capacity_contract(
        tmp_path: Path, entry: str) -> None:
    """配置显式严格时，删掉 sidecar 自报与容量字段不能让入口放行。"""
    from scripts.run_full_pipeline import StepFailed

    genesim, _ir, sidecar, config = _strict_contract_genesim(tmp_path)
    sidecar.pop("requires_pimir")
    sidecar.pop("hw_budget")
    (genesim / "models" / "sc.json").write_text(json.dumps(sidecar))
    with pytest.raises(StepFailed, match="hw_budget"):
        _run_post_refine_entry(entry, genesim, config)


@pytest.mark.parametrize("capacity", [4 * 1024 ** 3, 16 * 1024 ** 3])
@pytest.mark.parametrize("entry", ["freshness", "sim_ir", "trace"])
def test_strict_config_rejects_a_sidecar_with_a_wrong_capacity(
        tmp_path: Path, entry: str, capacity: int) -> None:
    """删掉自报再写 4/16 GiB 仍是同一个绕过路径，三个入口都必须拒绝。"""
    from scripts.run_full_pipeline import StepFailed

    genesim, _ir, sidecar, config = _strict_contract_genesim(tmp_path)
    sidecar.pop("requires_pimir")
    sidecar["hw_budget"]["mram_bytes_per_dpu"] = capacity
    (genesim / "models" / "sc.json").write_text(json.dumps(sidecar))
    with pytest.raises(StepFailed, match="mram_bytes_per_dpu|容量|不一致"):
        _run_post_refine_entry(entry, genesim, config)


def test_non_strict_config_keeps_the_legacy_compatibility_branch(
        tmp_path: Path) -> None:
    """调用方明确按非严格（配置 false、产物不自报）时才允许旧式 sidecar。"""
    from scripts.run_full_pipeline import verify_sidecar_freshness

    genesim, _ir, sidecar, config = _strict_contract_genesim(
        tmp_path, require=False)
    sidecar.pop("requires_pimir")
    sidecar.pop("hw_budget")
    (genesim / "models" / "sc.json").write_text(json.dumps(sidecar))
    verify_sidecar_freshness(genesim, config)


def test_audit_honours_an_explicit_strict_context() -> None:
    """共享验证器要接受调用方的严格上下文，而不是只看 sidecar 自报。"""
    from scripts.run_full_pipeline import StepFailed, audit_sidecar_coverage

    sidecar = _capacity_sidecar()
    sidecar.pop("requires_pimir")
    sidecar.pop("hw_budget")
    with pytest.raises(StepFailed, match="hw_budget"):
        audit_sidecar_coverage(
            _capacity_ir(), sidecar, strict_pimir=True)
    audit_sidecar_coverage(
        _capacity_ir(), sidecar, strict_pimir=False)


def test_audit_keeps_self_reported_strictness_when_the_caller_is_not_strict(
) -> None:
    """显式非严格不能关掉产物自己的声明：自报带产物时容量契约仍是硬约束。"""
    from scripts.run_full_pipeline import StepFailed, audit_sidecar_coverage

    sidecar = _capacity_sidecar()
    sidecar.pop("hw_budget")
    with pytest.raises(StepFailed, match="hw_budget"):
        audit_sidecar_coverage(
            _capacity_ir(), sidecar, strict_pimir=False)


def _manifest_only_case(tmp_path: Path, *, with_fingerprint: bool = False):
    """精化后删掉吸收编号的产物：融合清单还在，吸收集合为空。"""
    ir, sidecar = _contract_case(tmp_path, with_fingerprint=with_fingerprint)
    sidecar["fused_activations"] = [{
        "layer": 0, "role": "gate_proj", "expert": 0, "activation": "silu",
    }]
    return ir, sidecar


def _write_pimir_traces(genesim: Path) -> None:
    """给契约夹具的每个设备算子写一条带真实编译签名哈希的 pimir trace。

    带编译单位（B 路）的条目同时写入 pimir 相位元数据与运行期形状；没有单位
    的分片 GEMM 只写签名与指令数。
    """
    sidecar = json.loads((genesim / "models" / "sc.json").read_text())
    traces = genesim / "pim_traces"
    traces.mkdir(exist_ok=True)
    for raw_op_id, entry in sidecar["operators"].items():
        kwargs: dict = {}
        unit_keys = ("trace_tokens", "local_input_shapes",
                     "local_output_shapes")
        if all(entry.get(key) is not None for key in unit_keys):
            tokens = int(entry["trace_tokens"])

            def resolved(shape):
                dims = []
                for dim in shape:
                    if dim == "Tq":
                        dims.append(tokens)
                    elif dim in ("Tp", "Tp+Tq"):
                        # 与编译器 _pimir_unit_shapes 的历史维占位同源。
                        dims.append(128)
                    else:
                        dims.append(int(dim))
                return dims

            def total_elems(shapes) -> int:
                total = 0
                for shape in shapes:
                    count = 1
                    for dim in resolved(shape):
                        count *= dim
                    total += count
                return total

            def runtime_shape(shape):
                dims = list(shape)
                if dims and dims[0] == 1:
                    dims[0] = "Tq"
                return dims

            kwargs = {
                "pimir": {
                    "tokens": tokens,
                    "phase_count": 1,
                    "total_in_bytes": tokens * total_elems(
                        entry["local_input_shapes"]) * 2,
                    "total_out_bytes": tokens * total_elems(
                        entry["local_output_shapes"]) * 2,
                },
                "input_shapes": [
                    runtime_shape(shape)
                    for shape in entry["local_input_shapes"]
                ],
                "output_shapes": [
                    runtime_shape(shape)
                    for shape in entry["local_output_shapes"]
                ],
            }
        _write_trace(
            traces, int(raw_op_id), str(entry.get("op_type", "GEMM")),
            "pimir", pimir_sha256=entry.get("pimir_sha256"),
            num_instructions=1, **kwargs)


@pytest.mark.parametrize("entry", ["freshness", "sim_ir", "trace"])
def test_post_refine_entries_require_absorbed_closure(
        tmp_path: Path, entry: str) -> None:
    """A2 之后所有复核入口都要求融合清单与吸收编号双向闭合。"""
    from scripts.run_full_pipeline import (
        StepFailed, verify_sidecar_freshness, verify_sim_ir_matches_sidecar,
        verify_trace_provenance)

    genesim = tmp_path / "genesim"
    (genesim / "models").mkdir(parents=True)
    (genesim / "conf").mkdir()
    ir, sidecar = _manifest_only_case(tmp_path)
    (genesim / "models" / "sim.ir").write_text(json.dumps(ir))
    (genesim / "models" / "sc.json").write_text(json.dumps(sidecar))
    config = genesim / "conf" / "sim.yaml"
    config.write_text(
        'model:\n  ir_path: "models/sim.ir"\n'
        'scheduler:\n  compiler_placement_file: "models/sc.json"\n')
    _write_pimir_traces(genesim)
    call = {
        "freshness": lambda: verify_sidecar_freshness(genesim, config),
        "sim_ir": lambda: verify_sim_ir_matches_sidecar(genesim, config),
        "trace": lambda: verify_trace_provenance(genesim, config),
    }[entry]
    with pytest.raises(StepFailed, match="吸收|清单|闭合"):
        call()


def _exemption_contradiction_case(tmp_path: Path):
    """把真实 host 算子写成「豁免 + PIM 放置条目」的矛盾产物。"""
    ir, sidecar = _contract_case(tmp_path)
    sidecar["operators"]["1"]["device_hint"] = "pim"
    sidecar["operators"]["1"]["shards"] = [{
        "dpu_id": 0, "local_in_features": 64, "local_out_features": 2}]
    sidecar["pimir_exempt_ops"] = {"1": {
        "op_type": "GEMM", "device_hint": "cpu", "semantic_role": "router",
        "expert_id": -1,
        "weight": "model.layers.0.self_attn.router.weight",
        "reason": "host 执行",
    }}
    return ir, sidecar


@pytest.mark.parametrize("entry", ["freshness", "sim_ir", "trace"])
def test_post_refine_entries_reject_exemption_placement_contradictions(
        tmp_path: Path, entry: str) -> None:
    """五个入口共用的规则：豁免与 sidecar 放置矛盾时每个复核入口都失败。"""
    from scripts.run_full_pipeline import (
        StepFailed, verify_sidecar_freshness, verify_sim_ir_matches_sidecar,
        verify_trace_provenance)

    genesim = tmp_path / "genesim"
    (genesim / "models").mkdir(parents=True)
    (genesim / "conf").mkdir()
    ir, sidecar = _exemption_contradiction_case(tmp_path)
    (genesim / "models" / "sim.ir").write_text(json.dumps(ir))
    (genesim / "models" / "sc.json").write_text(json.dumps(sidecar))
    config = genesim / "conf" / "sim.yaml"
    config.write_text(
        'model:\n  ir_path: "models/sim.ir"\n'
        'scheduler:\n  compiler_placement_file: "models/sc.json"\n')
    _write_pimir_traces(genesim)
    call = {
        "freshness": lambda: verify_sidecar_freshness(genesim, config),
        "sim_ir": lambda: verify_sim_ir_matches_sidecar(genesim, config),
        "trace": lambda: verify_trace_provenance(genesim, config),
    }[entry]
    with pytest.raises(StepFailed, match="豁免"):
        call()


def _skip_simulation_sidecar(tmp_path, ir, sidecar, monkeypatch):
    """把主链跑到 --skip-simulation，返回退出码。"""
    from types import SimpleNamespace

    import scripts.run_full_pipeline as pipeline

    genesim = tmp_path / "genesim"
    models = genesim / "models"
    models.mkdir(parents=True)
    (models / "m_pimir.ir").write_text(json.dumps(ir))
    (models / "m_placement.json").write_text(json.dumps(sidecar))

    args = SimpleNamespace(
        backend="genesim", run_name=None, token_check="warn",
        decode_steps=1, num_stages=2, num_dpus=4,
        model_dir=str(tmp_path / "model"), moe_execution=None,
        max_requests=1, skip_simulation=True, ab_compare=False,
        log_dir=str(tmp_path / "logs"), no_pu_mapping=True,
    )
    monkeypatch.setattr(pipeline, "parse_args", lambda: args)
    monkeypatch.setattr(pipeline, "genesim_root", lambda: genesim)
    monkeypatch.setattr(
        pipeline, "llama2_7b_model_dir", lambda: tmp_path / "model")
    monkeypatch.setattr(
        pipeline, "architecture_of",
        lambda path: ("llama", SimpleNamespace(
            artifact="m", required_roles=frozenset(), sim_config=None)))
    monkeypatch.setattr(pipeline, "_resolve_moe_execution", lambda *a: "")
    monkeypatch.setattr(
        pipeline, "step_a_model_ir", lambda *a, **k: models / "m.ir")
    monkeypatch.setattr(
        pipeline, "step_bcd_export",
        lambda *a, **k: models / "m_placement.json")
    monkeypatch.setattr(
        pipeline, "step_a2_refine_pimir_ir", lambda *a, **k: models / "m_pimir.ir")
    return pipeline.main()


def test_skip_simulation_rejects_exemption_placement_contradictions(
        tmp_path, monkeypatch) -> None:
    """--skip-simulation 也走同一份豁免规则，矛盾放置不能签发通过。"""
    ir, sidecar = _exemption_contradiction_case(tmp_path)
    assert _skip_simulation_sidecar(tmp_path, ir, sidecar, monkeypatch) == 1


def test_skip_simulation_requires_the_absorbed_ids(tmp_path, monkeypatch) -> None:
    """--skip-simulation 跑到 A2 之后，同样要求清单与吸收编号闭合。"""
    ir, sidecar = _manifest_only_case(tmp_path)
    assert _skip_simulation_sidecar(tmp_path, ir, sidecar, monkeypatch) == 1


def test_export_step_validates_the_bcd_phase(tmp_path, monkeypatch) -> None:
    """BCD 只产融合清单、吸收编号由 A2 回写，导出步骤必须按宽松阶段校验。"""
    import scripts.run_full_pipeline as pipeline

    genesim = tmp_path / "genesim"
    models = genesim / "models"
    models.mkdir(parents=True)
    ir, sidecar = _contract_case(tmp_path)
    (models / "m.ir").write_text(json.dumps(ir))
    (models / "m_tp2_pp2_placement.json").write_text(json.dumps(sidecar))
    monkeypatch.setattr(pipeline, "_run", lambda *a, **k: None)
    monkeypatch.setattr(pipeline, "genesim_models_dir", lambda: models)
    seen: dict = {}

    def stop(loaded_ir, loaded_sidecar, **kwargs):
        seen.update(kwargs)
        raise pipeline.StepFailed("验证阶段已记录")

    monkeypatch.setattr(pipeline, "validate_sidecar_contract", stop)
    arch = SimpleNamespace(
        artifact="m", required_roles=frozenset(), sim_config=None)
    with pytest.raises(pipeline.StepFailed, match="验证阶段已记录"):
        pipeline.step_bcd_export(
            2, 4, tmp_path / "logs", tmp_path / "model", arch)
    assert seen.get("require_absorbed_closure") is False


def test_skip_simulation_requires_the_capacity_contract(
        tmp_path, monkeypatch) -> None:
    """--skip-simulation 自己刚编完算子，缺自报或容量字段同样不能签发通过。"""
    ir, sidecar = _contract_case(tmp_path)
    sidecar.pop("requires_pimir")
    sidecar.pop("hw_budget")
    assert _skip_simulation_sidecar(tmp_path, ir, sidecar, monkeypatch) == 1


def test_export_step_requires_the_capacity_contract(
        tmp_path, monkeypatch) -> None:
    """BCD 导出步骤必须按严格上下文核对容量，不能只看是否自报。"""
    import scripts.run_full_pipeline as pipeline

    genesim = tmp_path / "genesim"
    models = genesim / "models"
    models.mkdir(parents=True)
    ir, sidecar = _contract_case(tmp_path)
    sidecar.pop("requires_pimir")
    sidecar.pop("hw_budget")
    (models / "m.ir").write_text(json.dumps(ir))
    (models / "m_tp2_pp2_placement.json").write_text(json.dumps(sidecar))
    monkeypatch.setattr(pipeline, "_run", lambda *a, **k: None)
    monkeypatch.setattr(pipeline, "genesim_models_dir", lambda: models)
    arch = SimpleNamespace(
        artifact="m", required_roles=frozenset(), sim_config=None)

    with pytest.raises(pipeline.StepFailed, match="hw_budget"):
        pipeline.step_bcd_export(
            2, 4, tmp_path / "logs", tmp_path / "model", arch)


def test_export_step_uses_the_full_contract(tmp_path, monkeypatch) -> None:
    """BCD 导出步骤同样走完整契约，缺 moe_config 时不能写出就放行。"""
    import scripts.run_full_pipeline as pipeline

    genesim = tmp_path / "genesim"
    models = genesim / "models"
    models.mkdir(parents=True)
    ir, sidecar = _contract_case(tmp_path)
    sidecar.pop("moe_config")
    (models / "m.ir").write_text(json.dumps(ir))
    (models / "m_tp2_pp2_placement.json").write_text(json.dumps(sidecar))
    monkeypatch.setattr(pipeline, "_run", lambda *a, **k: None)
    monkeypatch.setattr(pipeline, "genesim_models_dir", lambda: models)
    arch = SimpleNamespace(
        artifact="m", required_roles=frozenset(), sim_config=None)

    with pytest.raises(pipeline.StepFailed, match="moe_config"):
        pipeline.step_bcd_export(
            2, 4, tmp_path / "logs", tmp_path / "model", arch)


# ── round13/14 问题3：严格入口的配置与 IR/sidecar 是必需输入 ──────────


def _write_sim_config(
    path: Path, *, ir: bool = True, sidecar: bool = True,
    require: bool = True, ir_path: str = "models/sim.ir",
    sidecar_path: str = "models/sc.json",
) -> None:
    """按需省略配置键，覆盖「缺键但配置仍显式严格」的绕过组合。"""
    text = f'model:\n  ir_path: "{ir_path}"\n' if ir else ""
    text += "scheduler:\n"
    if sidecar:
        text += f'  compiler_placement_file: "{sidecar_path}"\n'
    text += f"  require_compiler_pimir: {'true' if require else 'false'}\n"
    path.write_text(text)


@pytest.mark.parametrize("entry", ["freshness", "sim_ir", "trace"])
@pytest.mark.parametrize("case", [
    "no_sidecar_key", "no_ir_key", "missing_sidecar_file", "missing_ir_file"])
def test_strict_entry_requires_config_and_artifacts(
        tmp_path: Path, entry: str, case: str) -> None:
    """严格模式下删配置键或删产物文件，三个复核入口都必须失败。"""
    from scripts.run_full_pipeline import StepFailed

    genesim, _ir, _sidecar, config = _strict_contract_genesim(tmp_path)
    if case == "no_sidecar_key":
        _write_sim_config(config, sidecar=False)
    elif case == "no_ir_key":
        _write_sim_config(config, ir=False)
    elif case == "missing_sidecar_file":
        _write_sim_config(config, sidecar_path="models/nope.json")
    else:
        _write_sim_config(config, ir_path="models/nope.ir")
    with pytest.raises(
            StepFailed,
            match="sidecar 配置键|IR 配置键|sidecar 不存在|仿真 IR 不存在"):
        _run_post_refine_entry(entry, genesim, config)


@pytest.mark.parametrize("entry", ["freshness", "sim_ir", "trace"])
def test_strict_entry_rejects_a_missing_config_file(
        tmp_path: Path, entry: str) -> None:
    """调用方明确给了配置、文件却不存在时不能按「无对象」放行。"""
    from scripts.run_full_pipeline import StepFailed

    genesim, _ir, _sidecar, config = _strict_contract_genesim(tmp_path)
    config.unlink()
    with pytest.raises(StepFailed, match="仿真配置不存在"):
        _run_post_refine_entry(entry, genesim, config)


@pytest.mark.parametrize("entry", ["freshness", "sim_ir", "trace"])
def test_sidecar_self_report_requires_the_ir_key(
        tmp_path: Path, entry: str) -> None:
    """配置非严格但 sidecar 自报严格时，缺 IR 键同样不能放行。"""
    from scripts.run_full_pipeline import StepFailed

    genesim, _ir, _sidecar, config = _strict_contract_genesim(
        tmp_path, require=False)
    _write_sim_config(config, ir=False, require=False)
    with pytest.raises(StepFailed, match="IR 配置键"):
        _run_post_refine_entry(entry, genesim, config)


def test_non_strict_entry_keeps_the_no_sidecar_legacy_path(
        tmp_path: Path) -> None:
    """配置与产物都明确非严格时，无 sidecar 的旧式用法继续放行。"""
    from scripts.run_full_pipeline import (
        verify_sidecar_freshness, verify_sim_ir_matches_sidecar,
        verify_trace_provenance)

    genesim, _ir, sidecar, config = _strict_contract_genesim(
        tmp_path, require=False)
    sidecar.pop("requires_pimir")
    (genesim / "models" / "sc.json").write_text(json.dumps(sidecar))
    _write_sim_config(config, sidecar=False, require=False)
    verify_sidecar_freshness(genesim, config)
    verify_sim_ir_matches_sidecar(genesim, config)
    verify_trace_provenance(genesim, config)


# ── round13/14 问题4：trace 必须绑定源 IR 与 sidecar 的编译签名 ──────


def test_trace_entry_accepts_hash_bound_shards(tmp_path: Path) -> None:
    """TP 分片回指同一源条目且签名哈希一致时放行。"""
    from scripts.run_full_pipeline import verify_trace_provenance

    genesim, _ir, sidecar, config = _strict_contract_genesim(tmp_path)
    digest = sidecar["operators"]["1"]["pimir_sha256"]
    _write_trace(
        genesim / "pim_traces", 101, "GEMM", "pimir", parent=1,
        pimir_sha256=digest, num_instructions=3)
    verify_trace_provenance(genesim, config)


def _replace_trace(
    genesim: Path, op_id: int, op_type: str, **kwargs) -> None:
    traces = genesim / "pim_traces"
    for path in traces.glob(f"op_{op_id}_*.pim_trace"):
        path.unlink()
    _write_trace(traces, op_id, op_type, "pimir", **kwargs)


@pytest.mark.parametrize("case,match", [
    ("wrong_hash", "编译签名哈希"),
    ("missing_hash", "缺 compile_signature.pimir_sha256"),
    ("wrong_type", "类型"),
    ("wrong_parent", "不在源 IR"),
    ("wrong_expert", "expert_id"),
])
def test_trace_entry_rejects_unbound_pimir_traces(
        tmp_path: Path, case: str, match: str) -> None:
    """错哈希、缺哈希、错类型与错父编号都不能凭来源标签放行。"""
    from scripts.run_full_pipeline import StepFailed, verify_trace_provenance

    genesim, _ir, sidecar, config = _strict_contract_genesim(tmp_path)
    digest = sidecar["operators"]["1"]["pimir_sha256"]
    if case == "wrong_hash":
        _replace_trace(
            genesim, 1, "GEMM", pimir_sha256="0" * 64, num_instructions=1)
    elif case == "missing_hash":
        _replace_trace(genesim, 1, "GEMM", num_instructions=1)
    elif case == "wrong_type":
        _replace_trace(
            genesim, 1, "SILU", pimir_sha256=digest, num_instructions=1)
    elif case == "wrong_expert":
        digest3 = sidecar["operators"]["3"]["pimir_sha256"]
        _replace_trace(
            genesim, 3, "SILU", pimir_sha256=digest3, num_instructions=1,
            expert_id=1)
    else:
        _replace_trace(
            genesim, 101, "GEMM", parent=99, pimir_sha256=digest,
            num_instructions=1)
    with pytest.raises(StepFailed, match=match):
        verify_trace_provenance(genesim, config)


def _absorbed_genesim(tmp_path: Path):
    """在契约夹具上把 SILU 节点标成合法吸收，返回 (genesim, config)。

    融合清单声明 gate_proj 折了 silu，主算子的 pimir 必须真的带
    `act_spec<silu>`；否则审计会按「融合状态没有实产证明」拒绝。
    """
    import hashlib

    genesim, ir, sidecar = _contract_genesim(tmp_path)
    sidecar["fused_activations"] = [{
        "layer": 0, "role": "gate_proj", "expert": 0, "activation": "silu",
    }]
    sidecar["absorbed_ops"] = [3]
    fused = genesim / "models" / "gate_fused.pimir.mlir"
    _write_gemm_pimir(fused, 64, 16, tile_n=16, activation="silu")
    sidecar["operators"]["2"]["pimir_path"] = str(fused)
    sidecar["operators"]["2"]["pimir_sha256"] = hashlib.sha256(
        fused.read_bytes()).hexdigest()
    (genesim / "models" / "sc.json").write_text(json.dumps(sidecar))
    traces = genesim / "pim_traces"
    traces.mkdir(exist_ok=True)
    for op_id, op_type in ((1, "GEMM"), (2, "GEMM"), (4, "GEMM")):
        entry = sidecar["operators"][str(op_id)]
        _write_trace(
            traces, op_id, op_type, "pimir",
            pimir_sha256=entry["pimir_sha256"], num_instructions=2)
    return genesim, genesim / "conf" / "sim.yaml"


def test_trace_entry_accepts_a_proven_empty_absorbed_trace(
        tmp_path: Path) -> None:
    from scripts.run_full_pipeline import verify_trace_provenance

    genesim, config = _absorbed_genesim(tmp_path)
    _write_trace(
        genesim / "pim_traces", 3, "SILU", "absorbed", num_instructions=0)
    verify_trace_provenance(genesim, config)


def test_trace_entry_rejects_a_non_empty_absorbed_trace(
        tmp_path: Path) -> None:
    """absorbed 只接受零指令 trace；有指令就不是被折进主算子的空占位。"""
    from scripts.run_full_pipeline import StepFailed, verify_trace_provenance

    genesim, config = _absorbed_genesim(tmp_path)
    _write_trace(
        genesim / "pim_traces", 3, "SILU", "absorbed", num_instructions=1)
    with pytest.raises(StepFailed, match="不是空 trace"):
        verify_trace_provenance(genesim, config)


def test_trace_entry_rejects_an_absorbed_id_outside_the_sidecar(
        tmp_path: Path) -> None:
    """标成 absorbed 但 sidecar 没有吸收编号，同样不能免掉 pimir 覆盖。"""
    from scripts.run_full_pipeline import StepFailed, verify_trace_provenance

    genesim, config = _absorbed_genesim(tmp_path)
    sidecar = json.loads((genesim / "models" / "sc.json").read_text())
    sidecar["absorbed_ops"] = []
    sidecar["fused_activations"] = []
    # 清单撤掉后主算子也不能再带融合激活：pimir 与清单必须双向一致，
    # 否则先被 A 路激活核对拒绝，测不到本条要查的吸收编号。
    import hashlib

    plain = genesim / "models" / "gate.pimir.mlir"
    _write_gemm_pimir(plain, 64, 16, tile_n=16)
    sidecar["operators"]["2"]["pimir_path"] = str(plain)
    sidecar["operators"]["2"]["pimir_sha256"] = hashlib.sha256(
        plain.read_bytes()).hexdigest()
    (genesim / "models" / "sc.json").write_text(json.dumps(sidecar))
    # trace 的编译签名也要换成新 pimir 的哈希，先让覆盖与 lineage 都成立。
    _write_trace(
        genesim / "pim_traces", 2, "GEMM", "pimir",
        pimir_sha256=sidecar["operators"]["2"]["pimir_sha256"],
        num_instructions=2)
    _write_trace(
        genesim / "pim_traces", 3, "SILU", "absorbed", num_instructions=0)
    with pytest.raises(StepFailed, match="没有吸收编号|没有 pimir 来源"):
        verify_trace_provenance(genesim, config)


# ── round15 问题1：严格入口必须核对 pimir 产物的可读性与哈希 ──────────


def _break_pimir_artifact(genesim: Path, sidecar: dict, case: str) -> None:
    """按 case 破坏一个非豁免算子的 pimir 产物身份并写回 sidecar。"""
    entry = sidecar["operators"]["1"]
    pimir = Path(entry["pimir_path"])
    if case == "missing_key":
        entry.pop("pimir_path")
    elif case == "missing_file":
        entry["pimir_path"] = str(pimir.parent / "gone.pimir.mlir")
    elif case == "unreadable_file":
        broken = pimir.parent / "broken.pimir.mlir"
        broken.mkdir()
        entry["pimir_path"] = str(broken)
    elif case == "missing_hash":
        entry.pop("pimir_sha256")
    else:
        pimir.write_text("module { /* 内容被换过 */ }\n")
    (genesim / "models" / "sc.json").write_text(json.dumps(sidecar))


@pytest.mark.parametrize("entry", ["freshness", "sim_ir", "trace"])
@pytest.mark.parametrize("case", [
    "missing_key", "missing_file", "unreadable_file", "missing_hash",
    "stale_hash"])
def test_strict_entry_requires_a_valid_pimir_artifact(
        tmp_path: Path, entry: str, case: str) -> None:
    """产物校验不是可选参数：缺项/不可读/缺哈希/错哈希都不能放行。"""
    from scripts.run_full_pipeline import StepFailed

    genesim, _ir, sidecar, config = _strict_contract_genesim(tmp_path)
    _break_pimir_artifact(genesim, sidecar, case)
    with pytest.raises(
            StepFailed, match="pimir_path|pimir_sha256|读不到|哈希"):
        _run_post_refine_entry(entry, genesim, config)


@pytest.mark.parametrize("entry", ["freshness", "sim_ir", "trace"])
def test_sidecar_self_report_requires_a_valid_pimir_artifact(
        tmp_path: Path, entry: str) -> None:
    """配置非严格但 sidecar 自报严格时，pimir 产物同样是必需输入。"""
    from scripts.run_full_pipeline import StepFailed

    genesim, _ir, sidecar, config = _strict_contract_genesim(
        tmp_path, require=False)
    _break_pimir_artifact(genesim, sidecar, "missing_file")
    with pytest.raises(StepFailed, match="读不到|pimir_path"):
        _run_post_refine_entry(entry, genesim, config)


# ── round15 问题2：trace 验收的配置是必需输入 ───────────────────────


def test_trace_entry_requires_a_configuration(tmp_path: Path) -> None:
    """无配置只能做来源白名单，不能签发覆盖与 lineage 验收。"""
    from scripts.run_full_pipeline import StepFailed, verify_trace_provenance

    genesim, config = _trace_provenance_setup(tmp_path)
    _write_trace(genesim / "pim_traces", 1, "GEMM", "pimir",
                 pimir_sha256="0" * 64)
    _write_trace(genesim / "pim_traces", 2, "ROPE", "pimir",
                 pimir_sha256="0" * 64)
    with pytest.raises(StepFailed, match="配置"):
        verify_trace_provenance(genesim)
    verify_trace_provenance(genesim, config)


# ── round15 问题3：E 阶段的请求分布输入由主链生成 ───────────────────


def test_prepare_simulation_config_generates_a_missing_trace(
        tmp_path: Path) -> None:
    """干净检出没有 traces/ 时，E 阶段在本次运行目录生成请求分布输入。"""
    import yaml

    from scripts.run_full_pipeline import prepare_simulation_config

    source = tmp_path / "sim.yaml"
    source.write_text(
        'trace:\n  path: "traces/round15-missing.trace"\n'
        '  num_requests: 2\n')
    out = prepare_simulation_config(
        source, tmp_path / "logs", genesim=genesim_root())
    loaded = yaml.safe_load(out.read_text())
    generated = Path(loaded["trace"]["path"])
    assert generated.is_absolute() and generated.is_file()
    assert generated.read_bytes()[:4] == b"GSTR"


def test_prepare_simulation_config_keeps_an_existing_trace(
        tmp_path: Path) -> None:
    """配置指向的 trace 已存在时原样使用，不重新生成。"""
    import yaml

    from scripts.run_full_pipeline import prepare_simulation_config

    genesim = tmp_path / "genesim"
    (genesim / "traces").mkdir(parents=True)
    trace = genesim / "traces" / "req.trace"
    trace.write_bytes(b"sentinel")
    source = tmp_path / "sim.yaml"
    source.write_text('trace:\n  path: "traces/req.trace"\n')
    out = prepare_simulation_config(
        source, tmp_path / "logs", genesim=genesim)
    loaded = yaml.safe_load(out.read_text())
    assert loaded["trace"]["path"] == "traces/req.trace"
    assert trace.read_bytes() == b"sentinel"


def _qwen3_moe_hf_config() -> dict:
    """4 层真实权重切片的 MoE 身份字段（其余字段与本用例无关）。"""
    return {
        "model_type": "qwen3_moe",
        "num_experts": 128,
        "num_experts_per_tok": 8,
        "moe_intermediate_size": 768,
    }


def test_moe_identity_rejects_a_top_k_that_disagrees_with_the_config() -> None:
    """top-k 不进入权重字节，A 段必须拿 config 派生值逐字段核对。

    只比较 IR 与 sidecar 自洽时，把 top_k 写成 7 的 IR 能一路通过 A/BCD/E，
    动态分派比例与 MOE_ROUTE 成本随之失真。
    """
    from scripts.run_full_pipeline import StepFailed, _validate_moe_identity

    declared = {"num_experts": 128, "top_k": 7, "intermediate_size": 768}
    with pytest.raises(StepFailed, match="top_k.*7.*8"):
        _validate_moe_identity(
            declared, _qwen3_moe_hf_config(), stage="图骨架")

    # 真实 30B 的四元组正例：128 专家 top-8、专家宽 768。
    _validate_moe_identity(
        {"num_experts": 128, "top_k": 8, "intermediate_size": 768},
        _qwen3_moe_hf_config(), stage="图骨架")


def test_moe_identity_rejects_a_wrong_expert_width() -> None:
    """专家宽度同属身份字段；写错时也要在 A 段失败，不能只看参数总量。"""
    from scripts.run_full_pipeline import StepFailed, _validate_moe_identity

    with pytest.raises(StepFailed, match="intermediate_size"):
        _validate_moe_identity(
            {"num_experts": 128, "top_k": 8, "intermediate_size": 1024},
            _qwen3_moe_hf_config(), stage="图骨架")


def test_dense_config_rejects_moe_scalars_and_accepts_an_empty_identity() -> None:
    """非 MoE 配置不得携带 MoE 标量；全零身份按稠密模型放行。"""
    from scripts.run_full_pipeline import StepFailed, _validate_moe_identity

    dense = {"model_type": "qwen3", "hidden_size": 64}
    _validate_moe_identity(
        {"num_experts": 0, "top_k": 0, "intermediate_size": 0},
        dense, stage="图骨架")
    with pytest.raises(StepFailed, match="不是 MoE"):
        _validate_moe_identity(
            {"num_experts": 0, "top_k": 8, "intermediate_size": 0},
            dense, stage="图骨架")


def _bpath_unit_record(**overrides) -> dict:
    """一条 B 路 pimir trace 的元数据记录（1 token × 本地宽 192）。"""
    record = {
        "op_id": 100,
        "op_type": "VECTOR_MUL",
        "parent_op_id": 2,
        "trace_source": "pimir",
        "compile_signature": {"pimir_sha256": "abc123"},
        "pimir": {"tokens": 1, "phase_count": 1,
                  "total_in_bytes": 768, "total_out_bytes": 384},
        "input_shapes": [["Tq", 192], ["Tq", 192]],
        "output_shapes": [["Tq", 192]],
    }
    record.update(overrides)
    return record


def _bpath_unit_sidecar(**overrides) -> dict:
    """与 `_bpath_unit_record` 对应的源 IR 与 sidecar 条目。"""
    entry = {
        "op_type": "VECTOR_MUL",
        "device_hint": "pim",
        "shards": [],
        "pimir_path": "/x/op.pimir.mlir",
        "pimir_sha256": "abc123",
        "trace_tokens": 1,
        "local_input_shapes": [["Tq", 192], ["Tq", 192]],
        "local_output_shapes": [["Tq", 192]],
    }
    entry.update(overrides)
    return {
        "operators": {"2": entry},
        "absorbed_ops": [],
    }


def _bpath_unit_ir() -> dict:
    return {"operators": [{
        "op_id": 2, "op_type": "VECTOR_MUL", "expert_id": -1,
        "input_shapes": [["Tq", 768], ["Tq", 768]],
        "output_shapes": [["Tq", 768]],
    }]}


def test_trace_lineage_accepts_a_matching_bpath_unit() -> None:
    """trace 的 token 数、本地形状与字节都与 sidecar 单位一致时放行。"""
    from scripts.run_full_pipeline import _trace_source_lineage

    problems = _trace_source_lineage(
        [_bpath_unit_record()], _bpath_unit_ir(), _bpath_unit_sidecar(),
        strict=True)
    assert problems == [], problems


def test_trace_lineage_rejects_a_wide_token_bpath_trace() -> None:
    """trace 里 baked 了 128 个 token 时 token 会被重复计算，必须拒绝。"""
    from scripts.run_full_pipeline import _trace_source_lineage

    record = _bpath_unit_record(
        pimir={"tokens": 128, "phase_count": 1,
               "total_in_bytes": 98304, "total_out_bytes": 49152})
    problems = _trace_source_lineage(
        [record], _bpath_unit_ir(), _bpath_unit_sidecar(), strict=True)
    assert any("tokens" in problem for problem in problems), problems


def test_trace_lineage_rejects_a_global_width_bpath_trace() -> None:
    """trace 按全局 768 宽编、sidecar 单位是本地 192 宽时必须拒绝。"""
    from scripts.run_full_pipeline import _trace_source_lineage

    record = _bpath_unit_record(
        input_shapes=[["Tq", 768], ["Tq", 768]],
        output_shapes=[["Tq", 768]])
    problems = _trace_source_lineage(
        [record], _bpath_unit_ir(), _bpath_unit_sidecar(), strict=True)
    assert any("末维" in problem for problem in problems), problems


def test_trace_lineage_rejects_bpath_bytes_that_ignore_the_token_unit() -> None:
    """单相逐元素 trace 的输出字节必须等于声明形状 × token 数。"""
    from scripts.run_full_pipeline import _trace_source_lineage

    record = _bpath_unit_record(
        pimir={"tokens": 1, "phase_count": 1,
               "total_in_bytes": 768, "total_out_bytes": 196608})
    problems = _trace_source_lineage(
        [record], _bpath_unit_ir(), _bpath_unit_sidecar(), strict=True)
    assert any("字节" in problem for problem in problems), problems


def test_trace_lineage_requires_the_unit_fields_in_strict_mode() -> None:
    """严格产物必须回传 trace_tokens 与本地形状，缺了就没有可核对的不变式。"""
    from scripts.run_full_pipeline import _trace_source_lineage

    sidecar = _bpath_unit_sidecar()
    for key in ("trace_tokens", "local_input_shapes", "local_output_shapes"):
        sidecar["operators"]["2"].pop(key)
    problems = _trace_source_lineage(
        [_bpath_unit_record()], _bpath_unit_ir(), sidecar, strict=True)
    assert any("trace_tokens" in problem or "local_" in problem
               for problem in problems), problems
    # 非严格旧产物没有这些字段时不另报错，保持旧式兼容。
    assert _trace_source_lineage(
        [_bpath_unit_record()], _bpath_unit_ir(), sidecar, strict=False) == []


def test_trace_lineage_checks_split_as_a_pure_view() -> None:
    """SPLIT 的 IR 输出形状是代表切片；字节按读写相等核对，不能少算头数。"""
    from scripts.run_full_pipeline import _trace_source_lineage

    ir = {"operators": [{
        "op_id": 2, "op_type": "SPLIT", "expert_id": -1,
        "input_shapes": [[32, "Tq", 128]], "output_shapes": [["Tq", 128]],
    }]}
    entry = {
        "op_type": "SPLIT", "device_hint": "pim", "shards": [],
        "pimir_path": "/x/op.pimir.mlir", "pimir_sha256": "abc123",
        "trace_tokens": 1,
        "local_input_shapes": [[32, "Tq", 128]],
        "local_output_shapes": [["Tq", 128]],
    }
    sidecar = {"operators": {"2": entry}, "absorbed_ops": []}
    record = {
        "op_id": 100, "op_type": "SPLIT", "parent_op_id": 2,
        "trace_source": "pimir",
        "compile_signature": {"pimir_sha256": "abc123"},
        "pimir": {"tokens": 1, "phase_count": 1,
                  "total_in_bytes": 8192, "total_out_bytes": 8192},
        "input_shapes": [[32, "Tq", 128]],
        "output_shapes": [["Tq", 128]],
    }
    assert _trace_source_lineage(
        [record], ir, sidecar, strict=True) == []
    broken = dict(record, pimir={
        "tokens": 1, "phase_count": 1,
        "total_in_bytes": 8192, "total_out_bytes": 256,
    })
    problems = _trace_source_lineage([broken], ir, sidecar, strict=True)
    assert any("SPLIT" in problem for problem in problems), problems


_FOUR_NODE_UNIT_CASES = {
    # 算子类型: (IR 输入形状, IR 输出形状, 单位输入, 单位输出, 单相字节)
    "MASK": (
        [["Tq", "Tp+Tq"]], [["Tq", "Tp+Tq"]],
        [["Tq", "Tp+Tq"]], [["Tq", "Tp+Tq"]], 256),
    "SOFTMAX": (
        [["Tq", "Tp+Tq"]], [["Tq", "Tp+Tq"]],
        [["Tq", "Tp+Tq"]], [["Tq", "Tp+Tq"]], 256),
    "GEMV_SCORE": (
        [["Tq", 128], ["Tp+Tq", 128]], [["Tq", "Tp+Tq"]],
        [["Tq", 128], ["Tp+Tq", 128]], [["Tq", "Tp+Tq"]], None),
    "GEMV_CONTEXT": (
        [["Tq", "Tp+Tq"], ["Tp+Tq", 128]], [["Tq", 128]],
        [["Tq", "Tp+Tq"], ["Tp+Tq", 128]], [["Tq", 128]], None),
}


def _four_node_case(op_type: str):
    inputs, outputs, unit_in, unit_out, total_bytes = (
        _FOUR_NODE_UNIT_CASES[op_type])
    ir = {"operators": [{
        "op_id": 2, "op_type": op_type, "expert_id": -1,
        "input_shapes": inputs, "output_shapes": outputs,
    }]}
    entry = {
        "op_type": op_type, "device_hint": "pim", "shards": [],
        "pimir_path": "/x/op.pimir.mlir", "pimir_sha256": "abc123",
        "trace_tokens": 1,
        "local_input_shapes": unit_in,
        "local_output_shapes": unit_out,
    }
    pimir = {"tokens": 1, "phase_count": 1,
             "has_block_loop": True, "loop_unit_tokens": 128}
    if total_bytes is not None:
        pimir.update(total_in_bytes=total_bytes, total_out_bytes=total_bytes)
    record = {
        "op_id": 100, "op_type": op_type, "parent_op_id": 2,
        "trace_source": "pimir",
        "compile_signature": {"pimir_sha256": "abc123"},
        "pimir": pimir, "input_shapes": inputs, "output_shapes": outputs,
    }
    return ir, {"operators": {"2": entry}, "absorbed_ops": []}, record


@pytest.mark.parametrize("op_type", sorted(_FOUR_NODE_UNIT_CASES))
def test_trace_lineage_accepts_four_node_attention_units(op_type: str) -> None:
    """四节点注意力的历史维保留符号后，prefill/decode 的合法 trace 都放行。"""
    from scripts.run_full_pipeline import _trace_source_lineage

    ir, sidecar, record = _four_node_case(op_type)
    assert _trace_source_lineage(
        [record], ir, sidecar, strict=True) == []


def test_trace_lineage_rejects_a_history_dim_folded_into_a_placeholder() -> None:
    """把 Tp+Tq 固化成 128 会丢掉历史维角色，严格产物必须拒绝。"""
    from scripts.run_full_pipeline import _trace_source_lineage

    ir, sidecar, record = _four_node_case("MASK")
    sidecar["operators"]["2"]["local_input_shapes"] = [[1, 128]]
    problems = _trace_source_lineage(
        [record], ir, sidecar, strict=True)
    assert any("符号维" in problem for problem in problems), problems


def test_trace_lineage_rejects_a_wrong_feature_width_in_four_node_units(
) -> None:
    """历史维跳过比较，但 GEMV_CONTEXT 的 head_dim 仍必须与单位一致。"""
    from scripts.run_full_pipeline import _trace_source_lineage

    ir, sidecar, record = _four_node_case("GEMV_CONTEXT")
    sidecar["operators"]["2"]["local_input_shapes"][1] = ["Tp+Tq", 64]
    problems = _trace_source_lineage(
        [record], ir, sidecar, strict=True)
    assert any("末维" in problem for problem in problems), problems


# ── round17 问题3：B 路单位必须进共享严格验证器 ─────────────────────


@pytest.mark.parametrize("entry", ["freshness", "sim_ir", "trace"])
@pytest.mark.parametrize("case", ["missing", "half"])
def test_strict_entries_require_a_complete_bpath_unit(
        tmp_path: Path, entry: str, case: str) -> None:
    """缺单位/半缺单位在 freshness、仿真 IR 与 trace 入口都必须失败。

    单位是 trace 与运行期形状同源的唯一证据；只挂在 trace 入口会让前两者
    签发一份 GeneSim 严格入口必然拒绝的产物。
    """
    from scripts.run_full_pipeline import StepFailed

    genesim, _ir, sidecar, config = _strict_contract_genesim(tmp_path)
    unit = sidecar["operators"]["3"]
    keys = (("trace_tokens", "local_input_shapes", "local_output_shapes")
            if case == "missing" else ("local_output_shapes",))
    for key in keys:
        unit.pop(key)
    (genesim / "models" / "sc.json").write_text(json.dumps(sidecar))
    with pytest.raises(StepFailed, match="编译单位|trace_tokens"):
        _run_post_refine_entry(entry, genesim, config)


@pytest.mark.parametrize("entry", ["freshness", "sim_ir"])
def test_strict_entries_reject_units_that_drop_symbolic_dims(
        tmp_path: Path, entry: str) -> None:
    """sidecar 单位把 IR 的 Tq 固化成 1 时同样不能签发。"""
    from scripts.run_full_pipeline import StepFailed

    genesim, ir, sidecar, config = _strict_contract_genesim(tmp_path)
    ir["operators"][3]["input_shapes"] = [["Tq", 16]]
    ir["operators"][3]["output_shapes"] = [["Tq", 16]]
    (genesim / "models" / "sim.ir").write_text(json.dumps(ir))
    sidecar["operators"]["3"]["local_input_shapes"] = [[1, 16]]
    sidecar["operators"]["3"]["local_output_shapes"] = [[1, 16]]
    (genesim / "models" / "sc.json").write_text(json.dumps(sidecar))
    with pytest.raises(StepFailed, match="符号维|编译单位"):
        _run_post_refine_entry(entry, genesim, config)


def test_skip_simulation_requires_a_complete_bpath_unit(
        tmp_path, monkeypatch) -> None:
    """--skip-simulation 自己刚产出的 sidecar 缺单位时不能签发通过。"""
    ir, sidecar = _contract_case(tmp_path)
    for key in ("trace_tokens", "local_input_shapes", "local_output_shapes"):
        sidecar["operators"]["3"].pop(key)
    assert _skip_simulation_sidecar(tmp_path, ir, sidecar, monkeypatch) == 1


# ── round18 问题1：A/B 路分类必须取自源 IR ──────────────────────────


@pytest.mark.parametrize("entry", ["freshness", "sim_ir", "trace"])
def test_bpath_entry_with_a_fake_shard_is_rejected(
        tmp_path: Path, entry: str) -> None:
    """给 B 路条目塞一个假 shard 不能跳过编译单位核对。"""
    from scripts.run_full_pipeline import StepFailed

    genesim, _ir, sidecar, config = _strict_contract_genesim(tmp_path)
    unit = sidecar["operators"]["3"]
    unit["shards"] = [{"dpu_id": 0, "local_in_features": 16,
                       "local_out_features": 16}]
    for key in ("trace_tokens", "local_input_shapes", "local_output_shapes"):
        unit.pop(key)
    (genesim / "models" / "sc.json").write_text(json.dumps(sidecar))
    with pytest.raises(StepFailed, match="非 GEMM 却带分片|编译单位"):
        _run_post_refine_entry(entry, genesim, config)


@pytest.mark.parametrize("entry", ["freshness", "sim_ir", "trace"])
def test_device_gemm_without_shards_is_rejected(
        tmp_path: Path, entry: str) -> None:
    """设备 GEMM 的分片被置空时不能按算子级条目放行。"""
    from scripts.run_full_pipeline import StepFailed

    genesim, _ir, sidecar, config = _strict_contract_genesim(tmp_path)
    sidecar["operators"]["2"].pop("shards")
    (genesim / "models" / "sc.json").write_text(json.dumps(sidecar))
    with pytest.raises(StepFailed, match="GEMM 却没有分片"):
        _run_post_refine_entry(entry, genesim, config)


def test_skip_simulation_rejects_a_fake_shard_on_a_bpath_entry(
        tmp_path, monkeypatch) -> None:
    """--skip-simulation 同样按源 IR 分类，假分片不能绕过单位核对。"""
    ir, sidecar = _contract_case(tmp_path)
    unit = sidecar["operators"]["3"]
    unit["shards"] = [{"dpu_id": 0, "local_in_features": 16,
                       "local_out_features": 16}]
    for key in ("trace_tokens", "local_input_shapes", "local_output_shapes"):
        unit.pop(key)
    assert _skip_simulation_sidecar(tmp_path, ir, sidecar, monkeypatch) == 1


# ── round18 问题2：sidecar 单位必须与 pimir 实产绑定 ────────────────


@pytest.mark.parametrize("entry", ["freshness", "sim_ir"])
def test_unit_tokens_must_match_the_pimir(
        tmp_path: Path, entry: str) -> None:
    """op3 的 trace_tokens 改成 128、pimir 仍为 1 时必须拒绝。"""
    from scripts.run_full_pipeline import StepFailed

    genesim, _ir, sidecar, config = _strict_contract_genesim(tmp_path)
    sidecar["operators"]["3"]["trace_tokens"] = 128
    (genesim / "models" / "sc.json").write_text(json.dumps(sidecar))
    with pytest.raises(StepFailed, match="traceTokens|编译单位"):
        _run_post_refine_entry(entry, genesim, config)


@pytest.mark.parametrize("entry", ["freshness", "sim_ir"])
def test_unit_feature_width_must_match_the_pimir(
        tmp_path: Path, entry: str) -> None:
    """B 路本地宽改成 32、pimir 仍为 16 时必须拒绝。"""
    from scripts.run_full_pipeline import StepFailed

    genesim, _ir, sidecar, config = _strict_contract_genesim(tmp_path)
    sidecar["operators"]["3"]["local_input_shapes"] = [["Tq", 32]]
    sidecar["operators"]["3"]["local_output_shapes"] = [["Tq", 32]]
    (genesim / "models" / "sc.json").write_text(json.dumps(sidecar))
    with pytest.raises(StepFailed, match="末维|编译单位"):
        _run_post_refine_entry(entry, genesim, config)


@pytest.mark.parametrize("entry", ["freshness", "sim_ir"])
def test_unit_must_match_the_pimir_actually_referenced(
        tmp_path: Path, entry: str) -> None:
    """把 pimir_path 指向一份旧的 128-token 产物同样不能放行。"""
    import hashlib

    from scripts.run_full_pipeline import StepFailed

    genesim, _ir, sidecar, config = _strict_contract_genesim(tmp_path)
    old = tmp_path / "old-128.pimir.mlir"
    old.write_text(
        "module {\n"
        "  tt.func @op(%arg0: tensor<128x16xf16>) "
        "attributes {pim.traceTokens = 128 : i64} {\n"
        "    %0 = pim.lut %arg0 : tensor<128x16xf16> -> tensor<128x16xf16>\n"
        "    tt.return\n"
        "  }\n"
        "}\n")
    sidecar["operators"]["3"]["pimir_path"] = str(old)
    sidecar["operators"]["3"]["pimir_sha256"] = hashlib.sha256(
        old.read_bytes()).hexdigest()
    (genesim / "models" / "sc.json").write_text(json.dumps(sidecar))
    with pytest.raises(StepFailed, match="traceTokens|编译单位"):
        _run_post_refine_entry(entry, genesim, config)


@pytest.mark.parametrize("entry", ["freshness", "sim_ir"])
def test_pimir_without_trace_tokens_is_rejected(
        tmp_path: Path, entry: str) -> None:
    """严格产物的 pimir 没写 traceTokens 时不能靠形状自证。"""
    import hashlib

    from scripts.run_full_pipeline import StepFailed

    genesim, _ir, sidecar, config = _strict_contract_genesim(tmp_path)
    old = tmp_path / "no-tokens.pimir.mlir"
    old.write_text(
        "module {\n"
        "  tt.func @op(%arg0: tensor<1x16xf16>) {\n"
        "    %0 = pim.lut %arg0 : tensor<1x16xf16> -> tensor<1x16xf16>\n"
        "    tt.return\n"
        "  }\n"
        "}\n")
    sidecar["operators"]["3"]["pimir_path"] = str(old)
    sidecar["operators"]["3"]["pimir_sha256"] = hashlib.sha256(
        old.read_bytes()).hexdigest()
    (genesim / "models" / "sc.json").write_text(json.dumps(sidecar))
    with pytest.raises(StepFailed, match="traceTokens|编译单位"):
        _run_post_refine_entry(entry, genesim, config)


# ── round17 问题4：独立仿真入口的共享产物必须与编译默认一致 ─────────


def test_checked_in_qwen3_standalone_products_match_the_compiler_default(
) -> None:
    """`conf/sim_qwen3_moe_pimir.yaml` 引用的共享产物必须可直接独立运行。

    模板同时服务主链 E 阶段与 `./run.sh --config` 独立入口。共享 models/
    若停在其他执行语义或旧单位契约上，独立入口会在启动预检阶段失败；产物
    又不在模板注释给出的单步命令覆盖范围内，用户无从恢复。
    """
    import yaml

    from genesim_bridge.paths import genesim_root
    from runtime.exec_plan_gen import moe_execution_default
    from scripts.run_full_pipeline import audit_sidecar_coverage

    genesim = genesim_root(required=False)
    if genesim is None:
        pytest.skip("未配置 GeneSim 根目录")
    template = genesim / "conf" / "sim_qwen3_moe_pimir.yaml"
    if not template.is_file():
        pytest.skip("独立仿真模板不存在")
    loaded = yaml.safe_load(template.read_text()) or {}
    ir_path = genesim / str((loaded.get("model") or {}).get("ir_path", ""))
    sidecar_path = genesim / str(
        (loaded.get("scheduler") or {}).get("compiler_placement_file", ""))
    if not ir_path.is_file() or not sidecar_path.is_file():
        pytest.skip("共享 models 产物未生成")
    ir = json.loads(ir_path.read_text())
    sidecar = json.loads(sidecar_path.read_text())
    assert sidecar.get("moe_config", {}).get("execution") == ir.get(
        "moe_execution"), "IR 与 sidecar 的执行语义不一致"
    assert ir.get("moe_execution") == moe_execution_default(), (
        "共享产物的执行语义与编译器默认不一致；请重跑主链刷新 models/")
    assert sidecar.get("ir_fingerprint") == ir.get("model_fingerprint"), (
        "共享 IR 与 sidecar 不是同一次导出；请重跑主链")
    # 走一遍严格契约（容量、单位、豁免、专家身份与吸收结构）。
    audit_sidecar_coverage(ir, sidecar, strict_pimir=True)


# ── round19 问题1/2/3：A 路 GEMM 的 pimir 语义、host 证据与条目字段 ─────


def _point_gemms_at(sidecar: dict, pimir: Path) -> None:
    """把全部 GEMM 条目改指到同一份 pimir，并同步哈希（代价伪造探针）。"""
    import hashlib

    digest = hashlib.sha256(pimir.read_bytes()).hexdigest()
    for entry in sidecar["operators"].values():
        if entry.get("op_type") == "GEMM":
            entry["pimir_path"] = str(pimir)
            entry["pimir_sha256"] = digest


def test_audit_rejects_gemms_pointing_at_a_bpath_pimir(tmp_path: Path) -> None:
    """A 路 GEMM 指向 B 路逐元素产物时必须失败：否则 trace 代价可任意改写。"""
    from scripts.run_full_pipeline import StepFailed, validate_sidecar_contract

    ir, sidecar = _contract_case(tmp_path)
    bpath = Path(sidecar["operators"]["3"]["pimir_path"])
    _point_gemms_at(sidecar, bpath)
    with pytest.raises(StepFailed, match="分块 GEMM|A 路"):
        validate_sidecar_contract(ir, sidecar)


def test_audit_rejects_a_gemm_pimir_from_another_shard_width(
        tmp_path: Path) -> None:
    """GEMM 的 pimir 必须按该分片的本地宽度编：换一份别的宽度就失败。"""
    from scripts.run_full_pipeline import StepFailed, validate_sidecar_contract

    import hashlib

    ir, sidecar = _contract_case(tmp_path)
    other = tmp_path / "gate_other_width.pimir.mlir"
    _write_gemm_pimir(other, 64, 8, tile_n=8)
    sidecar["operators"]["2"]["pimir_path"] = str(other)
    sidecar["operators"]["2"]["pimir_sha256"] = hashlib.sha256(
        other.read_bytes()).hexdigest()
    with pytest.raises(StepFailed, match="tile-n|分块 GEMM|循环|N="):
        validate_sidecar_contract(ir, sidecar)


def test_audit_rejects_a_device_gemm_without_kernel_tile_n(
        tmp_path: Path) -> None:
    """删掉 kernel_tile_n 会让 GeneSim 退回 conf 的常量分块，必须拒绝。"""
    from scripts.run_full_pipeline import StepFailed, validate_sidecar_contract

    ir, sidecar = _contract_case(tmp_path)
    for entry in sidecar["operators"].values():
        if entry.get("op_type") == "GEMM":
            entry.pop("kernel_tile_n")
    with pytest.raises(StepFailed, match="kernel_tile_n"):
        validate_sidecar_contract(ir, sidecar)


def test_audit_rejects_a_fused_gemm_without_act_spec(tmp_path: Path) -> None:
    """融合清单声明折了激活，pimir 实产必须带同名 act_spec。"""
    from scripts.run_full_pipeline import StepFailed, validate_sidecar_contract

    ir, sidecar = _contract_case(tmp_path)
    sidecar["fused_activations"] = [{
        "layer": 0, "role": "gate_proj", "expert": 0, "activation": "silu",
    }]
    with pytest.raises(StepFailed, match="激活|act_spec"):
        validate_sidecar_contract(ir, sidecar, require_absorbed_closure=False)


def test_audit_rejects_a_host_gemm_claim_without_source_ir_evidence(
        tmp_path: Path) -> None:
    """条目自称 host 不能免掉 A 路检查：源 IR/豁免/host_gemms 缺一即失败。"""
    from scripts.run_full_pipeline import StepFailed, validate_sidecar_contract

    ir, sidecar = _contract_case(tmp_path)
    # 用非专家 GEMM（router）：专家 GEMM 先被身份核对拒绝，测不到 host 规则。
    entry = sidecar["operators"]["1"]
    entry["device_hint"] = "cpu"
    entry.pop("shards")
    entry.pop("kernel_tile_n")
    entry.pop("semantic_role")
    with pytest.raises(StepFailed, match="host|设备 GEMM 却没有分片"):
        validate_sidecar_contract(ir, sidecar)


@pytest.mark.parametrize("entry", ["freshness", "sim_ir", "trace"])
def test_strict_entries_reject_a_host_gemm_claim(
        tmp_path: Path, entry: str) -> None:
    """五入口共享同一判定：lm_head 式的 host 自报在三个后置入口都失败。"""
    from scripts.run_full_pipeline import StepFailed

    genesim, _ir, sidecar, config = _strict_contract_genesim(tmp_path)
    unit = sidecar["operators"]["1"]
    unit["device_hint"] = "cpu"
    unit.pop("shards")
    unit.pop("kernel_tile_n")
    unit.pop("semantic_role")
    (genesim / "models" / "sc.json").write_text(json.dumps(sidecar))
    with pytest.raises(StepFailed, match="host|设备 GEMM 却没有分片"):
        _run_post_refine_entry(entry, genesim, config)


def test_skip_simulation_rejects_a_host_gemm_claim(
        tmp_path, monkeypatch) -> None:
    """--skip-simulation 共用同一份 A 路与 host 证据规则。"""
    ir, sidecar = _contract_case(tmp_path)
    entry = sidecar["operators"]["1"]
    entry["device_hint"] = "cpu"
    entry.pop("shards")
    entry.pop("kernel_tile_n")
    entry.pop("semantic_role")
    assert _skip_simulation_sidecar(tmp_path, ir, sidecar, monkeypatch) == 1


@pytest.mark.parametrize("declared", ["VECTOR_ADD", "GELU"])
def test_audit_rejects_a_bpath_op_type_that_disagrees_with_the_source_ir(
        declared: str) -> None:
    """条目 op_type 必须等于源 IR 类型：SILU 改成别的类型不能签发。"""
    from scripts.run_full_pipeline import StepFailed, validate_sidecar_contract

    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        ir, sidecar = _contract_case(Path(tmp))
        sidecar["operators"]["3"]["op_type"] = declared
        with pytest.raises(StepFailed, match="op_type|op3"):
            validate_sidecar_contract(ir, sidecar)


def test_audit_rejects_a_device_gemm_without_semantic_role(
        tmp_path: Path) -> None:
    """设备 GEMM 的投影身份必须回传并与源 IR 一致。"""
    from scripts.run_full_pipeline import StepFailed, validate_sidecar_contract

    ir, sidecar = _contract_case(tmp_path)
    sidecar["operators"]["1"].pop("semantic_role")
    with pytest.raises(StepFailed, match="semantic_role"):
        validate_sidecar_contract(ir, sidecar)


def test_audit_rejects_a_bpath_eltwise_kind_that_disagrees_with_the_ir(
        tmp_path: Path) -> None:
    """同形同字节的 add/mul 互换：pimir 实产的逐元素语义必须与 IR 类型对应。"""
    import hashlib

    from scripts.run_full_pipeline import StepFailed, validate_sidecar_contract

    pimir = tmp_path / "add.pimir.mlir"
    pimir.write_text(
        "module {\n"
        "  tt.func @op(%arg0: tensor<1x16xf16>, %arg1: tensor<1x16xf16>) "
        "attributes {pim.traceTokens = 1 : i64} {\n"
        "    %0 = pim.eltwise %arg0, %arg1 {kind = #pim.eltwise<add>} : "
        "tensor<1x16xf16>, tensor<1x16xf16> -> tensor<1x16xf16>\n"
        "    tt.return\n  }\n}\n")
    digest = hashlib.sha256(pimir.read_bytes()).hexdigest()
    ir = {"operators": [
        {"op_id": 0, "op_type": "MODEL_INPUT"},
        {"op_id": 1, "op_type": "VECTOR_MUL", "expert_id": -1,
         "input_shapes": [[1, 16], [1, 16]], "output_shapes": [[1, 16]]},
        {"op_id": 2, "op_type": "MODEL_OUTPUT"},
    ], "dependencies": [], "subgraphs": [[1]]}
    sidecar = {
        "requires_pimir": True,
        "operators": {"1": {
            "op_type": "VECTOR_MUL", "device_hint": "pim", "shards": [],
            "pimir_path": str(pimir), "pimir_sha256": digest,
            "trace_tokens": 1, "local_input_shapes": [[1, 16], [1, 16]],
            "local_output_shapes": [[1, 16]],
        }},
        "hw_budget": _capacity_sidecar()["hw_budget"],
    }
    with pytest.raises(StepFailed, match="eltwise|逐元素语义|mul"):
        validate_sidecar_contract(ir, sidecar)


def test_trace_lineage_checks_bytes_by_the_source_ir_op_type() -> None:
    """trace 字节判据由源 IR 类型选择，不用条目自报的 op_type。"""
    from scripts.run_full_pipeline import _trace_source_lineage

    ir = {"operators": [{
        "op_id": 2, "op_type": "VECTOR_MUL", "expert_id": -1,
        "input_shapes": [["Tq", 768], ["Tq", 768]],
        "output_shapes": [["Tq", 768]],
    }]}
    entry = {
        # 条目自称 SPLIT：若按它选判据，只查读写相等（768==768）就会放过。
        "op_type": "SPLIT", "device_hint": "pim", "shards": [],
        "pimir_path": "/x/op.pimir.mlir", "pimir_sha256": "abc123",
        "trace_tokens": 1,
        "local_input_shapes": [["Tq", 768], ["Tq", 768]],
        "local_output_shapes": [["Tq", 768]],
    }
    sidecar = {"operators": {"2": entry}, "absorbed_ops": []}
    record = {
        "op_id": 100, "op_type": "VECTOR_MUL", "parent_op_id": 2,
        "trace_source": "pimir",
        "compile_signature": {"pimir_sha256": "abc123"},
        "pimir": {"tokens": 1, "phase_count": 1,
                  "total_in_bytes": 768, "total_out_bytes": 768},
        "input_shapes": [["Tq", 768], ["Tq", 768]],
        "output_shapes": [["Tq", 768]],
    }
    problems = _trace_source_lineage([record], ir, sidecar, strict=True)
    assert any("字节" in problem for problem in problems), problems


@pytest.mark.parametrize("entry", ["freshness", "sim_ir", "trace"])
@pytest.mark.parametrize("case", ["bpath", "notile"])
def test_strict_entries_reject_an_invalid_apath_pimir(
        tmp_path: Path, entry: str, case: str) -> None:
    """A 路语义在 freshness、仿真 IR 与 trace 三个共享入口同判。

    A 路核对只有挂在共享验证器上，才能让导出、skip-simulation 与三个
    后置入口对同一份 sidecar 给出同一结论。
    """
    from scripts.run_full_pipeline import StepFailed

    genesim, _ir, sidecar, config = _strict_contract_genesim(tmp_path)
    if case == "bpath":
        _point_gemms_at(sidecar, Path(sidecar["operators"]["3"]["pimir_path"]))
    else:
        for op_entry in sidecar["operators"].values():
            if op_entry.get("op_type") == "GEMM":
                op_entry.pop("kernel_tile_n")
    (genesim / "models" / "sc.json").write_text(json.dumps(sidecar))
    with pytest.raises(StepFailed, match="A 路|分块 GEMM|kernel_tile_n"):
        _run_post_refine_entry(entry, genesim, config)


@pytest.mark.parametrize("case", ["bpath", "notile"])
def test_skip_simulation_rejects_an_invalid_apath_pimir(
        tmp_path, monkeypatch, case: str) -> None:
    """--skip-simulation 也走同一份 A 路规则。"""
    ir, sidecar = _contract_case(tmp_path)
    if case == "bpath":
        _point_gemms_at(sidecar, Path(sidecar["operators"]["3"]["pimir_path"]))
    else:
        for op_entry in sidecar["operators"].values():
            if op_entry.get("op_type") == "GEMM":
                op_entry.pop("kernel_tile_n")
    assert _skip_simulation_sidecar(tmp_path, ir, sidecar, monkeypatch) == 1


# ── round20 问题1：B 路 pimir 的顶层语义必须与算子类型对应 ──────────

# 每个设备算子按**真实编译实产**取一份语义样本：顶层原语与关键属性都和
# 其他类型不同，宽度统一用 16，确保判据落在语义而不是形状上。
_BPATH_SEMANTIC_PIMIRS = {
    "RMSNORM": """module {
  tt.func @op(%a: tensor<1x16xf16>, %w: tensor<16xf16>) attributes {pim.traceTokens = 1 : i64} {
    %0 = pim.normalize %a, %w {axis = 1 : i64, rmsNorm} : tensor<1x16xf16>, tensor<16xf16> -> tensor<1x16xf16>
    tt.return
  }
}
""",
    "SILU": """module {
  tt.func @op(%a: tensor<1x16xf16>) attributes {pim.traceTokens = 1 : i64} {
    %0 = pim.lut %a {kind = #pim.activation<silu>} : tensor<1x16xf16> -> tensor<1x16xf16>
    tt.return
  }
}
""",
    "GELU": """module {
  tt.func @op(%a: tensor<1x16xf16>) attributes {pim.traceTokens = 1 : i64} {
    %0 = pim.lut %a {kind = #pim.activation<gelu>} : tensor<1x16xf16> -> tensor<1x16xf16>
    tt.return
  }
}
""",
    "VECTOR_ADD": """module {
  tt.func @op(%a: tensor<1x16xf16>, %b: tensor<1x16xf16>) attributes {pim.traceTokens = 1 : i64} {
    %0 = pim.eltwise %a, %b {kind = #pim.eltwise<add>} : tensor<1x16xf16>, tensor<1x16xf16> -> tensor<1x16xf16>
    tt.return
  }
}
""",
    "VECTOR_MUL": """module {
  tt.func @op(%a: tensor<1x16xf16>, %b: tensor<1x16xf16>) attributes {pim.traceTokens = 1 : i64} {
    %0 = pim.eltwise %a, %b {kind = #pim.eltwise<mul>} : tensor<1x16xf16>, tensor<1x16xf16> -> tensor<1x16xf16>
    tt.return
  }
}
""",
    "PAGED_ATTN": """module {
  tt.func @op(%a: tensor<1x16xf16>, %c: !pim.memdesc<4096xf16, #pim.mram>) attributes {pim.traceTokens = 1 : i64} {
    %0 = pim.paged_attn %a, %c {scale = 0.1 : f64, slotElems = 2048 : i64, slotTokens = 16 : i64} : tensor<1x16xf16>, !pim.memdesc<4096xf16, #pim.mram> -> tensor<1x16xf16>
    tt.return
  }
}
""",
    "ROPE": """module {
  tt.func @op(%a: tensor<1x1x1x16xf16>, %b: tensor<1x1x1x16xf16>, %c: tensor<1x1x1x16xf16>) attributes {pim.traceTokens = 1 : i64} {
    %0 = pim.eltwise %a, %b {kind = #pim.eltwise<mul>} : tensor<1x1x1x16xf16>, tensor<1x1x1x16xf16> -> tensor<1x1x1x16xf16>
    %1 = pim.eltwise %a, %c {kind = #pim.eltwise<mul>, rotateHalf} : tensor<1x1x1x16xf16>, tensor<1x1x1x16xf16> -> tensor<1x1x1x16xf16>
    %2 = pim.eltwise %0, %1 {kind = #pim.eltwise<add>, sourceSubBlocks = ["Llama2Activation_Add_Cos", "Llama2Activation_Add_Sin"]} : tensor<1x1x1x16xf16>, tensor<1x1x1x16xf16> -> tensor<1x1x1x16xf16>
    tt.return
  }
}
""",
    "TRANSPOSE": """module {
  tt.func @op(%a: tensor<1x16xf16>) attributes {pim.traceTokens = 1 : i64} {
    %0 = pim.transpose %a {axes = array<i64: 1, 0>} : tensor<1x16xf16> -> tensor<16x1xf16>
    tt.return
  }
}
""",
    "SPLIT": """module {
  tt.func @op(%a: tensor<1x16xf16>) attributes {pim.traceTokens = 1 : i64} {
    %0 = pim.split_heads %a {axis = 1 : i64, numHeads = 1 : i64} : tensor<1x16xf16> -> tensor<1x16xf16>
    tt.return
  }
}
""",
    "CONCAT": """module {
  tt.func @op(%a: tensor<1x16xf16>, %b: tensor<1x16xf16>) attributes {pim.traceTokens = 1 : i64} {
    %0 = pim.concat %a, %b {axis = 1 : i64} : tensor<1x16xf16>, tensor<1x16xf16> -> tensor<1x32xf16>
    tt.return
  }
}
""",
    "RESHAPE": """module {
  tt.func @op(%a: tensor<1x16xf16>) attributes {pim.traceTokens = 1 : i64} {
    %0 = pim.reshape %a : tensor<1x16xf16> -> tensor<1x4x4xf16>
    tt.return
  }
}
""",
    "GATHER": """module {
  tt.func @op(%a: tensor<1x16xf16>, %i: tensor<1x1xi32>) attributes {pim.traceTokens = 1 : i64} {
    %0 = pim.gather %a, %i : tensor<1x16xf16>, tensor<1x1xi32> -> tensor<1x1x16xf16>
    tt.return
  }
}
""",
    "MEM_COPY": """module {
  tt.func @op(%a: tensor<1x16xi8>, %c: !pim.memdesc<16xi8, #pim.mram>, %f: i32) attributes {pim.traceTokens = 1 : i64} {
    pim.kv_cache %a, %c, %f {isKey} : tensor<1x16xi8>, !pim.memdesc<16xi8, #pim.mram>, i32
    tt.return
  }
}
""",
    "QUANT": """module {
  tt.func @op(%a: tensor<1x16xf16>) attributes {pim.traceTokens = 1 : i64} {
    %0 = pim.global_pool %a {groupSize = 128 : i64, kind = #pim.pool_kind<absmax>} : tensor<1x16xf16> -> tensor<1x1xf16>
    %1 = pim.lut %0 {kind = #pim.activation<identity>} : tensor<1x1xf16> -> tensor<1x1xf16>
    %2 = pim.kantor %a scale %1 {spec = #pim.kantor_spec<mode = fp2int_converter, cardValue = 3>} : tensor<1x16xf16> scale tensor<1x1xf16> -> tensor<1x16xi8>
    tt.return
  }
}
""",
    "MASK": """module {
  tt.func @op(%a: tensor<128x16xf16>, %b: tensor<128x16xf16>) attributes {pim.kvBlockTokens = 128 : i64, pim.traceTokens = 1 : i64} {
    %0 = pim.mask %a, %b {layout = #pim.mask_layout<causal_tril>} : tensor<128x16xf16>, tensor<128x16xf16> -> tensor<128x16xf16>
    tt.return
  }
}
""",
    "SOFTMAX": """module {
  tt.func @op(%a: tensor<128x16xf16>) attributes {pim.kvBlockTokens = 128 : i64, pim.traceTokens = 1 : i64} {
    %0 = pim.reduce_axis %a {axis = 1 : i64, kind = #pim.eltwise<max>} : tensor<128x16xf16> -> tensor<128x1xf16>
    %1 = pim.eltwise %a, %0 {kind = #pim.eltwise<sub>} : tensor<128x16xf16>, tensor<128x1xf16> -> tensor<128x16xf16>
    %2 = pim.lut %1 {kind = #pim.activation<exp>} : tensor<128x16xf16> -> tensor<128x16xf16>
    %3 = pim.reduce_axis %2 {axis = 1 : i64, kind = #pim.eltwise<add>} : tensor<128x16xf16> -> tensor<128x1xf32>
    %4 = pim.lut %3 {kind = #pim.activation<reciprocal>} : tensor<128x1xf32> -> tensor<128x1xf16>
    %5 = pim.eltwise %2, %4 {kind = #pim.eltwise<mul>} : tensor<128x16xf16>, tensor<128x1xf16> -> tensor<128x16xf16>
    tt.return
  }
}
""",
    "GEMV_SCORE": """module {
  tt.func @op(%a: tensor<128x16xi8>, %b: tensor<16x16xi8>) attributes {pim.kvBlockTokens = 128 : i64, pim.kvFixedIn = 16384 : i64, pim.kvFixedOut = 0 : i64, pim.traceTokens = 1 : i64} {
    %0 = pim.matmul %a, %b {stationarity = #pim.stationarity<kv>} : tensor<128x16xi8>, tensor<16x16xi8> -> tensor<128x16xi8>
    tt.return
  }
}
""",
    "GEMV_CONTEXT": """module {
  tt.func @op(%a: tensor<128x16xi8>, %b: tensor<16x16xi8>) attributes {pim.kvBlockTokens = 128 : i64, pim.kvFixedIn = 0 : i64, pim.kvFixedOut = 16384 : i64, pim.traceTokens = 1 : i64} {
    %0 = pim.matmul %a, %b {stationarity = #pim.stationarity<kv>} : tensor<128x16xi8>, tensor<16x16xi8> -> tensor<128x16xi8>
    tt.return
  }
}
""",
}


def _bpath_semantic_case(
    tmp_path: Path, op_type: str, pimir_text: str,
) -> tuple[dict, dict]:
    """单算子 B 路产物：源 IR 类型与所引 pimir 的语义可以故意不一致。"""
    import hashlib

    from memory.mem_planner import default_hw_budget

    pimir = tmp_path / "cited.pimir.mlir"
    pimir.write_text(pimir_text)
    digest = hashlib.sha256(pimir.read_bytes()).hexdigest()
    shapes = [[1, 16]]
    ir = {
        "operators": [
            {"op_id": 0, "op_type": "MODEL_INPUT"},
            {"op_id": 1, "op_type": op_type, "expert_id": -1,
             "input_shapes": shapes, "output_shapes": shapes},
            {"op_id": 2, "op_type": "MODEL_OUTPUT"},
        ],
        "dependencies": [],
        "subgraphs": [[1]],
    }
    hw = default_hw_budget()
    sidecar = {
        "requires_pimir": True,
        "hw_budget": {
            "mram_bytes_per_dpu": hw.mram_bytes,
            "sys_reserve_bytes": hw.sys_reserve_bytes,
            "align": hw.align,
        },
        "operators": {"1": {
            "op_type": op_type, "device_hint": "pim", "shards": [],
            "pimir_path": str(pimir), "pimir_sha256": digest,
            "trace_tokens": 1,
            "local_input_shapes": shapes,
            "local_output_shapes": shapes,
        }},
    }
    return ir, sidecar


@pytest.mark.parametrize("op_type, cited", [
    ("ROPE", "RMSNORM"),
    ("PAGED_ATTN", "RMSNORM"),
    ("RMSNORM", "ROPE"),
    ("SILU", "GELU"),
    ("SILU", "VECTOR_MUL"),
    # 两次注意力矩阵乘形状相同，靠固定读写端区分：得分乘 q 只读一次，
    # 上下文乘的累加结果只写一次，互换会让块循环的固定开销记错算子。
    ("GEMV_SCORE", "GEMV_CONTEXT"),
    ("GEMV_CONTEXT", "GEMV_SCORE"),
])
def test_audit_rejects_a_bpath_pimir_with_another_operators_semantics(
        tmp_path: Path, op_type: str, cited: str) -> None:
    """同宽不同算子的 pimir 互换后，token 数与末维仍然一致，必须靠语义指纹拒绝。

    这是 round19 只绑定 add/mul 的同类根因：ROPE/PAGED_ATTN/RMSNORM 等
    互换了编译实产后，路径、哈希、token 数、特征维全部成立。
    """
    from scripts.run_full_pipeline import StepFailed, validate_sidecar_contract

    ir, sidecar = _bpath_semantic_case(
        tmp_path, op_type, _BPATH_SEMANTIC_PIMIRS[cited])
    with pytest.raises(StepFailed, match="顶层原语|语义|pimir"):
        validate_sidecar_contract(ir, sidecar)


def test_audit_rejects_a_bpath_operator_type_without_a_calibrated_fingerprint(
        tmp_path: Path) -> None:
    """没有按实产校准过的 B 路类型直接拒绝，不能只靠 op_type 自报。"""
    from scripts.run_full_pipeline import StepFailed, validate_sidecar_contract

    ir, sidecar = _bpath_semantic_case(
        tmp_path, "BRAND_NEW_OP", _BPATH_SEMANTIC_PIMIRS["SILU"])
    with pytest.raises(StepFailed, match="语义指纹|登记"):
        validate_sidecar_contract(ir, sidecar)


@pytest.mark.parametrize("op_type", sorted(_BPATH_SEMANTIC_PIMIRS))
def test_bpath_semantic_table_accepts_each_calibrated_product(
        op_type: str) -> None:
    """每个真实产物的形态都必须能通过它自己那一行指纹，避免表过严误杀。"""
    from scripts.run_full_pipeline import (
        _BPATH_PIMIR_SEMANTICS,
        _bpath_pimir_semantic_problems,
        _pimir_unit_facts,
    )

    assert op_type in _BPATH_PIMIR_SEMANTICS
    facts = _pimir_unit_facts(_BPATH_SEMANTIC_PIMIRS[op_type])
    problems = _bpath_pimir_semantic_problems(op_type, facts, f"op-{op_type}")
    assert problems == [], problems


def test_bpath_semantic_table_is_pinned_across_repos() -> None:
    """两个仓各自持有同一张指纹表，跨仓测试钉住逐值相等。"""
    from scripts.run_full_pipeline import _BPATH_PIMIR_SEMANTICS

    genesim_src = genesim_root() / "src"
    if str(genesim_src) not in sys.path:
        sys.path.insert(0, str(genesim_src))
    from pim.pimir_trace import BPATH_PIMIR_SEMANTICS

    assert _BPATH_PIMIR_SEMANTICS == BPATH_PIMIR_SEMANTICS


@pytest.mark.parametrize("entry", ["freshness", "sim_ir", "trace"])
@pytest.mark.parametrize("cited", ["GELU", "VECTOR_MUL"])
def test_strict_entries_reject_a_bpath_semantic_swap(
        tmp_path: Path, entry: str, cited: str) -> None:
    """三个后置入口与导出共用同一张 B 路语义指纹表。

    op3 是 SILU；把它的 pimir 换成同宽的 GELU 或 VECTOR_MUL 产物后，
    token 数与末维仍然一致，只有共享验证器的语义指纹能拦住。
    """
    import hashlib

    from scripts.run_full_pipeline import StepFailed

    genesim, _ir, sidecar, config = _strict_contract_genesim(tmp_path)
    pimir = tmp_path / f"{cited}.pimir.mlir"
    pimir.write_text(_BPATH_SEMANTIC_PIMIRS[cited])
    sidecar["operators"]["3"]["pimir_path"] = str(pimir)
    sidecar["operators"]["3"]["pimir_sha256"] = hashlib.sha256(
        pimir.read_bytes()).hexdigest()
    (genesim / "models" / "sc.json").write_text(json.dumps(sidecar))
    with pytest.raises(StepFailed, match="顶层原语|语义|pimir"):
        _run_post_refine_entry(entry, genesim, config)


def test_skip_simulation_rejects_a_bpath_semantic_swap(
        tmp_path: Path, monkeypatch) -> None:
    """--skip-simulation 与导出、后置入口同判。"""
    import hashlib

    ir, sidecar = _contract_case(tmp_path)
    pimir = tmp_path / "gelu.pimir.mlir"
    pimir.write_text(_BPATH_SEMANTIC_PIMIRS["GELU"])
    sidecar["operators"]["3"]["pimir_path"] = str(pimir)
    sidecar["operators"]["3"]["pimir_sha256"] = hashlib.sha256(
        pimir.read_bytes()).hexdigest()
    assert _skip_simulation_sidecar(tmp_path, ir, sidecar, monkeypatch) == 1


_SEMANTIC_FACT_KEYS = (
    "top_level_ops", "activation_kind", "eltwise_kind", "has_rms_norm",
    "has_rotate_half", "has_source_sub_blocks", "has_kv_block_tokens",
    "has_stationarity_kv", "kv_fixed_in_positive", "kv_fixed_out_positive",
    "has_slot_tokens", "has_slot_elems", "has_reciprocal_lut",
)


def test_genesim_pimir_facts_agree_on_each_calibrated_product() -> None:
    """两仓各自实现的 pimir 解析对同一份实产必须给出同一组语义事实。

    只钉住表相等不够：一侧把 `rotateHalf` 解析丢、另一侧保留，表仍然相等，
    判定却分叉。这里用 18 类样本逐键比较。
    """
    from scripts.run_full_pipeline import _pimir_unit_facts

    genesim_src = genesim_root() / "src"
    if str(genesim_src) not in sys.path:
        sys.path.insert(0, str(genesim_src))
    from pim.pimir_trace import (
        bpath_pimir_semantic_problems, pimir_unit_facts)

    for op_type, text in _BPATH_SEMANTIC_PIMIRS.items():
        ours = _pimir_unit_facts(text)
        theirs = pimir_unit_facts(text)
        for key in _SEMANTIC_FACT_KEYS:
            assert ours[key] == theirs[key], (op_type, key)
        assert bpath_pimir_semantic_problems(op_type, theirs, "op") == []
