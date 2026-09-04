#!/usr/bin/env python3
"""跨三个仓库跑通 Llama2-7B 的完整链路，并核对每一段真的生效了。

链路与每段的载体：

    HuggingFace 权重 + config
      │
      ├─(A)─> GeneSim model_parser        → 图骨架 IR（semantic_role 标好投影身份）
      │
      ├─(B)─> 图编译器 compile_llama2      → PIMTensorSpec（每个权重的切分与归属）
      │         策略由 --num-stages 决定
      │
      ├─(C)─> 算子编译器 FlagTree          → .so + pim mlir（真实分块由 WRAM 预算定）
      │         每种本地分片形状编一次
      │
      ├─(D)─> placement sidecar           → dpu_id / local_*_features /
      │                                      kernel_tile_n / pimir_path
      │
      └─(E)─> GeneSim 仿真                 → 照 pim mlir 生成 PIM trace，出代价

用法（先 source paths.json 里的 pytorch_env_script）：

    python scripts/run_full_pipeline.py --num-stages 4

    # 只想快速验证接线、不跑完整仿真：
    python scripts/run_full_pipeline.py --num-stages 4 --skip-simulation

    # A/B 对照：同一份 sidecar，关掉本地分片形状再跑一次
    python scripts/run_full_pipeline.py --num-stages 4 --ab-compare

每一步跑完都会核对产物，不满足就直接失败——这个脚本的用途是回答「链路通没通」，
所以任何一段静默退化都必须变成非零退出码。
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from genesim_bridge.paths import genesim_models_dir, genesim_root, llama2_7b_model_dir

_REPO_ROOT = Path(__file__).resolve().parent.parent
# 七个投影，缺一个都说明 IR 或匹配退化了。
_EXPECTED_ROLES = {
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
}


class StepFailed(RuntimeError):
    """某一段没达到判据。"""


def _run(cmd: list[str], *, cwd: Path, log_path: Path, env: Optional[dict] = None) -> None:
    """跑一条命令，输出落到 log_path；失败时把尾部贴出来再抛。"""
    print(f"    $ {' '.join(str(c) for c in cmd)}")
    print(f"      日志: {log_path}")
    with log_path.open("w") as handle:
        proc = subprocess.run(
            cmd, cwd=str(cwd), stdout=handle, stderr=subprocess.STDOUT,
            env={**os.environ, **(env or {})},
        )
    if proc.returncode != 0:
        tail = "\n".join(log_path.read_text().splitlines()[-30:])
        raise StepFailed(
            f"命令失败（exit {proc.returncode}）：{' '.join(str(c) for c in cmd)}\n"
            f"日志尾部：\n{tail}"
        )


def step_a_model_ir(genesim: Path, model_dir: Path, log_dir: Path) -> Path:
    """(A) 从 HF config 生成 GeneSim 图骨架，核对投影身份标全了。"""
    print("\n[A] HuggingFace config → GeneSim 图骨架 IR")
    ir_path = genesim / "models" / "llama2_7b.ir"
    _run(
        ["python", "scripts/model_parser.py",
         "--model_name", str(model_dir), "--output", str(ir_path)],
        cwd=genesim, log_path=log_dir / "a_model_parser.log",
    )

    ir = json.loads(ir_path.read_text())
    gemms = [op for op in ir["operators"] if op["op_type"] == "GEMM"]
    roles = {op.get("semantic_role", "") for op in gemms}
    missing = _EXPECTED_ROLES - roles
    if missing:
        raise StepFailed(
            f"图骨架里缺这些投影的 semantic_role: {sorted(missing)}。"
            "放置导出按语义标签匹配，缺了就无法确定 GEMM 身份。"
        )
    unlabeled = [op["op_id"] for op in gemms if not op.get("semantic_role")]
    if unlabeled:
        raise StepFailed(f"{len(unlabeled)} 个 GEMM 没有 semantic_role，如 {unlabeled[:5]}")
    print(f"    算子 {len(ir['operators'])} 个，GEMM {len(gemms)} 个，七种投影身份齐全")
    return ir_path


def step_zero_pu_mapping(
    genesim: Path, ir_path: Path, num_stages: int, num_dpus: int, log_dir: Path
) -> Path:
    """(0) GeneSim 给出固定 PU 映射，核对逐段的 Cluster 归属合理。

    这一段方向与其余三段相反：不是"编译器算完告诉 GeneSim"，而是"GeneSim 定下
    PU 映射、约束编译器"。产物是 PartitionPlan，由编译器侧转成 ShardStrategy。
    """
    print("\n[0] GeneSim 固定 PU 映射 → PartitionPlan")
    tp_width = num_dpus // num_stages
    plan_path = genesim / "models" / f"llama2_7b_tp{tp_width}_pp{num_stages}_plan.json"
    _run(
        ["python", "scripts/export_fixed_pu_mapping.py",
         "--num-stages", str(num_stages), "--num-dpus", str(num_dpus),
         "--ir", str(ir_path), "--out", str(plan_path)],
        cwd=genesim, log_path=log_dir / "0_pu_mapping.log",
    )

    plan = json.loads(plan_path.read_text())
    mapping = plan.get("dpu_to_cluster")
    if not mapping or len(mapping) != num_dpus:
        raise StepFailed(
            f"方案里的 dpu_to_cluster 应有 {num_dpus} 项，实际 "
            f"{len(mapping) if mapping else 0} 项"
        )
    if plan.get("num_stages") != num_stages or plan.get("num_dpus") != num_dpus:
        raise StepFailed(
            f"方案的段数/DPU 数与请求不一致：{plan.get('num_stages')}/{plan.get('num_dpus')}"
        )
    # 报告段内是否走快链路——同 Cluster 是 512 GB/s，跨 Cluster 是 128 GB/s。
    for stage in range(num_stages):
        members = [tuple(e) for e in mapping[stage * tp_width : (stage + 1) * tp_width]]
        shared = len(set(members)) == 1
        print(f"    stage{stage}: {members} "
              f"({'同 Cluster 快链路' if shared else '跨 Cluster'})")
    print(f"    方案: {plan_path.name}（source={plan.get('source', '未标注')}）")
    return plan_path


def step_bcd_export(
    num_stages: int, num_dpus: int, log_dir: Path,
    *, plan_path: Optional[Path] = None,
) -> Path:
    """(B)(C)(D) 图编译 + 算子编译 + 导出 sidecar，核对各类字段都写了。

    `plan_path` 给了就按 GeneSim 的方案编译（段数由方案决定），否则用 --num-stages。
    """
    print("\n[B+C+D] 图编译 → 算子编译（真实分块）→ placement sidecar")
    strategy_args = (
        ["--partition-plan", str(plan_path)] if plan_path is not None
        else ["--num-stages", str(num_stages), "--num-dpus", str(num_dpus)]
    )
    _run(
        ["python", "scripts/export_pp_placement.py",
         *strategy_args, "--measure-kernel-tiles"],
        cwd=_REPO_ROOT, log_path=log_dir / "bcd_export.log",
    )

    models_dir = genesim_models_dir()
    tp_width = num_dpus // num_stages
    name = f"tp{tp_width}_pp{num_stages}"
    sidecar_path = models_dir / f"llama2_7b_{name}_placement.json"
    if not sidecar_path.is_file():
        raise StepFailed(f"没找到 sidecar：{sidecar_path}")

    sidecar = json.loads(sidecar_path.read_text())
    ops = sidecar["operators"]
    if not ops:
        raise StepFailed("sidecar 里没有任何放置结果")

    # 四类字段各自对应链路上的一段，逐个核对，不能只看文件存在。
    checks = {
        "dpu_id（图切分归属）": lambda e: "dpu_id" in e,
        "local_in/out_features（本地分片形状）": lambda e: (
            e.get("local_in_features") and e.get("local_out_features")
        ),
        "semantic_role（投影身份）": lambda e: e.get("semantic_role") in _EXPECTED_ROLES,
        "kernel_tile_n（算子编译器实测分块）": lambda e: (e.get("kernel_tile_n") or 0) > 0,
        "pimir_path（pim mlir 落盘路径）": lambda e: bool(e.get("pimir_path")),
    }
    for label, predicate in checks.items():
        bad = [op_id for op_id, entry in ops.items() if not predicate(entry)]
        if bad:
            raise StepFailed(
                f"{len(bad)} 个算子缺 {label}，如 op{bad[:5]}"
            )

    pimir_files = {entry["pimir_path"] for entry in ops.values()}
    absent = [p for p in pimir_files if not Path(p).is_file()]
    if absent:
        raise StepFailed(f"pimir_path 指向的文件不存在：{absent[:3]}")

    tiles = sorted({int(e["kernel_tile_n"]) for e in ops.values()})
    print(f"    放置 {len(ops)} 个 GEMM，本地形状 {len(pimir_files)} 种 pim mlir")
    print(f"    算子编译器选出的分块: {tiles}（GeneSim 默认常量是 32）")

    # 走方案路径时，Cluster 映射必须随 sidecar 回传，否则 GeneSim 会退回取模换算。
    if plan_path is not None:
        carried = sidecar.get("dpu_to_cluster")
        if not carried:
            raise StepFailed(
                "按 GeneSim 方案编译，但 sidecar 里没有 dpu_to_cluster——"
                "映射没有回传，GeneSim 会退回取模换算。"
            )
        declared = json.loads(Path(plan_path).read_text())["dpu_to_cluster"]
        if carried != declared:
            raise StepFailed(
                f"sidecar 带回的 dpu_to_cluster 与方案声明的不一致：\n"
                f"  方案: {declared}\n  sidecar: {carried}"
            )
        print(f"    Cluster 映射已随 sidecar 回传，与方案一致（{len(carried)} 项）")
    return sidecar_path


def step_e_simulate(
    genesim: Path, config_name: str, log_dir: Path, *, label: str,
    results_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """(E) 跑仿真，核对 GEMM 的 trace 真的来自 pim mlir。"""
    print(f"\n[E] GeneSim 仿真（{label}）")
    # trace 缓存按签名判新旧，但这里要的是"这一轮确实重编过"，所以先清掉。
    traces = genesim / "pim_traces"
    if traces.is_dir():
        shutil.rmtree(traces)
    _run(
        ["./run.sh", "--config", f"conf/{config_name}"],
        cwd=genesim, log_path=log_dir / f"e_sim_{label}.log",
    )

    summary_path = genesim / "results" / "summary.json"
    if not summary_path.is_file():
        raise StepFailed(f"仿真没产出 {summary_path}")
    summary = json.loads(summary_path.read_text())
    if results_dir is not None:
        if results_dir.exists():
            shutil.rmtree(results_dir)
        shutil.copytree(genesim / "results", results_dir)

    print(f"    total_time_s = {summary['total_time_s']:.3f}")
    print(f"    tokens/s     = {summary['throughput_tokens_per_s']:.3f}")
    return summary


def verify_trace_provenance(genesim: Path, *, expect_pimir: bool) -> None:
    """核对 GEMM 的 trace 来源，避免"跑通了但其实走的是手写模板"。"""
    sys.path.insert(0, str(genesim / "src"))
    from pim.pim_trace_loader import load_trace_file

    sources: Dict[str, int] = {}
    tiles: Dict[Any, int] = {}
    for path in sorted((genesim / "pim_traces").glob("op_*_GEMM.pim_trace")):
        trace = load_trace_file(str(path), use_cache=False)
        source = trace.metadata.get("trace_source", "unknown")
        sources[source] = sources.get(source, 0) + 1
        pimir_meta = trace.metadata.get("pimir") or {}
        if pimir_meta:
            key = (pimir_meta.get("tile_n"), pimir_meta.get("k_iterations"))
            tiles[key] = tiles.get(key, 0) + 1

    if not sources:
        raise StepFailed("没有找到任何 GEMM trace，无法判断来源")
    print(f"    GEMM trace 来源: {sources}")
    if tiles:
        print(f"    (tile_n, k_iterations) 分布: {tiles}")
    if expect_pimir and sources.get("template"):
        raise StepFailed(
            f"{sources['template']} 个 GEMM 退回了手写模板；"
            "算子编译器的分块没有进入代价链。"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--num-stages", type=int, default=4,
                        help="流水段数：1=纯张量并行，num_dpus=纯流水")
    parser.add_argument("--num-dpus", type=int, default=8)
    parser.add_argument("--skip-simulation", action="store_true",
                        help="只验证到 sidecar，不跑仿真（几分钟 vs 十几分钟）")
    parser.add_argument("--ab-compare", action="store_true",
                        help="额外跑一次关掉本地分片形状的对照，输出比值")
    parser.add_argument("--log-dir", default=None,
                        help="日志目录，默认 <repo>/test-results/full-pipeline")
    parser.add_argument(
        "--no-pu-mapping", action="store_true",
        help=(
            "跳过第 [0] 步，不让 GeneSim 给 PU 映射，改由 --num-stages 直接指定切分。"
            "用于对照：验证走方案与直接给段数产出的 sidecar 语义相同。"
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.num_dpus % args.num_stages:
        print(f"错误: num_stages={args.num_stages} 不能整除 num_dpus={args.num_dpus}",
              file=sys.stderr)
        return 2

    log_dir = Path(args.log_dir) if args.log_dir else _REPO_ROOT / "test-results" / "full-pipeline"
    log_dir.mkdir(parents=True, exist_ok=True)
    genesim = genesim_root()
    model_dir = llama2_7b_model_dir()
    tp_width = args.num_dpus // args.num_stages

    print("=" * 78)
    print(f"Llama2-7B 全流程：tp{tp_width}_pp{args.num_stages}（{args.num_dpus} 台 DPU）")
    print(f"  图编译: {_REPO_ROOT}")
    print(f"  GeneSim: {genesim}")
    print("=" * 78)

    try:
        ir_path = step_a_model_ir(genesim, model_dir, log_dir)
        plan_path = None
        if not args.no_pu_mapping:
            plan_path = step_zero_pu_mapping(
                genesim, ir_path, args.num_stages, args.num_dpus, log_dir
            )
        step_bcd_export(
            args.num_stages, args.num_dpus, log_dir, plan_path=plan_path
        )

        if args.skip_simulation:
            print("\n跳过仿真（--skip-simulation）。到 sidecar 为止的链路已验证。")
            print("\n全流程验证通过（未含仿真）。")
            return 0

        config = f"sim_llama2_7b_pp_tp{tp_width}pp{args.num_stages}_globalcost.yaml"
        if not (genesim / "conf" / config).is_file():
            raise StepFailed(
                f"没有对应的 GeneSim 配置 conf/{config}。"
                f"当前只为部分策略准备了配置，可参照已有文件新建一份。"
            )
        main_summary = step_e_simulate(
            genesim, config, log_dir, label="pimir",
            results_dir=Path("/tmp/full_pipeline_pimir"),
        )
        verify_trace_provenance(genesim, expect_pimir=True)

        if args.ab_compare:
            ab_config = f"sim_llama2_7b_pp_tp{tp_width}pp{args.num_stages}_ab_global.yaml"
            if not (genesim / "conf" / ab_config).is_file():
                print(f"\n跳过 A/B：没有 conf/{ab_config}")
            else:
                ab_summary = step_e_simulate(
                    genesim, ab_config, log_dir, label="global",
                    results_dir=Path("/tmp/full_pipeline_global"),
                )
                ratio = main_summary["total_time_s"] / ab_summary["total_time_s"]
                print("\n[A/B] 本地分片形状 vs 模型级全局形状")
                print(f"    全局形状 total_time_s = {ab_summary['total_time_s']:.3f}")
                print(f"    本地分片 total_time_s = {main_summary['total_time_s']:.3f}")
                print(f"    比值 = {ratio:.4f}（tp_width={tp_width}，理论上 ≈ 1/{tp_width}）")
                if abs(ratio - 1.0) < 1e-6:
                    raise StepFailed(
                        "两次仿真结果完全相同——切分没有进入代价链。"
                    )
    except StepFailed as exc:
        print(f"\n失败: {exc}", file=sys.stderr)
        return 1

    print("\n" + "=" * 78)
    print("全流程验证通过：模型加载 → 图编译切分 → 算子编译 → GeneSim 代价")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
