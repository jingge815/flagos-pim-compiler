#!/usr/bin/env python3
"""跨三个仓库跑通 Llama2-7B 的完整链路，并核对每一段真的生效了。

链路与每段的载体：

    HuggingFace 权重 + config
      │
      ├─(A)─> GeneSim model_parser        → 图骨架 IR（semantic_role 标好投影身份）
      │
      ├─(A2)> refine_ir_with_flagtree     → 精化 pimir 成本 IR（仿真加载的那份，
      │                                      与图骨架同一次运行、同一套算子编号）
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
import hashlib
import json
import os
import shutil
import struct
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

# 仿真配置 `model.ir_path` 指向的精化 IR 文件名（pimir 成本口径），由 (A2) 步产出。
_PIMIR_IR_NAME = "llama2_7b_pimir.ir"
# 精化的序列长度口径：与 docs/llama-2.md 第六步、现有产物、仿真 trace 一致。
_REFINE_SEQ_LEN = "128"


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


def _ir_op_ids(ir_path: Path) -> list[int]:
    """读出 IR 里按出现顺序的算子编号序列。"""
    ir = json.loads(ir_path.read_text())
    return [int(op["op_id"]) for op in ir["operators"]]


def step_a2_refine_pimir_ir(genesim: Path, ir_path: Path, log_dir: Path) -> Path:
    """(A2) 从刚生成的图骨架 IR 精化出仿真要加载的 pimir 成本 IR。

    仿真配置加载的 `models/llama2_7b_pimir.ir` 历来是手动跑
    `refine_ir_with_flagtree.py` 的产物：图骨架改了结构（"支持全部算子"给
    注意力补了算子级节点）而没重跑精化时，sidecar 按新编号、仿真 IR 还是旧
    编号，调度器按 op_id 查不到 pimir_path，注意力算子整批退回手写模板
    （2026-10-07 的 6144 个模板 trace 就是这么来的）。放进主流程后，图骨架
    与精化 IR 永远是同一次运行里的同一套编号。

    已存在且编号序列与图骨架一致时跳过重新精化：refine 只回填成本、不改
    结构，结构一致就是同一份。注意成本口径不在比对范围——工具链重编后
    想强制重算成本，删掉这份文件再跑即可。
    """
    print("\n[A2] 图骨架 IR → 精化 pimir 成本 IR（仿真加载的那份）")
    pimir_ir = genesim / "models" / _PIMIR_IR_NAME
    if pimir_ir.is_file() and _ir_op_ids(pimir_ir) == _ir_op_ids(ir_path):
        print(f"    精化 IR 与图骨架同套编号（{pimir_ir.name}），跳过重新精化")
        return pimir_ir
    _run(
        ["python", "scripts/refine_ir_with_flagtree.py",
         "--ir", str(ir_path), "--out-ir", str(pimir_ir),
         "--sidecar", str(genesim / "models" / "llama2_7b_pimir_extensions.json"),
         "--seq-len", _REFINE_SEQ_LEN, "--ir-level", "pimir"],
        cwd=genesim, log_path=log_dir / "a2_refine_pimir.log",
    )
    if _ir_op_ids(pimir_ir) != _ir_op_ids(ir_path):
        raise StepFailed(
            f"精化 IR 的算子编号与图骨架不一致：{pimir_ir} vs {ir_path}。"
            "refine 应该只回填成本、不改结构，请检查 refine_ir_with_flagtree.py。")
    print(f"    精化完成：{pimir_ir.name}"
          f"（{len(_ir_op_ids(pimir_ir))} 个算子，编号与图骨架一致）")
    return pimir_ir


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


def check_sidecar_entries(ops: Dict[str, Dict[str, Any]]) -> tuple[int, int]:
    """按条目类别分流核对 sidecar，返回 (GEMM 条数, 算子级条数)。

    sidecar 里两类条目的契约不同，用一套谓词查会把好的判成坏的：

      GEMM（A 路）   按图切分归属到各台 DPU，所以有 `shards` / `semantic_role`
                     / `kernel_tile_n`，pim mlir 是分块 GEMM
      算子级（B 路）  不做张量并行切分，`shards` 是空列表、没有投影身份，
                     pim mlir 是整算子级的相位链或单相 op

    两类共有的判据只有一条：`pimir_path` 必须在。缺了就说明这个算子的原语没进
    GeneSim，仿真会静默退回手写模板——正是本脚本要防的退化。
    """
    from genesim_bridge.op_classify import MNEMONIC_OF

    # GEMM 条目：图切分那一段的四项证据，逐项都要在。
    gemm_checks = {
        "shards（图切分归属，每台参与 DPU 各一项）": lambda e: bool(e.get("shards")),
        "shards[*].local_in/out_features（本地分片形状）": lambda e: all(
            s.get("local_in_features") and s.get("local_out_features")
            for s in e.get("shards", [])
        ),
        "semantic_role（投影身份）": lambda e: e.get("semantic_role") in _EXPECTED_ROLES,
        "kernel_tile_n（算子编译器实测分块）": lambda e: (e.get("kernel_tile_n") or 0) > 0,
    }
    # 算子级条目：op_type 要是 mnemonic 单表认得的名字，否则它在 GeneSim 侧
    # 没有原语落点，得报错而不是当普通条目放过。
    bpath_checks = {
        "op_type（mnemonic 单表认得的算子名）":
            lambda e: e.get("op_type") in MNEMONIC_OF,
    }
    common_checks = {
        "pimir_path（pim mlir 落盘路径）": lambda e: bool(e.get("pimir_path")),
    }

    # 有 shards 列表且非空 = 走过图切分的 GEMM；空列表 = 算子级节点。
    gemm_ops = {k: e for k, e in ops.items() if e.get("shards")}
    bpath_ops = {k: e for k, e in ops.items() if not e.get("shards")}

    for label, entries in (("GEMM", gemm_ops), ("算子级", bpath_ops)):
        checks = dict(common_checks)
        checks.update(gemm_checks if label == "GEMM" else bpath_checks)
        for field_label, predicate in checks.items():
            bad = [op_id for op_id, entry in entries.items()
                   if not predicate(entry)]
            if bad:
                raise StepFailed(
                    f"{len(bad)} 个{label}算子缺 {field_label}，如 op{bad[:5]}"
                )
    return len(gemm_ops), len(bpath_ops)


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

    # 本步骤用 --measure-kernel-tiles 编了算子，所以 sidecar 必须自报带产物。
    # GeneSim 靠这个字段自动进严格模式；缺了就退回"看配置文件写没写对"，也就是
    # 本脚本要防的那种静默退化。
    if not sidecar.get("requires_pimir"):
        raise StepFailed(
            "sidecar 没有 requires_pimir=true。带算子编译产物的导出必须自报，"
            "否则 GeneSim 不会自动进严格模式，pim mlir 读不到会静默退回手写模板。"
        )

    gemm_count, bpath_count = check_sidecar_entries(ops)
    print(f"    条目分两类核对：GEMM {gemm_count} 条、算子级 {bpath_count} 条")

    pimir_files = {entry["pimir_path"] for entry in ops.values()}
    absent = [p for p in pimir_files if not Path(p).is_file()]
    if absent:
        raise StepFailed(f"pimir_path 指向的文件不存在：{absent[:3]}")

    # 内容哈希要和磁盘上的文件对得上：sidecar 记的是绝对路径，如果缓存被重建过、
    # 而 sidecar 还是旧的，路径可能存在但内容已经变了。哈希不符说明这份 sidecar
    # 与当前的算子编译产物不是一套。
    #
    # pimir_sha256 是后加的字段（见 docs/genesimsupporTp-20260905.md 第五之二节）。
    # 本步骤自己产出 sidecar，所以正常情况下一定有；缺了说明是早于该字段导出的旧
    # 产物，此时提示一句并跳过这一项核对——报"缺 pimir_sha256"信息量太低，看起来
    # 像 bug 而不是"重新导出即可"。注意只跳过哈希核对，后面的分块打印和
    # dpu_to_cluster 核对照常执行。
    without_hash = [op_id for op_id, e in ops.items() if not e.get("pimir_sha256")]
    if without_hash:
        print(
            f"    注意: {len(without_hash)}/{len(ops)} 个算子的 sidecar 没有 "
            "pimir_sha256（早于该字段导出的旧产物），跳过内容一致性核对；"
            "重新跑一次本脚本即可补上。"
        )
    else:
        stale = stale_pimir_entries(ops)
        if stale:
            raise StepFailed(_stale_sidecar_message(stale))

    # 分块是 GEMM 才有的字段（B 路算子级节点不做分块搜索），所以只在 GEMM
    # 条目上取；对全部条目取会在第一条 B 路条目上抛 KeyError。
    tiles = sorted({int(e["kernel_tile_n"]) for e in ops.values()
                    if e.get("kernel_tile_n")})
    print(f"    放置 {gemm_count} 个 GEMM 与 {bpath_count} 个算子级节点，"
          f"本地形状 {len(pimir_files)} 种 pim mlir")
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


def _config_text(genesim: Path, config: str | Path) -> Optional[str]:
    """读仿真配置的文本。没有这份配置就返回 None。

    传文件名时从 genesim 的 conf 目录找；传路径时直接读，
    临时收过请求数的配置不在那个目录里。
    """
    path = config if isinstance(config, Path) else genesim / "conf" / config
    if not path.is_file():
        return None
    return path.read_text()


def _sidecar_of(genesim: Path, config_text: str) -> Optional[Path]:
    """从配置里取出 `compiler_placement_file`，没有就返回 None。"""
    import yaml

    config = yaml.safe_load(config_text) or {}
    relative = (config.get("scheduler") or {}).get("compiler_placement_file")
    if not relative:
        return None
    path = Path(relative)
    return path if path.is_absolute() else genesim / path


def limited_config(source: Path, conf_dir: Path, max_requests: int) -> Path:
    """复制一份配置，把请求数收成 max_requests，原文件不动。

    仿真时间跟着 token 数走。10 个请求共 8084 个 token 要二十多分钟，
    而第 1 个请求已经同时有 prefill 和 decode。原配置留着，
    需要完整数字时显式把请求数加回去。
    """
    import yaml

    loaded = yaml.safe_load(source.read_text())
    loaded.setdefault("trace", {})["num_requests"] = max_requests
    conf_dir.mkdir(parents=True, exist_ok=True)
    out = conf_dir / source.name
    out.write_text(yaml.safe_dump(loaded, sort_keys=False))
    return out


def _genesim_sim_command(genesim: Path, config_path: Path) -> list[str]:
    """返回跑 GeneSim 仿真的命令。

    首选 `./run.sh`——那是 GeneSim 上游设计的入口。但它每个子命令都先 `check_uv` +
    `check_venv`，机器上没装 uv、或者没跑过 `install.sh` 建 `.venv` 时会直接退出。

    这种情况下退回"用当前 python 直接跑 src/main.py"：仿真读的是已经生成好的 pim
    mlir，不自己编译算子，所以不需要 GeneSim 那套独立 venv。实测两条路径的
    `total_time_s` 小数位完全一致（1418.1092459087413），uv 只是包管理工具。

    这条退路的用途是让纯 CPU 容器里的验证能跑完；交付给用户仍应走 `install.sh` +
    `run.sh`，因为 `run.sh` 的其余子命令（predictor、upmem_checker）没有这条退路。
    """
    uv_available = shutil.which("uv") is not None
    venv_ready = (genesim / ".venv" / "bin" / "python").is_file()
    if uv_available and venv_ready:
        return ["./run.sh", "--config", str(config_path)]

    missing = "uv" if not uv_available else ".venv"
    print(f"    （缺 {missing}，改用当前 python 直接跑 src/main.py；"
          "结果与 run.sh 一致，见 _genesim_sim_command 的说明）")
    return [sys.executable, "src/main.py", "--config", str(config_path)]


def step_e_simulate(
    genesim: Path, config_name: str, log_dir: Path, *, label: str,
    results_dir: Optional[Path] = None, max_requests: Optional[int] = None,
) -> Dict[str, Any]:
    """(E) 跑仿真，核对 GEMM 的 trace 真的来自 pim mlir。"""
    print(f"\n[E] GeneSim 仿真（{label}）")
    config_path = genesim / "conf" / config_name
    if max_requests is not None:
        # 临时配置放日志目录，不写进 genesim 仓，免得留下一份请求数被改过的副本。
        config_path = limited_config(config_path, log_dir, max_requests)
        print(f"    请求数收成 {max_requests}：{config_path.name}")
    # sidecar 与磁盘产物不是一套时，仿真跑出来的数字没有意义，先拦下来，
    # 省掉十几分钟的无效仿真。
    verify_sidecar_freshness(genesim, config_path)
    # 仿真 IR 与 sidecar 不是同一套算子编号时，注意力算子会整批退回手写
    # 模板，同样在启动前拦下。
    verify_sim_ir_matches_sidecar(genesim, config_path)
    # trace 缓存按签名判新旧，但这里要的是"这一轮确实重编过"，所以先清掉。
    traces = genesim / "pim_traces"
    if traces.is_dir():
        shutil.rmtree(traces)
    _run(_genesim_sim_command(genesim, config_path),
         cwd=genesim, log_path=log_dir / f"e_sim_{label}.log")

    # 来源校验挂在仿真入口上，而不是只挂在主流程里。慢速仿真测试也走这里，
    # 整批 trace 退回手写模板时必须在这里失败，不能只在主流程里才看得见。
    verify_trace_provenance(genesim)

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


def _trace_metadata(path: Path) -> Dict[str, Any]:
    """读一条 pim trace 的元数据。

    头部是 `PIMT` + version(4) + 指令条数(8) + 元数据长度(4)，与
    genesim `pim_isa.save_trace_file` 同一布局。头部是 `PIMT` 时按长度字段
    切 JSON，长度超出文件就是头部损坏，直接报错。头部不是 `PIMT` 的
    （测试里的简化头）退回按字节扫描。
    """
    raw = path.read_bytes()
    if len(raw) >= 20 and raw[:4] == b"PIMT":
        meta_len = struct.unpack_from("<I", raw, 16)[0]
        # 头部是 PIMT 就按长度字段切。长度超出文件说明头部损坏，
        # 不能退回扫描——文件里碰巧有 `{` 会让损坏被静默放过。
        if 20 + meta_len > len(raw):
            raise StepFailed(
                f"trace 头部损坏：{path.name} 声明元数据 {meta_len} 字节，"
                f"文件只有 {len(raw)} 字节")
        try:
            return json.loads(raw[20:20 + meta_len].decode("utf-8"))
        except json.JSONDecodeError:
            raise StepFailed(f"trace 文件损坏，读不出元数据：{path.name}")
    # 头部不是 PIMT（测试里的简化头）时退回扫描。
    start = raw.find(b"{")
    if start < 0:
        raise StepFailed(f"trace 文件损坏，读不出元数据：{path.name}")
    try:
        meta, _ = json.JSONDecoder().raw_decode(
            raw[start:].decode("utf-8", "ignore"))
    except json.JSONDecodeError:
        raise StepFailed(f"trace 文件损坏，读不出元数据：{path.name}")
    return meta


def stale_pimir_entries(ops: Dict[str, Dict[str, Any]]) -> list[str]:
    """返回 `pimir_sha256` 与磁盘文件对不上的算子 id。

    sidecar 记的是绝对路径：缓存重建后路径还在、内容已经变了，哈希是唯一
    能发现"这份 sidecar 与当前算子编译产物不是一套"的办法。
    """
    stale = []
    for op_id, entry in ops.items():
        digest = hashlib.sha256(
            Path(entry["pimir_path"]).read_bytes()
        ).hexdigest()
        if digest != entry["pimir_sha256"]:
            stale.append(op_id)
    return stale


def _stale_sidecar_message(stale: list[str]) -> str:
    return (
        f"{len(stale)} 个算子的 pimir_sha256 与磁盘上的 pim mlir 不一致"
        f"（如 op{stale[:5]}）：sidecar 与当前算子编译产物不是一套，"
        "请重新导出（scripts/run_full_pipeline.py 不带 --skip-simulation，"
        "或单独跑 scripts/export_pp_placement.py）。"
    )


def verify_sidecar_freshness(genesim: Path, config: str | Path) -> None:
    """仿真用的 sidecar 必须与磁盘上的 pim mlir 是一套。

    主流程的哈希核对只在导出步骤里，仿真入口绕得开它：缓存被重建过、
    sidecar 还是旧的时，仿真照样跑完，代价链吃的却不是当前的产物。
    配置里没有 `compiler_placement_file` 时没有 sidecar 可核对，直接返回。
    """
    config_text = _config_text(genesim, config)
    if config_text is None:
        return
    sidecar_path = _sidecar_of(genesim, config_text)
    if sidecar_path is None:
        return
    if not sidecar_path.is_file():
        raise StepFailed(f"配置指向的 sidecar 不存在：{sidecar_path}")
    ops = json.loads(sidecar_path.read_text())["operators"]
    without_hash = [op_id for op_id, e in ops.items() if not e.get("pimir_sha256")]
    if without_hash:
        raise StepFailed(
            f"{len(without_hash)}/{len(ops)} 个算子的 sidecar 没有 pimir_sha256，"
            "无法确认它与当前算子编译产物是一套，请重新导出。"
        )
    stale = stale_pimir_entries(ops)
    if stale:
        raise StepFailed(_stale_sidecar_message(stale))


def verify_sim_ir_matches_sidecar(genesim: Path, config: str | Path) -> None:
    """仿真加载的 IR 与 sidecar 必须是同一套算子编号。

    调度器按 op_id 查 sidecar 里的 pimir_path：精化 IR 是旧结构、sidecar 是
    新导出时，查不到的算子静默退回手写模板，仿真照样跑完（2026-10-07 的
    6144 个模板 trace 就是这么来的）。模型入口/出口没有设备算子，本来就不
    在 sidecar 里，其余算子缺一个都拦在仿真启动前，省十几分钟无效仿真。
    配置里没有 ir_path 或 sidecar 时没有可对拍的对象，直接返回。
    """
    config_text = _config_text(genesim, config)
    if config_text is None:
        return
    import yaml
    loaded = yaml.safe_load(config_text) or {}
    ir_rel = (loaded.get("model") or {}).get("ir_path")
    sidecar_path = _sidecar_of(genesim, config_text)
    if not ir_rel or sidecar_path is None:
        return
    ir_path = genesim / ir_rel
    if not ir_path.is_file():
        raise StepFailed(
            f"配置指向的仿真 IR 不存在：{ir_path}。先跑 (A2) 精化步骤生成"
            "（scripts/run_full_pipeline.py 主流程已包含这一步）。")
    if not sidecar_path.is_file():
        raise StepFailed(f"配置指向的 sidecar 不存在：{sidecar_path}")
    sidecar_ops = json.loads(sidecar_path.read_text())["operators"]
    ir = json.loads(ir_path.read_text())
    missing = [
        (int(op["op_id"]), str(op["op_type"])) for op in ir["operators"]
        if op["op_type"] not in ("MODEL_INPUT", "MODEL_OUTPUT")
        and not (sidecar_ops.get(str(op["op_id"])) or {}).get("pimir_path")
    ]
    if missing:
        kinds = sorted({t for _, t in missing})
        raise StepFailed(
            f"{len(missing)} 个算子在 sidecar 里没有 pimir_path（类型 {kinds}，"
            f"如 op{[i for i, _ in missing[:5]]}）：仿真 IR（{ir_path.name}）与 "
            f"sidecar（{sidecar_path.name}）不是同一套算子编号。"
            "精化 IR 多半是图骨架改结构之前生成的旧产物——重跑 "
            "scripts/run_full_pipeline.py（(A2) 会重新精化），或手动跑 "
            "genesim/scripts/refine_ir_with_flagtree.py。")


def verify_trace_provenance(genesim: Path) -> None:
    """核对每个算子的 trace 来源，避免"跑通了但其实走的是手写模板"。

    原先只扫 GEMM。注意力那一组（ROPE、MASK、SOFTMAX、GEMV）退回模板时
    这里看不见，仿真照样算通过，而那些算子的分块根本没进代价链。
    """
    sources: Dict[str, int] = {}
    by_op: Dict[str, Dict[str, int]] = {}
    tiles: Dict[Any, int] = {}
    for path in sorted((genesim / "pim_traces").glob("op_*.pim_trace")):
        meta = _trace_metadata(path)
        source = (meta.get("compile_signature") or {}).get("trace_source", "unknown")
        op_type = str(meta.get("op_type", "unknown"))
        sources[source] = sources.get(source, 0) + 1
        counts = by_op.setdefault(op_type, {})
        counts[source] = counts.get(source, 0) + 1
        pimir_meta = meta.get("pimir") or {}
        if pimir_meta:
            key = (pimir_meta.get("tile_n"), pimir_meta.get("k_iterations"))
            tiles[key] = tiles.get(key, 0) + 1

    if not sources:
        raise StepFailed("没有找到任何 trace，无法判断来源")
    print(f"    trace 来源: {sources}")
    if tiles:
        print(f"    (tile_n, k_iterations) 分布: {tiles}")
    if sources.get("template"):
        templated = sorted(op for op, counts in by_op.items() if counts.get("template"))
        raise StepFailed(
            f"{sources['template']} 个算子退回了手写模板（{', '.join(templated)}）；"
            "算子编译器的分块没有进入代价链。"
        )



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--num-stages", type=int, default=4,
                        help="流水段数：1=纯张量并行，num_dpus=纯流水")
    parser.add_argument("--num-dpus", type=int, default=8)
    parser.add_argument("--max-requests", type=int, default=None,
                        help="仿真的请求数上限。默认跟随配置；"
                             "慢速测试传 1，只要 prefill 与 decode 各一次")
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
        step_a2_refine_pimir_ir(genesim, ir_path, log_dir)
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
            max_requests=args.max_requests,
        )
        # 来源校验在 step_e_simulate 里已经做过，这里不再重扫一遍 trace。

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
