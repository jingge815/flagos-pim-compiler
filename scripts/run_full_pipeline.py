#!/usr/bin/env python3
"""跨三个仓库跑通模型（默认 Llama2-7B，支持 Qwen3-MoE）的完整链路。

模型类型、加载器、切分宽度、必需投影身份、产物前缀与仿真配置都从
`backend/architectures.py` 的注册表派生（`--model-dir` 指定模型目录）；
脚本里不再硬编模型名。MoE 的 router/专家身份、专家号与门控语义随 IR、
sidecar 与仿真配置逐段核对。

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

    # Qwen3-MoE：模型目录决定架构、产物名与仿真配置
    python scripts/run_full_pipeline.py --model-dir <模型目录> --num-stages 2

    # 只想快速验证接线、不跑完整仿真：
    python scripts/run_full_pipeline.py --num-stages 4 --skip-simulation

    # A/B 对照：同一份 sidecar，关掉本地分片形状再跑一次
    python scripts/run_full_pipeline.py --num-stages 4 --ab-compare

CEVA 后端（我方 GML/bin 走厂商 GC 与 CEVA 仿真）：

    bash scripts/run_ceva_pipeline.sh --run-name ours_e2e --decode-steps 10

    启动脚本自动加载 paths.json 中的编译环境。主流程导出 65 张子图并强制走
    FlagTree → 来源清单校验 → 工作区 run_all.sh --gml-bundle（GC、DDR、仿真、判定）。
    我方 bin 默认用 --token-check warn；需要逐 token 一致时显式传 strict。
    整轮约 10 小时，建议 nohup 后台运行。

每一步跑完都会核对产物，不满足就直接失败——这个脚本的用途是回答「链路通没通」，
所以任何一段静默退化都必须变成非零退出码。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import struct
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.architectures import Architecture, architecture_of
from genesim_bridge.cost_extractor import SIDECAR_VERSION
from runtime.exec_plan_gen import MOE_EXECUTION_MODES
from genesim_bridge.ir_fingerprint import (
    assert_ir_fingerprint, fingerprint_ir, fusion_signature)
from genesim_bridge.paths import (ceva_reference_dir, ceva_workspace, genesim_models_dir,
                                  genesim_root, llama2_7b_model_dir)

_REPO_ROOT = Path(__file__).resolve().parent.parent

# 精化的序列长度口径：与 docs/llama-2.md 第六步、现有产物、仿真 trace 一致。
_REFINE_SEQ_LEN = "128"


def _refine_ir_level() -> str:
    """没有 GPU 时精化走 B 路。

    pimir 级要真实执行 FlagGems 算子来测量成本，纯 CPU 机器上做不到，会在
    捕获 kernel 时直接报错。oplevel 是整算子 IR，只编译不执行，纯 CPU 可跑。
    """
    import torch

    if torch.cuda.is_available():
        return "pimir"
    print("    未检测到 GPU，精化改用 oplevel（只编译不执行）")
    return "oplevel"


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


def _resolve_moe_execution(model_dir: Path, requested: str | None) -> str:
    """与编译入口同一份规则：非 MoE 为空串，MoE 缺省跟随 PIM_MOE_DISPATCH。"""
    import types

    from runtime.exec_plan_gen import resolve_moe_execution

    config = json.loads((Path(model_dir) / "config.json").read_text())
    # 传完整配置：MoE 身份由 model_type 与有效字段共同判定，只给专家数
    # 会让 model_type 丢失，单专家/非 qwen3_moe 的判定随之失真。
    cfg = types.SimpleNamespace(**config)
    return resolve_moe_execution(cfg, requested)


def _configured_moe_identity(config: Dict[str, Any]) -> Optional[Dict[str, int]]:
    """从 HF config 派生 MoE 身份；非 MoE 模型返回 None。

    执行语义由 `_resolve_moe_execution` 单独解析，这里只取进入 IR/sidecar
    的三项身份字段。用 `is_moe_config` 判定而不是专家数阈值，1 专家/top-1
    同样是 MoE。
    """
    import types

    from runtime.exec_plan_gen import is_moe_config

    if not is_moe_config(types.SimpleNamespace(**config)):
        return None
    return {
        "num_experts": int(config.get("num_experts", 0) or 0),
        "top_k": int(config.get("num_experts_per_tok", 0) or 0),
        "intermediate_size": int(config.get("moe_intermediate_size", 0) or 0),
    }


def _validate_moe_identity(
    declared: Dict[str, Any], config: Dict[str, Any], *, stage: str,
) -> None:
    """IR/sidecar 的 MoE 身份必须与 HF config 逐字段一致。

    只比较 IR 与 sidecar 自身的一致性不够：top-k 不进入权重字节，写错的 IR
    可以一路通过 A/BCD/E，只表现为动态分派比例与 MOE_ROUTE 成本失真。期望值
    必须从源模型配置派生，新增 MoE 标量字段也走同一份期望元组。
    """
    expected = _configured_moe_identity(config)
    actual = {
        "num_experts": int(declared.get("num_experts", 0) or 0),
        "top_k": int(declared.get("top_k", 0) or 0),
        "intermediate_size": int(declared.get("intermediate_size", 0) or 0),
    }
    if expected is None:
        if any(actual.values()):
            raise StepFailed(
                f"{stage} 的 MoE 身份 {actual} 非零，但 HF config 不是 MoE 模型")
        return
    mismatch = {
        key: (actual[key], want)
        for key, want in expected.items()
        if actual[key] != want
    }
    if mismatch:
        raise StepFailed(
            f"{stage} 的 MoE 身份与 HF config 不一致（实际, 期望）：{mismatch}。"
            "top-k 与专家宽度不进入权重字节总量，只靠参数量校验发现不了。")


def step_a_model_ir(
    genesim: Path, model_dir: Path, log_dir: Path,
    model_type: str, arch: Architecture, moe_execution: str = "",
    out_dir: Path | None = None,
) -> Path:
    """(A) 从 HF config 生成 GeneSim 图骨架，核对模型身份与投影身份。"""
    print(f"\n[A] HuggingFace config → GeneSim 图骨架 IR（model_type={model_type}）")
    models_dir = Path(out_dir) if out_dir is not None else genesim / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    ir_path = models_dir / f"{arch.artifact}.ir"
    # 图编译器统一以 fp16 加载执行；检查点声明 bf16 时，若不告诉 parser 执行
    # dtype，图骨架会按 bf16 生成四节点注意力，而执行侧走融合 PAGED_ATTN，
    # 两段的注意力结构对不上。
    parser_cmd = ["python", "scripts/model_parser.py",
                  "--model_name", str(model_dir), "--output", str(ir_path),
                  "--execution_dtype", "float16"]
    if moe_execution:
        # 执行语义由主链统一选定，parser 不再自己读环境变量猜。
        parser_cmd += ["--moe-execution", moe_execution]
    _run(parser_cmd, cwd=genesim, log_path=log_dir / "a_model_parser.log")

    ir = json.loads(ir_path.read_text())
    # 图骨架的执行语义必须就是主链选定的那一份：两种语义的专家计费差 16 倍，
    # 靠 sidecar 自报或事后比对都会让链路分叉。
    if str(ir.get("moe_execution", "")) != moe_execution:
        raise StepFailed(
            f"图骨架的 moe_execution={ir.get('moe_execution')!r} 与主链选定的 "
            f"{moe_execution!r} 不一致，model_parser 没有按编译配置生成 IR。")
    gemms = [op for op in ir["operators"] if op["op_type"] == "GEMM"]
    roles = {op.get("semantic_role", "") for op in gemms}
    # MoE 身份必须与源模型一致：128 专家不能被写成 dense 的七个投影。
    config = json.loads((Path(model_dir) / "config.json").read_text())
    tied = bool(config.get("tie_word_embeddings", False))
    required_roles = set(arch.required_roles)
    if tied:
        # tied 模型的输出头与词嵌入共享权重，图里仍有一个 GEMM，但权重名
        # 不独立；这里只放宽角色集合，其余完整性检查照旧。
        required_roles.discard("lm_head")
    missing = required_roles - roles
    if missing:
        raise StepFailed(
            f"图骨架里缺这些投影的 semantic_role: {sorted(missing)}。"
            "放置导出按语义标签匹配，缺了就无法确定 GEMM 身份。"
        )
    unlabeled = [op["op_id"] for op in gemms if not op.get("semantic_role")]
    if unlabeled:
        raise StepFailed(f"{len(unlabeled)} 个 GEMM 没有 semantic_role，如 {unlabeled[:5]}")
    # 输出头必须真的通向词表维：形状写错时 role 齐了也照样漏算词表投影。
    vocab_size = int(config.get("vocab_size", 0))
    lm_heads = [op for op in gemms if op.get("semantic_role") == "lm_head"]
    if len(lm_heads) != 1:
        raise StepFailed(
            f"图骨架里 lm_head GEMM 有 {len(lm_heads)} 个，应恰好 1 个（整图一份）")
    if int(lm_heads[0]["output_shapes"][0][-1]) != vocab_size:
        raise StepFailed(
            f"lm_head 输出宽度 {lm_heads[0]['output_shapes'][0][-1]} 与 config "
            f"的 vocab_size {vocab_size} 不一致")
    # 图内权重必须覆盖完整执行图：缺 lm_head 或 q/k norm 时，parameter_bytes
    # 会比完整 HF 参数少一截，这条判据与 parser 的生成期检查互为闭环。
    dtype_bytes = {"float16": 2, "bfloat16": 2, "float32": 4}.get(
        str(ir.get("execution_dtype", "")))
    if dtype_bytes is None:
        raise StepFailed(
            f"图骨架的 execution_dtype={ir.get('execution_dtype')!r} 无法换算"
            "权重字节，只认 float16/bfloat16/float32")
    expected_bytes = int(ir.get("num_parameters", 0)) * dtype_bytes
    if int(ir.get("parameter_bytes", 0)) != expected_bytes:
        raise StepFailed(
            f"图骨架 parameter_bytes={ir.get('parameter_bytes')} 与 "
            f"num_parameters={ir.get('num_parameters')}×{dtype_bytes}B="
            f"{expected_bytes} 不一致：图里缺算子或权重。")
    # 完整身份从 config 派生后逐项比较：top_k 与专家宽都不进权重字节，
    # 不能只靠参数量与专家 GEMM 数间接覆盖。非 MoE 配置必须全零，否则
    # parser 把稠密模型写成 MoE 标量也会在这里失败。
    _validate_moe_identity({
        "num_experts": ir.get("moe_num_experts", 0),
        "top_k": ir.get("moe_top_k", 0),
        "intermediate_size": ir.get("moe_intermediate_size", 0),
    }, config, stage="图骨架")
    if int(ir.get("moe_num_experts", 0)) > 0:
        expected_experts = int(config.get("num_experts", 0))
        if int(ir["moe_num_experts"]) != expected_experts:
            raise StepFailed(
                f"IR 的 moe_num_experts={ir['moe_num_experts']} 与 config 的 "
                f"{expected_experts} 不一致")
        expert_gemms = [op for op in gemms if int(op.get("expert_id", -1)) >= 0]
        if len(expert_gemms) != (
                int(ir["num_layers"]) * expected_experts * 3):
            raise StepFailed(
                f"专家 GEMM 数 {len(expert_gemms)} != 层数×专家数×3 "
                f"({int(ir['num_layers'])}×{expected_experts}×3)")
        print(f"    MoE 几何: {ir['moe_num_experts']} 专家 top-{ir['moe_top_k']}，"
              f"专家宽 {ir['moe_intermediate_size']}，"
              f"专家 GEMM {len(expert_gemms)} 个")
    print(f"    算子 {len(ir['operators'])} 个，GEMM {len(gemms)} 个，"
          f"投影身份 {sorted(roles)}")
    return ir_path


def _ir_op_ids(ir_path: Path) -> list[int]:
    """读出 IR 里按出现顺序的算子编号序列。"""
    ir = json.loads(ir_path.read_text())
    return [int(op["op_id"]) for op in ir["operators"]]


def _placed_ir_of(sidecar_path: Path | str) -> Path | None:
    """按 sidecar 命名规则找同一轮导出的放置 IR；命名不匹配时返回 None。"""
    name = Path(sidecar_path).name
    if not name.endswith("_placement.json"):
        return None
    return Path(sidecar_path).with_name(
        name.replace("_placement.json", "_placed.ir"))


def _ir_device_hints(ir_path: Path) -> dict[int, str]:
    """读出每个算子的设备落点，供精化缓存判断落点证据是否需要刷新。"""
    ir = json.loads(Path(ir_path).read_text())
    return {
        int(op["op_id"]): str(op.get("device_hint", ""))
        for op in ir["operators"]
    }


def step_a2_refine_pimir_ir(
    genesim: Path, ir_path: Path, log_dir: Path, artifact: str,
    placement: Path | None = None, out_dir: Path | None = None,
) -> Path:
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
    models_dir = Path(out_dir) if out_dir is not None else genesim / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    pimir_ir = models_dir / f"{artifact}_pimir.ir"
    skeleton = json.loads(Path(ir_path).read_text())
    # 指纹由内容重算：骨架被手改过（词表/形状变了而指纹没更新）时在这里失败，
    # 不能拿旧精化产物继续跑。
    try:
        skeleton_fingerprint = assert_ir_fingerprint(skeleton, str(ir_path))
    except ValueError as exc:
        raise StepFailed(str(exc)) from exc
    # 精化输入优先用放置后的 IR：它带着 export_gemm 按 spec.device 写下的
    # 落点证据，GeneSim 的豁免校验才能与真实执行计划对上。放置不改结构，
    # 指纹必须与图骨架一致。
    refine_source = Path(ir_path)
    if placement is not None and Path(placement).is_file():
        placed = _placed_ir_of(placement)
        if placed is not None and placed.is_file():
            try:
                placed_fingerprint = assert_ir_fingerprint(
                    json.loads(placed.read_text()), str(placed))
            except ValueError as exc:
                raise StepFailed(str(exc)) from exc
            if placed_fingerprint != skeleton_fingerprint:
                raise StepFailed(
                    f"放置 IR（{placed.name}）的结构指纹与图骨架不一致："
                    "放置只允许改设备/成本字段，不能改变结构。")
            refine_source = placed
    fusion_entries: list = []
    if placement is not None and Path(placement).is_file():
        placement_data = json.loads(Path(placement).read_text())
        fusion_entries = placement_data.get("fused_activations") or []
    fusion_requested = bool(fusion_entries)
    expected_fusion = fusion_signature(fusion_entries)
    # sidecar 按实际档位命名：无 GPU 时走 oplevel，不能覆盖 pimir 那份。
    ir_level = _refine_ir_level()
    sidecar = models_dir / f"{artifact}_{ir_level}_extensions.json"
    # 缓存键 = 结构指纹 + 精化档位 + 融合清单 + 成本算法版本。只看编号序列
    # 会让「同算子序列、不同词表」的骨架错误复用旧精化 IR。
    sidecar_ok = False
    if sidecar.is_file():
        extension = json.loads(sidecar.read_text())
        sidecar_ok = (
            extension.get("ir_level") == ir_level
            and extension.get("ir_fingerprint") == skeleton_fingerprint
            and extension.get("fusion_signature") == expected_fusion
            and extension.get("version") == SIDECAR_VERSION
        )
    if (pimir_ir.is_file() and _ir_op_ids(pimir_ir) == _ir_op_ids(ir_path)
            and _ir_device_hints(pimir_ir) == _ir_device_hints(refine_source)
            and not fusion_requested and sidecar_ok):
        # 精化 IR 允许已存在，但结构必须与骨架一致；只是成本不同不算差异。
        try:
            refined_matches = (
                assert_ir_fingerprint(
                    json.loads(pimir_ir.read_text()), str(pimir_ir))
                == skeleton_fingerprint)
        except ValueError:
            refined_matches = False
        if refined_matches:
            print(f"    精化 IR 与图骨架同套编号（{pimir_ir.name}），"
                  f"{ir_level} sidecar 与指纹/融合清单/成本版本一致，跳过重新精化")
            return pimir_ir
    cmd = [
        "python", "scripts/refine_ir_with_flagtree.py",
        "--ir", str(refine_source), "--out-ir", str(pimir_ir),
        "--sidecar", str(sidecar),
        "--seq-len", _REFINE_SEQ_LEN, "--ir-level", ir_level,
    ]
    # 放置 sidecar 里带图编译器的融合清单。没有它就按未融合算，会把已被吸收的
    # 激活再计一遍——所以必须在放置导出之后跑，并且把这份 sidecar 传进来。
    if placement is not None and Path(placement).is_file():
        # 只取融合清单，不动形状口径：这份精化 IR 按模型级全局形状测，
        # 而融合与否是图编译器的决策，两种口径都要带上。
        cmd += ["--fusion-from", str(placement)]
    _run(cmd, cwd=genesim, log_path=log_dir / "a2_refine_pimir.log")
    if _ir_op_ids(pimir_ir) != _ir_op_ids(ir_path):
        raise StepFailed(
            f"精化 IR 的算子编号与图骨架不一致：{pimir_ir} vs {ir_path}。"
            "refine 应该只回填成本、不改结构，请检查 refine_ir_with_flagtree.py。")
    refined = json.loads(pimir_ir.read_text())
    try:
        refined_fingerprint = assert_ir_fingerprint(refined, str(pimir_ir))
    except ValueError as exc:
        raise StepFailed(str(exc)) from exc
    if refined_fingerprint != skeleton_fingerprint:
        raise StepFailed(
            f"精化 IR 的结构指纹与图骨架不一致：{pimir_ir} vs {ir_path}。"
            "refine 只允许改成本，结构必须逐项相同。")
    print(f"    精化完成：{pimir_ir.name}"
          f"（{len(_ir_op_ids(pimir_ir))} 个算子，编号与结构指纹均与图骨架一致）")
    return pimir_ir


def step_zero_pu_mapping(
    genesim: Path, ir_path: Path, num_stages: int, num_dpus: int, log_dir: Path,
    artifact: str, out_dir: Path | None = None,
) -> Path:
    """(0) GeneSim 给出固定 PU 映射，核对逐段的 Cluster 归属合理。

    这一段方向与其余三段相反：不是"编译器算完告诉 GeneSim"，而是"GeneSim 定下
    PU 映射、约束编译器"。产物是 PartitionPlan，由编译器侧转成 ShardStrategy。
    """
    print("\n[0] GeneSim 固定 PU 映射 → PartitionPlan")
    tp_width = num_dpus // num_stages
    models_dir = Path(out_dir) if out_dir is not None else genesim / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    plan_path = models_dir / f"{artifact}_tp{tp_width}_pp{num_stages}_plan.json"
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


def _entry_is_host(
    op_id: int, ir_op: Dict[str, Any], entry: Dict[str, Any],
    sidecar: Optional[Dict[str, Any]],
) -> bool:
    """条目按 host 跳过 A 路检查的唯一证据链。

    `device_hint` 是被验条目自报的开关，不能单独证明落点：源 IR 的
    `device_hint=cpu`、`pimir_exempt_ops` 的豁免登记、以及 GEMM 的
    `host_gemms` 角色/权重证据三者同时成立才算 host。缺任何一项时条目按
    设备算子处理，必须给出分片、投影身份与算子编译分块。
    """
    if sidecar is None:
        return False
    if str(ir_op.get("device_hint", "")).lower() != "cpu":
        return False
    if str(entry.get("device_hint", "")).lower() != "cpu":
        return False
    exemption = (sidecar.get("pimir_exempt_ops") or {}).get(str(op_id))
    if not isinstance(exemption, dict):
        return False
    if str(exemption.get("device_hint", "")).lower() != "cpu":
        return False
    if ir_op.get("op_type") != "GEMM":
        return True
    host_evidence = (sidecar.get("host_gemms") or {}).get(str(op_id))
    return _host_gemm_evidence_matches(ir_op, exemption, host_evidence)


def _entry_class_problems(
    ir_ops: Dict[int, Dict[str, Any]],
    ops: Dict[str, Dict[str, Any]],
    *,
    strict: bool = True,
    sidecar: Optional[Dict[str, Any]] = None,
) -> List[str]:
    """条目类别与字段必须与源 IR 一致：GEMM 走 A 路，其余算子都走 B 路。

    `shards` 与 `device_hint` 都是被验产物自报的字段，不能反过来决定类别或
    免掉 A 路检查：host 由源 IR 落点、豁免登记与 `host_gemms` 证据共同证明。
    与源 IR 矛盾时直接失败——B 路不得带分片；严格产物里的设备 GEMM 必须有
    分片；条目的 `op_type` 必须等于源 IR 的算子类型，设备 GEMM 的
    `semantic_role` 必须是源 IR 的投影身份。
    非严格旧式 sidecar 允许只写 `dpu_id` 的单 DPU GEMM 条目，这是旧格式的
    放置表达，不改变 A/B 路分类。
    """
    problems: List[str] = []
    for raw_op_id, entry in ops.items():
        op_id = int(raw_op_id)
        ir_op = ir_ops.get(op_id)
        if ir_op is None:
            continue  # 越界编号由覆盖审计的 extra 检查负责。
        source_type = str(ir_op.get("op_type", ""))
        declared_type = str(entry.get("op_type") or "")
        if declared_type and declared_type != source_type:
            problems.append(
                f"op{op_id} 的 op_type={declared_type!r} 与源 IR "
                f"{source_type!r} 不一致")
        elif strict and not declared_type:
            problems.append(f"op{op_id} 缺 op_type：条目类型必须与源 IR 一致")
        claims_host = str(entry.get("device_hint", "")).lower() == "cpu"
        host = claims_host and _entry_is_host(op_id, ir_op, entry, sidecar)
        if claims_host and not host:
            problems.append(
                f"op{op_id}（{source_type}）自称 host，但缺源 IR 的 "
                "device_hint=cpu、pimir_exempt_ops 登记或 host_gemms 证据")
        shards = entry.get("shards") or []
        if source_type == "GEMM":
            role = str(entry.get("semantic_role") or "")
            source_role = str(ir_op.get("semantic_role") or "")
            if strict and source_role and not role:
                problems.append(
                    f"op{op_id} 是设备 GEMM 却缺 semantic_role 投影身份")
            elif role != source_role:
                problems.append(
                    f"op{op_id} 的 semantic_role={role!r} 与源 IR "
                    f"{source_role!r} 不一致")
            if strict and not shards and not host:
                problems.append(f"op{op_id} 是设备 GEMM 却没有分片")
        elif shards:
            problems.append(
                f"op{op_id}（{source_type}）是非 GEMM 却带分片")
    return problems


def check_sidecar_entries(
    ir_ops: Dict[int, Dict[str, Any]],
    ops: Dict[str, Dict[str, Any]],
    required_roles: set[str] | frozenset[str],
    *,
    sidecar: Optional[Dict[str, Any]] = None,
) -> tuple[int, int]:
    """按条目类别分流核对 sidecar，返回 (GEMM 条数, 算子级条数)。

    sidecar 里两类条目的契约不同，用一套谓词查会把好的判成坏的：

      GEMM（A 路）   按图切分归属到各台 DPU，所以有 `shards` / `semantic_role`
                     / `kernel_tile_n`，pim mlir 是分块 GEMM
      算子级（B 路）  不做张量并行切分，`shards` 是空列表、没有投影身份，
                     pim mlir 是整算子级的相位链或单相 op

    类别只由**源 IR**决定，`shards` 是被验产物自报的字段，不能反过来定义类别；
    与 IR 矛盾的条目直接失败。两类共有的判据只有一条：`pimir_path` 必须在。
    缺了就说明这个算子的原语没进 GeneSim，仿真会静默退回手写模板——正是本脚本
    要防的退化。
    """
    from genesim_bridge.op_classify import MNEMONIC_OF

    class_problems = _entry_class_problems(ir_ops, ops, sidecar=sidecar)
    if class_problems:
        raise StepFailed(
            f"{len(class_problems)} 条 sidecar 条目的类别或字段与源 IR 矛盾"
            f"（如 {class_problems[0]}）：GEMM 走 A 路，其余算子都是 B 路，"
            "不能用 shards/device_hint 自报改写类别或跳过 host 证据")
    # GEMM 条目：图切分那一段的四项证据，逐项都要在。host GEMM 没有设备
    # 分片，由源 IR 落点、豁免登记与 `host_gemms` 证据共同核对。
    gemm_checks = {
        "shards（图切分归属，每台参与 DPU 各一项）": lambda e: bool(e.get("shards")),
        "shards[*].local_in/out_features（本地分片形状）": lambda e: all(
            s.get("local_in_features") and s.get("local_out_features")
            for s in e.get("shards", [])
        ),
        "semantic_role（投影身份）": lambda e: e.get("semantic_role") in required_roles,
        "kernel_tile_n（算子编译器实测分块）": lambda e: (e.get("kernel_tile_n") or 0) > 0,
    }
    # 算子级条目：op_type 要与源 IR 的类型一致（类别检查已钉住），并且是
    # mnemonic 单表认得的名字，否则它在 GeneSim 侧没有原语落点。
    bpath_checks = {
        "op_type（与源 IR 一致且在 mnemonic 单表里）":
            lambda e: e.get("op_type") in MNEMONIC_OF,
    }
    common_checks = {
        "pimir_path（pim mlir 落盘路径）": lambda e: bool(e.get("pimir_path")),
    }

    gemm_ops: Dict[str, Dict[str, Any]] = {}
    bpath_ops: Dict[str, Dict[str, Any]] = {}
    for raw_op_id, entry in ops.items():
        ir_op = ir_ops.get(int(raw_op_id))
        if ir_op is None:
            raise StepFailed(
                f"sidecar 里的 op{raw_op_id} 不在源 IR 中：编号已经错位，"
                "请用当前 IR 重新导出")
        target = gemm_ops if ir_op.get("op_type") == "GEMM" else bpath_ops
        target[raw_op_id] = entry

    for label, entries in (("GEMM", gemm_ops), ("算子级", bpath_ops)):
        checks = dict(common_checks)
        checks.update(gemm_checks if label == "GEMM" else bpath_checks)
        for field_label, predicate in checks.items():
            bad = [op_id for op_id, entry in entries.items()
                   if not (label == "GEMM" and _entry_is_host(
                       int(op_id), ir_ops[int(op_id)], entry, sidecar))
                   and not predicate(entry)]
            if bad:
                raise StepFailed(
                    f"{len(bad)} 个{label}算子缺 {field_label}，如 op{bad[:5]}"
                )
    return len(gemm_ops), len(bpath_ops)


def _placement_contradicts_exemption(placement: Any) -> bool:
    """豁免说 host，sidecar 的放置条目却给出设备执行证据。

    有分片、写了 `pimir_path` 或自称 `device_hint=pim` 都说明这个算子实际
    落在 DPU 上；两处证据矛盾时不能按 host 免掉 pimir。
    """
    if not isinstance(placement, dict) or not placement:
        return False
    if placement.get("shards") or placement.get("pimir_path"):
        return True
    return str(placement.get("device_hint", "")).lower() != "cpu"


def _host_gemm_evidence_matches(
    op: Dict[str, Any], entry: Dict[str, Any], host_evidence: Any,
) -> bool:
    """host GEMM 豁免必须与导出写下的 `host_gemms` 落点证据逐项一致。"""
    if int(op.get("expert_id", -1)) >= 0:
        return False
    if not isinstance(host_evidence, dict):
        return False
    if str(host_evidence.get("device_hint", "")).lower() != "cpu":
        return False
    role = str(op.get("semantic_role", ""))
    if str(host_evidence.get("semantic_role", "")) != role:
        return False
    weight = str(host_evidence.get("weight", ""))
    if not weight:
        return False
    declared = str(entry.get("weight", ""))
    return not declared or declared == weight


def _exemption_is_legitimate(
    op: Dict[str, Any], entry: Dict[str, Any], *,
    placement: Any = None, host_evidence: Any = None,
) -> bool:
    """豁免必须与源 IR 的真实落点一致；未知算子默认不放行。

    合法豁免只有模型入口/出口与 host 执行两类。host 证据由放置导出在
    `export_gemm` 里按 `spec.device` 写进 IR（`device_hint=cpu`）和 sidecar
    条目，不能用 `expert_id<0` 之类的身份推断——router、注意力投影与输出头
    同样满足那个条件，却明确落在 DPU 上。豁免还必须与 sidecar 的实际放置
    不矛盾：有 PIM 分片或 pimir 的算子不能同时按 host 免掉；host GEMM 的
    角色与权重身份要和 `host_gemms` 的导出证据对上。
    """
    declared = str(entry.get("op_type", ""))
    if declared and declared != str(op.get("op_type", "")):
        return False
    if _placement_contradicts_exemption(placement):
        return False
    if op.get("op_type") in ("MODEL_INPUT", "MODEL_OUTPUT"):
        return True
    if str(op.get("device_hint", "")).lower() != "cpu":
        return False
    if str(entry.get("device_hint", "")).lower() != "cpu":
        return False
    if op.get("op_type") == "GEMM":
        role = str(op.get("semantic_role", ""))
        if str(entry.get("semantic_role", "")) != role:
            return False
        return _host_gemm_evidence_matches(op, entry, host_evidence)
    return True


# 可被折进主算子的激活类型。与 `contracts/fusion_contract.py` 的两张融合表
# 一致：执行路径只折 linear 尾部激活（silu/relu/sigmoid）与门控激活（gelu）。
_ABSORBABLE_ACTIVATIONS = frozenset({"SILU", "GELU", "RELU", "SIGMOID"})


def _validate_sidecar_identity(
    ir: Dict[str, Any], sidecar: Dict[str, Any],
) -> None:
    """MoE 元数据与结构指纹必须与源 IR 一致。

    MoE 的 `moe_config` 不能缺席：128 专家与同形状的 dense sidecar 都可能有
    gate/up/down 投影，缺了身份就无法区分。结构指纹由内容重算，不看自报值。
    """
    experts = int(ir.get("moe_num_experts", 0) or 0)
    declared = sidecar.get("moe_config")
    if experts > 0:
        if not isinstance(declared, dict):
            raise StepFailed(
                f"源 IR 是 MoE 模型（{experts} 个专家），但 sidecar 没有 "
                "moe_config：无法核对专家几何与执行语义，请用当前 IR 重新导出。")
        expected = {
            "num_experts": experts,
            "top_k": int(ir.get("moe_top_k", 0) or 0),
            "intermediate_size": int(ir.get("moe_intermediate_size", 0) or 0),
            "execution": str(ir.get("moe_execution", "") or ""),
        }
        mismatch = {
            key: (declared.get(key), want)
            for key, want in expected.items()
            if declared.get(key) != want
        }
        if mismatch:
            raise StepFailed(
                "sidecar 的 moe_config 与源 IR 不一致："
                f"{mismatch}。sidecar 与 IR 不是同一个模型/同一套 MoE 语义。")
    elif isinstance(declared, dict):
        raise StepFailed(
            "sidecar 带 moe_config，但源 IR 不是 MoE 模型：两份产物不是同一次"
            "导出的同一模型。")

    try:
        actual = assert_ir_fingerprint(ir, "源 IR")
    except ValueError as exc:
        raise StepFailed(str(exc)) from exc
    declared_fingerprint = sidecar.get("ir_fingerprint")
    if declared_fingerprint:
        if str(declared_fingerprint) != actual:
            raise StepFailed(
                f"sidecar 的 ir_fingerprint={str(declared_fingerprint)[:16]} 与"
                f"源 IR 重算的 {actual[:16]} 不一致：两份产物不是同一个模型/"
                "同一套结构，放置结果会钉到别的算子上。")
    elif str(ir.get("model_fingerprint", "")):
        raise StepFailed(
            f"sidecar 没有 ir_fingerprint，而源 IR 带结构指纹 "
            f"{str(ir.get('model_fingerprint'))[:16]}：无法证明两份产物同源。")


def validate_sidecar_contract(
    ir: Dict[str, Any], sidecar: Dict[str, Any], *,
    require_absorbed_closure: bool = True,
    strict_pimir: Optional[bool] = None,
) -> None:
    """编译器侧唯一的严格产物验证器。

    导出、skip-simulation、仿真前复核、仿真 IR 对照与 trace 验收共用这一份
    规则：模型/MoE 元数据、结构指纹、容量三件套、逐条豁免、吸收结构、逐
    op_id 专家身份与 pimir 产物的可读性/哈希缺一项都不放行。默认按精化后的
    阶段要求吸收集合与融合清单双向闭合；只有 B+C+D 导出步骤显式放宽
    `require_absorbed_closure`。

    `strict_pimir` 是调用方的严格性上下文，与 sidecar 自报取并集：配置显式
    要求严格时，删掉自报也不能退回兼容分支。None 表示调用方没有声明，只看
    产物自报（单元测试与覆盖审计组件用）。直接 GeneSim 仍保留独立复核。
    """
    _validate_sidecar_identity(ir, sidecar)
    audit_sidecar_coverage(
        ir, sidecar,
        require_absorbed_closure=require_absorbed_closure,
        strict_pimir=strict_pimir)


def _config_requires_pimir(loaded: Dict[str, Any]) -> bool:
    """仿真配置是否显式要求严格 pimir 契约。"""
    scheduler = loaded.get("scheduler") or {}
    return bool(scheduler.get("require_compiler_pimir", False))


def _validate_sidecar_hw_budget(sidecar: Dict[str, Any]) -> None:
    """严格产物必须回传完整容量契约：总容量、系统预留与对齐缺一不可。

    三者共同决定「三区 + 内核 tile」的可用预算；缺一个或任一偏离都会让编译器
    与 GeneSim 各算一套容量。严格性来自调用方配置与 sidecar 自报的并集，
    只有双方都明确非严格的旧式 sidecar 才允许没有这个字段。
    """
    from memory.mem_planner import default_hw_budget

    declared = sidecar.get("hw_budget")
    if not isinstance(declared, dict):
        raise StepFailed(
            "严格产物（配置显式要求或 sidecar 自报 requires_pimir）没有 "
            "hw_budget：容量契约不能缺席。请用当前主链重新导出。")
    expected = default_hw_budget()
    problems = []
    for field, want in (
            ("mram_bytes_per_dpu", expected.mram_bytes),
            ("sys_reserve_bytes", expected.sys_reserve_bytes),
            ("align", expected.align)):
        value = declared.get(field)
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            problems.append(f"{field}={value!r} 不是正整数")
            continue
        if value != want:
            problems.append(f"{field}={value} 与容量契约 {want} 不一致")
    if problems:
        raise StepFailed(
            "sidecar 的 hw_budget 与容量契约不一致："
            + "；".join(problems)
            + "。需求 §2.1 P0-3 的 8 DPU × 8 GiB、64 MiB 系统预留与 64 字节"
            "对齐是硬约束，不能用更宽松的预算进仿真。")


def _validate_absorbed_ops(
    ir: Dict[str, Any], sidecar: Dict[str, Any], *,
    require_closure: bool = True,
) -> None:
    """被吸收节点必须由融合清单与 IR 依赖边唯一证明。

    `absorbed_ops` 是 sidecar 自报的，不能当作无需证明的事实：伪造一个设备
    算子编号就能让它从覆盖校验里消失。合法吸收只有一种结构——主算子在融合
    清单里按 (层, 角色, 专家) 声明了同名激活，被吸收节点是该主算子在 IR 里的
    唯一消费者，且主算子自己有 pimir。

    `require_closure` 区分产物阶段：B+C+D 只产出融合清单，`absorbed_ops` 由
    精化步骤回写，此时允许清单单独存在；A2 之后、skip-simulation、仿真前
    复核、trace 验收与直接 GeneSim 则要求清单与吸收编号双向闭合，删掉吸收
    编号会让主算子折激活与独立激活 trace 同时进入代价链。
    """
    absorbed = {int(op_id) for op_id in (sidecar.get("absorbed_ops") or [])}
    manifest = {
        (int(entry["layer"]), str(entry["role"]),
         int(entry.get("expert", -1)), str(entry["activation"]).lower())
        for entry in (sidecar.get("fused_activations") or [])
    }
    if not absorbed and not manifest:
        return
    if absorbed and not manifest:
        raise StepFailed(
            f"sidecar 声明了 {len(absorbed)} 个被吸收算子，但没有 "
            "fused_activations：无法证明它们真的被折进了主算子。")

    by_id = {int(op["op_id"]): op for op in ir.get("operators", [])}
    layer_of = {
        int(op_id): layer
        for layer, group in enumerate(ir.get("subgraphs") or [])
        for op_id in group
    }
    inputs: Dict[int, list] = {}
    consumers: Dict[int, set] = {}
    for edge in ir.get("dependencies") or []:
        src, dst = int(edge["src_op_id"]), int(edge["dst_op_id"])
        inputs.setdefault(dst, []).append(src)
        consumers.setdefault(src, set()).add(dst)
    entries = sidecar.get("operators") or {}

    problems: list[str] = []
    absorbed_signatures: set = set()
    for op_id in sorted(absorbed):
        op = by_id.get(op_id)
        if op is None:
            problems.append(f"op{op_id} 不在源 IR 里")
            continue
        if op.get("op_type") not in _ABSORBABLE_ACTIVATIONS:
            problems.append(
                f"op{op_id} 是 {op.get('op_type')}，只有激活可以被吸收")
            continue
        producers = inputs.get(op_id) or []
        if len(producers) != 1:
            problems.append(
                f"op{op_id} 有 {len(producers)} 个生产者，无法唯一确定被折进"
                "哪个主算子")
            continue
        main = by_id.get(producers[0])
        if main is None or main.get("op_type") != "GEMM":
            problems.append(f"op{op_id} 的生产者不是主算子 GEMM")
            continue
        main_id = int(main["op_id"])
        readers = consumers.get(main_id) or set()
        if readers != {op_id}:
            others = sorted(readers - {op_id})
            problems.append(
                f"op{op_id} 的主算子 op{main_id} 还有其它消费者 "
                f"op{others[:5]}：折叠会改写另一个读者要的激活前值")
            continue
        if layer_of.get(main_id) != layer_of.get(op_id):
            problems.append(f"op{op_id} 与主算子不在同一层")
            continue
        triple = (
            layer_of.get(main_id, -1),
            str(main.get("semantic_role", "")),
            int(main.get("expert_id", -1)),
            str(op["op_type"]).lower(),
        )
        if triple not in manifest:
            problems.append(
                f"op{op_id} 的生产主算子 {triple[:3]} 不在融合清单里")
            continue
        main_entry = entries.get(str(main_id)) or {}
        if not main_entry.get("pimir_path"):
            problems.append(
                f"op{op_id} 的主算子 op{main_id} 没有 pimir_path："
                "激活折进了一个没有真实产物的算子")
            continue
        absorbed_signatures.add(triple)

    if require_closure:
        unclosed = sorted(manifest - absorbed_signatures)
        if unclosed:
            problems.append(
                f"{len(unclosed)} 条融合清单没有被吸收编号闭合（如 "
                f"{unclosed[:3]}）：精化后清单与 absorbed_ops 必须双向对应")

    if problems:
        raise StepFailed(
            f"absorbed_ops 与融合清单不自洽：{'；'.join(problems)}。"
            "被吸收编号只能由 (层, 角色, 专家) + 主算子 pimir + IR 依赖边"
            "唯一证明，不能自报。")
    # B+C+D 阶段反向（清单 → 被吸收编号）不要求：`absorbed_ops` 由精化步骤
    # 回写。清单本身的真实性由 `fused_kernels_missing_act_spec` 核对——
    # 折过激活的 kernel 必须带 act_spec。


def _invalid_exemptions(
    ir: Dict[str, Any], sidecar: Dict[str, Any],
) -> list[int]:
    """逐条核对 `pimir_exempt_ops` 与源 IR 的真实落点，返回不合法编号。

    覆盖审计、skip-simulation、仿真前复核与 trace 验收共用这一份规则：
    豁免必须落在 host（`device_hint=cpu`）或是模型边界，且 sidecar 条目
    带有导出时写下的落点证据。仅凭 `expert_id<0` 之类的身份推断不算证据。
    同一算子的实际放置条目与 `host_gemms` 证据也在这里交叉核对：一边按
    host 免掉 pimir、另一边却写着 PIM 分片或 pimir，两处证据矛盾即失败。
    """
    ir_ops = {int(op["op_id"]): op for op in ir.get("operators", [])}
    placements = sidecar.get("operators") or {}
    host_gemms = sidecar.get("host_gemms") or {}
    invalid: list[int] = []
    for raw_op_id, entry in (sidecar.get("pimir_exempt_ops") or {}).items():
        op_id = int(raw_op_id)
        op = ir_ops.get(op_id)
        if op is None or not _exemption_is_legitimate(
                op, entry or {}, placement=placements.get(raw_op_id),
                host_evidence=host_gemms.get(raw_op_id)):
            invalid.append(op_id)
    return sorted(invalid)


def _moe_expert_triples(
    ir: Dict[str, Any],
) -> Dict[int, tuple[int, str, int]]:
    """源 IR 里专家 GEMM 的 (层号, 投影身份, 专家号)，按 op_id 索引。"""
    layer_of = {
        int(op_id): layer
        for layer, group in enumerate(ir.get("subgraphs") or [])
        for op_id in group
    }
    triples: Dict[int, tuple[int, str, int]] = {}
    for op in ir.get("operators", []):
        expert = int(op.get("expert_id", -1))
        if expert < 0 or op.get("op_type") != "GEMM":
            continue
        op_id = int(op["op_id"])
        triples[op_id] = (
            layer_of.get(op_id, -1),
            str(op.get("semantic_role", "")),
            expert,
        )
    return triples


def _moe_identity_problems(
    ir: Dict[str, Any], sidecar: Dict[str, Any],
) -> list[str]:
    """逐 op_id 核对专家 GEMM 的 (层, 角色, 专家, 权重) 身份。

    编译器导出审计与 trace 验收共用这一份规则：集合相等不能证明产物同源，
    同层两个专家互换后三元组集合仍然相同。
    """
    expected = _moe_expert_triples(ir)
    if not expected:
        return []
    ops = sidecar.get("operators") or {}
    problems: list[str] = []
    for op_id, (layer, role, expert) in sorted(expected.items()):
        entry = ops.get(str(op_id)) or {}
        declared = (
            str(entry.get("semantic_role", "")),
            int(entry.get("expert_id", -1)),
        )
        if declared != (role, expert):
            problems.append(
                f"op{op_id}(IR {role}/{expert} vs sidecar "
                f"{declared[0]}/{declared[1]})")
            continue
        weight = str(entry.get("weight", ""))
        if (f"layers.{layer}." not in weight
                or f"experts.{expert}.{role}.weight" not in weight):
            problems.append(
                f"op{op_id}(期望 layers.{layer}…experts.{expert}."
                f"{role}.weight，实际 {weight or '缺 weight'})")
    return problems


_PIMIR_TRACE_TOKENS_RE = re.compile(r"pim\.traceTokens\s*=\s*(\d+)")
_PIMIR_FUNC_ARGS_RE = re.compile(
    r"tt\.func\s+@\w+\s*\((.*?)\)\s*attributes", re.S)
_PIMIR_ARROW_RE = re.compile(r"->\s*([^\n]+)")
_PIMIR_TENSOR_RE = re.compile(r"tensor<([0-9]+(?:x[0-9]+)*)x[a-z0-9]+>")
_PIMIR_GEMM_OP_RE = re.compile(r"\b(tt\.dot|pim\.matmul)\b")
_PIMIR_TILE_N_RE = re.compile(r'"?pim\.tile-n"?\s*=\s*(\d+)')
_PIMIR_TILE_K_RE = re.compile(r'"?pim\.tile-k"?\s*=\s*(\d+)')
_PIMIR_ACT_SPEC_RE = re.compile(r"#pim\.act_spec<kind\s*=\s*(\w+)>")
_PIMIR_ELTWISE_RE = re.compile(r"#pim\.eltwise<(\w+)>")
_PIMIR_ACTIVATION_RE = re.compile(r"#pim\.activation<(\w+)>")
# 顶层算子行：`%0 = pim.lut %a ...` 或 `pim.kv_cache %a ...`。`#pim.unit<>`
# 这类属性引用前面紧跟 `#`，不是算子；`pim.target = ...` 后面不是操作数。
_PIMIR_TOP_LEVEL_OP_RE = re.compile(r"(?<![\w#])pim\.(\w+)\s+%")
_PIMIR_KV_FIXED_IN_RE = re.compile(r"pim\.kvFixedIn\s*=\s*(\d+)")
_PIMIR_KV_FIXED_OUT_RE = re.compile(r"pim\.kvFixedOut\s*=\s*(\d+)")
_PIMIR_SCF_FOR_RE = re.compile(
    r"scf\.for\s+%\S+\s*=\s*%(\w+?)\s+to\s+%(\w+?)\s+step\s+%(\w+?)\s")
_PIMIR_CONST_RE = re.compile(
    r"%(\S+)\s*=\s*arith\.constant\s+(-?\d+)\s*:\s*i\d+")


def _split_top_level(text: str) -> List[str]:
    """按顶层逗号切分 MLIR 片段：尖括号/方括号/花括号内的逗号不切。"""
    parts: List[str] = []
    depth = 0
    start = 0
    for index, char in enumerate(text):
        if char in "<([{":
            depth += 1
        elif char in ">)]}":
            depth -= 1
        elif char == "," and depth == 0:
            parts.append(text[start:index])
            start = index + 1
    parts.append(text[start:])
    return parts


def _positive_attr(pattern: "re.Pattern[str]", text: str) -> bool:
    """pimir 的整数属性是否出现且为正：0 表示这部分不随 KV 历史重复。"""
    match = pattern.search(text)
    return match is not None and int(match.group(1)) > 0


def _pimir_unit_facts(text: str) -> Dict[str, Any]:
    """解析 pimir 的编译单位：trace token 数与参数/结果张量形状。

    B 路 pimir 的函数参数按位置对应操作数（memdesc/scalar 记 None），最后
    一条带 `->` 的 op 的结果是函数输出。`pim.traceTokens` 缺失时 `tokens`
    为 None，由严格产物校验拒绝。同时解析 A 路需要的分块 GEMM 事实（原语、
    tile 属性、`tt.dot` 的三维与循环嵌套、融合激活种类）与 B 路语义指纹
    事实（顶层原语、激活/逐元素种类、块循环与 KV 固定开销标注）：两种产物
    共用同一份文本解析，避免一个入口解析、另一个入口只信 sidecar 自报。
    """
    tokens_match = _PIMIR_TRACE_TOKENS_RE.search(text)
    tokens = int(tokens_match.group(1)) if tokens_match else None

    def dims(match) -> List[int]:
        return [int(dim) for dim in match.group(1).split("x")]

    input_shapes: List[Optional[List[int]]] = []
    args_text = _PIMIR_FUNC_ARGS_RE.search(text)
    if args_text:
        for arg in _split_top_level(args_text.group(1)):
            match = _PIMIR_TENSOR_RE.search(arg)
            input_shapes.append(dims(match) if match else None)
    output_shapes: List[List[int]] = []
    arrows = _PIMIR_ARROW_RE.findall(text)
    if arrows:
        for part in _split_top_level(arrows[-1]):
            output_shapes.extend(dims(match)
                                 for match in _PIMIR_TENSOR_RE.finditer(part))

    gemm_lines = [
        line for line in text.splitlines() if _PIMIR_GEMM_OP_RE.search(line)
    ]
    primitive = (
        _PIMIR_GEMM_OP_RE.search(gemm_lines[0]).group(1)
        if gemm_lines else None
    )
    m = k = n = None
    if gemm_lines:
        shapes = [tuple(int(dim) for dim in match.groups())
                  for match in re.finditer(
                      r"tensor<(\d+)x(\d+)x[a-z0-9]+", gemm_lines[0])]
        if len(shapes) >= 3:
            m, k, n = shapes[0][0], shapes[0][1], shapes[2][1]
    tile_n = _PIMIR_TILE_N_RE.search(text)
    tile_k = _PIMIR_TILE_K_RE.search(text)
    activation = _PIMIR_ACT_SPEC_RE.search(gemm_lines[0]) if gemm_lines else None
    eltwise = _PIMIR_ELTWISE_RE.search(text)
    constants = {
        match.group(1): int(match.group(2))
        for match in _PIMIR_CONST_RE.finditer(text)
    }
    loops: List[tuple[int, int]] = []
    for match in _PIMIR_SCF_FOR_RE.finditer(text):
        bound = constants.get(match.group(2))
        step = constants.get(match.group(3))
        if bound is not None and step is not None:
            loops.append((bound, step))
    return {
        "tokens": tokens,
        "input_shapes": input_shapes,
        "output_shapes": output_shapes,
        "primitive": primitive,
        "gemm_ops": len(gemm_lines),
        "m": m,
        "k": k,
        "n": n,
        "tile_n": int(tile_n.group(1)) if tile_n else None,
        "tile_k": int(tile_k.group(1)) if tile_k else None,
        "activation": activation.group(1) if activation else None,
        "eltwise_kind": eltwise.group(1) if eltwise else None,
        "loops": loops,
        "has_phase": "#pim.phase_spec" in text,
        # B 路语义指纹：顶层原语与关键属性必须与源 IR 的算子类型对应。
        "top_level_ops": [
            match.group(1) for match in _PIMIR_TOP_LEVEL_OP_RE.finditer(text)
        ],
        "activation_kind": (
            match.group(1) if (match := _PIMIR_ACTIVATION_RE.search(text))
            else None),
        "has_rms_norm": "rmsNorm" in text,
        "has_rotate_half": "rotateHalf" in text,
        "has_source_sub_blocks": "sourceSubBlocks" in text,
        "has_kv_block_tokens": "pim.kvBlockTokens" in text,
        "has_stationarity_kv": "#pim.stationarity<kv>" in text,
        "kv_fixed_in_positive": _positive_attr(_PIMIR_KV_FIXED_IN_RE, text),
        "kv_fixed_out_positive": _positive_attr(_PIMIR_KV_FIXED_OUT_RE, text),
        "has_slot_tokens": bool(re.search(r"slotTokens\s*=\s*\d+", text)),
        "has_slot_elems": "slotElems" in text,
        "has_reciprocal_lut": "#pim.activation<reciprocal>" in text,
    }


def _pimir_unit_problems(
    op_type: str, entry: Dict[str, Any], facts: Dict[str, Any], label: str,
) -> List[str]:
    """sidecar 的编译单位必须与它引用的 pimir 实产一致。

    `trace_tokens` 要与 pimir 的 `pim.traceTokens` 相等；`local_*_shapes`
    的末维（具体特征/本地宽）要与 pimir 对应参数或结果的末维相等。pimir
    是单位的唯一真源，sidecar 自报字段只允许作为它的同源副本。
    SPLIT 的 pimir 是纯视图，输出类型记的是完整输入形状，而 IR 只记一个
    代表切片，所以输出末维对 SPLIT 不适用。
    """
    problems: List[str] = []
    declared_tokens = int(entry["trace_tokens"])
    pimir_tokens = facts.get("tokens")
    if pimir_tokens is None:
        problems.append(
            f"{label} 的 pimir 没有 pim.traceTokens：编译单位无法与实产绑定")
    elif int(pimir_tokens) != declared_tokens:
        problems.append(
            f"{label} 的 pimir traceTokens={pimir_tokens} 与 sidecar 的 "
            f"trace_tokens={declared_tokens} 不一致")
    comparisons = [("输入", "local_input_shapes", facts.get("input_shapes") or [])]
    if op_type != "SPLIT":
        comparisons.append(
            ("输出", "local_output_shapes", facts.get("output_shapes") or []))
    for kind, key, actual_shapes in comparisons:
        for index, shape in enumerate(entry.get(key) or []):
            if index >= len(actual_shapes):
                continue
            actual = actual_shapes[index]
            if not actual:
                continue
            feature = shape[-1] if shape else None
            if not isinstance(feature, int) or isinstance(feature, bool):
                continue
            if actual[-1] != feature:
                problems.append(
                    f"{label} 的{kind}[{index}] 特征维 {feature} 与 pimir "
                    f"末维 {actual[-1]} 不一致")
    return problems


# B 路算子类型 -> (必需顶层原语的多重集, 允许的顶层原语集合, 必需事实)。
# 只登记按真实编译实产校准过的类型；表是闭合的，未登记类型在严格产物里
# 直接拒绝。事实名对应 `_pimir_unit_facts` 的键，布尔事实的期望值为 True。
# GeneSim 的 `pim.pimir_trace.BPATH_PIMIR_SEMANTICS` 持有同一份表，跨仓
# 测试逐值钉住；只改一侧会失败。
_BPATH_PIMIR_SEMANTICS: Dict[
    str, Tuple[Tuple[str, ...], frozenset, Tuple[Tuple[str, Any], ...]]
] = {
    "RMSNORM": (("normalize",), frozenset({"normalize"}),
                (("has_rms_norm", True),)),
    "SILU": (("lut",), frozenset({"lut"}),
             (("activation_kind", "silu"),)),
    "GELU": (("lut",), frozenset({"lut"}),
             (("activation_kind", "gelu"),)),
    "VECTOR_ADD": (("eltwise",), frozenset({"eltwise"}),
                   (("eltwise_kind", "add"),)),
    "VECTOR_MUL": (("eltwise",), frozenset({"eltwise"}),
                   (("eltwise_kind", "mul"),)),
    "PAGED_ATTN": (("paged_attn",), frozenset({"paged_attn"}),
                   (("has_slot_tokens", True), ("has_slot_elems", True))),
    # ROPE 的实产是三相 pim.eltwise 链，`pim.rope` 只出现在发射端 IR 里。
    "ROPE": (("eltwise", "eltwise", "eltwise"), frozenset({"eltwise"}),
             (("has_rotate_half", True), ("has_source_sub_blocks", True))),
    "TRANSPOSE": (("transpose",), frozenset({"transpose"}), ()),
    "SPLIT": (("split_heads",), frozenset({"split_heads"}), ()),
    "CONCAT": (("concat",), frozenset({"concat"}), ()),
    "RESHAPE": (("reshape",), frozenset({"reshape"}), ()),
    "GATHER": (("gather",), frozenset({"gather"}), ()),
    "MEM_COPY": (("kv_cache",), frozenset({"kv_cache"}), ()),
    "QUANT": (("global_pool", "lut"),
              frozenset({"global_pool", "lut", "kantor"}), ()),
    # 四节点注意力的真实产物：MASK 单原语；SOFTMAX 展开成
    # reduce/exp/reciprocal 链；两次矩阵乘靠固定读写端区分得分与上下文。
    "MASK": (("mask",), frozenset({"mask"}),
             (("has_kv_block_tokens", True),)),
    "SOFTMAX": (("reduce_axis", "eltwise", "lut"),
                frozenset({"reduce_axis", "eltwise", "lut"}),
                (("activation_kind", "exp"),
                 ("has_reciprocal_lut", True),
                 ("has_kv_block_tokens", True))),
    "GEMV_SCORE": (("matmul",), frozenset({"matmul"}),
                   (("has_kv_block_tokens", True),
                    ("has_stationarity_kv", True),
                    ("kv_fixed_in_positive", True))),
    "GEMV_CONTEXT": (("matmul",), frozenset({"matmul"}),
                     (("has_kv_block_tokens", True),
                      ("has_stationarity_kv", True),
                      ("kv_fixed_out_positive", True))),
}


def _bpath_pimir_semantic_problems(
    op_type: str, facts: Dict[str, Any], label: str,
) -> List[str]:
    """核对 B 路 pimir 的顶层原语与关键属性是否真是该算子的编译实产。

    只比 token 数与特征末维发现不了同宽不同类型的产物互换：ROPE 的全部
    条目指向一份 RMSNORM pimir 后，路径、哈希、形状与字节都成立，代价却
    按另一种原语计算。这里按闭合表核对顶层原语多重集、允许集合与必需事实；
    未登记的类型直接拒绝，宁可要求重新校准也不放行未知语义。
    """
    spec = _BPATH_PIMIR_SEMANTICS.get(op_type)
    if spec is None:
        return [
            f"{label}（{op_type}）没有登记 B 路 pimir 语义指纹：严格契约"
            "只接受按真实实产校准过的算子类型"
        ]
    required_ops, allowed_ops, markers = spec
    ops = [str(op) for op in (facts.get("top_level_ops") or [])]
    problems: List[str] = []
    unexpected = sorted({op for op in ops if op not in allowed_ops})
    if unexpected:
        problems.append(
            f"{label}（{op_type}）的 pimir 顶层原语 {unexpected} 不在 "
            f"{sorted(allowed_ops)} 里：引用的不是该算子的编译实产")
    remaining = list(ops)
    missing: List[str] = []
    for required in required_ops:
        if required in remaining:
            remaining.remove(required)
        else:
            missing.append(required)
    if missing:
        problems.append(
            f"{label}（{op_type}）的 pimir 缺少顶层原语 {sorted(missing)}："
            f"该类型必须由 {list(required_ops)} 这些原语实现")
    for key, expected in markers:
        actual = facts.get(key)
        matched = (
            actual is expected if isinstance(expected, bool)
            else actual == expected)
        if not matched:
            problems.append(
                f"{label}（{op_type}）的 pimir 语义事实 {key}={actual!r}，"
                f"应为 {expected!r}：该属性缺失或属于另一个算子")
    return problems


def _bpath_unit_contract_problems(
    ir: Dict[str, Any], sidecar: Dict[str, Any], *, strict: bool,
    pimir_facts: Optional[Dict[str, Dict[str, Any]]] = None,
) -> List[str]:
    """严格产物里每条 B 路 pimir 都必须带完整的编译单位。

    单位是 trace 与运行期形状同源的唯一证据：`trace_tokens` 决定重复次数，
    `local_*_shapes` 用 Tq/Tp+Tq 标注 token 维与历史维、其余维是本地宽。
    条目类别只由源 IR 决定（GEMM 走 A 路，其余都是 B 路），B 路自报的非空
    `shards` 不能跳过这份检查。单位数值还必须与所引用 pimir 的
    `pim.traceTokens` 和特征末维一致，否则 sidecar 只是自证。
    导出、skip-simulation、仿真前复核与仿真 IR 对照共用这一份检查；只挂在
    trace 入口会让前三者签发一份 GeneSim 严格入口必然拒绝的产物。
    """
    if not strict:
        return []
    ops = sidecar.get("operators") or {}
    absorbed = {int(op_id) for op_id in (sidecar.get("absorbed_ops") or [])}
    exempt = {int(op_id) for op_id in
              (sidecar.get("pimir_exempt_ops") or {})}
    unit_keys = ("trace_tokens", "local_input_shapes", "local_output_shapes")
    facts_by_path = pimir_facts or {}
    problems: List[str] = []
    for op in ir.get("operators", []):
        op_id = int(op["op_id"])
        if op.get("op_type") in ("MODEL_INPUT", "MODEL_OUTPUT"):
            continue
        if op_id in absorbed or op_id in exempt:
            continue
        entry = ops.get(str(op_id)) or {}
        if op.get("op_type") == "GEMM":
            # GEMM 的 pimir 按本地分片形状编译；缺 pimir 由覆盖审计报。
            continue
        if entry.get("shards"):
            problems.append(
                f"op{op_id}（{op.get('op_type')}）是非 GEMM 却带分片：B 路"
                "不能借分片跳过编译单位")
            continue
        if not entry.get("pimir_path"):
            continue
        missing = [key for key in unit_keys if entry.get(key) is None]
        if missing:
            problems.append(
                f"op{op_id}（{op.get('op_type')}）缺编译单位 {missing}")
            continue
        tokens = entry["trace_tokens"]
        if (not isinstance(tokens, int) or isinstance(tokens, bool)
                or tokens <= 0):
            problems.append(f"op{op_id} 的 trace_tokens={tokens!r} 不是正整数")
            continue
        problems.extend(_unit_shape_role_problems(
            op, entry, f"op{op_id}"))
        facts = facts_by_path.get(entry["pimir_path"])
        if facts is None:
            problems.append(
                f"op{op_id} 的 pimir 产物不可读：无法核对编译单位")
            continue
        problems.extend(_pimir_unit_problems(
            op.get("op_type", ""), entry, facts, f"op{op_id}"))
        # 顶层原语与关键属性必须与算子类型对应（与 GeneSim 的
        # `_assert_pimir_operator_semantics` 同一张表）：同宽不同类型的
        # 产物互换只靠 op_type 自报和形状核对发现不了。
        problems.extend(_bpath_pimir_semantic_problems(
            str(op.get("op_type", "")), facts, f"op{op_id}"))
    return problems


def _apath_facts_problems(
    entry: Dict[str, Any], facts: Dict[str, Any], tile_n: int, label: str,
) -> List[str]:
    """把一条设备 GEMM 的 pimir 事实与 sidecar 分片逐项对齐。"""
    problems: List[str] = []
    if facts.get("primitive") is None:
        return [
            f"{label} 的 pimir 不是分块 GEMM 原语（没有 tt.dot/pim.matmul）："
            "A 路条目指向了 B 路产物"
        ]
    if facts.get("gemm_ops") != 1:
        problems.append(
            f"{label} 的 pimir 含 {facts.get('gemm_ops')} 个 GEMM 原语："
            "编译单位必须是单个 tt.dot")
    if facts.get("tokens") is not None:
        problems.append(
            f"{label} 的 pimir 带 pim.traceTokens：这是 B 路的 1-token 编译"
            "单位，不是该分片的实测分块 GEMM")
    if facts.get("has_phase"):
        problems.append(f"{label} 的 pimir 带相位链：这是 B 路产物")
    if facts.get("tile_n") != tile_n:
        problems.append(
            f"{label} 的 pimir tile-n={facts.get('tile_n')!r} 与 sidecar 的 "
            f"kernel_tile_n={tile_n} 不一致")
    tile_k = facts.get("tile_k")
    if not isinstance(tile_k, int) or isinstance(tile_k, bool) or tile_k <= 0:
        problems.append(f"{label} 的 pimir 没有正整数 pim.tile-k")
        return problems
    if facts.get("m") != 1:
        problems.append(
            f"{label} 的 tt.dot M={facts.get('m')!r} 不是 1：编译单位不是"
            "单 token 的分块 GEMM")
    if facts.get("k") != tile_k:
        problems.append(
            f"{label} 的 tt.dot K={facts.get('k')!r} 与 pimir tile-k={tile_k}"
            " 不一致")
    if facts.get("n") != tile_n:
        problems.append(
            f"{label} 的 tt.dot N={facts.get('n')!r} 与 kernel_tile_n="
            f"{tile_n} 不一致")
    loops = set(facts.get("loops") or [])
    for shard in entry.get("shards") or []:
        local_in = shard.get("local_in_features")
        local_out = shard.get("local_out_features")
        if (not isinstance(local_in, int) or isinstance(local_in, bool)
                or local_in <= 0 or not isinstance(local_out, int)
                or isinstance(local_out, bool) or local_out <= 0):
            problems.append(
                f"{label} 的分片本地形状 {local_in!r}×{local_out!r} 不是"
                "正整数")
            continue
        if facts.get("n") is not None and facts["n"] > local_out:
            problems.append(
                f"{label} 的 tt.dot N={facts['n']} 大于本地输出宽 "
                f"{local_out}：pimir 不是这个分片的产物")
        expected = set()
        if local_in > tile_k:
            expected.add((local_in, tile_k))
        if local_out > tile_n:
            expected.add((local_out, tile_n))
        if facts.get("primitive") == "tt.dot":
            missing = sorted(expected - loops)
            if missing:
                problems.append(
                    f"{label} 的 tt.dot 循环缺少 {missing}：pimir 的 K/N "
                    f"分块与本地形状 {local_in}×{local_out} 不符")
            extra = sorted(loops - expected)
            if extra:
                problems.append(
                    f"{label} 的 tt.dot 循环多出 {extra}：pimir 不是该分片"
                    "的单层 K/N 分块 GEMM")
    return problems


def _apath_unit_problems(
    ir: Dict[str, Any], sidecar: Dict[str, Any], *,
    pimir_facts: Dict[str, Dict[str, Any]],
) -> List[str]:
    """严格产物里每条设备 GEMM 的 pimir 必须是该分片的实测分块 GEMM。

    A 路与 round18 的 B 路对称：sidecar 只回传路径与哈希时，把全部 GEMM
    指向一份 B 路逐元素产物、或换成另一分片宽度的 GEMM，五个编译器入口
    都会签发，仿真代价可以任意伪造。这里按源 IR 逐条核对：

      (a) 产物是分块 GEMM 原语（`tt.dot`/`pim.matmul`），不带
          `pim.traceTokens` 与相位链；
      (b) `tt.dot` 的 M/K/N 与分片 `local_in/out_features`、
          `kernel_tile_n` 和 pimir 的 tile 属性一致，K/N 循环嵌套与之对应；
      (c) 融合清单声明了激活的投影，产物必须带同名 `act_spec`，反向也不能
          多带——融合状态会改写发射语义与代价。

    host GEMM（源 IR `device_hint=cpu`、豁免登记与 `host_gemms` 证据齐全）
    没有 pimir，不参与本项；豁免与路径缺失由覆盖审计另行报告。
    """
    ops = sidecar.get("operators") or {}
    exempt = {int(op_id) for op_id in
              (sidecar.get("pimir_exempt_ops") or {})}
    manifest = {
        (int(entry["layer"]), str(entry["role"]),
         int(entry.get("expert", -1))): str(entry["activation"]).lower()
        for entry in (sidecar.get("fused_activations") or [])
    }
    layer_of = {
        int(op_id): layer
        for layer, group in enumerate(ir.get("subgraphs") or [])
        for op_id in group
    }
    problems: List[str] = []
    for op in ir.get("operators", []):
        if op.get("op_type") != "GEMM":
            continue
        op_id = int(op["op_id"])
        if op_id in exempt:
            continue
        entry = ops.get(str(op_id)) or {}
        if not entry.get("pimir_path"):
            continue  # 缺路径由覆盖审计报告。
        if str(entry.get("device_hint", "")).lower() == "cpu":
            # 合法 host GEMM 不写 pimir；自称 host 却缺证据的条目已由类别
            # 检查拒绝，能走到这里的只剩合法 host。
            continue
        if not entry.get("shards"):
            continue  # 空分片由类别检查报告。
        tile_n = entry.get("kernel_tile_n")
        if (not isinstance(tile_n, int) or isinstance(tile_n, bool)
                or tile_n <= 0):
            problems.append(
                f"op{op_id} 的 kernel_tile_n={tile_n!r} 不是正整数："
                "A 路条目必须回传算子编译器实测分块")
            continue
        facts = pimir_facts.get(entry["pimir_path"])
        if facts is None:
            problems.append(
                f"op{op_id} 的 pimir 产物不可读：无法核对 A 路编译单位")
            continue
        label = f"op{op_id}"
        problems.extend(_apath_facts_problems(entry, facts, tile_n, label))
        triple = (
            layer_of.get(op_id),
            str(op.get("semantic_role", "")),
            int(op.get("expert_id", -1)),
        )
        expected_act = str(manifest.get(triple) or "")
        actual_act = str(facts.get("activation") or "")
        if expected_act != actual_act:
            problems.append(
                f"{label} 的 pimir 激活 {actual_act or '无'} 与融合清单的 "
                f"{expected_act or '无'} 不一致（{triple[:2]}，expert="
                f"{triple[2]}）：融合状态必须由 pimir 实产证明")
    return problems


def audit_sidecar_coverage(
    ir: Dict[str, Any], sidecar: Dict[str, Any], *,
    require_absorbed_closure: bool = True,
    strict_pimir: Optional[bool] = None,
) -> None:
    """以源 IR 为准核对 sidecar 覆盖，与算子最终落到哪种设备无关。

    覆盖集合由源 IR 定义：除模型边界、被吸收节点与显式豁免外，每个算子都
    必须有 pimir_path；严格上下文（sidecar 自报或调用方显式严格）还必须核对
    文件可读、哈希正确，产物校验不允许由入口关闭。豁免只允许模型边界/host/
    非 DPU 稠密 GEMM，未知设备算子直接失败。MoE 再按 (层号, 投影身份, 专家号)
    核对 gate/up/down 完整集合，不能只取专家号并集。主链导出、skip-simulation
    与仿真前核对共用这一份契约。

    `strict_pimir` 与 sidecar 自报取并集，任一成立都要求完整容量契约与可读
    且哈希匹配的 pimir 产物。
    """
    strict = bool(sidecar.get("requires_pimir")) or bool(strict_pimir)
    if strict:
        _validate_sidecar_hw_budget(sidecar)
    _validate_absorbed_ops(
        ir, sidecar, require_closure=require_absorbed_closure)
    ops = sidecar.get("operators") or {}
    exempt = {
        int(op_id): (entry or {})
        for op_id, entry in (sidecar.get("pimir_exempt_ops") or {}).items()
    }
    absorbed = {int(op_id) for op_id in (sidecar.get("absorbed_ops") or [])}
    ir_ops = {int(op["op_id"]): op for op in ir.get("operators", [])}

    missing: list[int] = []
    unreadable: list[int] = []
    unhashed: list[int] = []
    stale: list[int] = []
    illegitimate = _invalid_exemptions(ir, sidecar)
    # 多个算子共用同一份 pim mlir，按路径缓存哈希与解析后的编译单位。
    pimir_cache: dict[str, tuple[str, Dict[str, Any]]] = {}
    for op_id, op in sorted(ir_ops.items()):
        if op.get("op_type") in ("MODEL_INPUT", "MODEL_OUTPUT"):
            continue
        if op_id in absorbed:
            continue
        if op_id in exempt:
            # 合法性已由 `_invalid_exemptions` 逐条核对，这里只跳过产物检查。
            continue
        entry = ops.get(str(op_id)) or {}
        path_text = entry.get("pimir_path")
        if not path_text:
            missing.append(op_id)
            continue
        if not strict:
            continue
        path = Path(path_text)
        if not path.is_file():
            unreadable.append(op_id)
            continue
        digest = entry.get("pimir_sha256")
        if not digest:
            unhashed.append(op_id)
            continue
        cached = pimir_cache.get(path_text)
        if cached is None:
            try:
                data = path.read_bytes()
            except OSError:
                unreadable.append(op_id)
                continue
            cached = (
                hashlib.sha256(data).hexdigest(),
                _pimir_unit_facts(data.decode("utf-8", "ignore")),
            )
            pimir_cache[path_text] = cached
        if cached[0] != digest:
            stale.append(op_id)

    problems = []
    if missing:
        kinds = sorted({
            str(ir_ops[op_id].get("op_type", "unknown"))
            for op_id in missing
        })
        problems.append(
            f"{len(missing)} 个算子（{', '.join(kinds)}）没有 pimir_path"
            f"（如 op{missing[:8]}）")
    if unreadable:
        problems.append(
            f"{len(unreadable)} 个 pimir_path 读不到（如 op{unreadable[:5]}）")
    if unhashed:
        problems.append(
            f"{len(unhashed)} 个缺 pimir_sha256（如 op{unhashed[:5]}）")
    if stale:
        problems.append(f"{len(stale)} 个哈希不符（如 op{stale[:5]}）")
    if illegitimate:
        problems.append(
            f"{len(illegitimate)} 个豁免与源 IR 的落点不符（如 "
            f"op{illegitimate[:5]}）：只有模型边界与 host 执行（带导出时"
            "写下的 device_hint=cpu 证据）可以没有 pimir")

    mismatched = _moe_identity_problems(ir, sidecar)
    if mismatched:
        problems.append(
            f"{len(mismatched)} 个专家 GEMM 的 (层, 角色, 专家, 权重) "
            f"身份不一致（如 {', '.join(mismatched[:3])}）")
    class_problems = _entry_class_problems(ir_ops, ops, strict=strict,
                                           sidecar=sidecar)
    if class_problems:
        problems.append(
            f"{len(class_problems)} 条条目的类别或字段与源 IR 矛盾（如 "
            f"{class_problems[0]}）：GEMM 走 A 路，其余算子都是 B 路，"
            "不能用 shards/device_hint 自报改写类别或跳过 host 证据")
    unit_problems = _bpath_unit_contract_problems(
        ir, sidecar, strict=strict,
        pimir_facts={path: facts for path, (_digest, facts)
                     in pimir_cache.items()})
    if unit_problems:
        problems.append(
            f"{len(unit_problems)} 条 B 路条目的编译单位不完整或口径不符"
            f"（如 {unit_problems[0]}）：严格产物必须回传 trace_tokens 与"
            "保留 Tq/Tp+Tq 符号的形状模板，请用当前主链重新导出 sidecar")
    if strict:
        apath_problems = _apath_unit_problems(
            ir, sidecar,
            pimir_facts={path: facts for path, (_digest, facts)
                         in pimir_cache.items()})
        if apath_problems:
            problems.append(
                f"{len(apath_problems)} 条 A 路 GEMM 的 pimir 语义/形状不符"
                f"（如 {apath_problems[0]}）：pimir 必须是该分片的实测分块"
                "GEMM，分块与分片宽度、融合激活都要与实产一致")

    if problems:
        raise StepFailed(
            "sidecar 对源 IR 的覆盖不完整：" + "；".join(problems)
            + "。覆盖与算子最终落到 GPU/CPU/PIM 无关；合法没有原语的节点"
            "必须写入 sidecar 的 pimir_exempt_ops，且未知设备算子不得豁免。")

    if strict:
        extra = sorted(
            op_id for op_id in (int(raw) for raw in ops)
            if op_id not in ir_ops)
        if extra:
            raise StepFailed(
                f"sidecar 引用了源 IR 里不存在的算子 op{extra[:8]}："
                "op_id 编号已经错位，请用当前 IR 重新导出。")


def fused_kernels_missing_act_spec(sidecar: Dict[str, Any]) -> list[str]:
    """融合清单非空时，折激活的投影 kernel 必须带 act_spec。

    旧产物把同形状的 gate 与 up 编成同一份不带激活的 kernel，仿真时延取自
    那份 trace，激活一分不付。
    """
    from genesim_bridge.ir_cost import has_fused_act_spec

    fused_roles = {entry["role"] for entry in sidecar.get("fused_activations") or []}
    if not fused_roles:
        return []
    missing = []
    seen = set()
    for entry in sidecar.get("operators", {}).values():
        if entry.get("semantic_role") not in fused_roles:
            continue
        path = entry.get("pimir_path")
        if not path or path in seen or not Path(path).is_file():
            continue
        seen.add(path)
        if not has_fused_act_spec(Path(path).read_text()):
            missing.append(path)
    return missing


def step_bcd_export(
    num_stages: int, num_dpus: int, log_dir: Path, model_dir: Path,
    arch: Architecture, *, plan_path: Optional[Path] = None,
    moe_execution: str = "", ir_path: Optional[Path] = None,
    out_dir: Optional[Path] = None,
) -> Path:
    """(B)(C)(D) 图编译 + 算子编译 + 导出 sidecar，核对各类字段都写了。

    `plan_path` 给了就按 GeneSim 的方案编译（段数由方案决定），否则用 --num-stages。
    模型类型、加载器与产物名都从架构注册表派生，不再固定 Llama2。
    `ir_path`/`out_dir` 给了就用本次运行的图骨架与专属产物目录，不写共享位置。
    """
    print("\n[B+C+D] 图编译 → 算子编译（真实分块）→ placement sidecar")
    strategy_args = (
        ["--partition-plan", str(plan_path)] if plan_path is not None
        else ["--num-stages", str(num_stages), "--num-dpus", str(num_dpus)]
    )
    export_cmd = ["python", "scripts/export_pp_placement.py",
                  "--model-dir", str(model_dir), *strategy_args,
                  "--measure-kernel-tiles"]
    if moe_execution:
        export_cmd += ["--moe-execution", moe_execution]
    if ir_path is not None:
        export_cmd += ["--ir", str(ir_path)]
    if out_dir is not None:
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        export_cmd += ["--out-dir", str(out_dir)]
    _run(export_cmd, cwd=_REPO_ROOT, log_path=log_dir / "bcd_export.log")

    models_dir = Path(out_dir) if out_dir is not None else genesim_models_dir()
    tp_width = num_dpus // num_stages
    name = f"tp{tp_width}_pp{num_stages}"
    sidecar_path = models_dir / f"{arch.artifact}_{name}_placement.json"
    if not sidecar_path.is_file():
        raise StepFailed(f"没找到 sidecar：{sidecar_path}")

    sidecar = json.loads(sidecar_path.read_text())
    ops = sidecar["operators"]
    if not ops:
        raise StepFailed("sidecar 里没有任何放置结果")

    # 覆盖校验以源 IR 为准，与算子最终落到哪种设备无关：条目缺失的算子
    # 改放到 GPU 后不会从校验对象里消失。skip-simulation 也依赖这一条。
    source_ir_path = (
        Path(ir_path) if ir_path is not None
        else models_dir / f"{arch.artifact}.ir")
    if not source_ir_path.is_file():
        raise StepFailed(
            f"源 IR 不存在：{source_ir_path}。覆盖校验必须对着生成 sidecar "
            "的那份图骨架做，请先跑 (A) 步骤。")
    # 用放置后的 IR 做审计：它带着 export_gemm 按 spec.device 写下的真实
    # 落点（device_hint=pim/cpu），host GEMM 的豁免才有证据。结构指纹必须
    # 与图骨架一致——放置只允许改设备/成本字段。
    source_ir = json.loads(source_ir_path.read_text())
    placed_ir_path = _placed_ir_of(sidecar_path)
    if placed_ir_path is not None and placed_ir_path.is_file():
        placed_ir = json.loads(placed_ir_path.read_text())
        if fingerprint_ir(placed_ir) != fingerprint_ir(source_ir):
            raise StepFailed(
                f"放置 IR（{placed_ir_path.name}）与图骨架的结构指纹不一致："
                "放置只允许改设备/成本字段，结构变了说明两份产物不同源。")
        source_ir = placed_ir
    # B+C+D 只产出融合清单；`absorbed_ops` 由 A2 精化回写。这一步按导出阶段
    # 校验，吸收集合与清单的双向闭合交给 A2 之后的各复核入口。
    validate_sidecar_contract(
        source_ir, sidecar,
        require_absorbed_closure=False, strict_pimir=True)

    # sidecar 与源 IR 自洽还不够：top-k/专家宽来自 IR，写错的 IR 会让两者
    # 一起错。这里再拿 HF config 派生的身份逐字段核对，守住 A 段之后的
    # 第二条入口。
    _validate_moe_identity(
        sidecar.get("moe_config") or {},
        json.loads((Path(model_dir) / "config.json").read_text()),
        stage="放置 sidecar")

    # 本步骤用 --measure-kernel-tiles 编了算子，所以 sidecar 必须自报带产物。
    # GeneSim 靠这个字段自动进严格模式；缺了就退回"看配置文件写没写对"，也就是
    # 本脚本要防的那种静默退化。
    if not sidecar.get("requires_pimir"):
        raise StepFailed(
            "sidecar 没有 requires_pimir=true。带算子编译产物的导出必须自报，"
            "否则 GeneSim 不会自动进严格模式，pim mlir 读不到会静默退回手写模板。"
        )

    ir_ops = {int(op["op_id"]): op for op in source_ir.get("operators", [])}
    gemm_count, bpath_count = check_sidecar_entries(
        ir_ops, ops, arch.required_roles, sidecar=sidecar)
    print(f"    条目分两类核对：GEMM {gemm_count} 条、算子级 {bpath_count} 条")

    # MoE：执行语义与专家身份都必须与主链选定的那一份一致。sidecar 的
    # moe_config.execution 来自编译配置，不是从 IR 抄的。
    if moe_execution:
        declared_exec = str(
            (sidecar.get("moe_config") or {}).get("execution", ""))
        if declared_exec != moe_execution:
            raise StepFailed(
                f"sidecar 的 moe_config.execution={declared_exec!r} 与主链选定的 "
                f"{moe_execution!r} 不一致：仿真会按另一种语义计费。")
    if int(sidecar.get("moe_config", {}).get("num_experts", 0)) > 0:
        expert_entries = [e for e in ops.values() if e.get("expert_id") is not None]
        experts = {int(e["expert_id"]) for e in expert_entries}
        declared = int(sidecar["moe_config"]["num_experts"])
        if experts != set(range(declared)):
            raise StepFailed(
                f"sidecar 的专家号 {sorted(experts)[:8]} 未覆盖 0..{declared - 1}")
        print(f"    MoE: {declared} 专家 top-{sidecar['moe_config']['top_k']}，"
              f"专家 GEMM {len(expert_entries)} 条")

    pimir_files = {entry["pimir_path"] for entry in ops.values()}
    absent = [p for p in pimir_files if not Path(p).is_file()]
    if absent:
        raise StepFailed(f"pimir_path 指向的文件不存在：{absent[:3]}")

    # 内容哈希要和磁盘上的文件对得上：sidecar 记的是绝对路径，如果缓存被重建过、
    # 而 sidecar 还是旧的，路径可能存在但内容已经变了。哈希不符说明这份 sidecar
    # 与当前的算子编译产物不是一套。
    #
    # 本步骤自己刚产出 sidecar（带 --measure-kernel-tiles），缺哈希说明导出实现
    # 有缺陷或产物被替换过，不能沿用"旧产物"豁免放行——那会让严格模式在
    # --skip-simulation 这条路上形同虚设。旧 sidecar 应重新导出，而不是在这里
    # 被接受。
    without_hash = [op_id for op_id, e in ops.items() if not e.get("pimir_sha256")]
    if without_hash:
        raise StepFailed(
            f"{len(without_hash)}/{len(ops)} 个算子的 sidecar 没有 pimir_sha256"
            f"（如 op{without_hash[:5]}）：本次导出没有写全内容身份，"
            "无法确认 sidecar 与 pim mlir 是同一套。请检查放置导出实现后重跑。"
        )
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
    stale_fused = fused_kernels_missing_act_spec(sidecar)
    if stale_fused:
        raise StepFailed(
            "融合清单非空，但折激活的投影 kernel 没有 act_spec："
            f"{stale_fused[:3]}。按当前代码重跑放置导出。"
        )
    return sidecar_path


def _config_text(genesim: Path, config: str | Path) -> str:
    """读仿真配置的文本；调用方明确给了配置却不存在时直接失败。

    传文件名时从 genesim 的 conf 目录找；传路径时直接读，
    临时收过请求数的配置不在那个目录里。配置本身是严格性的唯一来源，
    缺失时不能按「没有可核对对象」放行，否则三个复核入口都能空跑。
    """
    path = config if isinstance(config, Path) else genesim / "conf" / config
    if not path.is_file():
        raise StepFailed(
            f"仿真配置不存在：{path}。复核入口必须先拿到本次配置，"
            "才能判定严格性与待核对产物。")
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


def _generate_synthetic_trace(
    genesim: Path, config: Dict[str, Any], out_path: Path,
) -> Path:
    """配置引用的请求分布缺失时，在本次运行目录生成一份合成 trace。

    trace 只描述请求数与 token 长度，与模型结构无关。用 GeneSim 的官方
    生成脚本与确定性种子，保证格式、版本与仿真加载器一致。
    """
    trace_cfg = config.get("trace") or {}
    requests = int(trace_cfg.get("num_requests", 10) or 10)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    _run([sys.executable, "scripts/trace_generator.py", "--synthetic",
          "--num_requests", str(requests), "--seed", "0",
          "--output", str(out_path), "--log_level", "WARNING"],
         cwd=genesim, log_path=out_path.with_name(
             out_path.name + ".generate.log"))
    if not out_path.is_file() or out_path.stat().st_size == 0:
        raise StepFailed(
            f"合成的请求分布 trace 没有生成：{out_path}。E 阶段不能在缺少 "
            "trace 输入时启动仿真。")
    return out_path


def prepare_simulation_config(
    source: Path, conf_dir: Path, *, max_requests: int | None = None,
    placement: Path | None = None, ir_path: Path | None = None,
    traces_dir: Path | None = None, results_dir: Path | None = None,
    output_name: str | None = None, genesim: Path | None = None,
) -> Path:
    """复制一份配置到本次运行的日志目录，绑定本次放置与专属输出目录。

    模板里的 `compiler_placement_file`、`model.ir_path` 与 `output.*` 都是
    静态默认值；主链必须改成本次运行的产物与独立目录，并回读复核。否则
    `tp1_pp4` 会消费模板里写死的 `tp4_pp2` 产物，两次运行还会互删共享
    `pim_traces`/`results`。trace 目录必须是本次的空目录，仿真才会全量重编。
    `genesim` 给定时，配置引用的请求分布 trace 若不存在（干净检出里
    `traces/` 被 gitignore），就在本次运行目录生成并绑定绝对路径。
    """
    import yaml

    loaded = yaml.safe_load(Path(source).read_text()) or {}
    if max_requests is not None:
        loaded.setdefault("trace", {})["num_requests"] = max_requests
    trace_input: Path | None = None
    if genesim is not None:
        configured = (loaded.get("trace") or {}).get("path")
        if configured:
            trace_path = Path(str(configured))
            if not trace_path.is_absolute():
                trace_path = Path(genesim) / trace_path
            if not trace_path.is_file():
                trace_input = _generate_synthetic_trace(
                    Path(genesim), loaded,
                    Path(conf_dir) / "traces"
                    / f"{Path(source).stem}.trace")
                loaded.setdefault("trace", {})["path"] = str(
                    trace_input.resolve())
    if placement is not None:
        loaded.setdefault("scheduler", {})["compiler_placement_file"] = str(
            Path(placement).resolve())
    if ir_path is not None:
        loaded.setdefault("model", {})["ir_path"] = str(
            Path(ir_path).resolve())
    if traces_dir is not None:
        traces_path = Path(traces_dir)
        if traces_path.exists() and any(traces_path.glob("op_*.pim_trace")):
            raise StepFailed(
                f"本次运行的 trace 目录已有 trace：{traces_path}。"
                "每个运行号必须使用独立空目录，不能复用别的任务的产物。")
        traces_path.mkdir(parents=True, exist_ok=True)
        loaded.setdefault("output", {})["pim_traces_dir"] = str(
            traces_path.resolve())
    if results_dir is not None:
        results_path = Path(results_dir)
        results_path.mkdir(parents=True, exist_ok=True)
        loaded.setdefault("output", {})["results_dir"] = str(
            results_path.resolve())
        # 日志也写进本次运行目录：共享 results/genesim.log 会被并发任务交错写。
        loaded["output"]["log_file"] = str(
            (results_path / "genesim.log").resolve())
    conf_dir.mkdir(parents=True, exist_ok=True)
    out = conf_dir / (output_name or Path(source).name)
    out.write_text(yaml.safe_dump(loaded, sort_keys=False))
    written = yaml.safe_load(out.read_text()) or {}
    if placement is not None:
        declared = (written.get("scheduler") or {}).get(
            "compiler_placement_file")
        if not declared or Path(declared).resolve() != Path(placement).resolve():
            raise StepFailed(
                f"临时仿真配置没有绑定本次放置产物：期望 {placement}，"
                f"实际 {declared!r}。")
    for section, key, wanted in (
            ("model", "ir_path", ir_path),
            ("trace", "path", trace_input),
            ("output", "pim_traces_dir", traces_dir),
            ("output", "results_dir", results_dir),
            ("output", "log_file",
             None if results_dir is None else Path(results_dir) / "genesim.log")):
        if wanted is None:
            continue
        declared = (written.get(section) or {}).get(key)
        if not declared or Path(declared).resolve() != Path(wanted).resolve():
            raise StepFailed(
                f"临时仿真配置没有绑定本次运行的 {section}.{key}："
                f"期望 {wanted}，实际 {declared!r}。")
    return out


def limited_config(source: Path, conf_dir: Path, max_requests: int) -> Path:
    """复制一份配置，把请求数收成 max_requests，原文件不动。

    仿真时间跟着 token 数走。10 个请求共 8084 个 token 要二十多分钟，
    而第 1 个请求已经同时有 prefill 和 decode。原配置留着，
    需要完整数字时显式把请求数加回去。
    """
    return prepare_simulation_config(
        source, conf_dir, max_requests=max_requests)


def run_directory(
    log_dir: Path, artifact: str, num_dpus: int, num_stages: int,
    *, moe_execution: str = "", run_id: str = "",
) -> Path:
    """本次运行的专属目录，按架构/策略/语义/运行号派生，跨任务互不覆盖。

    只有架构与策略时旧调用仍得到原来的名字；主链会传入执行语义与唯一
    运行号，同策略的并发运行也不会共用产物。
    """
    tp_width = num_dpus // num_stages
    parts = [artifact, f"tp{tp_width}_pp{num_stages}"]
    if moe_execution:
        parts.append(moe_execution)
    if run_id:
        parts.append(run_id)
    return Path(log_dir) / "_".join(parts)


def _new_run_id() -> str:
    """返回进程级唯一运行号，避免同策略的并发运行互相覆盖。"""
    return time.strftime("%Y%m%d-%H%M%S") + f"-{os.getpid()}"


def _configured_dir(
    genesim: Path, loaded: Dict[str, Any], key: str, default: str,
) -> Path:
    """从仿真配置解析输出目录；相对路径按 GeneSim 根解析。"""
    value = (loaded.get("output") or {}).get(key, default)
    path = Path(value)
    return path if path.is_absolute() else genesim / path


def _ir_layer_of(ir: Dict[str, Any]) -> Dict[int, int]:
    """算子编号 → 层号；不在任何层子图里的模型级算子不出现。"""
    return {
        int(op_id): layer
        for layer, group in enumerate(ir.get("subgraphs") or [])
        for op_id in group
    }


def _assert_placement_matches_run(
    sidecar: Dict[str, Any], ir: Dict[str, Any], *,
    plan: Any = None, num_dpus: int | None = None,
    num_stages: int | None = None,
) -> None:
    """核对 sidecar 的切分身份就是本次运行的策略。

    结构指纹刻意不含放置字段，所以另一份合法 sidecar 同样能通过指纹。
    这里按 PartitionPlan/命令行的 DPU 数、段数、Cluster 映射、逐 GEMM
    分片数与本地形状、逐层所属流水段逐项核对，错配在仿真启动前失败。
    """
    if plan is not None:
        if num_dpus is None:
            num_dpus = int(plan.num_dpus)
        if num_stages is None:
            num_stages = int(plan.num_stages)
        if (int(plan.num_dpus), int(plan.num_stages)) != (
                int(num_dpus), int(num_stages)):
            raise StepFailed(
                f"切分方案 ({plan.num_dpus}, {plan.num_stages}) 与本次请求 "
                f"({num_dpus}, {num_stages}) 不一致。")
    if num_dpus is None or num_stages is None:
        raise StepFailed("核对放置产物需要 DPU 数与流水段数。")
    if num_dpus % num_stages:
        raise StepFailed(
            f"num_stages={num_stages} 不能整除 num_dpus={num_dpus}")
    tp_width = num_dpus // num_stages
    dpu_ids = tuple(plan.dpu_ids) if plan is not None and plan.dpu_ids else tuple(
        range(num_dpus))
    if len(dpu_ids) != num_dpus:
        raise StepFailed(f"方案的 DPU 编号有 {len(dpu_ids)} 个，应为 {num_dpus}")
    if plan is not None and plan.dpu_to_cluster is not None:
        declared = sidecar.get("dpu_to_cluster")
        expected = [list(entry) for entry in plan.dpu_to_cluster]
        if declared != expected:
            raise StepFailed(
                f"sidecar 的 Cluster 映射与本次方案不一致：期望 {expected}，"
                f"实际 {declared!r}。")
    if int(ir.get("num_layers", len(ir.get("subgraphs") or []))) != len(
            ir.get("subgraphs") or []):
        raise StepFailed("源 IR 的 num_layers 与层子图数量不一致。")
    num_layers = len(ir.get("subgraphs") or [])
    if num_layers % num_stages:
        raise StepFailed(
            f"num_layers={num_layers} 不能被 num_stages={num_stages} 整除")
    per_stage = num_layers // num_stages
    stage_width = num_dpus // num_stages
    ops = {int(op["op_id"]): op for op in ir.get("operators", [])}
    layer_of = _ir_layer_of(ir)
    problems: list[str] = []
    for raw_op_id, entry in (sidecar.get("operators") or {}).items():
        op_id = int(raw_op_id)
        op = ops.get(op_id)
        if op is None:
            continue
        shards = entry.get("shards") or []
        if op.get("op_type") != "GEMM":
            if shards:
                problems.append(f"op{op_id} 是非 GEMM 却带分片")
            continue
        if not shards:
            if not _entry_is_host(op_id, op, entry, sidecar):
                problems.append(f"op{op_id} 是设备 GEMM 却没有分片")
            continue
        if len(shards) != tp_width:
            problems.append(
                f"op{op_id} 的分片数 {len(shards)} != tp_width={tp_width}")
            continue
        layer = layer_of.get(op_id)
        if layer is None:
            stage = num_stages - 1
        else:
            stage = layer // per_stage
        expected_dpus = set(
            dpu_ids[stage * stage_width:(stage + 1) * stage_width])
        got_dpus = {int(shard["dpu_id"]) for shard in shards}
        if got_dpus != expected_dpus:
            problems.append(
                f"op{op_id} 的分片落在 {sorted(got_dpus)}，"
                f"期望 stage{stage} 的 {sorted(expected_dpus)}")
            continue
        in_width = int(op.get("input_shapes", [[0]])[0][-1])
        out_width = int(op.get("output_shapes", [[0]])[0][-1])
        for shard in shards:
            local_in = int(shard.get("local_in_features", 0))
            local_out = int(shard.get("local_out_features", 0))
            col_ok = (
                local_in == in_width and out_width % tp_width == 0
                and local_out == out_width // tp_width)
            row_ok = (
                local_out == out_width and in_width % tp_width == 0
                and local_in == in_width // tp_width)
            if not (col_ok or row_ok):
                problems.append(
                    f"op{op_id} 的本地形状 {local_in}×{local_out} 与全局 "
                    f"{in_width}×{out_width}、tp_width={tp_width} 不符")
                break
    if problems:
        raise StepFailed(
            f"放置产物与本次切分不一致（{len(problems)} 项，如 "
            f"{'；'.join(problems[:3])}）：仿真不能消费另一份策略的 sidecar。")


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
    genesim: Path, config_name: str | Path, log_dir: Path, *, label: str,
    results_dir: Optional[Path] = None, max_requests: Optional[int] = None,
    moe_execution: str = "", placement: Optional[Path] = None,
    plan_path: Optional[Path] = None, num_dpus: Optional[int] = None,
    num_stages: Optional[int] = None, ir_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """(E) 跑仿真，核对 trace 真的来自 pim mlir。

    `placement` 是本次 BCD 返回的 sidecar；`ir_path` 是本次精化 IR。给了就
    把它们写进临时配置并复核。trace 与 results 使用本次运行号下的独立目录，
    不清空也不覆盖 GeneSim 的共享 `pim_traces`/`results`。
    """
    print(f"\n[E] GeneSim 仿真（{label}）")
    source = config_name if isinstance(config_name, Path) else (
        genesim / "conf" / config_name)
    if not source.is_file():
        raise StepFailed(f"仿真配置不存在：{source}")
    if results_dir is None:
        results_dir = log_dir / f"results_{label}"
    traces_dir = log_dir / f"pim_traces_{label}"
    config_path = prepare_simulation_config(
        source, log_dir, max_requests=max_requests, placement=placement,
        ir_path=ir_path, traces_dir=traces_dir, results_dir=results_dir,
        output_name=f"{source.stem}_{label}{source.suffix}", genesim=genesim)
    if max_requests is not None:
        # 临时配置放日志目录，不写进 genesim 仓，免得留下一份请求数被改过的副本。
        print(f"    请求数收成 {max_requests}：{config_path.name}")
    if placement is not None:
        sidecar = json.loads(Path(placement).read_text())
        import yaml
        loaded = yaml.safe_load(config_path.read_text()) or {}
        ir_rel = (loaded.get("model") or {}).get("ir_path")
        ir_path = genesim / str(ir_rel)
        if not ir_path.is_file():
            raise StepFailed(
                f"配置指向的仿真 IR 不存在：{ir_path}，无法核对放置产物。")
        plan = None
        if plan_path is not None:
            from contracts.partition_plan import PartitionPlan

            plan = PartitionPlan.read(Path(plan_path))
        expected_dpus = int(plan.num_dpus) if plan is not None else num_dpus
        expected_stages = int(plan.num_stages) if plan is not None else num_stages
        if expected_dpus and expected_stages:
            tag = f"tp{expected_dpus // expected_stages}_pp{expected_stages}"
            if tag not in Path(placement).name:
                raise StepFailed(
                    f"放置产物 {Path(placement).name} 不是本次 {tag} 的产物："
                    "仿真不能消费另一份策略的 sidecar。")
        _assert_placement_matches_run(
            sidecar, json.loads(ir_path.read_text()), plan=plan,
            num_dpus=num_dpus, num_stages=num_stages)
        print(f"    放置产物与本次切分一致：{Path(placement).name}")
    # sidecar 与磁盘产物不是一套时，仿真跑出来的数字没有意义，先拦下来，
    # 省掉十几分钟的无效仿真。
    verify_sidecar_freshness(genesim, config_path)
    # 仿真 IR 与 sidecar 不是同一套算子编号时，注意力算子会整批退回手写
    # 模板，同样在启动前拦下。
    verify_sim_ir_matches_sidecar(genesim, config_path)
    # trace 目录是本次运行号下的独立空目录（prepare_simulation_config 已
    # 校验），不需要删除共享目录，也不会误清别的任务的 trace。
    _run(_genesim_sim_command(genesim, config_path),
         cwd=genesim, log_path=log_dir / f"e_sim_{label}.log")

    # 来源校验挂在仿真入口上，而不是只挂在主流程里。慢速仿真测试也走这里，
    # 整批 trace 退回手写模板时必须在这里失败，不能只在主流程里才看得见。
    verify_trace_provenance(genesim, config_path)

    summary_path = Path(results_dir) / "summary.json"
    if not summary_path.is_file():
        raise StepFailed(f"仿真没产出 {summary_path}")
    summary = json.loads(summary_path.read_text())
    if moe_execution:
        # 仿真实际采用的语义必须与编译配置一致：summary 是 GeneSim 从加载的
        # IR 里读出来的，不是主链自己写进去的副本。
        got = ((summary.get("model") or {}).get("moe") or {}).get("execution")
        if got != moe_execution:
            raise StepFailed(
                f"GeneSim summary 的 moe.execution={got!r} 与编译配置 "
                f"{moe_execution!r} 不一致：仿真加载的 IR 不是主链产物。")

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
    能发现"这份 sidecar 与当前算子编译产物不是一套"的办法。缺路径或读不到
    文件同样说明产物身份不成立，统一转成 `StepFailed` 并给出 op 与路径。
    """
    stale = []
    for op_id, entry in ops.items():
        path_text = entry.get("pimir_path")
        if not path_text:
            raise StepFailed(
                f"op{op_id} 的 sidecar 条目没有 pimir_path：无法确认它与当前"
                "算子编译产物是一套，请重新导出。")
        try:
            data = Path(path_text).read_bytes()
        except OSError as exc:
            raise StepFailed(
                f"op{op_id} 的 pimir_path 读不到：{path_text}"
                f"（{exc.strerror or exc}）：sidecar 与算子编译产物不是一套，"
                "请重新导出。") from exc
        if hashlib.sha256(data).hexdigest() != entry.get("pimir_sha256"):
            stale.append(op_id)
    return stale


def _stale_sidecar_message(stale: list[str]) -> str:
    return (
        f"{len(stale)} 个算子的 pimir_sha256 与磁盘上的 pim mlir 不一致"
        f"（如 op{stale[:5]}）：sidecar 与当前算子编译产物不是一套，"
        "请重新导出（scripts/run_full_pipeline.py 不带 --skip-simulation，"
        "或单独跑 scripts/export_pp_placement.py）。"
    )


def _strict_sidecar_context(loaded: Dict[str, Any],
                            sidecar: Dict[str, Any] | None) -> bool:
    """严格性 = 仿真配置显式要求 ∪ sidecar 自报；缺一不能降级。"""
    return bool(_config_requires_pimir(loaded)) or bool(
        (sidecar or {}).get("requires_pimir"))


def _require_simulation_input(what: str, strict: bool) -> None:
    """严格模式下 IR/sidecar 是必需输入；缺失时不能按旧式兼容跳过。"""
    if strict:
        raise StepFailed(
            f"严格模式下配置缺少 {what}：无法完成同源核对。"
            "请补齐本次运行的仿真配置，或显式关闭 require_compiler_pimir。")


def verify_sidecar_freshness(genesim: Path, config: str | Path) -> None:
    """仿真用的 sidecar 必须与磁盘上的 pim mlir 是一套。

    主流程的哈希核对只在导出步骤里，仿真入口绕得开它：缓存被重建过、
    sidecar 还是旧的时，仿真照样跑完，代价链吃的却不是当前的产物。
    严格性（配置显式要求或 sidecar 自报）一旦成立，配置键与文件都必须存在；
    只有配置与产物都明确非严格时才允许无 sidecar 的旧式用法。
    """
    import yaml

    config_text = _config_text(genesim, config)
    loaded = yaml.safe_load(config_text) or {}
    config_strict = _config_requires_pimir(loaded)
    sidecar_path = _sidecar_of(genesim, config_text)
    if sidecar_path is None:
        _require_simulation_input("sidecar 配置键", config_strict)
        return
    if not sidecar_path.is_file():
        raise StepFailed(f"配置指向的 sidecar 不存在：{sidecar_path}")
    sidecar_data = json.loads(sidecar_path.read_text())
    strict = _strict_sidecar_context(loaded, sidecar_data)
    ops = sidecar_data["operators"]
    without_hash = [op_id for op_id, e in ops.items() if not e.get("pimir_sha256")]
    if without_hash:
        raise StepFailed(
            f"{len(without_hash)}/{len(ops)} 个算子的 sidecar 没有 pimir_sha256，"
            "无法确认它与当前算子编译产物是一套，请重新导出。"
        )
    stale = stale_pimir_entries(ops)
    if stale:
        raise StepFailed(_stale_sidecar_message(stale))
    # 哈希只证明文件没变，不能证明 MoE 元数据、豁免、吸收与专家身份没被
    # 换过：仿真前复核用与主编译相同的完整产物契约。
    ir_rel = (loaded.get("model") or {}).get("ir_path")
    if not ir_rel:
        _require_simulation_input("仿真 IR 配置键", strict)
        return
    ir_path = genesim / ir_rel
    if not ir_path.is_file():
        raise StepFailed(
            f"配置指向的仿真 IR 不存在：{ir_path}。先跑 (A2) 精化步骤生成"
            "（scripts/run_full_pipeline.py 主流程已包含这一步）。")
    validate_sidecar_contract(
        json.loads(ir_path.read_text()), sidecar_data,
        strict_pimir=config_strict)


def verify_sim_ir_matches_sidecar(genesim: Path, config: str | Path) -> None:
    """仿真加载的 IR 与 sidecar 必须是同一套算子编号。

    调度器按 op_id 查 sidecar 里的 pimir_path：精化 IR 是旧结构、sidecar 是
    新导出时，查不到的算子静默退回手写模板，仿真照样跑完（2026-10-07 的
    6144 个模板 trace 就是这么来的）。模型入口/出口没有设备算子，本来就不
    在 sidecar 里，其余算子缺一个都拦在仿真启动前，省十几分钟无效仿真。
    严格性成立时 IR 与 sidecar 的配置键和文件都是必需输入；只有配置与产物
    都明确非严格时，才允许没有可对拍对象的旧式用法。
    """
    import yaml

    config_text = _config_text(genesim, config)
    loaded = yaml.safe_load(config_text) or {}
    config_strict = _config_requires_pimir(loaded)
    ir_rel = (loaded.get("model") or {}).get("ir_path")
    sidecar_path = _sidecar_of(genesim, config_text)
    if sidecar_path is None:
        _require_simulation_input("sidecar 配置键", config_strict)
        return
    if not sidecar_path.is_file():
        raise StepFailed(f"配置指向的 sidecar 不存在：{sidecar_path}")
    sidecar_data = json.loads(sidecar_path.read_text())
    strict = _strict_sidecar_context(loaded, sidecar_data)
    if not ir_rel:
        _require_simulation_input("仿真 IR 配置键", strict)
        return
    ir_path = genesim / ir_rel
    if not ir_path.is_file():
        raise StepFailed(
            f"配置指向的仿真 IR 不存在：{ir_path}。先跑 (A2) 精化步骤生成"
            "（scripts/run_full_pipeline.py 主流程已包含这一步）。")
    ir = json.loads(ir_path.read_text())
    # 仿真 IR 与 sidecar 不仅要编号对得上，还要共用完整的产物契约：
    # MoE 元数据、结构指纹、容量、豁免、吸收与专家身份缺一项都不行。
    try:
        validate_sidecar_contract(
            ir, sidecar_data, strict_pimir=config_strict)
    except StepFailed as exc:
        raise StepFailed(
            f"{exc}（仿真 IR {ir_path.name} 与 sidecar {sidecar_path.name}"
            " 不是同一套产物；精化 IR 多半是图骨架改结构之前生成的旧产物，"
            "重跑 scripts/run_full_pipeline.py 让 (A2) 重新精化）") from exc


# B 路 pimir 的 trace 以「1 token × 本地宽」为单位，运行期按实际 Tq 重复。
# 这两个集合里的单相算子，trace 字节与声明形状一一对应，可以精确核对；
# 多相算子（ROPE/QUANT/PAGED_ATTN）的中间相会额外写工作区，只核 token 数
# 与形状。集合与 GeneSim `_validate_pimir_unit` 保持同一口径。
_UNIT_BALANCED_BYTES_OPS = frozenset(
    {"SILU", "GELU", "VECTOR_ADD", "VECTOR_MUL"})
_UNIT_OUTPUT_BYTES_OPS = frozenset({
    "RMSNORM", "TRANSPOSE", "RESHAPE", "CONCAT", "MASK", "GATHER",
})


def _unit_shape_conflict(actual_shapes, declared_shapes, kind: str) -> str | None:
    """比较 trace/IR 形状与 sidecar 编译单位形状。

    符号维（Tq / Tp+Tq）表示 token 维与历史维，由重复次数和 KV 块循环
    分别绑定，不在这里比较；只有两侧都是具体整数的特征维/本地宽必须逐一
    相等。GeneSim 的 `_unit_shape_conflict` 使用同一语义，两个消费方不能
    一个按符号、一个按绑定形状各判一套。
    """
    if len(actual_shapes) != len(declared_shapes):
        return (f"{kind}形状数量 {len(actual_shapes)} != "
                f"编译单位的 {len(declared_shapes)}")
    for index, (actual, declared) in enumerate(
            zip(actual_shapes, declared_shapes)):
        if len(actual) != len(declared):
            return f"{kind}[{index}] 秩 {len(actual)} != {len(declared)}"
        for position, (left, right) in enumerate(zip(actual, declared)):
            if (isinstance(left, int) and not isinstance(left, bool)
                    and isinstance(right, int)
                    and not isinstance(right, bool)
                    and left != right):
                suffix = "末维" if position == len(actual) - 1 else (
                    f"第 {position} 维")
                return f"{kind}[{index}] {suffix} {left} != {right}"
    return None


def _unit_shape_role_problems(ir_op, entry, label: str) -> List[str]:
    """声明单位必须保留源 IR 的符号维；特征维只要求是正整数本地宽。"""
    problems: List[str] = []
    for kind, key in (("输入", "input_shapes"), ("输出", "output_shapes")):
        ir_shapes = ir_op.get(key) or []
        declared_shapes = entry.get(f"local_{key}") or []
        if len(declared_shapes) != len(ir_shapes):
            problems.append(
                f"{label} 的{kind}形状数量 {len(declared_shapes)} != 源 IR 的 "
                f"{len(ir_shapes)}")
            continue
        for index, (declared, source) in enumerate(
                zip(declared_shapes, ir_shapes)):
            if len(declared) != len(source):
                problems.append(
                    f"{label} 的{kind}[{index}] 秩 {len(declared)} != 源 IR 的 "
                    f"{len(source)}")
                continue
            for position, (want, got) in enumerate(zip(source, declared)):
                if isinstance(want, str):
                    if got != want:
                        problems.append(
                            f"{label} 的{kind}[{index}][{position}] 符号维 "
                            f"{want!r} 被写成 {got!r}（丢失维角色）")
                    continue
                if (not isinstance(got, int) or isinstance(got, bool)
                        or got <= 0):
                    problems.append(
                        f"{label} 的{kind}[{index}][{position}] 特征维 "
                        f"{got!r} 不是正整数本地宽")
    return problems


def _resolved_unit_shape(shape, tokens: int, history: int | None,
                         label: str) -> List[int]:
    """把声明单位的符号维换成编译占位，供字节核对使用。"""
    dims: List[int] = []
    for dim in shape:
        if isinstance(dim, int) and not isinstance(dim, bool):
            dims.append(dim)
        elif dim == "Tq":
            dims.append(tokens)
        elif dim in ("Tp", "Tp+Tq"):
            if history is None:
                raise StepFailed(
                    f"{label} 含历史维 {dim!r}，但 pimir 没有 KV 块循环"
                    "元数据：无法核对编译单位的字节口径。")
            dims.append(history)
        else:
            raise StepFailed(
                f"{label} 含未知符号维 {dim!r}：编译单位只能保留 Tq / "
                "Tp+Tq 两个符号。")
    return dims


def _bpath_unit_problems(
    label: str, entry: Dict[str, Any], meta: Dict[str, Any], *,
    strict: bool, ir_op: Dict[str, Any],
) -> List[str]:
    """核对 B 路 trace 的 token 数、形状与字节是否等于 sidecar 编译单位。

    严格产物必须回传 `trace_tokens` 与 `local_*_shapes`；缺了就没有任何不变式
    能阻止「按全局形状/128 token 编、按本地形状执行」的代价失真。形状模板
    里的 Tq/Tp+Tq 是 token 维与历史维的角色标注，trace 侧只比较特征维。
    条目类别由源 IR 决定：GEMM 的 pimir 按本地分片形状编译，其余算子都必须
    有「1 token × 本地宽」单位；B 路自报的 shards 不能把它变成 GEMM。
    """
    if ir_op.get("op_type") == "GEMM":
        return []
    if entry.get("shards"):
        return [f"{label}（{ir_op.get('op_type')}）是非 GEMM 却带分片："
                "不能用分片 GEMM 的 pimir 跳过编译单位"]
    unit_keys = ("trace_tokens", "local_input_shapes", "local_output_shapes")
    present = [key for key in unit_keys if entry.get(key) is not None]
    if not present:
        if strict:
            return [
                f"{label} 的 sidecar 条目缺 trace_tokens/local_*_shapes："
                "严格产物无法证明 trace 与运行时形状同源"]
        return []
    missing = [key for key in unit_keys if entry.get(key) is None]
    if missing:
        return [f"{label} 的 sidecar 编译单位不完整，缺 {missing}"]

    problems: List[str] = []
    problems.extend(_unit_shape_role_problems(ir_op, entry, label))
    pimir = meta.get("pimir") or {}
    tokens = pimir.get("tokens")
    declared_tokens = int(entry["trace_tokens"])
    if tokens is None or int(tokens) != declared_tokens:
        problems.append(
            f"{label} 的 trace tokens={tokens!r} 与 sidecar 编译单位 "
            f"{declared_tokens} 不一致：运行期会重复计算 token")
    for kind, key in (("输入", "input_shapes"), ("输出", "output_shapes")):
        conflict = _unit_shape_conflict(
            meta.get(key) or [], entry.get(f"local_{key}") or [], kind)
        if conflict:
            problems.append(
                f"{label} 的 trace {conflict}：pimir 不是按该 DPU 的本地"
                "宽度编的")

    op_type = ir_op.get("op_type")
    if pimir.get("phase_count") != 1:
        return problems
    if op_type == "SPLIT":
        # SPLIT 是纯视图：IR 只记一个代表输出形状，pimir 一次搬完整份输入。
        if (int(pimir.get("total_in_bytes", -1))
                != int(pimir.get("total_out_bytes", -2))):
            problems.append(
                f"{label} 的 SPLIT trace 读写字节 "
                f"{pimir.get('total_in_bytes')}/{pimir.get('total_out_bytes')}"
                " 不相等：视图算子的搬入与搬出元素总数必须相同")
        return problems
    if op_type in (_UNIT_BALANCED_BYTES_OPS | _UNIT_OUTPUT_BYTES_OPS):
        history = pimir.get("loop_unit_tokens")

        def total_elems(shapes, kind: str) -> int:
            total = 0
            for shape in shapes:
                dims = _resolved_unit_shape(
                    shape, declared_tokens, history, f"{label} 的{kind}")
                count = 1
                for dim in dims:
                    count *= int(dim)
                total += count
            return total

        expected_out = declared_tokens * total_elems(
            entry["local_output_shapes"], "输出") * 2
        if int(pimir.get("total_out_bytes", -1)) != expected_out:
            problems.append(
                f"{label} 的 trace 输出字节 {pimir.get('total_out_bytes')} != "
                f"编译单位 {expected_out}（{declared_tokens} token × 本地形状"
                " × f16）")
        if op_type in _UNIT_BALANCED_BYTES_OPS:
            expected_in = declared_tokens * total_elems(
                entry["local_input_shapes"], "输入") * 2
            if int(pimir.get("total_in_bytes", -1)) != expected_in:
                problems.append(
                    f"{label} 的 trace 输入字节 {pimir.get('total_in_bytes')} "
                    f"!= 编译单位 {expected_in}")
    return problems


def _trace_source_lineage(
    records: list[Dict[str, Any]], ir: Dict[str, Any],
    sidecar: Dict[str, Any], *, strict: bool,
) -> List[str]:
    """逐条核对 trace 的父算子、类型、编译签名哈希与吸收状态。

    pimir trace 必须回指源 IR 里存在的算子，且 `compile_signature.pimir_sha256`
    与 sidecar 指向的那份 MLIR 相同；absorbed trace 必须是零指令、且编号在
    sidecar 的吸收集合里。只凭 `trace_source` 和父编号不能证明 trace 真是
    由 sidecar 那套产物生成的。
    """
    operators = {int(op["op_id"]): op for op in ir["operators"]}
    absorbed = {int(op_id) for op_id in (sidecar.get("absorbed_ops") or [])}
    entries = sidecar.get("operators") or {}
    problems: List[str] = []

    for meta in records:
        source = meta.get("trace_source") or (
            meta.get("compile_signature") or {}).get("trace_source", "unknown")
        parent = int(meta.get("parent_op_id", -1))
        source_id = parent if parent >= 0 else int(meta.get("op_id", -1))
        label = f"op{source_id}"
        if source not in ("pimir", "absorbed"):
            continue
        if source_id not in operators:
            problems.append(f"{label} 不在源 IR 里")
            continue
        expected_type = operators[source_id].get("op_type")
        if meta.get("op_type") != expected_type:
            problems.append(
                f"{label} 的 trace 类型 {meta.get('op_type')!r} 与源 IR "
                f"{expected_type!r} 不一致")
        expected_expert = operators[source_id].get("expert_id")
        if (expected_expert is not None and "expert_id" in meta
                and int(meta["expert_id"]) != int(expected_expert)):
            problems.append(
                f"{label} 的 trace expert_id={meta['expert_id']!r} 与源 IR "
                f"{expected_expert!r} 不一致")
        if source == "absorbed":
            if source_id not in absorbed:
                problems.append(f"{label} 的 trace 标为 absorbed，但 sidecar 没有吸收编号")
            if int(meta.get("num_instructions", -1)) != 0:
                problems.append(
                    f"{label} 的 absorbed trace 不是空 trace"
                    f"（num_instructions={meta.get('num_instructions')}）")
            continue

        entry = entries.get(str(source_id))
        if entry is None:
            problems.append(f"{label} 有 pimir trace，但 sidecar 没有对应条目")
            continue
        declared = entry.get("pimir_sha256")
        actual = (meta.get("compile_signature") or {}).get("pimir_sha256")
        if not declared:
            if strict:
                problems.append(f"{label} 的 sidecar 条目缺 pimir_sha256")
            continue
        if not actual:
            problems.append(
                f"{label} 的 pimir trace 缺 compile_signature.pimir_sha256")
        elif actual != declared:
            problems.append(
                f"{label} 的 trace 编译签名哈希与 sidecar 不一致")
        problems.extend(
            _bpath_unit_problems(
                label, entry, meta, strict=strict,
                ir_op=operators[source_id]))
    return problems


def verify_trace_provenance(
    genesim: Path, config: str | Path | None = None,
) -> None:
    """核对 trace 来源与覆盖，避免"跑通了但其实走的是手写模板"。

    来源只接受 `pimir`（真实算子编译产物）与 `absorbed`（被折进主算子的空
    trace）；`unknown`（无来源字段的旧 trace）与 `template` 一律失败。配置是
    验收的必需输入：没有配置就没有源 IR 与 sidecar 可对拍，只能做来源白名单，
    不能签发"覆盖完整、编译签名一致"。配置能定位到仿真 IR 与 sidecar 时，
    逐条核对父算子、类型与编译签名哈希，并把源 IR 要求的算子集合与实际拿到
    pimir trace 的集合对拍：缺算子同样失败，不能只看"有些 trace 是 pimir"。
    严格性成立时 IR 与 sidecar 的配置键和文件都是必需输入，缺失不能按旧式
    兼容放行。

    TP 展开出的分片按 trace 元数据的 `parent_op_id` 回指源算子。
    """
    if config is None:
        raise StepFailed(
            "trace 验收缺少仿真配置：无配置只能核对来源标签，无法核对源 IR "
            "覆盖与逐条编译签名。请传入本次运行的仿真配置。")
    sources: Dict[str, int] = {}
    by_op: Dict[str, Dict[str, int]] = {}
    tiles: Dict[Any, int] = {}
    covered: set = set()
    records: list[Dict[str, Any]] = []
    config_text = _config_text(genesim, config)
    import yaml

    loaded = yaml.safe_load(config_text) or {}
    # 只验收本次配置指定的目录；共享 pim_traces 里可能还有别的任务
    # 的旧 trace，不能拿它们的来源充数。
    trace_dir = _configured_dir(
        genesim, loaded, "pim_traces_dir", "pim_traces")
    for path in sorted(trace_dir.glob("op_*.pim_trace")):
        meta = _trace_metadata(path)
        records.append(meta)
        # 顶层 trace_source 记录这条 trace 实际怎么来的（pimir/absorbed/
        # template）；compile_signature 里的是编译输入口径（有没有 pimir），
        # 旧 trace 没有顶层字段时才退回它。
        source = meta.get("trace_source") or (
            meta.get("compile_signature") or {}).get("trace_source", "unknown")
        op_type = str(meta.get("op_type", "unknown"))
        sources[source] = sources.get(source, 0) + 1
        counts = by_op.setdefault(op_type, {})
        counts[source] = counts.get(source, 0) + 1
        pimir_meta = meta.get("pimir") or {}
        if pimir_meta:
            key = (pimir_meta.get("tile_n"), pimir_meta.get("k_iterations"))
            tiles[key] = tiles.get(key, 0) + 1
        if source == "pimir":
            parent = int(meta.get("parent_op_id", -1))
            covered.add(parent if parent >= 0
                        else int(meta.get("op_id", -1)))

    if not sources:
        raise StepFailed("没有找到任何 trace，无法判断来源")
    print(f"    trace 来源: {sources}")
    if tiles:
        print(f"    (tile_n, k_iterations) 分布: {tiles}")
    unexpected = sorted(
        source for source in sources if source not in ("pimir", "absorbed"))
    if unexpected:
        kinds = sorted(
            op_type for op_type, counts in by_op.items()
            if any(counts.get(source) for source in unexpected))
        raise StepFailed(
            f"trace 来源里出现 {unexpected}（{', '.join(kinds)}）：只接受 "
            "pimir 或 absorbed；退回手写模板/来源不明的 trace 不能进入"
            "代价链。")

    # 配置已由入口强制：按源 IR 的应覆盖集合与 lineage 对拍，没有配置就不
    # 存在"只签来源白名单"的旁路。
    config_strict = _config_requires_pimir(loaded)
    sidecar_path = _sidecar_of(genesim, config_text)
    if sidecar_path is None:
        _require_simulation_input("sidecar 配置键", config_strict)
        return
    if not sidecar_path.is_file():
        raise StepFailed(f"配置指向的 sidecar 不存在：{sidecar_path}")
    sidecar = json.loads(sidecar_path.read_text())
    strict = _strict_sidecar_context(loaded, sidecar)
    ir_rel = (loaded.get("model") or {}).get("ir_path")
    if not ir_rel:
        _require_simulation_input("仿真 IR 配置键", strict)
        return
    ir_path = genesim / ir_rel
    if not ir_path.is_file():
        raise StepFailed(f"配置指向的仿真 IR 不存在：{ir_path}")
    ir = json.loads(ir_path.read_text())
    # trace 验收与主编译共用同一份完整产物契约：MoE 元数据、容量、豁免、
    # 吸收与专家身份缺一项都可能在 trace 里表现为"覆盖了但其实不是这套"。
    validate_sidecar_contract(
        ir, sidecar, strict_pimir=config_strict)
    lineage = _trace_source_lineage(records, ir, sidecar, strict=strict)
    if lineage:
        raise StepFailed(
            f"{len(lineage)} 条 trace 无法证明来自本次 sidecar 的产物"
            f"（{'; '.join(lineage[:8])}）：父算子、类型或编译签名哈希不符。")
    absorbed = {int(op_id) for op_id in (sidecar.get("absorbed_ops") or [])}
    exempt = {int(op_id) for op_id in (sidecar.get("pimir_exempt_ops") or {})}
    required = {
        int(op["op_id"])
        for op in ir["operators"]
        if op.get("op_type") not in ("MODEL_INPUT", "MODEL_OUTPUT")
        and int(op["op_id"]) not in absorbed
        and int(op["op_id"]) not in exempt
    }
    missing = sorted(required - covered)
    if missing:
        raise StepFailed(
            f"{len(missing)} 个源 IR 要求的算子在 trace 里没有 pimir 来源"
            f"（如 op{missing[:8]}）：仿真没有为每个设备算子真实消费算子编译"
            "产物，缺的算子多半落到了 GPU/CPU 或退回了手写模板。")



def ceva_verdict(verdict_path: Path) -> list[str]:
    """读 run_all 的判定报告，返回 token 行；结论不是 PASS 就失败。"""
    if not verdict_path.is_file():
        raise StepFailed(f"没有判定报告 {verdict_path}")
    text = verdict_path.read_text()
    if "结论: PASS" not in text or "[FAIL]" in text or "[SKIP]" in text:
        raise StepFailed(f"CEVA 判定未通过：{verdict_path}")
    return [line for line in text.splitlines() if "token 与 baseline" in line]


def run_ceva(args: argparse.Namespace, log_dir: Path) -> Path:
    """CEVA 后端：我方导出 65 图，交给工作区 run_all.sh 编译、仿真并判定。"""
    workspace = ceva_workspace()
    bundle = workspace / "bundles" / args.run_name
    verdict_path = workspace / "runs" / args.run_name / "results" / "verdict.txt"
    print("\n[1] 图编译器导出 65 张 CEVA 子图（厂商反量化权重作 checkpoint，bin 全部我方生成）")
    _run([sys.executable, "scripts/export_gml.py", "--ceva-reference", str(ceva_reference_dir()),
          "--use-opcompiler", "--out-dir", str(bundle)],
         cwd=_REPO_ROOT, log_path=log_dir / "ceva_export.log")
    print("\n[2] 清单校验（全部文件由我方生成，尺寸与摘要一致）")
    _run([sys.executable, "scripts/verify_gml_artifact.py", str(bundle),
          "--ceva-manifest", str(bundle / "bundle.json")],
         cwd=_REPO_ROOT, log_path=log_dir / "ceva_manifest.log")
    print("\n[3] 厂商 GC → DDR 布局 → CEVA 仿真 → token 判定")
    _run(["bash", "run_all.sh", "-n", args.run_name, "--gml-bundle", str(bundle),
          "-s", str(args.decode_steps), "--token-check", args.token_check],
         cwd=workspace, log_path=log_dir / "ceva_run_all.log")
    for line in ceva_verdict(verdict_path):
        print(f"    {line.strip()}")
    return verdict_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--backend", choices=("genesim", "ceva"), default="genesim",
                        help="genesim=多 DPU 代价仿真；ceva=我方 GML 走厂商 GC 与 CEVA 仿真")
    parser.add_argument("--run-name", default=None,
                        help="运行号：CEVA 的包/运行目录名；genesim 用作运行"
                             "目录后缀，缺省按时间与进程号自动生成")
    parser.add_argument("--token-check", choices=("strict", "warn"), default="warn",
                        help="CEVA：strict 要求 token 与厂商基线一致；warn 不一致只告警（默认）")
    parser.add_argument("--decode-steps", type=int, default=10,
                        help="CEVA 后端的 decode 步数，至少 1（厂商基线为 10）")
    parser.add_argument("--num-stages", type=int, default=4,
                        help="流水段数：1=纯张量并行，num_dpus=纯流水")
    parser.add_argument("--num-dpus", type=int, default=8)
    parser.add_argument("--model-dir", default=None,
                        help="模型目录；架构、产物名与仿真配置从注册表派生。"
                             "默认取 paths 里的 llama2_7b_model_dir")
    parser.add_argument("--moe-execution", default=None,
                        choices=MOE_EXECUTION_MODES,
                        help="MoE 执行语义；缺省跟随 PIM_MOE_DISPATCH。"
                             "同一次运行里图骨架、放置 sidecar 与仿真都按它。")
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
    if args.backend == "ceva":
        if not args.run_name:
            print("错误: --backend ceva 需要 --run-name", file=sys.stderr)
            return 2
        if args.decode_steps < 1:
            print("错误: CEVA 全流程至少需要 1 步 decode", file=sys.stderr)
            return 2
        log_dir = Path(args.log_dir) if args.log_dir else _REPO_ROOT / "test-results" / f"ceva-{args.run_name}"
        log_dir.mkdir(parents=True, exist_ok=True)
        try:
            verdict_path = run_ceva(args, log_dir)
            token_lines = ceva_verdict(verdict_path)
        except StepFailed as exc:
            print(f"\n失败: {exc}", file=sys.stderr)
            return 1
        print("\nCEVA 全流程通过：图编译 → 算子编译 → GML/bin 生成 → 厂商 GC → CEVA 仿真")
        # 告警档的 PASS 只说明仿真正常，token 是否与基线一致要单独报。
        matched = bool(token_lines) and all(line.lstrip().startswith("[PASS]")
                                            for line in token_lines)
        print("token 对拍：通过（与厂商基线逐个一致）" if matched
              else "token 对拍：告警（与厂商基线不一致，见判定报告）")
        print(f"判定报告: {verdict_path}")
        return 0
    if args.num_dpus % args.num_stages:
        print(f"错误: num_stages={args.num_stages} 不能整除 num_dpus={args.num_dpus}",
              file=sys.stderr)
        return 2

    log_dir = Path(args.log_dir) if args.log_dir else _REPO_ROOT / "test-results" / "full-pipeline"
    genesim = genesim_root()
    model_dir = Path(args.model_dir) if args.model_dir else llama2_7b_model_dir()
    model_type, arch = architecture_of(model_dir)
    moe_execution = _resolve_moe_execution(model_dir, args.moe_execution)
    tp_width = args.num_dpus // args.num_stages
    # 本次运行的日志与产物按架构/策略/语义/唯一运行号隔离：另一份切分或
    # 另一语义的任务不能删掉这里的产物，本任务也不写任何共享固定目录。
    run_id = args.run_name or _new_run_id()
    run_dir = run_directory(
        log_dir, arch.artifact, args.num_dpus, args.num_stages,
        moe_execution=moe_execution, run_id=run_id)
    run_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir = run_dir / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print(f"{arch.artifact} 全流程（model_type={model_type}）："
          f"tp{tp_width}_pp{args.num_stages}（{args.num_dpus} 台 DPU）")
    print(f"  模型:   {model_dir}")
    print(f"  图编译: {_REPO_ROOT}")
    print(f"  GeneSim: {genesim}")
    print(f"  运行目录: {run_dir}（运行号 {run_id}）")
    if moe_execution:
        print(f"  MoE 执行语义: {moe_execution}")
    print("=" * 78)

    try:
        ir_path = step_a_model_ir(
            genesim, model_dir, run_dir, model_type, arch, moe_execution,
            out_dir=artifacts_dir)
        plan_path = None
        if not args.no_pu_mapping:
            plan_path = step_zero_pu_mapping(
                genesim, ir_path, args.num_stages, args.num_dpus, run_dir,
                arch.artifact, out_dir=artifacts_dir,
            )
        placement = step_bcd_export(
            args.num_stages, args.num_dpus, run_dir, model_dir, arch,
            plan_path=plan_path, moe_execution=moe_execution,
            ir_path=ir_path, out_dir=artifacts_dir,
        )
        # 精化放在放置导出之后：融合清单随放置 sidecar 一起去，先跑就没有它。
        pimir_ir = step_a2_refine_pimir_ir(
            genesim, ir_path, run_dir, arch.artifact, placement=placement,
            out_dir=artifacts_dir)

        if args.skip_simulation:
            # 只跑到 sidecar 也必须做覆盖校验：跳过仿真后没有 trace 可查，
            # 缺条目的设备算子会随流水线"成功"退出而完全不可见。
            validate_sidecar_contract(
                json.loads(pimir_ir.read_text()),
                json.loads(placement.read_text()),
                strict_pimir=True)
            print("\n跳过仿真（--skip-simulation）。到 sidecar 为止的链路已验证。")
            print("\n全流程验证通过（未含仿真）。")
            return 0

        if arch.sim_config is None:
            raise StepFailed(
                f"{arch.artifact} 还没有 GeneSim 仿真配置模板；"
                "请在架构注册表补 sim_config。")
        config = arch.sim_config.format(
            tp_width=tp_width, num_stages=args.num_stages)
        if not (genesim / "conf" / config).is_file():
            raise StepFailed(
                f"没有对应的 GeneSim 配置 conf/{config}。"
                f"当前只为部分策略准备了配置，可参照已有文件新建一份。"
            )
        main_summary = step_e_simulate(
            genesim, config, run_dir, label="pimir",
            results_dir=run_dir / "results_pimir",
            max_requests=args.max_requests,
            moe_execution=moe_execution,
            placement=placement, plan_path=plan_path,
            num_dpus=args.num_dpus, num_stages=args.num_stages,
            ir_path=pimir_ir,
        )
        # 来源校验在 step_e_simulate 里已经做过，这里不再重扫一遍 trace。

        if args.ab_compare and arch.sim_config_ab is None:
            print("\n跳过 A/B：该架构没有 A/B 对照配置模板")
        elif args.ab_compare:
            ab_config = arch.sim_config_ab.format(
                tp_width=tp_width, num_stages=args.num_stages)
            if not (genesim / "conf" / ab_config).is_file():
                print(f"\n跳过 A/B：没有 conf/{ab_config}")
            else:
                ab_summary = step_e_simulate(
                    genesim, ab_config, run_dir, label="global",
                    results_dir=run_dir / "results_global",
                    placement=placement, plan_path=plan_path,
                    num_dpus=args.num_dpus, num_stages=args.num_stages,
                    ir_path=pimir_ir,
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
