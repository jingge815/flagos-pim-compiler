#!/usr/bin/env python3
"""把流水切分策略的 GEMM 放置结果导出成 GeneSim 可读的 sidecar。

链路：模型权重（注册表加载器）+ ShardStrategy
        → compile_llama2（图编译，给每个权重标注 PIMTensorSpec）
        → export_placement_to_genesim（读 shard_map，写 device_hint 和 dpu_id）
        → GeneSim 的 scheduler.compiler_placement_file

用法（先 source paths.json 里的 pytorch_env_script）：

    python scripts/export_pp_placement.py --num-stages 8

    # Qwen3-MoE：模型类型、加载器、切分宽度与产物名都从架构注册表派生
    python scripts/export_pp_placement.py --model-dir <30B 目录> --num-stages 2

产物默认写到 GeneSim 的 models/ 目录，文件名带模型前缀与策略名，便于并存：

    models/<artifact>_<策略名>_placed.ir
    models/<artifact>_<策略名>_placement.json   ← 填进 GeneSim 配置的就是这个

注意：sidecar 只表达“哪个 GEMM 在哪台 DPU”。算子成本仍由
`refine_ir_with_flagtree.py` 按单卡全局形状测量，不反映切分后的本地形状，
也不包含跨段通信开销——详见 docs/ 里的遗留问题说明。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.architectures import architecture_of
from contracts.partition_plan import PartitionPlan
from genesim_bridge.paths import genesim_models_dir, llama2_7b_model_dir
from genesim_bridge.placement_export import (
    count_sidecar_classes,
    export_placement_to_genesim,
)
from graph.strategy import (
    format_strategy,
    llama_strategy,
    strategy_from_partition_plan,
)
from graph.fuse_pim import fuse_for_pim
from memory.mem_planner import default_site_capacity
from runtime.compile import compile_llama2
from runtime.exec_plan_gen import MOE_EXECUTION_MODES

NUM_DPUS = 8
NUM_TASKLETS = 4
PREFILL_SEQ_LEN = 16
MAX_SEQ = 64


def _fuse_for_placement(gm):
    """给 GeneSim 放置导出图补上融合标记。"""
    return fuse_for_pim(gm)


def report_capacity(compiled, sidecar) -> None:
    """打印每台 DPU 的权重/KV/激活/内核 tile 所需与可用字节。

    容量判据已在编译期执行；这里把实际数字落到日志，48 层这类贴近预算的
    模型才有可核对的所需/可用证据，而不是只看"没报错"。
    """
    budget = compiled.hw.mram_bytes - compiled.hw.sys_reserve_bytes
    kernel = max(
        (int(entry.get("placed_mram_bytes", 0) or 0)
         for entry in (sidecar.get("operators") or {}).values()),
        default=0)
    print("\n每台 DPU 容量（权重 + KV + 激活 + 内核 tile / 可用）:")
    for dpu_id, plan in sorted(compiled.mem_plans.items()):
        weight = plan.kv_base
        kv = plan.act_base - plan.kv_base
        activation = plan.total - plan.act_base
        need = plan.total + kernel
        print(f"  dpu{dpu_id}: {weight} + {kv} + {activation} + {kernel} "
              f"= {need} / {budget} B")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--num-stages", type=int, default=None,
        help=(
            "流水段数：1=纯张量并行，num_dpus=纯流水，中间值为混合切分。"
            "与 --partition-plan 二选一。"
        ),
    )
    parser.add_argument(
        "--partition-plan", default=None,
        help=(
            "GeneSim 给定的切分方案（scripts/export_partition_plan.py 的产物）。"
            "给了它就由方案决定段数和 DPU 数，不再看 --num-stages/--num-dpus。"
        ),
    )
    parser.add_argument("--num-dpus", type=int, default=NUM_DPUS)
    parser.add_argument(
        "--model-dir", default=None,
        help="模型目录（读 config.json 定架构），默认取 paths 里的 llama2_7b",
    )
    parser.add_argument(
        "--ir", default=None,
        help="输入 GeneSim 图骨架，默认取 <genesim>/models/<artifact>.ir",
    )
    parser.add_argument(
        "--out-dir", default=None,
        help="产物目录，默认取 <genesim>/models",
    )
    parser.add_argument(
        "--moe-execution", default=None,
        choices=MOE_EXECUTION_MODES,
        help=(
            "MoE 执行语义；缺省跟随 PIM_MOE_DISPATCH。这份值随 sidecar 的 "
            "moe_config.execution 下发给 GeneSim，必须与图骨架 IR 一致。"
        ),
    )
    parser.add_argument(
        "--measure-kernel-tiles", action="store_true",
        help=(
            "对每种本地分片形状真跑一次算子编译，把 pim mlir 里选定的输出分块写进 "
            "sidecar 的 kernel_tile_n；GeneSim 用它替掉 conf 里的 tile_size 常量。"
            "需要 CUDA 和 FlagTree。"
        ),
    )
    args = parser.parse_args()
    if (args.num_stages is None) == (args.partition_plan is None):
        parser.error("必须给 --num-stages 或 --partition-plan，且只能给一个")
    return args


def resolve_strategy(
    args: argparse.Namespace, arch, cfg, ir_path: Path, model_dir: Path,
):
    """从命令行或 PartitionPlan 解析策略，再按架构约束统一校验。

    两条入口都走注册表的 `strategy_check`：专家数整除、KV 头、层数与专家
    宽度由同一份规则判定。非法方案在这里明确失败，合法方案不会被静默替换。
    """
    shard_kwargs = arch.widths(cfg)
    plan: PartitionPlan | None = None
    if args.partition_plan:
        # GeneSim 给定切分方案：段数和 DPU 数由方案决定，命令行不再参与。
        plan = PartitionPlan.read(Path(args.partition_plan))
        # 方案是 GeneSim 从某份图骨架导出的，身份必须与本次导出用的骨架一致；
        # 不一致说明拿了别的模型的方案，切分会张冠李戴。没有 model_id 的旧
        # 方案（或旧 IR）退回按目录名的弱提示。
        ir_model_id = str(json.loads(ir_path.read_text()).get("model_id", ""))
        if plan.model_id and ir_model_id:
            if plan.model_id != ir_model_id:
                raise SystemExit(
                    f"方案标注的 model_id={plan.model_id!r} 与图骨架的 "
                    f"{ir_model_id!r} 不一致：方案不是这份 IR 的产物，请重新生成。")
        elif plan.model_id and plan.model_id not in str(model_dir):
            print(
                f"提示: 方案标注的 model_id={plan.model_id!r} 与本次加载的模型目录"
                f"（{model_dir}）看起来不一致，请确认没有张冠李戴。"
            )
        strategy = strategy_from_partition_plan(plan, **shard_kwargs)
        num_dpus = plan.num_dpus
        print(f"切分方案来自 {args.partition_plan}（source={plan.source or '未标注'}）")
    else:
        strategy = llama_strategy(
            args.num_dpus, num_stages=args.num_stages, **shard_kwargs
        )
        num_dpus = args.num_dpus
    arch.check_strategy(cfg, strategy)
    return strategy, num_dpus, plan


def main() -> None:
    args = parse_args()
    model_dir = Path(args.model_dir) if args.model_dir else llama2_7b_model_dir()
    model_type, arch = architecture_of(model_dir)
    models_dir = genesim_models_dir()
    ir_path = Path(args.ir) if args.ir else models_dir / f"{arch.artifact}.ir"
    out_dir = Path(args.out_dir) if args.out_dir else models_dir
    if not ir_path.is_file():
        raise SystemExit(
            f"找不到 GeneSim 图骨架 {ir_path}\n"
            "请先在 GeneSim 仓库跑 scripts/model_parser.py 生成 "
            f"{arch.artifact}.ir。"
        )

    torch.set_grad_enabled(False)
    model = arch.load(str(model_dir))
    cfg = model.config
    print(f"模型 {model_dir}（model_type={model_type}）")

    strategy, num_dpus, plan = resolve_strategy(args, arch, cfg, ir_path, model_dir)
    print(format_strategy(strategy, cfg.num_hidden_layers))

    # 站点容量契约（8 DPU × 8 GiB）是唯一真源；容量与硬件配置从同一工厂派生。
    hw, hardware = default_site_capacity(
        num_dpus=num_dpus, num_tasklets=NUM_TASKLETS)
    compiled = compile_llama2(
        model, strategy, prefill_seq_len=PREFILL_SEQ_LEN, max_seq=MAX_SEQ,
        hw=hw, hardware=hardware, moe_execution=args.moe_execution,
    )
    if compiled.moe_execution:
        print(f"MoE 执行语义: {compiled.moe_execution}")
    fusion = _fuse_for_placement(compiled.prefill_gm)
    print(f"GeneSim 代价评估融合: {fusion}")

    out_ir = out_dir / f"{arch.artifact}_{strategy.name}_placed.ir"
    sidecar_path = out_dir / f"{arch.artifact}_{strategy.name}_placement.json"
    sidecar = export_placement_to_genesim(
        compiled.prefill_gm, ir_path, out_ir, sidecar_path,
        dtype="float16",
        hardware=hardware,
        measure_kernel_tiles=args.measure_kernel_tiles,
        # GeneSim 给了方案就把它的 Cluster 映射原样带进 sidecar，往返用同一份声明。
        dpu_to_cluster=plan.dpu_to_cluster if plan is not None else None,
        # 非 GEMM 算子按所在层钉到该层流水段的第一台 DPU。
        strategy=strategy,
        # KV 规格决定哪些算子读的是 KV 区，随 sidecar 传给 GeneSim 选每字节周期。
        kv_specs=compiled.kv_specs,
        # 执行语义由编译配置决定，随 sidecar 交给仿真侧独立核对。
        moe_execution=compiled.moe_execution or None,
        # 容量契约随 sidecar 回传，GeneSim 核对两侧是同一份预算。
        hw_budget=hw,
    )

    # 按 stage 汇总，便于肉眼核对流水段和 DPU 的对应关系。一个 GEMM 可能切在多台
    # DPU 上（TP 宽度 > 1），每台参与的 DPU 都记一次，而不是只数代表 DPU。
    gemm_count, bpath_count, by_dpu = count_sidecar_classes(sidecar)
    report_capacity(compiled, sidecar)
    print(f"\n放置的 GEMM 算子数: {gemm_count}")
    print(f"带 pim mlir 的算子级节点数: {bpath_count}")
    print("每台 DPU 承担的 GEMM 分片数:")
    for dpu_id in sorted(by_dpu):
        print(f"  dpu{dpu_id}: {by_dpu[dpu_id]}")

    print(f"\n产物:\n  {out_ir}\n  {sidecar_path}")
    print(
        "\n在 GeneSim 配置里加一行即可生效:\n"
        f"  scheduler.compiler_placement_file: \"{sidecar_path}\""
    )


if __name__ == "__main__":
    main()
