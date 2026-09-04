#!/usr/bin/env python3
"""把流水切分策略的 GEMM 放置结果导出成 GeneSim 可读的 sidecar。

链路：Llama2 权重 + ShardStrategy
        → compile_llama2（图编译，给每个权重标注 PIMTensorSpec）
        → export_placement_to_genesim（读 shard_map，写 device_hint 和 dpu_id）
        → GeneSim 的 scheduler.compiler_placement_file

用法（先 source paths.json 里的 pytorch_env_script）：

    python scripts/export_pp_placement.py --num-stages 8

产物默认写到 GeneSim 的 models/ 目录，文件名带策略名，便于并存多种策略：

    models/llama2_7b_<策略名>_placed.ir
    models/llama2_7b_<策略名>_placement.json   ← 填进 GeneSim 配置的就是这个

注意：sidecar 只表达“哪个 GEMM 在哪台 DPU”。算子成本仍由
`refine_ir_with_flagtree.py` 按单卡全局形状测量，不反映切分后的本地形状，
也不包含跨段通信开销——详见 docs/ 里的遗留问题说明。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from contracts.op_contract import PIMHardwareConfig
from contracts.partition_plan import PartitionPlan
from genesim_bridge.paths import genesim_models_dir, llama2_7b_model_dir
from genesim_bridge.placement_export import export_placement_to_genesim
from graph.strategy import (
    format_strategy,
    llama_strategy,
    strategy_from_partition_plan,
)
from memory.mem_planner import HwBudget
from runtime.compile import compile_llama2

NUM_DPUS = 8
NUM_TASKLETS = 4
PREFILL_SEQ_LEN = 16
MAX_SEQ = 64


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
        "--ir", default=None,
        help="输入 GeneSim 图骨架，默认取 <genesim>/models/llama2_7b.ir",
    )
    parser.add_argument(
        "--out-dir", default=None,
        help="产物目录，默认取 <genesim>/models",
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


def main() -> None:
    args = parse_args()
    models_dir = genesim_models_dir()
    ir_path = Path(args.ir) if args.ir else models_dir / "llama2_7b.ir"
    out_dir = Path(args.out_dir) if args.out_dir else models_dir
    if not ir_path.is_file():
        raise SystemExit(
            f"找不到 GeneSim 图骨架 {ir_path}\n"
            "请先在 GeneSim 仓库跑 scripts/model_parser.py 生成 llama2_7b.ir。"
        )

    model_dir = llama2_7b_model_dir()
    torch.set_grad_enabled(False)
    from transformers import LlamaForCausalLM

    model = LlamaForCausalLM.from_pretrained(model_dir, dtype=torch.float16).eval()
    cfg = model.config

    shard_kwargs = dict(
        num_heads=cfg.num_attention_heads,
        num_kv_heads=cfg.num_key_value_heads,
        intermediate_size=cfg.intermediate_size,
        vocab_size=cfg.vocab_size,
        num_layers=cfg.num_hidden_layers,
    )
    plan: PartitionPlan | None = None
    if args.partition_plan:
        # GeneSim 给定切分方案：段数和 DPU 数由方案决定，命令行不再参与。
        plan = PartitionPlan.read(Path(args.partition_plan))
        if plan.model_id and plan.model_id not in str(model_dir):
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
    print(format_strategy(strategy, cfg.num_hidden_layers))

    hw = HwBudget(mram_bytes=4 * 2**30, align=1024, sys_reserve_bytes=64 * 2**20)
    hardware = PIMHardwareConfig(
        num_dpus=num_dpus, num_tasklets=NUM_TASKLETS,
        mram_bytes_per_dpu=hw.mram_bytes, wram_bytes_per_dpu=65536, dma_align=64,
    )
    compiled = compile_llama2(
        model, strategy, prefill_seq_len=PREFILL_SEQ_LEN, max_seq=MAX_SEQ,
        hw=hw, hardware=hardware,
    )

    out_ir = out_dir / f"llama2_7b_{strategy.name}_placed.ir"
    sidecar_path = out_dir / f"llama2_7b_{strategy.name}_placement.json"
    sidecar = export_placement_to_genesim(
        compiled.prefill_gm, ir_path, out_ir, sidecar_path,
        dtype="float16",
        hardware=hardware,
        measure_kernel_tiles=args.measure_kernel_tiles,
        # GeneSim 给了方案就把它的 Cluster 映射原样带进 sidecar，往返用同一份声明。
        dpu_to_cluster=plan.dpu_to_cluster if plan is not None else None,
    )

    # 按 stage 汇总，便于肉眼核对流水段和 DPU 的对应关系。
    by_dpu: dict[int, int] = {}
    for entry in sidecar["operators"].values():
        by_dpu[entry["dpu_id"]] = by_dpu.get(entry["dpu_id"], 0) + 1
    print(f"\n放置的 GEMM 算子数: {len(sidecar['operators'])}")
    print("每台 DPU 承担的 GEMM 数:")
    for dpu_id in sorted(by_dpu):
        print(f"  dpu{dpu_id}: {by_dpu[dpu_id]}")

    print(f"\n产物:\n  {out_ir}\n  {sidecar_path}")
    print(
        "\n在 GeneSim 配置里加一行即可生效:\n"
        f"  scheduler.compiler_placement_file: \"{sidecar_path}\""
    )


if __name__ == "__main__":
    main()
