#!/usr/bin/env python3
"""把流水切分策略的 GEMM 放置结果导出成 GeneSim 可读的 sidecar。

链路：Llama2 权重 + ShardStrategy
        → compile_llama2（图编译，给每个权重标注 PIMTensorSpec）
        → export_placement_to_genesim（读 shard_map，写 device_hint 和 dpu_id）
        → GeneSim 的 scheduler.compiler_placement_file

用法（先 source paths.json 里的 pytorch_env_script）：

    python scripts/export_pp_placement.py --num-stages 8

硬件参数（DPU 数、MRAM 预算、系统预留）缺省从 CEVA 画像推导，不再各自
内置默认值——曾因脚本写死 NUM_DPUS=8 而画像是 64 核，导出的放置只钉住
8 个核，仿真静默少用核。显式传参仍可覆盖。

产物默认写到 GeneSim 的 models/ 目录，文件名带策略名，便于并存多种策略：

    models/llama2_7b_<策略名>_placed.ir
    models/llama2_7b_<策略名>_placement.json   ← 填进 GeneSim 配置的就是这个

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

from contracts.hardware_profile import HardwareProfile, load_profile
from contracts.op_contract import PIMHardwareConfig
from contracts.partition_plan import PartitionPlan
from genesim_bridge.paths import genesim_models_dir, llama2_7b_model_dir
from genesim_bridge.placement_export import (
    count_sidecar_classes,
    export_placement_for_strategy,
)
from graph.strategy import (
    format_strategy,
    llama_strategy,
    strategy_from_partition_plan,
)
from memory.mem_planner import HwBudget

# 档位解析与 run_full_pipeline 共用一份实现：两个 CLI 的 --batch 口径单点定义。
from scripts.run_full_pipeline import parse_batch_list

# 缺省画像：CEVA 64 核 4e，与 GeneSim 的 conf/ceva_64core_4e.yaml 成对。
DEFAULT_PROFILE = Path(__file__).resolve().parent.parent / "contracts" / "profiles" / "ceva_64core_4e.json"
NUM_TASKLETS = 4
PREFILL_SEQ_LEN = 16
MAX_SEQ = 64


def apply_profile_defaults(args: argparse.Namespace) -> HardwareProfile:
    """num-dpus/mram-bytes/sys-reserve-bytes 未显式给出时从画像推导。"""
    profile = load_profile(Path(args.profile))
    if args.num_dpus is None:
        args.num_dpus = profile.total_clusters
    if args.mram_bytes is None:
        args.mram_bytes = profile.cluster_memory_bytes
    if args.sys_reserve_bytes is None:
        args.sys_reserve_bytes = profile.sys_reserve_bytes
    return profile


def build_hw_budget(mram_bytes: int, sys_reserve_bytes: int) -> HwBudget:
    """按显式字节预算构造单台 DPU 的 MRAM 约束。"""
    if mram_bytes <= 0:
        raise SystemExit("--mram-bytes 必须为正")
    if sys_reserve_bytes < 0 or sys_reserve_bytes >= mram_bytes:
        raise SystemExit("--sys-reserve-bytes 必须非负且小于 --mram-bytes")
    return HwBudget(
        mram_bytes=mram_bytes,
        align=64,
        sys_reserve_bytes=sys_reserve_bytes,
    )


def _fuse_for_placement(gm):
    """保留旧的私有入口，内部仍走同一条融合路径。"""
    from graph.fuse_pim import fuse_for_pim

    return fuse_for_pim(gm)


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
        "--batch", type=parse_batch_list, default=(1,),
        help=(
            "batch 扫参档列表（逗号分隔，如 1,4）：算子按相位逐档出产物，"
            "图按最大档编译。缺省单档 1。"
        ),
    )
    parser.add_argument(
        "--max-seq", type=int, default=None,
        help=(
            "KV 布局与容量表的最大序列长度。缺省沿用 64（经典单档路径）；"
            "扫参由 run_full_pipeline 按仿真 trace 的 seq_len+output_length "
            "传入，保证容量墙与真实序列同源。"
        ),
    )
    parser.add_argument(
        "--replicas", type=int, default=1,
        help=(
            "DP 副本数（R8 多 batch 并行）：每副本持有全部权重、KV/激活按"
            "副本划分，物理 DPU 总数 = num_dpus × replicas；缺省 1 等价现状。"
        ),
    )
    parser.add_argument(
        "--kv-head-pin", action="store_true",
        help=(
            "执行侧 KV head 分片：KV 读取算子按 head 归属钉到持有该 head 的"
            "DPU（缺省整段首 DPU 钉扎，B=1 执行放置不变）。"
        ),
    )
    parser.add_argument(
        "--per-slot-prefill", action="store_true",
        help=(
            "执行策略作为编译输入（R25-03）：额外导出 [1,S] 逐槽图并构建 "
            "B 份槽位平移 prefill 计划，sidecar 记录策略声明；服务 "
            "pipeline_overlap/weight_reuse 关的逐槽执行策略。缺省合批。"
        ),
    )
    parser.add_argument(
        "--partition-plan", default=None,
        help=(
            "GeneSim 给定的切分方案（scripts/export_partition_plan.py 的产物）。"
            "给了它就由方案决定段数和 DPU 数，不再看 --num-stages/--num-dpus。"
        ),
    )
    parser.add_argument(
        "--num-dpus", type=int, default=None,
        help="逻辑 DPU 数；缺省取画像的设备数 × 每设备簇数（CEVA 64 核 = 64）。",
    )
    parser.add_argument(
        "--profile", default=str(DEFAULT_PROFILE),
        help="硬件画像 JSON；num-dpus/mram-bytes/sys-reserve-bytes 缺省从它推导。",
    )
    parser.add_argument(
        "--ir", default=None,
        help="输入 GeneSim 图骨架，默认取 <genesim>/models/llama2_7b.ir",
    )
    parser.add_argument(
        "--out-dir", default=None,
        help="产物目录，默认取 <genesim>/models",
    )
    parser.add_argument(
        "--mram-bytes", type=int, default=None,
        help="单台 DPU 的 MRAM 字节预算；缺省取画像的 cluster_memory_bytes。",
    )
    parser.add_argument(
        "--sys-reserve-bytes", type=int, default=None,
        help="系统预留字节数（内核、栈、workspace）；缺省取画像的 sys_reserve_bytes。",
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
    if args.replicas <= 0:
        parser.error("--replicas 必须为正")
    return args


def main() -> None:
    args = parse_args()
    profile = apply_profile_defaults(args)
    print(
        f"硬件画像: {profile.name}（{profile.num_pim_devices} 设备 × "
        f"{profile.clusters_per_device} 核，MRAM {profile.cluster_memory_bytes} B）"
    )
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
        if args.replicas > 1 and args.replicas != plan.replicas:
            raise SystemExit(
                f"--replicas={args.replicas} 与方案声明的 replicas="
                f"{plan.replicas} 不一致；请重新生成方案或改用方案的副本数"
            )
        strategy = strategy_from_partition_plan(plan, **shard_kwargs)
        num_dpus = plan.num_dpus
        print(f"切分方案来自 {args.partition_plan}（source={plan.source or '未标注'}）")
    else:
        strategy = llama_strategy(
            args.num_dpus, num_stages=args.num_stages, replicas=args.replicas,
            **shard_kwargs,
        )
        num_dpus = args.num_dpus
    print(format_strategy(strategy, cfg.num_hidden_layers))

    hw = build_hw_budget(args.mram_bytes, args.sys_reserve_bytes)
    max_seq = MAX_SEQ if args.max_seq is None else int(args.max_seq)
    if max_seq <= 0:
        raise SystemExit("--max-seq 必须为正")
    # 字节预算允许命令行覆盖；WRAM/DMA 对齐/编译目标一律与画像同源。
    hardware = PIMHardwareConfig(
        num_dpus=num_dpus, num_tasklets=NUM_TASKLETS,
        mram_bytes_per_dpu=hw.mram_bytes,
        wram_bytes_per_dpu=profile.wram_bytes_per_dpu,
        dma_align=profile.dma_align,
        compiler_target=profile.compiler_target,
    )
    print(
        f"MRAM 预算: 物理 {hw.mram_bytes} B，"
        f"预留 {hw.sys_reserve_bytes} B，"
        f"可用 {hw.mram_bytes - hw.sys_reserve_bytes} B"
    )
    result = export_placement_for_strategy(
        model, ir_path, out_dir,
        strategy=strategy, plan=plan, hw=hw, hardware=hardware,
        profile=profile,
        prefill_seq_len=PREFILL_SEQ_LEN, max_seq=max_seq,
        force_compile=False,
        measure_kernel_tiles=args.measure_kernel_tiles,
        batch_sizes=args.batch,
        kv_head_pin=args.kv_head_pin,
        per_slot_prefill=args.per_slot_prefill,
        # 旧单次链路沿用带策略名的文件名，不能和共享目录里的其他策略相撞。
        file_stem=f"llama2_7b_{strategy.name}",
    )
    # 开关开启时逐槽主机计划必须真实生成（生产链证据，不是只写声明）。
    if args.per_slot_prefill:
        slots = max(args.batch)
        plans = result.compiled.per_slot_prefill if result.compiled else None
        if slots > 1 and (plans is None or len(plans) != slots):
            raise SystemExit(
                f"--per-slot-prefill 开启但没有生成 {slots} 份逐槽主机计划"
                f"（got {0 if plans is None else len(plans)}）：逐槽编译没有"
                "贯穿到 compile_llama2，产物不完整。")
        print(f"逐槽 prefill 主机计划: {0 if plans is None else len(plans)} 份"
              f"（slots={slots}）")
    out_ir = result.placed_ir
    sidecar_path = result.sidecar
    sidecar = json.loads(sidecar_path.read_text())

    # 按 stage 汇总，便于肉眼核对流水段和 DPU 的对应关系。一个 GEMM 可能切在多台
    # DPU 上（TP 宽度 > 1），每台参与的 DPU 都记一次，而不是只数代表 DPU。
    gemm_count, bpath_count, by_dpu = count_sidecar_classes(sidecar)
    print(f"\n放置的 GEMM 算子数: {gemm_count}")
    print(f"带 pim mlir 的算子级节点数: {bpath_count}")
    print("每台 DPU 承担的 GEMM 分片数:")
    for dpu_id in sorted(by_dpu):
        print(f"  dpu{dpu_id}: {by_dpu[dpu_id]}")

    # 编译期容量表落进导出日志：逐档「容量占用」与墙位判定都以它为源。
    capacity = sidecar.get("capacity") or {}
    if capacity.get("table"):
        print(f"\n{capacity['table']}")
        if capacity.get("wall_batch") is not None:
            print(f"容量墙位（编译期折算）：B={capacity['wall_batch']} 起 KV+激活超预算")

    print(f"\n产物:\n  {out_ir}\n  {sidecar_path}")
    print(
        "\n在 GeneSim 配置里加一行即可生效:\n"
        f"  scheduler.compiler_placement_file: \"{sidecar_path}\""
    )


if __name__ == "__main__":
    main()
