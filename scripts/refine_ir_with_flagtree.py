#!/usr/bin/env python3
"""用 FlagTree 编译产物精化 GeneSim ModelIR 的算子成本（本仓入口）。

genesim 仓的 `scripts/refine_ir_with_flagtree.py` 是同一个入口，但它的
`--ir-level` 只有 `ttir` / `pimir` 两档，都是 A 路。B 路（整算子级 PIM IR →
融合 + 展开成相位链 → 抽成本）新增在本脚本里，**不改 genesim 仓本体**：
那边照旧用 `--ir-level pimir`，这里多一档 `--ir-level oplevel`。

三档的差别：

    ttir      A 路第 1 步。FlagGems 算子 → Triton → 原生 TTIR。需要 GPU。
    pimir      A 路最终产物。再加 convert-triton-to-pim / tile-to-budget /
              explicit-dma，sidecar 里多 mram_traffic_bytes 与 WRAM 用量。
    oplevel   B 路。不跑 FlagGems、不碰 GPU：按算子类型发一段整算子级 IR，
              交给 triton-opt 展开成相位链再量。sidecar 的 `source_name`
              记的是 mnemonic（`pim.softmax`），不是 `flag_gems.ops.softmax`。

用法（先 source paths.json 里的 pytorch_env_script）：

    python scripts/refine_ir_with_flagtree.py \
        --ir  /media/disk/fengjingge/src/genesim/models/llama2_7b.ir \
        --out-ir models/llama2_7b_oplevel.ir \
        --sidecar models/llama2_7b_oplevel_extensions.json \
        --seq-len 128 \
        --ir-level oplevel
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from genesim_bridge import export_costs_to_genesim, load_local_shapes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ir", required=True, help="输入 .ir（GeneSim 图骨架）")
    parser.add_argument("--out-ir", required=True, help="输出精化成本后的 .ir")
    parser.add_argument("--sidecar", required=True, help="输出 sidecar JSON")
    parser.add_argument(
        "--seq-len", type=int, default=128,
        help="prefill 代表点的 Tq（decode 点固定为 Tq=1, Tp=seq_len）",
    )
    parser.add_argument(
        "--ir-level", default="pimir", choices=["ttir", "pimir", "oplevel"],
        help="成本从哪层 IR 抽：pimir（默认，A 路终产物）/ ttir（A 路第 1 步）"
             "/ oplevel（B 路，不需 GPU）",
    )
    parser.add_argument(
        "--placement", default=None,
        help="图编译器放置结果 sidecar（scripts/export_pp_placement.py 的产物）。"
             "给了就按其中的本地分片形状测量被放置的算子，不给则全部按模型级"
             "全局形状。B 路不用 FlagGems，本地形状只影响 data_bytes 那一项。",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    # A 路要真的把 FlagGems 算子跑起来，得先修好 triton 的 CUDA 头文件与
    # ptxas 路径；B 路只调 triton-opt 可执行文件，不 import triton，不需要这步。
    if args.ir_level != "oplevel":
        from genesim_bridge import assert_pim_passes_available, prepare_triton_env

        prepare_triton_env(pim=args.ir_level == "pimir")
        if args.ir_level == "pimir":
            assert_pim_passes_available()

    local_shapes = None
    if args.placement:
        local_shapes = load_local_shapes(Path(args.placement))
        if not local_shapes:
            raise SystemExit(
                f"{args.placement} 里没有 local_in_features/local_out_features。"
                "这是 version 1 的旧放置 sidecar，请用当前的 "
                "export_pp_placement.py 重新导出。"
            )

    sidecar = export_costs_to_genesim(
        ir_path=Path(args.ir),
        out_ir_path=Path(args.out_ir),
        sidecar_path=Path(args.sidecar),
        seq_len=args.seq_len,
        cross_validate=args.ir_level != "oplevel",
        ir_level=args.ir_level,
        local_shapes=local_shapes,
    )

    bridged = len(sidecar["coverage"]["bridged"])
    template = len(sidecar["coverage"]["template"])
    print(f"[{args.ir_level}] 桥接 {bridged} 个算子，{template} 个保留模板成本")
    if args.ir_level == "oplevel":
        names = sorted({e["source_name"] for e in sidecar["operators"].values()})
        print(f"[oplevel] mnemonic: {names}")
    elif "pim_options" in sidecar:
        print(f"[{args.ir_level}] PIM pass 参数: {sidecar['pim_options']}")
    print(f"精化后的 IR: {args.out_ir}")
    print(f"sidecar:     {args.sidecar}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
