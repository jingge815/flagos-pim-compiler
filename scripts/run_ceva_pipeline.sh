#!/usr/bin/env bash
# CEVA 全流程一键入口（单核与多核级联共用）：
#   图编译（按策略导出 bundle）→ 清单校验 → 算子编译（厂商 GC）→ GML+bin
#   → 按核 DDR 布局 → 级联蓝图 → 仿真 → token 判定 → 性能评估
#
# 用法：
#   bash scripts/run_ceva_pipeline.sh --run-name NAME [--decode-steps 10]
#        [--num-ceva 2] [--strategy pp2|tp2] [--token-check warn|strict]
#        [--dry-run] [--fresh]
#
# 不带 --num-ceva 时保持现有单核行为不变（需求 R6）：
# 默认等价旧版命令 `run_ceva_pipeline.sh --run-name NAME --decode-steps 10`。
# 多核必须显式传 --num-ceva 2 --strategy pp2|tp2。
# 若确需用受保护基线名（如 ours_e2e）跑多核，先备份基线，再设
# ALLOW_PROTECTED_RUN=1 显式放行。
# 产物：<workspace>/runs/NAME/results/{verdict.txt, cascade_eval.json, cascade_eval.txt}
#
# 说明：单核基线 runs/single_ours 存在时，级联评估自动用它作同口径参照
# （厂商 app 与段式 app 每层开销差 2.29 倍，跨口径比较会把偏差放大到 55%）。

set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PYTORCH_ENV=${PYTORCH_ENV_SCRIPT:-$(python3 -c \
    'import json, sys; print(json.load(open(sys.argv[1]))["pytorch_env_script"])' \
    "$REPO_ROOT/paths.json")}

if [[ ! -f "$PYTORCH_ENV" ]]; then
    echo "错误：找不到 PyTorch 环境脚本 $PYTORCH_ENV" >&2
    exit 1
fi

# --run-name 必填：缺省会写进 bundles/None，产物无处可寻
if [[ "$*" != *"--run-name"* ]]; then
    echo "错误：必须指定 --run-name <名字>" >&2
    exit 2
fi

# 多核调用才做下面的防呆；单核调用（无 --num-ceva）直接透传，行为与旧版一致。
num_ceva=1
prev=""
for a in "$@"; do
    if [[ "$prev" == "--num-ceva" ]]; then num_ceva=$a; fi
    case "$a" in
        --num-ceva=*) num_ceva=${a#--num-ceva=};;
    esac
    prev="$a"
done
if [[ "$num_ceva" -ge 2 ]]; then
    # 保护在用基线：多核覆盖会毁掉 token 对拍与评估的参照物
    # （单核重跑基线是需求引用的合法命令，不拦）。
    # ALLOW_PROTECTED_RUN=1 是显式逃生口：调用方须先备份基线并自行承担参照物切换。
    if [[ "${ALLOW_PROTECTED_RUN:-0}" != "1" ]]; then
        for protected in ours_e2e pp2_e2e pp2_v3 single_ours; do
            if [[ "$*" == *"--run-name $protected"* || "$*" == *"--run-name=$protected"* ]]; then
                echo "错误：$protected 是在用基线，不能作为多核运行的 --run-name" >&2
                exit 2
            fi
        done
    else
        echo "警告：ALLOW_PROTECTED_RUN=1，允许用受保护基线名启动多核运行" >&2
    fi
    if [[ "$num_ceva" -ne 2 ]]; then
        # cascade 编排按两核写死（pp2 的 c0/c1、tp2 的分片对）
        echo "错误：级联运行时只支持 2 核（收到 $num_ceva）" >&2
        exit 2
    fi
fi

source "$PYTORCH_ENV"
exec python "$REPO_ROOT/scripts/run_full_pipeline.py" --backend ceva "$@"
