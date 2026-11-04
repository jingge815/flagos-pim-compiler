#!/usr/bin/env bash
# 自动加载编译环境，执行图编译、FlagTree、GML/bin 生成、厂商 GC 和 CEVA 仿真。
# 用法：bash scripts/run_ceva_pipeline.sh --run-name NAME [--decode-steps 10]

set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PYTORCH_ENV=${PYTORCH_ENV_SCRIPT:-$(python3 -c \
    'import json, sys; print(json.load(open(sys.argv[1]))["pytorch_env_script"])' \
    "$REPO_ROOT/paths.json")}

if [[ ! -f "$PYTORCH_ENV" ]]; then
    echo "错误：找不到 PyTorch 环境脚本 $PYTORCH_ENV" >&2
    exit 1
fi

source "$PYTORCH_ENV"
exec python "$REPO_ROOT/scripts/run_full_pipeline.py" --backend ceva "$@"
