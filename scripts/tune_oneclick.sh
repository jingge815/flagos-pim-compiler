#!/usr/bin/env bash
# 一键调优：激活环境，先 dry-run 检查候选组合，再跑完整调优。
# 用法：scripts/tune_oneclick.sh [选项] [配置] [输出目录]
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="$ROOT/examples/tune/quick.yaml"
OUTPUT=""
DRY_RUN_ONLY=false
POSITIONAL=()

usage() {
  cat <<'EOF'
用法：scripts/tune_oneclick.sh [选项] [配置] [输出目录]

选项：
  --config PATH       调优配置，默认 examples/tune/quick.yaml
  --output-dir DIR    输出目录；不填则用配置里的 output.dir
  --dry-run           只检查候选组合，不编译不仿真
  -h, --help          显示本帮助

示例：
  scripts/tune_oneclick.sh
  scripts/tune_oneclick.sh --output-dir /tmp/my-tune
  scripts/tune_oneclick.sh --config examples/tune/quick.yaml --dry-run
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)
      [[ $# -ge 2 ]] || { echo "缺少 --config 的路径" >&2; exit 2; }
      CONFIG="$2"
      shift 2
      ;;
    --output-dir)
      [[ $# -ge 2 ]] || { echo "缺少 --output-dir 的目录" >&2; exit 2; }
      OUTPUT="$2"
      shift 2
      ;;
    --dry-run)
      DRY_RUN_ONLY=true
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    -*)
      echo "未知参数：$1" >&2
      usage >&2
      exit 2
      ;;
    *)
      POSITIONAL+=("$1")
      shift
      ;;
  esac
done

if [[ ${#POSITIONAL[@]} -ge 1 ]]; then
  CONFIG="${POSITIONAL[0]}"
fi
if [[ ${#POSITIONAL[@]} -ge 2 ]]; then
  OUTPUT="${POSITIONAL[1]}"
fi
if [[ ${#POSITIONAL[@]} -gt 2 ]]; then
  echo "位置参数最多两个：配置、输出目录" >&2
  usage >&2
  exit 2
fi

if [[ ! -f "$CONFIG" ]]; then
  echo "找不到配置：$CONFIG" >&2
  exit 2
fi
if [[ "$CONFIG" != /* ]]; then
  CONFIG="$(cd "$(dirname "$CONFIG")" && pwd)/$(basename "$CONFIG")"
fi

ENV_SCRIPT="$(python3 - "$ROOT/paths.json" <<'PY'
import json
import sys

print(json.load(open(sys.argv[1]))["pytorch_env_script"])
PY
)"
# shellcheck disable=SC1090
source "$ENV_SCRIPT"

ARGS=(--config "$CONFIG")
if [[ -n "$OUTPUT" ]]; then
  ARGS+=(--output-dir "$OUTPUT")
fi

cd "$ROOT"
echo "[1/2] 检查候选组合：$CONFIG"
python scripts/tune_pipeline.py "${ARGS[@]}" --dry-run
if [[ "$DRY_RUN_ONLY" == true ]]; then
  echo "仅检查候选组合，未启动编译与仿真。"
  exit 0
fi
echo "[2/2] 开始完整调优"
python scripts/tune_pipeline.py "${ARGS[@]}"
