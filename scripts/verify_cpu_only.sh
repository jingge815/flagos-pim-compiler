#!/usr/bin/env bash
# 在一个真正没有 GPU 的 Ubuntu 22.04 容器里验证整条链路。
#
# 为什么要用容器：`CUDA_VISIBLE_DEVICES=""` 只让 torch.cuda.is_available() 返回
# False，驱动、libcuda.so、/dev/nvidia* 都还在，测不出纯 CPU 机器的真实行为。本轮
# 就是靠容器才发现 driver shim 会因为 `libcuda.so cannot found!` 直接挂掉——那个缺陷
# 在宿主机上用环境变量怎么测都不会暴露。
#
# 容器里没有设备节点、没有 libcuda、没有 nvidia-smi，与甲方的机器一致。
#
# 用法：
#     bash scripts/verify_cpu_only.sh              # 跑全部
#     bash scripts/verify_cpu_only.sh quick        # 只跑不需要模型的快速项
#
# 不会改动宿主环境：源码目录以只读挂载，编译缓存和 HOME 都指向容器内的 /tmp。

set -uo pipefail

IMAGE=pim-cputest:22.04
MODE=${1:-full}

PIM_COMPILER=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
GENESIM=${GENESIM_ROOT:-/media/disk/fengjingge/src/genesim}
INSTALLED=${FLAGOS_INSTALLED:-/media/disk/fengjingge/src/flagOS/flagOS-installed}
MODEL_DIR=${LLAMA2_7B_MODEL_DIR:-$INSTALLED/model-inference/models/Llama-2-7b-hf}

for path in "$PIM_COMPILER" "$GENESIM" "$INSTALLED"; do
  [[ -d $path ]] || { echo "错误：找不到目录 $path" >&2; exit 1; }
done

command -v docker >/dev/null 2>&1 || {
  echo "错误：需要 docker。没有 docker 时可以在真实的纯 CPU 机器上直接跑本脚本里的命令。" >&2
  exit 1
}

# 容器只需要编译器和 git，其余依赖来自挂载进去的 flagOS-installed。
if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
  echo "==> 构建验证镜像 $IMAGE"
  docker build -q -t "$IMAGE" - <<'DOCKERFILE' >/dev/null
FROM ubuntu:22.04
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update -qq && apt-get install -y -qq \
      gcc g++ make git curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*
DOCKERFILE
fi

run_in_container() {
  # 两个源码目录都以只读挂到 /src，再在容器内复制成可写副本。两处都必须可写：
  #   - pim-compiler：pytest 的 conftest 要写 test-results/
  #   - genesim：全流程脚本要往 models/ 写 IR、方案和 sidecar
  # 复制而不是直接可写挂载，是为了保证宿主目录一个字节都不会被改。
  #
  # GENESIM_ROOT 指向容器内的副本，否则 genesim_bridge 会顺着只读的宿主路径去写。
  docker run --rm \
    -v "$INSTALLED":"$INSTALLED":ro \
    -v "$PIM_COMPILER":/src/pim-compiler:ro \
    -v "$GENESIM":/src/genesim:ro \
    -e HOME=/tmp \
    -e OPCOMPILER_CACHE_DIR=/tmp/opcache \
    -e PIM_COMPILER_ROOT=/work/pim-compiler \
    -e PYTORCH_ENV_SCRIPT="$INSTALLED/pytorch/env-pytorch.sh" \
    -e LLAMA2_7B_MODEL_DIR="$MODEL_DIR" \
    -e GENESIM_ROOT=/work/genesim \
    "$IMAGE" bash -c "
      set -euo pipefail
      mkdir -p /work
      cp -r /src/pim-compiler /work/pim-compiler
      cp -r /src/genesim /work/genesim
      . '$INSTALLED/pytorch/env-pytorch.sh' >/dev/null 2>&1
      cd /work/pim-compiler
      $1
    "
}

FAILED=()
step() {
  local label=$1 script=$2
  echo
  echo "════════════════════════════════════════════════════════════"
  echo "  $label"
  echo "════════════════════════════════════════════════════════════"
  if run_in_container "$script"; then
    echo "  ✓ $label"
  else
    echo "  ✗ $label"
    FAILED+=("$label")
  fi
}

# ── 0. 先确认容器里真的没有 GPU，否则后面的结论不成立 ──────────────
step "环境自检：容器里确实没有 GPU" '
python3 - <<PY
import os, torch
assert not os.path.exists("/dev/nvidia0"), "容器里居然有 /dev/nvidia0，隔离没生效"
assert not torch.cuda.is_available(), "torch 居然报告有 CUDA，隔离没生效"
print("  /dev/nvidia0:", os.path.exists("/dev/nvidia0"))
print("  torch.cuda.is_available():", torch.cuda.is_available())
print("  torch:", torch.__version__)
PY
# grep -c 在没有匹配时返回 1，`set -e` 下会让整步失败——用 `|| true` 兜住；
# 计数本身由 Python 那段的 assert 负责判定，这两行只是给人看的信息。
echo "  libcuda 条目数: $(ldconfig -p 2>/dev/null | grep -c libcuda || true)"
if command -v nvidia-smi >/dev/null 2>&1; then
  echo "  nvidia-smi: 有（不该有）"
  exit 1
fi
echo "  nvidia-smi: 无"
'

# ── 1. 算子编译产物与有卡路径逐字节一致（本方案的核心判据）──────────
step "算子编译：pim mlir 与有卡产物 sha256 一致" '
python3 - <<PY
import dataclasses, hashlib
from contracts.op_contract import DEFAULT_HARDWARE_CONFIG, OpCompileRequest
from opcompiler_bridge.driver import compile_op

# 硬件参数取自有卡时导出的那份产物的 module attributes，两边必须一致才能比。
hw = dataclasses.replace(DEFAULT_HARDWARE_CONFIG, num_dpus=8, num_tasklets=4,
        wram_bytes_per_dpu=65536, mram_bytes_per_dpu=4294967296, dma_align=64)
r = compile_op(OpCompileRequest(op="linear", arg_shapes=[(1, 4096), (4096, 4096)],
        hardware=hw, dtype="float16", num_tasklets=hw.num_tasklets), force=True)
got = hashlib.sha256(r.pimir.encode()).hexdigest()
want = "26a1a970d0c648d89e3f707267dd0077bbadccf86bbbe7a441690c7069b6000c"
print("  pim mlir sha256:", got)
assert got == want, f"与有卡产物不一致！期望 {want}"
print("  与有卡产物逐字节一致 ✓")
PY
'

# ── 2. 快速回归 ────────────────────────────────────────────────
step "快速回归（不含 7B 全量）" '
set -o pipefail
python3 -m pytest tests/ -q -p no:cacheprovider -k "not llama2_7b" 2>&1 | tail -4
'

# ── 3. 算子编译单元测试（改动前无卡时全部 skip）──────────────────
step "算子编译单元测试" '
set -o pipefail
python3 -m pytest tests/test_opcompiler_linear.py -q -p no:cacheprovider 2>&1 | tail -4
'

if [[ $MODE == quick ]]; then
  echo
  echo "（quick 模式：跳过需要 7B 模型的项）"
else
  [[ -d $MODE ]] || true

  # ── 4. 7B 全量 ──────────────────────────────────────────────
  step "7B 全量测试" '
set -o pipefail
python3 -m pytest tests/ -q -p no:cacheprovider -k "llama2_7b" 2>&1 | tail -4
'

  # ── 5. 7B CPU 推理与峰值内存 ────────────────────────────────
  step "7B CPU 推理与峰值内存" '
python3 - <<PY
import os, resource, time, torch
from transformers import LlamaForCausalLM
torch.set_grad_enabled(False)
m = LlamaForCausalLM.from_pretrained(os.environ["LLAMA2_7B_MODEL_DIR"],
                                    dtype=torch.float16).eval()
t0 = time.time()
out = m(input_ids=torch.arange(128).unsqueeze(0), use_cache=True)
past, nxt = out.past_key_values, int(out.logits[0, -1].argmax())
for _ in range(4):
    o = m(input_ids=torch.tensor([[nxt]]), past_key_values=past, use_cache=True)
    past, nxt = o.past_key_values, int(o.logits[0, -1].argmax())
peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1048576
print(f"  prefill 128 + decode 4: {time.time() - t0:.1f}s")
print(f"  峰值 RSS: {peak:.1f} GiB")
PY
'

  # ── 6. 全流程闭环（这是"链路通没通"的唯一判据）────────────────
  step "全流程闭环 GeneSim→图编译→算子编译→GeneSim" '
set -o pipefail
python3 scripts/run_full_pipeline.py --num-stages 4 2>&1 | tail -25
'
fi

echo
echo "════════════════════════════════════════════════════════════"
if (( ${#FAILED[@]} == 0 )); then
  echo "  全部通过（真无 GPU 的 Ubuntu 22.04 容器）"
  exit 0
fi
echo "  以下项目失败："
printf '    - %s\n' "${FAILED[@]}"
exit 1
