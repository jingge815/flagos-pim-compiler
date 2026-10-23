"""用 PIMExecutor 启动 vLLM 的 OpenAI 兼容服务。

引擎与 PIMWorker 必须同进程，所以关闭 vLLM 的多进程执行。
并发上限由 `--max-num-seqs` 决定，它等于编译期的 slot 数；
`--max-num-batched-tokens` 至少要能放下一步内所有并发请求的整段 prompt，
否则 vLLM 会切分 prefill，而静态图不支持分块 prefill。

示例：
    python scripts/serve_pim.py --model /path/to/Llama-2-7b-hf \\
        --max-model-len 512 --max-num-seqs 8 --max-num-batched-tokens 4096
"""

from __future__ import annotations

import asyncio
import os

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vllm.entrypoints.openai.api_server import run_server
from vllm.entrypoints.openai.cli_args import (
    FlexibleArgumentParser,
    make_arg_parser,
    validate_parsed_serve_args,
)

from backend.vllm_executor import PIMExecutor


def main() -> None:
    parser = make_arg_parser(FlexibleArgumentParser())
    args = parser.parse_args()
    args.distributed_executor_backend = PIMExecutor
    args.enable_prefix_caching = False
    validate_parsed_serve_args(args)
    asyncio.run(run_server(args))


if __name__ == "__main__":
    main()
