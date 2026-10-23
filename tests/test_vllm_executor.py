"""验证 PIMExecutor 能让 vLLM 引擎跑通两个并发请求的贪心生成。"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
import torch

NUM_SLOTS = 2
MAX_SEQ = 16
PREFILL_SEQ_LEN = 4
MAX_NEW_TOKENS = 3
PROMPTS = [[1, 2, 3, 4], [5, 6, 7, 8]]

# 引擎与 worker 必须在同一进程里，PIMWorker 才能直接驱动 NumpyBackend。
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
os.environ.setdefault("PIM_PREFILL_SEQ_LEN", str(PREFILL_SEQ_LEN))
os.environ.setdefault("PIM_MRAM_BYTES", str(1 << 28))

sys.path.insert(0, str(Path(__file__).parent.parent))

from transformers import LlamaConfig, LlamaForCausalLM  # noqa: E402

from backend.vllm_executor import PIMExecutor  # noqa: E402


@pytest.fixture(scope="module")
def tiny_model_dir(tmp_path_factory) -> Path:
    """保存一个随机初始化的微型 Llama，作为 vLLM 的模型目录。"""
    # 不要改全局梯度开关：会让 torch.export 导出不同的图。
    torch.manual_seed(0)
    config = LlamaConfig(
        hidden_size=64, intermediate_size=128, num_hidden_layers=4,
        num_attention_heads=4, num_key_value_heads=4, vocab_size=128,
        max_position_embeddings=MAX_SEQ,
    )
    model = LlamaForCausalLM(config).eval().half()
    path = tmp_path_factory.mktemp("tiny-llama")
    model.save_pretrained(path)
    return path


def test_llm_runs_two_requests(tiny_model_dir: Path) -> None:
    """vLLM 引擎经 PIMExecutor 生成，两个 prompt 各自产出一个非空结果。"""
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=str(tiny_model_dir),
        skip_tokenizer_init=True,
        dtype="float16",
        max_model_len=MAX_SEQ,
        max_num_seqs=NUM_SLOTS,
        # 一步内能放下所有并发请求的整段 prompt，避免分块 prefill。
        max_num_batched_tokens=NUM_SLOTS * PREFILL_SEQ_LEN,
        enable_prefix_caching=False,
        gpu_memory_utilization=0.2,
        distributed_executor_backend=PIMExecutor,
    )
    outputs = llm.generate(
        [{"prompt_token_ids": p} for p in PROMPTS],
        SamplingParams(max_tokens=MAX_NEW_TOKENS, temperature=0.0),
    )

    assert len(outputs) == len(PROMPTS)
    for output in outputs:
        assert output.outputs, f"{output.request_id} 没有输出"
        assert 1 <= len(output.outputs[0].token_ids) <= MAX_NEW_TOKENS
