"""激活标定：分位数 scale 与按层的 scale 表。"""

from __future__ import annotations

import numpy as np
import pytest

from quant.calibration import ActivationScales, percentile_scale


def test_percentile_scale_ignores_rare_outliers() -> None:
    """取 99.999% 分位的 |x| 除以 128（厂商 percentile 标定，int8 量程 2^7）。"""
    values = np.linspace(-1.0, 1.0, 1_000_001, dtype=np.float32)
    values[0] = -1000.0   # 一个离群点不该决定 scale
    assert percentile_scale(values) == pytest.approx(1.0 / 128, rel=1e-3)


def test_percentile_scale_of_tiny_sample_is_its_max() -> None:
    values = np.array([0.5, -2.0, 1.0], dtype=np.float32)
    assert percentile_scale(values) == pytest.approx(2.0 / 128)


def test_percentile_scale_rejects_all_zero() -> None:
    with pytest.raises(ValueError, match="全零"):
        percentile_scale(np.zeros(16, dtype=np.float32))


def test_activation_scales_lookup() -> None:
    scales = ActivationScales(key=0.04, value=0.003, static_input={"norm": 0.03})
    assert scales.kv(is_key=True) == 0.04
    assert scales.kv(is_key=False) == 0.003
    assert scales.static("norm") == 0.03
    with pytest.raises(KeyError, match="没有标定"):
        scales.static("other")


def _tiny_model():
    import torch
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(0)
    return LlamaForCausalLM(LlamaConfig(
        vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=2, num_key_value_heads=2, max_position_embeddings=32)).eval()


def test_kv_cache_scales_follow_the_cache_values() -> None:
    """每层 K/V scale 等于该层 cache 全部标定值的分位数 / 128；同时记下 final norm 的输入。"""
    import torch

    from quant.calibration import calibrate_llama

    model = _tiny_model()
    samples = [torch.tensor([[1, 5, 9, 3]]), torch.tensor([[2, 7]])]
    result = calibrate_llama(model, samples)

    keys = [[], []]
    values = [[], []]
    for ids in samples:
        with torch.no_grad():
            cache = model(ids, use_cache=True).past_key_values
        for layer in range(2):
            keys[layer].append(cache.layers[layer].keys.numpy().ravel())
            values[layer].append(cache.layers[layer].values.numpy().ravel())
    for layer in range(2):
        assert result.kv[layer][0] == pytest.approx(percentile_scale(np.concatenate(keys[layer])))
        assert result.kv[layer][1] == pytest.approx(percentile_scale(np.concatenate(values[layer])))
    assert [h.shape for h in result.head_inputs] == [(1, 4, 32), (1, 2, 32)]


def test_graph_static_scales_run_the_graph() -> None:
    """静态入口的 scale 在导出的图上实跑标定输入，按 FX 节点名取那条边的激活。"""
    import torch

    from quant.calibration import graph_static_scales

    class Net(torch.nn.Module):
        def forward(self, x):
            return (x * 2.0).relu()

    gm = torch.export.export(Net(), (torch.randn(1, 4),)).module()
    name = next(n.name for n in gm.graph.nodes if n.op == "call_function")
    samples = [torch.tensor([[1.0, -3.0, 0.5, 2.0]])]
    scales = graph_static_scales(gm, [name], samples)
    assert scales[name] == pytest.approx(6.0 / 128)


def test_calibration_prompts_are_fixed_and_general() -> None:
    """标定语料是固定的多领域短文本，不含验收 prompt 以外的特殊依赖，可复现。"""
    from quant.calibration import CALIBRATION_PROMPTS

    assert len(CALIBRATION_PROMPTS) >= 4
    assert len(set(CALIBRATION_PROMPTS)) == len(CALIBRATION_PROMPTS)


def test_bundle_scales_index_by_layer() -> None:
    from quant.calibration import LlamaCalibration

    cal = LlamaCalibration(kv=[(0.1, 0.2), (0.3, 0.4)])
    assert cal.scales_for(1).kv(is_key=True) == 0.3
    assert cal.scales_for(None).key is None


def test_graph_static_scales_splits_samples_to_graph_width() -> None:
    """导出图按单 token 定形时，多 token 样本按第 1 维切开逐个跑。"""
    import torch

    from quant.calibration import graph_static_scales

    class Net(torch.nn.Module):
        def forward(self, x):
            return x * 2.0

    gm = torch.export.export(Net(), (torch.randn(1, 1, 4),)).module()
    name = next(n.name for n in gm.graph.nodes if n.op == "call_function")
    sample = torch.tensor([[[1.0, 0, 0, 0], [0, -5.0, 0, 0], [0, 0, 2.0, 0]]])
    assert graph_static_scales(gm, [name], [sample])[name] == pytest.approx(10.0 / 128)
