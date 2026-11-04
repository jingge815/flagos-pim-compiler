"""激活标定：按真实前向的激活统计静态量化 scale。

动态量化（DQ）在硬件上逐组现算，不需要标定；需要编译期定值的只有两类：

- KV cache：int8 落盘，进出同一块 cache 共用一个 scale；
- 静态 int8 入口：输出头的 norm 输出直接量化成 int8 喂矩阵乘。

口径取厂商量化器（`calibrate_mode=percentile`）：|x| 的 99.999% 分位除以 2^7。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# 固定的标定语料：对话、叙述、史实、代码、数学各一类，保证结果可复现、不偏某一领域。
CALIBRATION_PROMPTS = (
    "hello how are you?",
    "The quick brown fox jumps over the lazy dog. It was a sunny day in the park, "
    "and children were playing games near the lake.",
    "In 1492, Columbus sailed across the Atlantic Ocean and reached the Americas.",
    "def add(a, b):\n    return a + b\n\nprint(add(2, 3))",
    "The derivative of x squared is two x, and the integral of one over x is the "
    "natural logarithm of x.",
    "Please summarize the following paragraph in one sentence for a busy reader.",
)

PERCENTILE = 0.99999
# int8 对称量化的量程 2^7：scale = 分位值 / 128。
INT8_RANGE = 128


def percentile_scale(values: np.ndarray, percentile: float = PERCENTILE) -> float:
    """一组标定激活的 int8 scale：|x| 的分位值 / 128。

    分位下标与厂商 `_find_scale_by_percentile` 相同，样本少时退化为取最大值。
    """
    flat = np.abs(np.asarray(values, dtype=np.float32).ravel())
    if flat.size == 0 or not flat.any():
        raise ValueError("标定激活全零，定不出 scale")
    k = min(max(int(flat.size * percentile), 1), flat.size - 1) if flat.size > 1 else 0
    return float(np.partition(flat, k)[k]) / INT8_RANGE


@dataclass(frozen=True)
class ActivationScales:
    """一张子图的静态量化 scale。`static_input` 按静态 int8 入口的标定键索引。"""

    key: float | None = None
    value: float | None = None
    static_input: dict[str, float] = field(default_factory=dict)

    def kv(self, *, is_key: bool) -> float:
        scale = self.key if is_key else self.value
        if scale is None:
            raise KeyError(f"{'K' if is_key else 'V'} cache 没有标定 scale")
        return scale

    def static(self, name: str) -> float:
        if name not in self.static_input:
            raise KeyError(f"静态 int8 入口 {name!r} 没有标定 scale")
        return self.static_input[name]


@dataclass
class LlamaCalibration:
    """一次整模型标定的结果：每层 (K, V) scale，以及喂输出头的 final norm 输入样本。"""

    kv: list[tuple[float, float]]
    head_inputs: list = field(default_factory=list)

    def scales_for(self, layer_id: int | None) -> ActivationScales:
        """某层子图的 scale；输出头（layer_id=None）没有 KV cache。"""
        if layer_id is None:
            return ActivationScales()
        key, value = self.kv[layer_id]
        return ActivationScales(key=key, value=value)


def calibrate_llama(model, samples) -> LlamaCalibration:
    """整模型跑标定 token，统计每层 KV cache 的 scale，并截下 final norm 的输入。

    final norm 输入留作输出头子图的标定输入：输出头的静态 int8 入口在 norm 之后，
    要在导出的子图上实跑才能对上它的 FX 节点。
    """
    import torch

    keys: list[list[np.ndarray]] = []
    values: list[list[np.ndarray]] = []
    head_inputs = []
    hook = model.model.norm.register_forward_pre_hook(
        lambda module, args: head_inputs.append(args[0].detach().clone()))
    try:
        for ids in samples:
            with torch.no_grad():
                cache = model(ids, use_cache=True).past_key_values
            for layer, entry in enumerate(cache.layers):
                if layer == len(keys):
                    keys.append([])
                    values.append([])
                keys[layer].append(entry.keys.float().numpy().ravel())
                values[layer].append(entry.values.float().numpy().ravel())
    finally:
        hook.remove()
    kv = [(percentile_scale(np.concatenate(k)), percentile_scale(np.concatenate(v)))
          for k, v in zip(keys, values)]
    return LlamaCalibration(kv=kv, head_inputs=head_inputs)


def graph_static_scales(gm, names, samples) -> dict[str, float]:
    """在导出的图上逐样本实跑，取 `names` 这些 FX 节点输出的分位数 scale。"""
    import torch

    wanted = set(names)
    caught: dict[str, list[np.ndarray]] = {name: [] for name in wanted}

    class _Recorder(torch.fx.Interpreter):
        def run_node(self, node):
            result = super().run_node(node)
            if node.name in wanted:
                caught[node.name].append(result.detach().float().numpy().ravel())
            return result

    placeholder = next(n for n in gm.graph.nodes if n.op == "placeholder")
    width = placeholder.meta["val"].shape[1]
    for sample in samples:
        # 导出图的 token 维是定长的（输出头为 1），长样本切成同宽的片逐个跑。
        for start in range(0, sample.shape[1] - width + 1, width):
            with torch.no_grad():
                _Recorder(gm).run(sample[:, start:start + width])
    missing = [name for name, got in caught.items() if not got]
    if missing:
        raise KeyError(f"图里没有这些标定节点：{missing}")
    return {name: percentile_scale(np.concatenate(got)) for name, got in caught.items()}
