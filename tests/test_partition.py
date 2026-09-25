"""验证 FX 图的 DPU 标记和连通子图划分。"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch.fx import Graph, GraphModule, Node
from transformers import LlamaConfig, LlamaForCausalLM

sys.path.insert(0, str(Path(__file__).parent.parent))

from contracts.graph_meta import DEVICE_DPU, DEVICE_HOST, DEVICE_META_KEY, PART_ID_META_KEY
from graph.partition import HOST_ONLY, partition_graph


def _module_with_host_break() -> tuple[GraphModule, dict[str, Node]]:
    graph = Graph()
    input_node = graph.placeholder("input")
    add = graph.call_function(torch.ops.aten.add.Tensor, (input_node, 1))
    # 主机断点要用**真正**留主机的算子。`relu` 不行：黑名单口径下它是设备算子
    # （逐元素、有 numpy 内核）；`embedding` 原先也留主机，现在下了设备。挑
    # `_assert_tensor_metadata` —— 它是编译脚手架，黑名单里最稳定的一条。
    relu = graph.call_function(
        torch.ops.aten._assert_tensor_metadata.default, (add,))
    relu.meta["val"] = add.meta.get("val")
    mul = graph.call_function(torch.ops.aten.mul.Tensor, (relu, 2))
    graph.output(mul)
    return GraphModule({}, graph), {
        "input": input_node,
        "add": add,
        "relu": relu,
        "mul": mul,
    }


def test_partition_marks_device_by_default_and_host_breaks_components() -> None:
    """默认下设备，只有黑名单里的留主机。

    反过来（白名单）会让每个没想到的算子静默落主机——图照样跑完、数值也对，
    只是那些算子根本没在设备上执行。黑名单下漏一个新算子会在 `spec_prop`
    的规则表那儿直接报错。

    这里断言黑名单的**语义**而不是逐一列出成员：成员会随命令编码能力变化
    （`cat` / `slice` 等编码支持了就该移出去），逐一列出等于每次都要改测试。
    """
    gm, nodes = _module_with_host_break()

    partitions = partition_graph(gm)

    # 注意力也已下设备：读写 KV 缓存那段搬进了 `runtime.kernels.sdpa_kernel`，
    # 缓存本来就在 DPU 本地 MRAM 里。留主机的只剩编译脚手架与图入口。
    assert torch.ops.aten.scaled_dot_product_attention.default not in HOST_ONLY
    # 广播视图留主机：`expand` / `repeat` 的输出元素数大于输入，而命令按
    # `arg_shapes` 读输入、按 `out_shape` 写输出，广播倍数没有地方放。
    assert torch.ops.aten.expand.default in HOST_ONLY
    # 视图族与 `to.dtype` 已下设备：逐参 dtype 读入修好后，解码对拍通过
    # （`test_strategy_sweep`）。它们不再留在黑名单里。
    assert torch.ops.aten.view.default not in HOST_ONLY
    from graph.partition import _is_host_only
    assert not _is_host_only(torch.ops.aten.to.dtype)
    assert not _is_host_only(torch.ops.aten.reshape.default)
    assert not _is_host_only(torch.ops.aten.transpose.int)
    # 切片、拼接、查表也已下设备：列表实参摊平、子区间实参按字面量编码之后，
    # 命令编码都表达得了。
    assert not _is_host_only(torch.ops.aten.cat.default)
    assert not _is_host_only(torch.ops.aten.slice.Tensor)
    assert not _is_host_only(torch.ops.aten.embedding.default)
    # 真正留主机的只剩脚手架与图入口的位置序列。
    assert _is_host_only(torch.ops.aten._assert_tensor_metadata.default)
    assert _is_host_only(torch.ops.aten.arange.start)
    assert not _is_host_only(torch.ops.aten.scaled_dot_product_attention.default)
    # 逐元素算术**不**在黑名单里——它们是设备算子。
    assert torch.ops.aten.mul.Tensor not in HOST_ONLY
    assert torch.ops.aten.silu.default not in HOST_ONLY
    assert nodes["input"].meta[DEVICE_META_KEY] == DEVICE_HOST
    assert nodes["add"].meta[DEVICE_META_KEY] == DEVICE_DPU
    assert nodes["relu"].meta[DEVICE_META_KEY] == DEVICE_HOST
    assert nodes["mul"].meta[DEVICE_META_KEY] == DEVICE_DPU
    assert [(partition.part_id, partition.nodes) for partition in partitions] == [
        (0, [nodes["add"]]),
        (1, [nodes["mul"]]),
    ]
    assert nodes["add"].meta[PART_ID_META_KEY] == 0
    assert nodes["mul"].meta[PART_ID_META_KEY] == 1
    assert PART_ID_META_KEY not in nodes["relu"].meta


def test_partition_keeps_direct_dpu_fork_and_join_together() -> None:
    graph = Graph()
    input_node = graph.placeholder("input")
    source = graph.call_function(torch.ops.aten.add.Tensor, (input_node, 1))
    left = graph.call_function(torch.ops.aten.mul.Tensor, (source, 2))
    right = graph.call_function(torch.ops.aten.tanh.default, (source,))
    joined = graph.call_function(torch.ops.aten.add.Tensor, (left, right))
    graph.output(joined)
    gm = GraphModule({}, graph)

    partitions = partition_graph(gm)

    assert len(partitions) == 1
    assert partitions[0].part_id == 0
    assert partitions[0].nodes == [source, left, right, joined]
    assert {node.meta[PART_ID_META_KEY] for node in partitions[0].nodes} == {0}


def test_partition_does_not_merge_dpu_consumers_across_a_host_fan_out() -> None:
    graph = Graph()
    input_node = graph.placeholder("input")
    # 扇出点要用真正留主机的算子（同 `_module_with_host_break` 的理由）。
    relu = graph.call_function(
        torch.ops.aten._assert_tensor_metadata.default, (input_node,))
    left = graph.call_function(torch.ops.aten.add.Tensor, (relu, 1))
    right = graph.call_function(torch.ops.aten.mul.Tensor, (relu, 2))
    graph.output((left, right))
    gm = GraphModule({}, graph)

    partitions = partition_graph(gm)

    assert [(partition.part_id, partition.nodes) for partition in partitions] == [
        (0, [left]),
        (1, [right]),
    ]


def test_partition_numbers_disconnected_components_by_fx_order() -> None:
    graph = Graph()
    first_input = graph.placeholder("first_input")
    second_input = graph.placeholder("second_input")
    first = graph.call_function(torch.ops.aten.add.Tensor, (first_input, 1))
    second = graph.call_function(torch.ops.aten.mul.Tensor, (second_input, 2))
    graph.output((first, second))
    gm = GraphModule({}, graph)

    partitions = partition_graph(gm)

    assert [(partition.part_id, partition.nodes) for partition in partitions] == [
        (0, [first]),
        (1, [second]),
    ]


def test_partition_replaces_stale_metadata_without_touching_other_metadata() -> None:
    gm, nodes = _module_with_host_break()
    nodes["relu"].meta[PART_ID_META_KEY] = 99
    nodes["relu"].meta["sentinel"] = "preserve-me"
    nodes["add"].meta["sentinel"] = "preserve-me-too"

    first = partition_graph(gm)
    second = partition_graph(gm)

    assert [partition.nodes for partition in second] == [partition.nodes for partition in first]
    assert nodes["relu"].meta[DEVICE_META_KEY] == DEVICE_HOST
    assert PART_ID_META_KEY not in nodes["relu"].meta
    assert nodes["relu"].meta["sentinel"] == "preserve-me"
    assert nodes["add"].meta["sentinel"] == "preserve-me-too"


class _FixedMaskLlama(torch.nn.Module):
    def __init__(self, model: LlamaForCausalLM) -> None:
        super().__init__()
        self.model = model

    def forward(self, input_ids: torch.Tensor, causal_mask: torch.Tensor) -> torch.Tensor:
        return self.model(
            input_ids=input_ids,
            attention_mask=causal_mask,
            use_cache=False,
            return_dict=True,
        ).logits


def _export_random_llama() -> GraphModule:
    sequence_length = 16
    torch.manual_seed(0)
    model = LlamaForCausalLM(
        LlamaConfig(
            vocab_size=32000,
            hidden_size=64,
            intermediate_size=176,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=4,
            max_position_embeddings=sequence_length,
            bos_token_id=1,
            eos_token_id=2,
            pad_token_id=0,
        )
    ).eval()
    input_ids = torch.arange(sequence_length, dtype=torch.long).unsqueeze(0)
    blocked = torch.triu(torch.ones(sequence_length, sequence_length, dtype=torch.bool), diagonal=1)
    causal_mask = torch.zeros((1, 1, sequence_length, sequence_length), dtype=torch.float32)
    causal_mask.masked_fill_(blocked, torch.finfo(causal_mask.dtype).min)
    return torch.export.export(
        _FixedMaskLlama(model),
        (input_ids, causal_mask),
        strict=True,
    ).module()


def test_partition_covers_the_strictly_exported_random_llama_graph() -> None:
    gm = _export_random_llama()

    partitions = partition_graph(gm)
    nodes = list(gm.graph.nodes)
    dpu_nodes = [node for node in nodes if node.meta[DEVICE_META_KEY] == DEVICE_DPU]
    host_nodes = [node for node in nodes if node.meta[DEVICE_META_KEY] == DEVICE_HOST]
    listed_nodes = [node for partition in partitions for node in partition.nodes]

    assert dpu_nodes
    assert host_nodes
    assert set(listed_nodes) == set(dpu_nodes)
    assert len(listed_nodes) == len(set(listed_nodes))
    assert {partition.part_id for partition in partitions} == set(range(len(partitions)))
    assert {node.meta[PART_ID_META_KEY] for node in dpu_nodes} == set(range(len(partitions)))
    assert all(PART_ID_META_KEY not in node.meta for node in host_nodes)


def test_every_device_op_has_a_shard_rule_and_a_kernel() -> None:
    """图上每个设备侧算子都要**同时**有切分规则和 numpy 内核。

    这是黑名单口径的闭合条件。白名单时代漏一个算子的后果是它静默留在主机上：
    图照样跑完、数值也对，只是那个算子根本没在设备上执行——正是本项目要消掉的
    失真。黑名单把这种漏变成了两种显式失败：

      - 缺切分规则 → `propagate_specs` 抛错（本测试第一段断言）
      - 缺内核     → 后端 `submit` 时找不到 kernel

    所以这两张表必须与分区口径同步。这里按**真实导出的 llama 图**来查，
    而不是按手写的合成图——只有真实图才会带出 GQA 的广播视图、dtype 转换的
    各种重载这些容易漏的形态。
    """
    from graph.spec_prop import RULE_TABLE
    from runtime.kernels import _KERNELS

    gm = _export_random_llama()
    partition_graph(gm)

    device_targets = {
        node.target
        for node in gm.graph.nodes
        if node.op == "call_function"
        and node.meta[DEVICE_META_KEY] == DEVICE_DPU
    }
    assert device_targets, "这张图上应当有设备侧算子"

    missing_rules = sorted(str(t) for t in device_targets if t not in RULE_TABLE)
    assert not missing_rules, (
        f"这些算子被判为设备侧但没有切分规则: {missing_rules}。"
        f"要么补 RULE_TABLE，要么列进 partition.HOST_ONLY 并说明理由"
    )

    missing_kernels = sorted(
        str(t) for t in device_targets if str(t) not in _KERNELS)
    assert not missing_kernels, (
        f"这些算子被判为设备侧但没有 numpy 内核: {missing_kernels}"
    )


def test_rms_norm_and_silu_run_on_the_device() -> None:
    """RMSNorm 的三步与 SiLU 必须在设备上，不能落主机。

    它们原来全在主机：白名单只有 linear/add/mul/tanh/addmm 五个。这条断言是
    「不准静默落主机」的具体化——RMSNorm 在 aten 里是
    `pow -> mean -> add -> rsqrt -> mul` 一串，五步里有三步原来没人管。
    """
    gm = _export_random_llama()
    partition_graph(gm)
    device_targets = {
        node.target
        for node in gm.graph.nodes
        if node.op == "call_function"
        and node.meta[DEVICE_META_KEY] == DEVICE_DPU
    }

    for target in (
        torch.ops.aten.pow.Tensor_Scalar,   # RMSNorm: x²
        torch.ops.aten.mean.dim,            # RMSNorm: 均方
        torch.ops.aten.rsqrt.default,       # RMSNorm: 1/√
        torch.ops.aten.silu.default,        # MLP 的门控激活
        torch.ops.aten.neg.default,         # RoPE 的旋转半区
    ):
        assert target in device_targets, f"{target} 应当在设备上执行"


def test_attention_runs_on_the_device() -> None:
    """注意力是**设备**节点：读写 KV 缓存那段在设备内核里。

    它一度留在主机（`make_sdpa_handler`），理由是「通用设备通路会绕开它、
    KV 缓存不再被更新」。现在那段计算搬进了 `runtime/kernels.sdpa_kernel`，
    缓存本来就在 DPU 本地 MRAM 里，所以不再需要主机回调。

    留主机的是脚手架与图入口的位置序列——真正动张量的算子都走 PIM。
    """
    gm = _export_random_llama()
    partition_graph(gm)
    sdpa = [
        node for node in gm.graph.nodes
        if node.op == "call_function"
        and node.target is torch.ops.aten.scaled_dot_product_attention.default
    ]
    assert sdpa, "这张图上应当有注意力节点"
    for node in sdpa:
        assert node.meta[DEVICE_META_KEY] == DEVICE_DPU


def test_only_scaffolding_stays_on_the_host() -> None:
    """除脚手架与位置序列外，图上没有算子留在主机。

    这是「几乎所有算子都走 PIM」的判据本身：黑名单里每多一个真算子，这条就红。
    """
    gm = _export_random_llama()
    partition_graph(gm)
    host_targets = {
        str(node.target) for node in gm.graph.nodes
        if node.op == "call_function"
        and node.meta.get(DEVICE_META_KEY) == DEVICE_HOST
    }
    scaffolding = {
        "aten._assert_tensor_metadata.default",
        "aten.arange.start",
        "aten.arange.default",
        # 高阶算子与内置函数：`getitem` 取输出元组的一项，
        # `wrap_with_set_grad_enabled` 是导出的包装，都没有张量计算。
        "<built-in function getitem>",
        "wrap_with_set_grad_enabled",
    }
    assert host_targets <= scaffolding, (
        f"这些算子留在了主机，但它们不是脚手架：{sorted(host_targets - scaffolding)}")


def test_no_kernel_is_registered_for_a_host_only_op() -> None:
    """反向闭合：`_KERNELS` 里不许有主机侧算子的条目（评审 20260923 的 P0-3）。

    正向那条（上面 `test_every_device_op_has_a_shard_rule_and_a_kernel`）查的是
    「设备算子有没有内核」。只有正向时，反向的错是**按构造的死代码**：内核写了、
    注册了、`register_all` 也挂上去了，但那个算子永远被判成主机，后端 `submit`
    永远收不到它。既不报错，也没有任何测试会失败——它只是永远不执行。

    实测这一度有 8 条：cat / embedding / permute / reshape / slice / to.dtype /
    transpose.int / view；后来注意力也搬下了设备。它们现在**全部下设备**，
    各自的规则与内核都在这两个表里，所以这条反向闭合不再有豁免项。

    判据用 `_is_host_only` 而不是 `HOST_ONLY` 集合：`to` 这类是按**算子名整族**
    留主机的，只查集合会漏掉它们。
    """
    import torch

    from graph.partition import _is_host_only
    from runtime.kernels import _KERNELS

    # `_KERNELS` 的键是 `str(OpOverload)`；反查回目标才能问 `_is_host_only`。
    by_str = {}
    for packet_name in dir(torch.ops.aten):
        packet = getattr(torch.ops.aten, packet_name)
        for overload in getattr(packet, "overloads", lambda: [])():
            target = getattr(packet, overload)
            by_str[str(target)] = target

    unreachable = sorted(
        key for key in _KERNELS
        if key in by_str and _is_host_only(by_str[key])
    )
    assert not unreachable, (
        f"这些内核挂在主机侧算子上，按构造永不执行: {unreachable}。"
        f"要么让算子下设备，要么把内核从 _KERNELS 里删掉——两头都写就是死代码"
    )
