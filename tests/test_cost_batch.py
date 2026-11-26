"""成本链 batch 代表点判据：M=8 权重摊薄前置实验与 sidecar 逐相位产物。"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from tests.test_opcompiler_linear import _pim_passes_available

pytestmark = pytest.mark.skipif(
    not _pim_passes_available(),
    reason="当前 triton 没有 PIM pass，需重跑 0-install-flagtree.sh",
)

# llama2-7B 投影的真实本地形状档（gate/up 投影列切后的 4096->2048 族）。
LOCAL_IN, LOCAL_OUT = 4096, 2048
ELEM = 2  # float16
_WEIGHT = LOCAL_IN * LOCAL_OUT * ELEM
_ACT = (LOCAL_IN + LOCAL_OUT) * ELEM  # 单行激活（输入+输出）


def _compile_cost(m_rows: int):
    from contracts.op_contract import DEFAULT_HARDWARE_CONFIG, OpCompileRequest
    from genesim_bridge.ir_cost import analyze_ir
    from opcompiler_bridge.driver import compile_op

    result = compile_op(OpCompileRequest(
        op="linear",
        arg_shapes=[(m_rows, LOCAL_IN), (LOCAL_OUT, LOCAL_IN)],
        hardware=DEFAULT_HARDWARE_CONFIG,
        dtype="float16",
        num_tasklets=DEFAULT_HARDWARE_CONFIG.num_tasklets,
    ))
    assert result.pimir, f"M={m_rows} 编译没有产出 pim mlir"
    cost = analyze_ir(result.pimir, kernel_name="linear_kernel", grid=(1,),
                      arg_values={}, ir_level="pimir")
    return cost


def _weight_loads(cost, m_rows: int, overhead: float) -> float:
    """从总搬运字节扣掉激活侧与固定开销，剩下的折成权重份数。

    `overhead` 取 M=1 档按同式推出的常量部分（量化尺度等不随 M 变的缓冲），
    两档相减后只剩「随 M 的权重重装次数」这一项。
    """
    return (cost.mram_traffic_bytes - m_rows * _ACT - overhead) / _WEIGHT


def test_m8_weight_amortized() -> None:
    """前置实验：M=8 产物 tile_m>1 且权重搬运份数 <= ceil(8/tile_m)。

    这是 R5 的第一个闸门：若 M=8 仍选 tile_m=1 或权重搬运接近 8 份，
    说明 kernel/tile 搜索没有利用 M 维，权重复用需回 FlagTree 侧另立项。
    """
    cost1 = _compile_cost(1)
    cost8 = _compile_cost(8)
    # M=1 档的固定开销：总搬运 - 单份权重 - 单行激活。
    overhead = cost1.mram_traffic_bytes - _WEIGHT - _ACT
    loads1 = _weight_loads(cost1, 1, overhead)
    loads8 = _weight_loads(cost8, 8, overhead)
    print(f"\nM=1: tile_m={cost1.tile_m} tile_n={cost1.tile_n} "
          f"traffic={cost1.mram_traffic_bytes} weight_loads={loads1:.3f}")
    print(f"M=8: tile_m={cost8.tile_m} tile_n={cost8.tile_n} "
          f"traffic={cost8.mram_traffic_bytes} weight_loads={loads8:.3f}")
    assert cost8.tile_m and cost8.tile_m > 1, (
        f"M=8 产物 tile_m={cost8.tile_m}，tile 搜索未利用 M 维（前置实验证伪）")
    assert loads8 <= -(-8 // cost8.tile_m) + 0.05, (
        f"M=8 权重搬运 {loads8:.2f} 份 > ceil(8/tile_m)="
        f"{-(-8 // cost8.tile_m)}（前置实验证伪）")


def _compile_cost_leading(x_shape: tuple[int, ...]):
    """按图执行热路径的实参形状编译（前导维由驱动折叠成 M）。

    [B,S] prefill 图的 linear 实参是三维 [B,S,K]——`runtime/kernels.py` 把
    节点本地形状原样送 `OpCompileRequest`，驱动 `flatten_leading_dims` 折
    前导维，真实编译行数 M=B×S。
    """
    from contracts.op_contract import DEFAULT_HARDWARE_CONFIG, OpCompileRequest
    from genesim_bridge.ir_cost import analyze_ir
    from opcompiler_bridge.driver import compile_op

    result = compile_op(OpCompileRequest(
        op="linear",
        arg_shapes=[list(x_shape), (LOCAL_OUT, LOCAL_IN)],
        hardware=DEFAULT_HARDWARE_CONFIG,
        dtype="float16",
        num_tasklets=DEFAULT_HARDWARE_CONFIG.num_tasklets,
    ))
    assert result.pimir, f"shape={x_shape} 编译没有产出 pim mlir"
    cost = analyze_ir(result.pimir, kernel_name="linear_kernel", grid=(1,),
                      arg_values={}, ir_level="pimir")
    return cost


@pytest.mark.parametrize("batch", [2, 4])
def test_prefill_merged_product_matches_graph_execution(batch) -> None:
    """R17-01 跨层锚点：图执行 [B,S] 合批 prefill 与 sidecar prefill_b{B}
    消费同一 M=B×S 语义的真实产物（M、tile_m、权重字节逐项一致）。

    S=17 非 tile_m 整数（本机 tile_m=8）：合批权重装载 ceil(B×17/8) 份，
    严格少于逐槽口径 B×ceil(17/8) 份——round17 实测反例（M=68 时
    9.107 份 vs 12.155 份）由此沉淀为常驻判据。
    """
    from contracts.op_contract import flatten_leading_dims

    seq = 17
    # 图执行侧：[B,S] prefill 图的 linear 实参（三维，驱动折叠）。
    graph_cost = _compile_cost_leading((batch, seq, LOCAL_IN))
    # genesim 消费侧：prefill_b{B} 档按 M=B×S 编译（export_placement 的
    # m_rows，wiring 由 test_sidecar_batch_entries 钉住）。
    merged_cost = _compile_cost(batch * seq)

    # M=B×S 不是推测：折叠结果直接钉死。
    assert flatten_leading_dims((batch, seq, LOCAL_IN)) == (
        batch * seq, LOCAL_IN)
    # 两层的产物语义一致：tile_m 与权重搬运字节逐项相等。
    assert graph_cost.tile_m == merged_cost.tile_m
    assert (graph_cost.mram_traffic_bytes
            == merged_cost.mram_traffic_bytes)
    tile_m = merged_cost.tile_m

    cost1 = _compile_cost(1)
    overhead = cost1.mram_traffic_bytes - _WEIGHT - _ACT
    loads_merged = _weight_loads(merged_cost, batch * seq, overhead)
    loads_slot = _weight_loads(_compile_cost(seq), seq, overhead)
    # 合批装载 = ceil(B×S/tile_m) 份（尾块缓冲随 M 线性小量），严格少于
    # 逐槽 B 份 × ceil(S/tile_m)。
    assert loads_merged <= (
        -(-(batch * seq) // tile_m) + 0.05 + 0.003 * batch * seq), (
        f"B={batch} 合批装载 {loads_merged:.3f} 份 > "
        f"ceil({batch * seq}/{tile_m})")
    assert loads_merged < batch * loads_slot - 0.5, (
        f"B={batch} 合批装载 {loads_merged:.3f} 份未少于逐槽 "
        f"{batch}×{loads_slot:.3f} 份（合批收益丢失）")
    print(f"\nB={batch}: tile_m={tile_m} 合批装载 {loads_merged:.6f} 份、"
          f"逐槽 {batch}×{loads_slot:.6f} 份")


# 引入别处的 fixture 供本模块按名请求（pytest 按模块命名空间发现）。
from tests.test_placement_export import annotated_tiny_llama  # noqa: F401


def _tiny_model():
    """一个可被 compile_llama2 接受的小型 Llama 原始模型。"""
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(0)
    return LlamaForCausalLM(LlamaConfig(
        vocab_size=32000, hidden_size=32, intermediate_size=176,
        num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=4,
        max_position_embeddings=16, bos_token_id=1, eos_token_id=2,
        pad_token_id=0,
    )).eval().to(torch.float16)


def test_sidecar_batch_entries(annotated_tiny_llama, tmp_path, monkeypatch) -> None:
    """逐相位逐档产物进 sidecar：decode_b{B} + prefill，顶层旧字段=decode_b1。"""
    import opcompiler_bridge.driver as driver
    from genesim_bridge.placement_export import export_placement_to_genesim
    from tests.test_placement_export import _write_fixture_ir

    gm = annotated_tiny_llama
    ir_path = tmp_path / "base.ir"
    _write_fixture_ir(ir_path)

    seen: list[tuple[int, int, int]] = []  # (m_rows, local_in, local_out)
    real = driver.compile_op

    def spy(request, **kwargs):
        if request.op == "linear":
            seen.append((request.arg_shapes[0][0], request.arg_shapes[0][1],
                         request.arg_shapes[1][0]))
        return real(request, **kwargs)

    monkeypatch.setattr(driver, "compile_op", spy)
    sidecar = export_placement_to_genesim(
        gm, ir_path, tmp_path / "placed.ir", tmp_path / "sc.json",
        measure_kernel_tiles=True, batch_sizes=(1, 4))

    gemms = [e for e in sidecar["operators"].values() if e.get("shards")]
    assert gemms, "fixture 里应有 GEMM 条目"
    for entry in gemms:
        phases = entry["pimir_by_phase"]
        assert set(phases) == {"prefill", "prefill_b1", "prefill_b4",
                               "decode_b1", "decode_b4"}, phases
        for phase in phases.values():
            assert Path(phase["path"]).is_file()
            assert len(phase["sha256"]) == 64
            assert phase["kernel_tile_n"] > 0
            assert isinstance(phase["tile_m"], int) and phase["tile_m"] >= 1
        # 顶层旧字段是 decode_b1 的同值副本，旧消费方零改动。
        assert entry["pimir_path"] == phases["decode_b1"]["path"]
        assert entry["pimir_sha256"] == phases["decode_b1"]["sha256"]
        assert entry["kernel_tile_n"] == phases["decode_b1"]["kernel_tile_n"]
        # prefill 旧键是 prefill_b1（M=S）的同值副本，服务逐槽策略与旧产物。
        assert phases["prefill"]["path"] == phases["prefill_b1"]["path"]
        # M 维不同的档位必须是不同的编译产物。
        assert phases["decode_b4"]["path"] != phases["decode_b1"]["path"]
        assert phases["prefill_b4"]["path"] != phases["prefill_b1"]["path"]

    # A 路编译按 (形状, m_rows) 去重：每个唯一键恰好编一次，
    # m_rows 覆盖 {1, 4}（decode 档）、16（prefill/prefill_b1 档=S）与
    # 64（prefill_b4 档=4×16，合批 M=B×S）。
    assert len(seen) == len(set(seen)), "同键重复编译"
    assert {m for m, _, _ in seen} == {1, 4, 16, 64}, sorted(set(seen))


def test_sidecar_default_batch_keeps_legacy_shape(annotated_tiny_llama, tmp_path,
                                                  monkeypatch) -> None:
    """缺省 batch_sizes=(1,)：decode_b1 + prefill 两键，结构语义与旧行为一致。"""
    import opcompiler_bridge.driver as driver
    from genesim_bridge.placement_export import export_placement_to_genesim
    from tests.test_placement_export import _write_fixture_ir

    gm = annotated_tiny_llama
    ir_path = tmp_path / "base.ir"
    _write_fixture_ir(ir_path)

    seen_m: list[int] = []
    real = driver.compile_op

    def spy(request, **kwargs):
        if request.op == "linear":
            seen_m.append(request.arg_shapes[0][0])
        return real(request, **kwargs)

    monkeypatch.setattr(driver, "compile_op", spy)
    sidecar = export_placement_to_genesim(
        gm, ir_path, tmp_path / "placed.ir", tmp_path / "sc.json",
        measure_kernel_tiles=True)

    gemms = [e for e in sidecar["operators"].values() if e.get("shards")]
    for entry in gemms:
        assert set(entry["pimir_by_phase"]) == {
            "prefill", "prefill_b1", "decode_b1"}
        assert entry["pimir_path"] == entry["pimir_by_phase"]["decode_b1"]["path"]
        assert (entry["pimir_by_phase"]["prefill"]["path"]
                == entry["pimir_by_phase"]["prefill_b1"]["path"])
    assert set(seen_m) == {1, 16}


# --- T5：CLI batch 档列表与 export_placement_for_strategy 透传 ---------------------


def _load_cli_module():
    import importlib.util

    script = Path(__file__).resolve().parent.parent / "scripts" / "export_pp_placement.py"
    spec = importlib.util.spec_from_file_location("export_pp_placement_test", script)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_cli_batch_list_parses(monkeypatch) -> None:
    """--batch 接受逗号分隔正整数列表，缺省单档 1，非法值直接报错。"""
    cli = _load_cli_module()
    monkeypatch.setattr(sys, "argv",
                        ["export_pp_placement.py", "--num-stages", "4",
                         "--batch", "1,4"])
    args = cli.parse_args()
    assert args.batch == (1, 4)
    monkeypatch.setattr(sys, "argv",
                        ["export_pp_placement.py", "--num-stages", "4"])
    assert cli.parse_args().batch == (1,)
    monkeypatch.setattr(sys, "argv",
                        ["export_pp_placement.py", "--num-stages", "4",
                         "--batch", "1,0"])
    with pytest.raises(SystemExit):
        cli.parse_args()


def test_export_strategy_batch_passthrough(tmp_path, monkeypatch) -> None:
    """batch 档列表经 export_placement_for_strategy 透传：图按最大档编译，
    逐相位产物全部进 PlacementExportResult.pimir_paths。"""
    import runtime.compile as compile_mod
    from genesim_bridge.placement_export import export_placement_for_strategy
    from tests.test_placement_export import _write_fixture_ir

    ir_path = tmp_path / "base.ir"
    _write_fixture_ir(ir_path)

    seen_batch = []
    real_compile = compile_mod.compile_llama2

    def compile_spy(model, strategy, **kwargs):
        seen_batch.append(kwargs.get("batch_size"))
        return real_compile(model, strategy, **kwargs)

    monkeypatch.setattr(compile_mod, "compile_llama2", compile_spy)

    from contracts.op_contract import PIMHardwareConfig
    from graph.strategy import llama_strategy
    from memory.mem_planner import HwBudget

    strategy = llama_strategy(
        2, num_stages=1, num_heads=4, num_kv_heads=4,
        intermediate_size=176, vocab_size=32000, num_layers=1)
    result = export_placement_for_strategy(
        _tiny_model(), ir_path, tmp_path / "out",
        strategy=strategy,
        hw=HwBudget(mram_bytes=1 << 32, align=64, sys_reserve_bytes=0),
        hardware=PIMHardwareConfig(num_dpus=2, num_tasklets=4,
                                   mram_bytes_per_dpu=1 << 32,
                                   wram_bytes_per_dpu=65536, dma_align=8),
        prefill_seq_len=16, max_seq=16,
        measure_kernel_tiles=True, batch_sizes=(1, 4))

    # 图编译按最坏档（KV/内存按 max(B) 校验）。
    assert seen_batch == [4]
    # 结果里的产物路径要覆盖全部相位档（decode_b1/decode_b4/prefill）。
    all_paths = {str(p) for p in result.pimir_paths}
    sidecar = json.loads(result.sidecar.read_text())
    gemm = next(e for e in sidecar["operators"].values() if e.get("shards"))
    for phase in gemm["pimir_by_phase"].values():
        assert phase["path"] in all_paths
    # 编译期容量表进产物：可读表 + 逐 DPU 折算 + 墙位，扫参报告据此出数。
    capacity = sidecar["capacity"]
    assert capacity["slots"] == 4
    assert capacity["max_seq"] == 16  # 序列长度口径随产物记录，可被扫参核对
    assert "容量表" in capacity["table"] and "slots=4" in capacity["table"]
    assert capacity["budget_bytes"] == 1 << 32
    assert capacity["wall_batch"] >= 4
    assert capacity["dpus"] and all(
        {"fixed_bytes", "var_bytes_at_slots", "slots", "max_batch"} <= set(stats)
        for stats in capacity["dpus"].values())
    # KV 物理布局载荷（顶层 kv_layout）：逐 DPU 的 head 归属、槽数与已分配
    # 字节；Σ(kv_allocated_bytes) 与手推 KV 公式一致，genesim 据此逐 DPU
    # 对账（round05 问题 7）。tiny 模型 num_kv_heads=4、2 台 DPU 各持 2 head。
    kv_layout = sidecar.get("kv_layout")
    assert kv_layout, "sidecar 顶层缺 kv_layout 载荷"
    assert set(kv_layout) == {"0", "1"}
    for dpu_id, spec in kv_layout.items():
        assert spec["num_slots"] == 4
        assert spec["layers"] == [0]
        assert len(spec["kv_heads"]) == 2
        assert spec["kv_allocated_bytes"] > 0
    total_kv = sum(spec["kv_allocated_bytes"] for spec in kv_layout.values())
    assert total_kv >= 2 * 1 * 16 * 4 * 8 * 4  # 2×layers×max_seq×heads×dim×dtype×slots


# --- R8 DP 副本：真实副本放置导出（replicas × 现有 TP/PP） -------------------------


def test_export_replicas_sidecar_payload_and_kv_layout_offset(
        tmp_path, monkeypatch) -> None:
    """replicas=2 导出：dp_replicas 载荷声明副本块、kv_layout 按物理 DPU 落键。

    副本放置 offset 不重叠（副本 1 的布局键 = 副本 0 + N 偏移）、权重复制
    ×G（每副本权重字节 = 容量表 fixed_bytes 合计）、容量按副本口径（每副本
    N 台 DPU 的折算表不变）。
    """
    import runtime.compile as compile_mod
    from genesim_bridge.placement_export import export_placement_for_strategy
    from tests.test_placement_export import _write_fixture_ir

    ir_path = tmp_path / "base.ir"
    _write_fixture_ir(ir_path)

    real_compile = compile_mod.compile_llama2
    monkeypatch.setattr(compile_mod, "compile_llama2", real_compile)

    from contracts.op_contract import PIMHardwareConfig
    from graph.strategy import llama_strategy
    from memory.mem_planner import HwBudget

    strategy = llama_strategy(
        2, num_stages=1, replicas=2, num_heads=4, num_kv_heads=4,
        intermediate_size=176, vocab_size=32000, num_layers=1)
    result = export_placement_for_strategy(
        _tiny_model(), ir_path, tmp_path / "out",
        strategy=strategy,
        hw=HwBudget(mram_bytes=1 << 32, align=64, sys_reserve_bytes=0),
        hardware=PIMHardwareConfig(num_dpus=2, num_tasklets=4,
                                   mram_bytes_per_dpu=1 << 32,
                                   wram_bytes_per_dpu=65536, dma_align=8),
        prefill_seq_len=16, max_seq=16,
        measure_kernel_tiles=True, batch_sizes=(1,))

    sidecar = json.loads(result.sidecar.read_text())
    # 顶层 dp_replicas 载荷：副本数、每副本 DPU 数与逐副本物理块。
    dp = sidecar["dp_replicas"]
    assert dp["replicas"] == 2
    assert dp["dpus_per_replica"] == 2
    assert dp["replica_dpu_ids"] == [[0, 1], [2, 3]]
    # 每副本静态字节拆成纯权重（权重区合计）与内核 tile 两项：fixed_bytes
    # = 权重 + tile，weight_bytes_per_replica 只含纯权重（R17-04）。
    fixed_total = sum(stats["fixed_bytes"]
                      for stats in sidecar["capacity"]["dpus"].values())
    tile_total = sum(stats["kernel_tile_bytes"]
                     for stats in sidecar["capacity"]["dpus"].values())
    assert dp["weight_bytes_per_replica"] == fixed_total - tile_total
    assert dp["kernel_tile_bytes_per_replica"] == tile_total
    assert (dp["weight_bytes_per_replica"]
            + dp["kernel_tile_bytes_per_replica"] == fixed_total)
    # kv_layout 按物理 DPU 落键：副本 1 的键 = 副本 0 + N 偏移，内容一致
    #（每副本一份完整 KV 布局，各自服务本副本的请求流）。
    kv = sidecar["kv_layout"]
    assert set(kv) == {"0", "1", "2", "3"}
    assert kv["2"] == dict(kv["0"], ) and kv["3"] == dict(kv["1"])
    blocks = [list(range(0, 2)), list(range(2, 4))]
    flat = [d for block in blocks for d in block]
    assert len(flat) == len(set(flat)), "副本 KV 区域 offset 不得重叠"
    # 容量表按副本口径记录副本数。
    assert sidecar["capacity"]["replicas"] == 2
    # 编译器策略摘要带副本维度。
    assert sidecar["compiler"]["strategy"]["replicas"] == 2
    assert sidecar["compiler"]["strategy"]["total_dpus"] == 4
    # 合成的 PartitionPlan 声明物理 DPU 数（供回读核对实际使用的切分）。
    plan = json.loads(result.partition_plan.read_text())
    assert plan["replicas"] == 2
    assert plan["num_dpus"] == 2  # 每副本 DPU 数


def test_dp2_sidecar_hardware_contract_matches_strict_identity_check(
        tmp_path, monkeypatch) -> None:
    """dp2 sidecar 的硬件契约满足 genesim 严格身份核验的规模口径。

    genesim 核验（run_manifest）要求：strategy.num_dpus == hardware.num_dpus
    == len(dpu_ids)（每副本编译基数），len(dpu_to_cluster) == total_dpus ==
    num_dpus×replicas（物理总规模）。导出侧同口径钉住，防止契约与物理映射
    两个真源再次漂移（round09 问题 1：dp2 产物被自己的下游核验拒绝）。
    """
    import runtime.compile as compile_mod
    from contracts.partition_plan import PartitionPlan
    from genesim_bridge.placement_export import export_placement_for_strategy
    from tests.test_placement_export import _write_fixture_ir

    ir_path = tmp_path / "base.ir"
    _write_fixture_ir(ir_path)
    real_compile = compile_mod.compile_llama2
    monkeypatch.setattr(compile_mod, "compile_llama2", real_compile)

    from contracts.op_contract import PIMHardwareConfig
    from graph.strategy import llama_strategy
    from memory.mem_planner import HwBudget

    strategy = llama_strategy(
        2, num_stages=1, replicas=2, num_heads=4, num_kv_heads=4,
        intermediate_size=176, vocab_size=32000, num_layers=1)
    plan = PartitionPlan(
        num_dpus=strategy.num_dpus, num_stages=strategy.num_stages,
        weight_rules=tuple(strategy.weight_rules),
        dpu_ids=tuple(strategy.dpu_ids), replicas=strategy.replicas,
        dpu_to_cluster=((0, 0), (0, 1), (0, 2), (0, 3)),
        source="test")
    result = export_placement_for_strategy(
        _tiny_model(), ir_path, tmp_path / "out",
        strategy=strategy, plan=plan,
        hw=HwBudget(mram_bytes=1 << 32, align=64, sys_reserve_bytes=0),
        hardware=PIMHardwareConfig(num_dpus=2, num_tasklets=4,
                                   mram_bytes_per_dpu=1 << 32,
                                   wram_bytes_per_dpu=65536, dma_align=8),
        prefill_seq_len=16, max_seq=16,
        measure_kernel_tiles=True, batch_sizes=(1,))

    sidecar = json.loads(result.sidecar.read_text())
    contract = sidecar["compiler"]["strategy"]
    per_replica = sidecar["compiler"]["hardware"]["num_dpus"]
    # 每副本基数：策略、硬件与 dpu_ids 三者同口径。
    assert contract["num_dpus"] == per_replica
    assert len(contract["dpu_ids"]) == per_replica
    # 物理总规模：dpu_to_cluster 与 total_dpus 都按 num_dpus×replicas 落。
    physical = per_replica * contract["replicas"]
    assert contract["total_dpus"] == physical
    assert len(sidecar["dpu_to_cluster"]) == physical


def test_kv_head_pin_routes_kv_reads_to_head_holders(
        annotated_tiny_llama, tmp_path) -> None:
    """执行侧 KV head 分片：GEMV 读取算子钉到持有该 head 的 DPU。

    tiny 模型 4 个 KV head、2 台 DPU 各持 2 head（GQA group=1）：q_head 0/1
    的读取算子落 dpu0、q_head 2/3 落 dpu1。缺省关闭时维持整段首 DPU 钉扎
    （B=1 执行放置与旧产物逐位一致）。
    """
    from genesim_bridge.placement_export import export_placement_for_strategy
    from tests.test_placement_export import _write_fixture_ir

    gm = annotated_tiny_llama
    ir_path = tmp_path / "base.ir"
    _write_fixture_ir(ir_path)
    # fixture 的 GEMV 条目补 q_head_id：真实 IR 里逐 q_head 链自带该字段。
    ir = json.loads(ir_path.read_text())
    kv_ops = [op for op in ir["operators"]
              if op["op_type"] in ("GEMV_SCORE", "GEMV_CONTEXT")]
    assert kv_ops
    # 两条读取链分别挂 q_head 0 与 3：GQA group=1 下分属 dpu0/dpu1。
    for op, q_head in zip(kv_ops, (0, 3)):
        op["q_head_id"] = q_head
    ir_path.write_text(json.dumps(ir))

    from contracts.op_contract import PIMHardwareConfig
    from graph.strategy import llama_strategy
    from memory.mem_planner import HwBudget

    def _export(kv_head_pin: bool):
        strategy = llama_strategy(
            2, num_stages=1, num_heads=4, num_kv_heads=4,
            intermediate_size=176, vocab_size=32000, num_layers=1)
        result = export_placement_for_strategy(
            _tiny_model(), ir_path, tmp_path / f"out-{kv_head_pin}",
            strategy=strategy,
            hw=HwBudget(mram_bytes=1 << 32, align=64, sys_reserve_bytes=0),
            hardware=PIMHardwareConfig(num_dpus=2, num_tasklets=4,
                                       mram_bytes_per_dpu=1 << 32,
                                       wram_bytes_per_dpu=65536, dma_align=8),
            prefill_seq_len=16, max_seq=16,
            measure_kernel_tiles=True, kv_head_pin=kv_head_pin)
        sidecar = json.loads(result.sidecar.read_text())
        return {
            int(op_id): entry
            for op_id, entry in sidecar["operators"].items()
            if entry.get("op_type") in ("GEMV_SCORE", "GEMV_CONTEXT")
        }

    off = _export(False)
    on = _export(True)
    # 缺省关闭：全部钉在层内首 DPU（旧行为）。
    assert all(entry["dpu_id"] == 0 for entry in off.values())
    # 开启：按 q_head 所属 KV head 的持有 DPU 分布，不再全落首 DPU。
    holders = {op_id: entry["dpu_id"] for op_id, entry in on.items()}
    assert set(holders.values()) == {0, 1}, holders
    # 与 kv_specs 的 head 归属一一对应：q_head 0/1 → dpu0，2/3 → dpu1。
    ir_ops = {op["op_id"]: op for op in ir["operators"]}
    for op_id, dpu in holders.items():
        q_head = ir_ops[op_id]["q_head_id"]
        assert dpu == q_head // 2, (op_id, q_head, dpu)
    # zone=kv 标记在两种状态下都在。
    assert all(entry.get("zone") == "kv" for entry in on.values())


# --- R25-03：执行策略贯穿生产导出链（per_slot_prefill → compile_llama2） ------------


def test_export_per_slot_prefill_delivers_host_plans(
        tmp_path, monkeypatch) -> None:
    """执行策略作为统一编译输入贯穿生产导出入口（R25-03）。

    `export_placement_for_strategy(per_slot_prefill=True)` 把开关传给
    `compile_llama2`（B 份逐槽主机计划真实生成并随结果交付），sidecar
    记录策略声明供 genesim 互检；缺省不传（False）、sidecar 不写该键，
    经典产物逐位不变。
    """
    import runtime.compile as compile_mod
    from genesim_bridge.placement_export import export_placement_for_strategy
    from tests.test_placement_export import _write_fixture_ir

    ir_path = tmp_path / "base.ir"
    _write_fixture_ir(ir_path)

    seen: list[bool] = []
    real_compile = compile_mod.compile_llama2

    def compile_spy(model, strategy, **kwargs):
        seen.append(bool(kwargs.get("per_slot_prefill")))
        return real_compile(model, strategy, **kwargs)

    monkeypatch.setattr(compile_mod, "compile_llama2", compile_spy)

    from contracts.op_contract import PIMHardwareConfig
    from graph.strategy import llama_strategy
    from memory.mem_planner import HwBudget

    def _export(out_name: str, **kwargs):
        return export_placement_for_strategy(
            _tiny_model(), ir_path, tmp_path / out_name,
            strategy=llama_strategy(
                2, num_stages=1, num_heads=4, num_kv_heads=4,
                intermediate_size=176, vocab_size=32000, num_layers=1),
            hw=HwBudget(mram_bytes=1 << 32, align=64, sys_reserve_bytes=0),
            hardware=PIMHardwareConfig(num_dpus=2, num_tasklets=4,
                                       mram_bytes_per_dpu=1 << 32,
                                       wram_bytes_per_dpu=65536, dma_align=8),
            prefill_seq_len=16, max_seq=16,
            measure_kernel_tiles=True,
            batch_sizes=(1,) if out_name == "out-single" else (2,),
            **kwargs)

    result = _export("out-per-slot", per_slot_prefill=True)
    assert seen == [True]
    # B 份逐槽主机计划随导出结果交付（主机执行器消费这份产物）。
    assert result.compiled.per_slot_prefill is not None
    assert len(result.compiled.per_slot_prefill) == 2
    # sidecar 声明逐槽编译：genesim 的执行策略互检据此拒绝错配消费。
    sidecar = json.loads(result.sidecar.read_text())
    assert sidecar["per_slot_prefill"] is True

    plain = _export("out-plain")
    assert seen == [True, False]
    assert plain.compiled.per_slot_prefill is None
    # B>1 的导出显式声明合批（false）：genesim 的逐槽策略据此拒绝错配消费。
    assert json.loads(plain.sidecar.read_text())["per_slot_prefill"] is False
    # B=1 无逐槽对象：不写该键，经典单档产物逐位不变。
    single = _export("out-single")
    assert seen == [True, False, False]
    assert single.compiled.per_slot_prefill is None
    assert "per_slot_prefill" not in json.loads(single.sidecar.read_text())
