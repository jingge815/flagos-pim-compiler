# 技术设计文档：接入 Qwen3 MoE（Qwen3MoeForCausalLM）到 PIM 推理链路
> 文档编号：design-qwen3-moe-20261105
> 创建日期：2026-11-05
> 关联需求文档：request-qwen3-moe-20261105
> 关联评审文档：review-qwen3-moe-20261109-round01（§十为其补充契约）
> 关联项目：flagos-pim-compiler（Qwen3 dense 基线 263fcda；MoE 改动区间 3ce60b6..daef534）

## 一、设计概述

### 1.1 需求背景回顾
链路已支持 llama/qwen2/qwen3 dense；Qwen3 MoE（30B-A3B：48 层全 MoE、128 专家
top-8、`norm_topk_prob=True`）因 HF 前向数据依赖分派而无法导出，且不在架构白名单。

### 1.2 设计目标
`qwen3_moe` 注册进白名单；小随机 MoE 与真实 30B-A3B 全链路（加载→编译→执行）
贪心生成与 HF 逐 token 一致；P1 增加主机动态分派原语与专家并行放置维度。

### 1.3 设计范围
- 包含范围：静态稠密模拟前向、MoE 算子图规则、专家权重张量并行、聚合路径、
  真实 30B 端到端、动态分派原语、专家并行（16 DPU 验证）。
- 不包含范围：vLLM serve 实测、235B、量化、滑窗 MoE、tied 词嵌入。genesim MoE 计费
  原列为不包含，round01 评审后追加到 §十 并纳入交付（2026-11-09）。

## 二、现状基线（实测，均已在设计前验证）

1. **静态改写等价性**：把 HF `Qwen3MoeSparseMoeBlock.forward` 的「命中专家循环 +
   token 子集 gather」改为「topk/softmax/scatter + 全专家加权累加」后，4 专家小模型
   逐元素 **max|Δ|=0.00e+00**（`norm_topk_prob` 两种取值都验证）。
2. **导出算子集**（4 专家/2 层实测 export）：每层仅新增 5 种算子
   `aten.topk.default`、`aten.scatter.src`、`aten.softmax.int`、
   `aten.sum.dim_IntList`、`aten.zeros/zeros_like`；专家本体是既有
   linear/mul/add。30B 规模估算 ≈ 4.7 万节点/图（389 linear/层 × 48 层）。
3. **下沉机制**：`graph/partition.py:26` 的 `HOST_ONLY` 是**黑名单**——不在黑名单的
   OpOverload 一律判 DPU，`graph/spec_prop.py:645` 查不到规则直接报错（不静默）。
   host 节点由 `runtime/exec_plan_gen.py:345` 的通用回退闭包直调 aten target，
   输入侧自动插 all_gather/scatter 重分布边（spec_prop.py:613）——**host 路径零注册**。
4. **权重规则**：`graph/strategy.py:13` `LLAMA_WEIGHT_RULES` 按子串匹配，
   `experts.{i}.gate_proj.weight` 天然命中 `gate_proj.weight` 列切规则；
   `mlp.gate.weight`（router）不含任何现有子串。策略可行性实测：30B 几何在
   8 DPU 首个可行策略是 **tp4_pp2**（4 KV 头挡掉 tp8），16 DPU 为 tp4_pp4 等。
5. **30B-A3B 几何**（ModelScope config 实取）：hidden 2048、moe_intermediate 768、
   128 专家 top-8、`tie=False`、无滑窗；专家权重 fp16 54 GiB，总权重 ≈ 57 GiB。
6. **权重装载**：`runtime/compile.py:451 write_weight_shards` 按 get_attr 节点通用
   遍历，专家权重自动覆盖，无需改动。

## 三、总体设计

### 3.1 总体设计概述
- 解决的问题：MoE 的数据依赖分派进不了静态蓝图。
- 现状：黑名单式图下沉 + host 通用镜像已具备，缺的只是 MoE 算子的归类与注册。
- 方案：**门控链（router/softmax/topk/加权）整体留 host**（复用通用回退），
  **专家 SwiGLU 走 DPU 既有 linear 路径**（TP 切分复用现有规则）；P0 稠密模拟
  （全专家算完按门控加权，数值与 HF 位级一致的改写已验证）；P1 用 gate 标志位
  让主机跳过未命中专家的 launch。
- 数据流（★=新增，其余复用）：

```
hidden(Tok×H, DPU Replicate)
 ├─ router linear (DPU, mlp.gate.weight col 切★规则) ─ all_gather → host
 │    → softmax·fp32 → topk → renorm(sum) → scatter → routing (T,E)
 │                                        └→ scatter True → selected_mask (T,E, bool)
 │      [全 host★HOST_ONLY；命中只看 selected_mask，routing 只参与加权]
 └─ for e in experts (静态展开):
      gate/up linear (DPU col, 既有规则) → silu·mul (DPU) → down (DPU row→Partial→reduce)
      → where(selected_mask[:,e], ·, 0) + × routing[:,e] + 累加 add（host 闭包）
        [选择/加权/累加是一段共享主机工作；PIM 原语集没有逐元素 select]
 → 残差 add (DPU, 既有)                          P0 到此为止
 → [P1★] dispatch host_op 读 selected_mask → 置跳过标志 → 未命中专家 launch 直接返回
```

### 3.2 与现有逻辑的衔接
- 入口不变：`backend/vllm_worker.py:227` 按 `model_type` 查注册表，加 `"qwen3_moe"`
  即全链路生效；`validate_engine_config`（vllm_worker.py:87）白名单自动放行。
- 导出入口 `runtime/compile.py:84 export_annotated_graph` 不改：静态化在
  **加载时**完成（换 forward），图上只见普通 aten 算子。
- 测试入口对齐 `tests/test_architectures.py` 的 `_compile_and_load` 模式。

### 3.3 模块划分与职责边界
- `backend/architectures.py`：MoE 加载 + 静态化（唯一的新模型侧代码）。
- `graph/partition.py`+`graph/strategy.py`+`graph/spec_prop.py`：算子归类与规则。
- `runtime/executor.py`+`backend/hal_numpy.py`+`contracts/`：仅 P1 原语。
- `memory/`：不改（专家权重走既有三区规划，容量判据现成）。
- 不做：MoE 专用 DPU 内核（P0 数值不需要；softmax/topk 在 host）。

## 四、模块详细设计

### 4.1 静态 MoE 前向改写（需求 P0-1｜`backend/architectures.py`｜新增）
**功能点**：加载 qwen3_moe 后把每个 `Qwen3MoeSparseMoeBlock` 的 forward 换成
静态形状版（topk/softmax/scatter + 全专家加权累加），使 torch.export 可导出。
**原因**：HF 原版用 `expert_hit.nonzero()` 与按 token 子集 gather，形状依赖数据，
`torch.export` 直接报 `data-dependent expression`（实测）。
**代码对比**：
```python
# 原有（HF，transformers 4.57.6）
expert_hit = torch.greater(expert_mask.sum(dim=(-1,-2)), 0).nonzero()
for expert_idx in expert_hit:
    idx, top_x = torch.where(expert_mask[expert_idx].squeeze(0))
    current_state = hidden_states[None, top_x].reshape(-1, hidden_dim)
    ...index_add_(0, top_x, ...)

# 目标（make_static_moe_forward，闭包捕获 blk）
logits = blk.gate(xs)
probs = F.softmax(logits, dim=1, dtype=torch.float)
tw, sel = torch.topk(probs, blk.top_k, dim=-1)
if blk.norm_topk_prob:
    tw = tw / tw.sum(dim=-1, keepdim=True)
# 命中只由 top-k 选中索引决定；路由权重只参与加权。
selected_mask = torch.zeros_like(probs, dtype=torch.bool).scatter(1, sel, True)
routing = torch.zeros_like(probs).scatter(1, sel, tw.to(probs.dtype))
acc = _expert_out(0) * routing[:, 0:1].to(xs.dtype)          # 首专家直接做初值
for e in range(1, blk.num_experts):
    acc = acc + _expert_out(e) * routing[:, e:e+1].to(xs.dtype)
# _expert_out(e) = torch.where(selected_mask[:, e:e+1], blk.experts[e](xs), 0.0)
return acc.view(b, s, h), logits
```
**解决思路**：非选中专家由布尔选中掩码掩零、乘加恒等；被 top-k 选中但
权重下溢为 0 的专家仍然执行（HF 会算 `0×输出`，结果为 nan 时同样保留），
不能用 `routing != 0` 兼任命中条件。按专家号升序累加与 HF index_add 顺序
一致，已实测位级相等（`norm_topk_prob` 两态均验）。

### 4.2 注册 qwen3_moe 与 MoE 策略入口（需求 P0-1｜`backend/architectures.py:48`｜修改）
**功能点**：`ARCHITECTURES` 加 `"qwen3_moe"`；新增 `_moe_strategy` 把
`moe_intermediate_size`（而非遗留的 `intermediate_size`）传给策略枚举，
并要求 TP 宽度能整除 `num_experts`。
**原因**：30B 的 dense `intermediate_size=6144` 是配置残留，真正要按 TP 切的 FFN
是专家的 768；沿用 `_dense_strategy` 会切错宽度并误判容量。router 权重按
专家维列切，专家数不能被 TP 宽度整除时 1 专家模型选不出合法策略。
**代码对比**：
```python
# 原有
"qwen3": Architecture(_load_qwen3, _dense_strategy),

# 目标
def _load_qwen3_moe(path):
    model = Qwen3MoeForCausalLM.from_pretrained(path, dtype=torch.float16).eval()
    for layer in model.model.layers:
        layer.mlp.forward = make_static_moe_forward(layer.mlp)
    return model

def _moe_strategy(num_dpus, cfg):
    return llama_strategies(num_dpus, num_heads=cfg.num_attention_heads,
        num_kv_heads=cfg.num_key_value_heads,
        intermediate_size=cfg.moe_intermediate_size,   # 专家 FFN 宽
        vocab_size=cfg.vocab_size, num_layers=cfg.num_hidden_layers)[0]

"qwen3_moe": Architecture(_load_qwen3_moe, _moe_strategy),
```
**解决思路**：静态化在 from_pretrained 之后做，权重引用不变、save/load 语义不变；
`tests/test_architectures.py::test_qwen3_moe_rejected` 翻转为
`test_qwen3_moe_config_passes`（白名单防呆测试对齐新现实）。

### 4.3 门控算子留主机（需求 P0-2｜`graph/partition.py:26 HOST_ONLY`｜修改）
**功能点**：`HOST_ONLY` 增补 5 个条目：`aten.topk.default`、`aten.softmax.int`、
`aten.sum.dim_IntList`、`aten.scatter.src`、`aten.zeros_like.default`（zeros 已在
别处出现则不重复加）。
**原因**：这些算子不在黑名单会被判 DPU，`spec_prop.py:645` 因无规则直接编译失败
（黑名单哲学：宁可吵闹）；门控张量只有 T×E（30B 下 1×128），放 host 代价可忽略。
**代码对比**：
```python
# 原有（HOST_ONLY 片段）
HOST_ONLY = (torch.ops.aten.arange.default, ...)

# 目标（追加，每条带一行注释说明属 MoE 门控链）
HOST_ONLY = (
    ...,
    # MoE 门控链留主机：topk/scatter 依赖完整专家维，门控张量仅 T×E。
    torch.ops.aten.topk.default,
    torch.ops.aten.softmax.int,
    torch.ops.aten.sum.dim_IntList,
    torch.ops.aten.scatter.src,
    torch.ops.aten.zeros_like.default,
)
```
**解决思路**：host 节点走 `exec_plan_gen.py:345` 通用回退闭包直调 aten，输入由
spec_prop 自动插重分布边，无任何新内核或新 RULE_TABLE 条目。

### 4.4 router 权重切分规则（需求 P0-2｜`graph/strategy.py:13 LLAMA_WEIGHT_RULES`｜修改）
**功能点**：规则表追加 `("mlp.gate.weight", "col")`。
**原因**：router linear 被判 DPU，其权重无规则时 `spec_prop.py:233` 报
「被 linear 消费，但策略未指定其切法」；子串 `gate_proj.weight` 匹配不到
`mlp.gate.weight`（已核对无歧义命中）。
**代码对比**：
```python
# 原有
LLAMA_WEIGHT_RULES = (..., ("lm_head.weight", "col"), ...)

# 目标
LLAMA_WEIGHT_RULES = (
    ...,
    # MoE router：按专家维列切，各 DPU 出 E/tp 份 logits，host all_gather 后门控。
    ("mlp.gate.weight", "col"),
)
```
**解决思路**：router 输出 Shard(col) 经既有 all_gather 边拼成完整 (T,E) 给 host
softmax/topk；softmax 需要完整专家维，这一顺序保证语义正确。

### 4.5 专家累加链的布局打通（需求 P0-2/P0-4｜`graph/spec_prop.py`｜修改）
**功能点**：为「down_proj 的 Partial 输出 × routing 列」补一条 mul 规则
（Partial×Replicate 标量列 → Partial），复用既有 `_rule_*` 结构；E 路累加与残差
add 复用现有规则。
**原因**：dense 路径只有「Partial + Replicate 的残差 add」先例，加权乘
（Partial × host 标量列经 scatter 边下发）是新组合，缺规则会在 spec_prop 报错。
**代码对比**：
```python
# 原有（_rule_linear 的 Partial 分支节选）
return [(x_node, _dpu_req(Placement("Shard", out_dim))), ...], Placement("Partial", reduce_type="sum")

# 目标（新增 _rule_weighted_partial：标量列乘 Partial 不改变分片语义）
def _rule_weighted_partial(node, x, w):
    return [(x[0], _dpu_req(x[1].placement)), (w[0], _dpu_req(REPLICATE))], x[1].placement
# 注册到 RULE_TABLE：aten.mul.Tensor 仅当某一操作数是 routing 切片时命中
```
**解决思路**：标量按 (token, expert) 加权的部分和满足 `w·Σpartial ≡ Σ(w·partial)`，
乘法可与归约交换，故 Partial 直乘、末路归约一次，通信量不随专家数放大。
（具体匹配条件在小模型测试中校准，若 mul.Tensor 通用规则已能覆盖则删除此条。）

### 4.6 小随机 MoE 用例（需求 P0-1~P0-4 验收｜`tests/test_architectures.py`｜修改）
**功能点**：新增 `qwen3_moe_model` fixture（8 专家 top-2、GQA、head_dim=32、
`norm_topk_prob` 显式 True）与 `test_qwen3_moe_generate_matches_hf`。
**原因**：需求验收以小模型全链路对拍 HF 为 P0 判据；沿用既有架构测试形态。
**代码对比**：
```python
# 目标（对齐 qwen3_model fixture 模式）
model = Qwen3MoeForCausalLM(Qwen3MoeConfig(
    max_position_embeddings=MAX_SEQ, hidden_size=64, moe_intermediate_size=32,
    num_hidden_layers=NUM_LAYERS, num_attention_heads=4, num_key_value_heads=2,
    head_dim=32, vocab_size=128, num_experts=8, num_experts_per_tok=2,
    norm_topk_prob=True)).eval().to(torch.float16)
loaded = ARCHITECTURES["qwen3_moe"].load(str(path))     # 注册表路径
assert got == ref[0][len(PROMPTS[0]):].tolist()          # 贪心逐 token 对拍
```
**解决思路**：随机权重 + 门控真值激活不同专家，等价性断言即覆盖路由/加权/累加全链。

### 4.7 真实 30B-A3B 端到端（需求 P0-5｜新增 `tests/test_qwen3_30b_real.py`、`genesim_bridge/paths.py`、`paths.json`｜修改/新增）
**功能点**：站点键 `qwen3_30b_model_dir`（env `QWEN3_30B_MODEL_DIR`）指向
`paths.json` 的站点路径（本工作区为 `/data2/pengcheng-models/Qwen3-30B-A3B`，
4 层切片为 `/data2/pengcheng-work/Qwen3-30B-A3B-4L`）；测试走注册表加载 →
tp4_pp2 编译 → 贪心 6 token 对拍 HF（形态对齐 `tests/test_qwen3_8b_real.py`）。
**原因**：真实几何（48 层×128 专家、4 KV 头、vocab 151936）是小模型覆盖不了的
验收面；tp4_pp2 是 8 DPU 下首个可行策略（实测枚举）。
**代码对比**：
```python
# 目标（与 8B 用例的唯一差异：策略与预算）
strategy = ARCHITECTURES["qwen3_moe"].strategy(8, loaded.config)   # → tp4_pp2
# 单一容量契约：8 DPU × 8 GiB、系统预留 64 MiB、对齐 64 B；预算与硬件同源派生
hw, hardware = default_site_capacity(num_dpus=8, num_tasklets=4)
compiled = compile_llama2(loaded, strategy, prefill_seq_len=8, max_seq=32,
                          hw=hw, hardware=hardware, ...)
```
**解决思路**：需求 §2.1 P0-3 的 8 DPU × 8 GiB 是硬约束；每 DPU 权重
≈7.2 GiB（54/8 专家，另加注意力/嵌入/输出头），余下约 0.4～0.9 GiB 给 KV、
激活与内核 tile，由 `default_site_capacity()` 同时派生 `HwBudget` 与
`PIMHardwareConfig`，测试、主链、执行基准与 GeneSim 不再各写一套预算。
风险：~4.7 万节点图的 spec_prop/exec_plan_gen 为纯 Python 遍历，编译耗时若超
分钟级属预期，用例不设时限。

### 4.8 P1 动态分派原语（需求 P1-1｜`runtime/exec_plan_gen.py`、`backend/hal_numpy.py`｜修改）
**功能点**：每个 MoE 层的专家 launch 命令带 `gate` 字段指向本层 dispatch host_op；
dispatch 读 top-k 选中索引生成的布尔 `selected_mask`，把「本步无任何 token 命中」
的专家写入跳过集，被跳过的 launch 直接返回（路由权重只参与加权，不判命中）。
**原因**：P0 稠密模拟对 128 专家全算，decode（T=1）时只有 8 个专家被选中，16× 计算浪费；
蓝图又是静态预提交（Phase 0a 起全量 submit），需要运行期跳过机制。
**代码对比**：
```python
# 原有（exec_plan_gen.py:498 发射）
builder.append("launch", dpu_id, payload, reads, writes, waits)

# 目标
# dispatch：host_op，fn = lambda hal, cmd: 依据 result_of(route_cmd) 更新 hal.gate_skips[cmd.id]
builder.append("host_op", None, {"node": f"moe_dispatch_{layer}", "fn": make_dispatch(route_cmd_id, expert_cmd_ids)}, [], [], waits)
# 专家 launch：payload 增加 "gate": dispatch_cmd_id 与 "expert": e
def run():                                  # hal_numpy._submit_on_dpu 内
    if cmd.id in hal.gate_skips.get(cmd.payload.get("gate"), ()):  # 未 gate 的命令不受影响
        return None
    ...
```
**解决思路**：跳过是纯运行期行为（命令仍在计划里，事件照常完成），计划文本、
内存布局、数值路径全部不变；`gate_skips` 默认空 dict，等价于改动前行为。
专家归属必须按层隔离：上一层输出（包括多专家累加状态）是下一层的中性激活
输入，不能与下一层专家权重相遇后污染候选；层内多个专家的汇合点（共享累加
出口）没有单一专家身份，不带 gate。验收：小模型 token 与 P0 完全一致 +
逐层断言每个 dispatch 都拦下本层命令 + 记录执行时间对比（期望 decode 显著下降）。

### 4.9 P1 专家并行维度（需求 P1-2｜`graph/strategy.py`｜修改）
**功能点**：`ShardStrategy` 增 `ep_width: int = 1`（专家并行宽，默认 1 等价现状）；
专家 e 的权重只放在其 EP 组 DPU（整专家不切），attention/KV 仍按 TP/PP。
**原因**：TP 切专家要求每个 DPU 都参与每个专家（通信随专家数放大）；EP 让专家
整块落位，为更大 MoE 与更少通信铺路；需求要求 16 DPU 验证。
**代码对比**：
```python
# 原有
@dataclass(frozen=True)
class ShardStrategy:
    name: str; num_dpus: int; num_stages: int = 1; ...

# 目标
    ep_width: int = 1   # 每段内专家并行宽；1 = 关闭，行为与现状逐字节一致

# spec_prop._weight_spec：专家权重命中 "experts." 前缀时 dpus 取 EP 组
```
**解决思路**：`ep_width` 参与 `__post_init__` 的整除校验（tp_width*ep_width*stages=
num_dpus）；host 聚合不关心专家在哪个 DPU（routing 列照常 scatter 到对应 DPU）；
16 DPU 用 tp2×pp2×ep4 一类组合验证，断言每 DPU 专家数差 ≤1 且 token 与 TP 路径一致。

## 五、改动汇总

### 5.1 文件级改动清单
| 文件路径 | 类型 | 内容 | 影响范围 | 优先级 |
|---|---|---|---|---|
| backend/architectures.py | 修改 | `make_static_moe_forward`、`_load_qwen3_moe`、`_moe_strategy`、注册 | vLLM 白名单自动跟随 | P0 |
| graph/partition.py | 修改 | HOST_ONLY 追加 5 算子 | 仅含这些算子的图（MoE） | P0 |
| graph/strategy.py | 修改 | `mlp.gate.weight` 规则；（P1）`ep_width` | 切分与容量 | P0/P1 |
| graph/spec_prop.py | 修改 | （如需）`_rule_weighted_partial`；（P1）专家 EP 放置 | 规格传播 | P0/P1 |
| tests/test_architectures.py | 修改 | MoE fixture+用例、翻转 rejected 用例 | — | P0 |
| genesim_bridge/paths.py、paths.json | 修改 | `qwen3_30b_model_dir` 键 | 站点配置 | P0 |
| tests/test_qwen3_30b_real.py | 新增 | 真实 30B 对拍 | — | P0 |
| runtime/exec_plan_gen.py、backend/hal_numpy.py、contracts/ | 修改 | gate/dispatch 原语 | 执行计划契约 | P1 |
| tests/test_moe_dispatch.py、tests/test_moe_ep.py | 修改 | P1 两项验证 | — | P1 |

### 5.2 新增/修改的数据结构
- `ShardStrategy.ep_width: int = 1`——默认 1 时与现状逐字节一致。
- `Command.payload["gate"]: int|None`（缺省 None = 无门控，行为不变）、
  `payload["expert"]: int`。
- `NumpyBackend.gate_skips: dict[int, set[int]]`（缺省空 = 不跳过任何命令）。

### 5.3 接口变更
- `ARCHITECTURES` 新 key `qwen3_moe`（既有 key 不动）。无既有签名变化；
  `ep_width` 为可选新参，默认值兼容全部既有调用。

## 六、实施计划
1. **P0-a**（4.1–4.5 + 4.6）：静态改写→注册→HOST_ONLY→规则→小模型对拍；
   门槛：`test_architectures.py` 全绿 + dense 三架构用例不回归。
2. **P0-b**（4.7）：30B 下载完成后跑真实对拍；门槛：逐 token 一致。
3. **P1-a**（4.8）：gate 原语；门槛：token 与 P0 一致 + 时间对比记录。
4. **P1-b**（4.9）：ep_width；门槛：16 DPU 用例 + 映射均衡断言。
每阶段结束跑 `python -m pytest tests/ -q -k "llama2_7b"`（2026-11-05 历史基线
48 passed；本轮未重跑该慢速子集）。

## 七、验证与测试方案
### 7.1 单元测试清单
- `test_architectures.py::test_qwen3_moe_generate_matches_hf`：8 专家小模型，
  `run_decode_loop` 与 `model.generate` 逐 token 相等。
- `test_architectures.py::test_qwen3_moe_config_passes`（原 rejected 翻转）：
  `model_type="qwen3_moe"` 通过校验。
- `test_qwen3_30b_real.py::test_qwen3_30b_generate_matches_hf`：真实 30B 贪心
  6 token 与 HF 一致（"The capital of France is" 类提示词）。
- P1：`tests/test_moe_dispatch.py::test_gate_skipped_launch_runs_no_kernel`（未命中
  专家不执行内核）、`::test_dispatch_skips_and_matches_hf`（与 P0/HF 逐 token 一致）、
  `tests/test_moe_ep.py::test_ep_16_dpu_generate_matches_hf`（16 DPU pp4×ep4 与 HF
  逐 token 一致，函数内断言每 DPU 专家数差 ≤1）。

### 7.2 集成验证
小模型 + 30B 各跑一次 `compile→load_weights→run_decode_loop` 全链；
`scripts/status.sh` 确认无未预期脏文件。

### 7.3 验收标准
需求文档第五节逐条：回归基线 `-k llama2_7b` 48 passed（历史值）不变；小模型与 30B 逐
token 一致；P1 记录加速比；Llama/Qwen3-8B 既有用例结果不变。

## 八、对需求文档的修正记录
1. 需求 2.1 P0-2 提到「topk/gather/index_select 图规则」——实测静态改写后图里
   **不出现 gather/index_select/one_hot**（scatter 替代），规则面缩小为 5 个
   HOST_ONLY 条目 + 1 条权重规则 + 至多 1 条 mul 规则。
2. 需求 P0-4「MoE 执行路径（内核）」——实测 host 通用回退已覆盖门控与聚合，
   **无需任何新 DPU 内核**，该条收敛为布局规则打通（4.5）。

## 九、待确认事项
1. `_rule_weighted_partial` 是否必要：若既有 mul.Tensor 规则已覆盖
   Partial×Replicate，则删除（P0-a 编码时以小模型实测定）。【已关闭：实测
   既有规则覆盖，未新增】
2. 30B 编译耗时若超出可接受范围（>30 分钟），是否引入「专家子图复用」优化
   ——默认不做，先测再说。【已验证：全程 8:43:25，瓶颈在稠密模拟执行而非编译】
3. P1 gate 原语是否同步暴露给 `hal_vendor`（当前为 stub，暂不需要）。
4. **§4.9 路线修正（2026-11-05 经用户确认）**：实施盘点发现 TP×EP 复合放置
   与现有契约冲突三处（spec 的 Shard 假设分片互斥、`comm/lowering` 的
   all_gather 假设分片不相交、KV 覆盖检查要求每头恰好一次）。本轮改为
   **PP×EP（tp=1）**：专家整块落单 DPU；attention/KV/路由器在段内 REPLICATE
   （30B 类模型 attention 仅占 6% 权重，复制代价小）；跨 DPU 专家求和把
   MoE 累加子链改标 HOST 走 host 聚合。16 DPU 以 pp4×ep4 验证。TP×EP
   复合放置（attention Shard(tp)×Replicate(ep)、EP 感知通信）列为后续课题。

## 十、GeneSim MoE 成本契约（2026-11-09 评审 round01 补充）

评审要求 GeneSim 的图骨架与执行语义一致，并把专家几何、驻留与产物链路写进
接口契约。本轮按下列约定实现（细节见实施文档 §8）：

1. **模型级元数据**：`ModelIR` 增 `moe_num_experts`、`moe_top_k`、
   `moe_intermediate_size`、`moe_execution`（`dense_sim` / `dynamic_dispatch`），
   以及模型身份 `model_type`、`vocab_size`、`num_kv_heads`、`execution_dtype`、
   `tie_word_embeddings` 与结构指纹 `model_fingerprint`。MoE 身份由
   `model_type=qwen3_moe` 与有效 MoE 字段共同判定，专家数必须是正整数、
   top-k 必须在 `[1, num_experts]`；1 专家/top-1 也是合法 MoE，不能用
   「专家数 > 1」当阈值。非 MoE 模型 MoE 四项为 0/空；带 MoE 配置但模型
   类型不受支持时显式报错，禁止退回 dense FFN。
   执行 dtype 由 `build_from_hf_config(execution_dtype=)` 显式给出，与图编译器
   的 fp16 加载对齐（决定融合注意力分支）。图骨架必须覆盖完整 HF 执行图：
   输出头 lm_head GEMM 与 Qwen3 的 q_norm/k_norm 都在图内，`parameter_bytes`
   与完整 HF 参数逐一对齐，缺一项即在 parser/主链失败。
2. **算子级身份**：`Operator.expert_id` 标记专家子图归属，`moe_cost_role`
   标记计费角色（`expert_work` / `expert_comm` / `shared_reduce`）；router 独立
   `semantic_role="router"`，输出头独立 `semantic_role="lm_head"`。放置 sidecar
   按 (层号, 角色, 专家号) 唯一定位权重，条目里回传 `expert_id`，顶层回传
   `moe_config`。
3. **执行语义**：`MOE_ROUTE`（softmax/top-k/重归一化/scatter，host）与
   `MOE_ACCUM`（选择 + 路由加权 + 累加，host，成本角色 `shared_reduce`）不产生
   pimir，列入 sidecar 的 `pimir_exempt_ops`；专家 SwiGLU 三件套、激活与专家
   内乘法走设备侧，按 any-hit 门控。选择必须留在 host：PIM 原语集（FlagTree
   的 eltwise 加/减/乘/除与加性 mask）没有逐元素 select，`where` 下设备只能
   借一个语义不同的原语；`ne.Scalar` 与 `where.ScalarOther` 因此在
   `graph/partition.HOST_ONLY` 里，与随后的加权乘、累加在同一个主机闭包中
   完成。首专家没有上一累加值，其输入与搬运比其余专家少一份累加副本。
4. **产物契约**：sidecar 自报 `requires_pimir=true` 时，GeneSim 在生成 trace 前
   核对结构指纹（`ir_fingerprint`，由 IR 内容重算，覆盖模型几何、完整算子
   编号/类型/形状/角色、依赖端点/张量身份与层归属）、`moe_config`、算子编号、
   每条 `pimir_path`/`pimir_sha256` 与显式豁免；缺项、缺指纹、缺哈希、不可读或
   哈希不符直接失败，不退回模板。严格性由仿真配置的
   `scheduler.require_compiler_pimir` 与 sidecar 自报取并集，配置显式严格时
   删掉自报也不能降级；导出、skip-simulation、仿真前复核、仿真 IR 与 trace
   五个编译器入口共用同一判定。覆盖集合以源 IR 为准，与算子最终落到
   GPU/CPU/PIM 无关。豁免只有模型边界与 host 执行两类：host 落点必须由
   放置导出按 `spec.device` 写进 IR（`device_hint=cpu`）与 sidecar
   （host GEMM 用独立的 `host_gemms` 条目，带 `device_hint`/角色/权重身份；
   非 GEMM host 算子由源 IR 的 `device_hint=cpu` 证明），审计、skip-simulation、仿真前复核、trace
   验收与直接 GeneSim 逐条核对同一份证据，未知设备算子默认拒绝；不能再从
   `expert_id<0` 推断「非 DPU GEMM」，否则 router、注意力投影与输出头会一起
   被放行。豁免还必须与 sidecar 的实际放置不矛盾：同一算子的 `operators`
   条目若带 PIM 分片、`pimir_path` 或 `device_hint=pim`，就不能再按 host
   免掉 pimir；host GEMM 的角色与权重身份还要和 `host_gemms` 的导出证据一致。
   MoE 专家按源 IR 的每个 `op_id` 逐条核对 (层号, 投影身份, 专家号)
   与权重路径，不能用三元组集合相等代替——同层两个专家互换后集合不变。
   被吸收节点只能由融合清单、主算子 pimir 与 IR 依赖边唯一证明：被吸收的
   激活必须是主算子的唯一消费者（`graph/fuse.py` 的单消费者规则），否则
   折叠会改写另一个读者需要的激活前值。B+C+D 阶段只产出融合清单，
   `absorbed_ops` 由 A2 精化回写；A2 之后、skip-simulation、仿真前复核、
   trace 验收与直接 GeneSim 要求清单与吸收编号按源 `op_id` 双向闭合，删掉
   吸收编号会让折激活与独立激活 trace 同时进入代价链。
   trace 验收按源 IR 应覆盖集合与实际 `pimir` 来源对拍，`unknown`、缺算子
   与非 pimir/absorbed 来源一律失败。精化缓存键同时包含结构指纹、精化档位、
   融合清单与成本算法版本。
   条目类别必须由源 IR 决定：`op_type == GEMM` 的条目走 A 路（分块 GEMM），
   其余算子都走 B 路（1 token × 本地宽的编译单位）；`shards` 是被验产物
   自报的字段，不得反过来定义类别。B 路条目带非空 `shards`、严格产物里的
   设备 GEMM 缺分片都直接失败，编译器共享验证器与 GeneSim 使用同一分类
   规则。B 路编译单位的唯一真源是该条目引用的 pimir 实产：sidecar 的
   `trace_tokens` 必须等于 pimir 的 `pim.traceTokens`，`local_*_shapes` 的
   具体特征末维必须等于对应操作数/结果的末维（SPLIT 是纯视图，只比较输入
   末维）；严格产物缺 `pim.traceTokens` 即失败。共享验证器按路径只解析一次
   pimir，导出、skip-simulation、仿真前复核、仿真 IR 与 trace 五个入口
   继承同一判定，GeneSim 在生成 trace 时对同一份单位再做一次 pimir 实产
   核对，禁止 sidecar 自报字段自证。
   A 路编译单位与 B 路对称：每个源 IR 设备 GEMM（非 host/豁免/吸收）按路径
   只解析一次其 pimir 实产，必须是分块 GEMM 原语（`tt.dot`/`pim.matmul`），
   不得带 `pim.traceTokens` 或相位链；`tt.dot` 的 M 必须为 1，K/N 必须等于
   pimir 的 `pim.tile-k`/`pim.tile-n`，`pim.tile-n` 必须等于 sidecar 的
   `kernel_tile_n`，K/N 循环嵌套与各分片的 `local_in_features`/
   `local_out_features` 对应，且 `tt.dot` N 不得大于本地输出宽。融合清单
   `fused_activations` 声明了激活的 (层, 角色, 专家)，该 GEMM 的 `tt.dot`
   必须带同名 `act_spec`，反向也不能多带。同一规则写入编译器共享验证器
   （导出、skip-simulation、仿真前复核、仿真 IR 与 trace）与 GeneSim 的
   覆盖校验和 trace 单位校验，两仓不互相 import。
   条目按 host 跳过 A 路检查必须同时具备源 IR 的 `device_hint=cpu`、
   `pimir_exempt_ops` 的豁免登记、以及 GEMM 的 `host_gemms` 角色/权重
   证据；`operators` 条目自称 `device_hint=cpu` 却缺任一证据直接失败，
   不能用一个自报开关免掉分片、投影身份与实测分块。条目的 `op_type` 必须
   等于源 IR 的算子类型（B 路还要求该类型在 mnemonic 单表里），设备 GEMM
   的 `semantic_role` 必须是源 IR 的投影身份；trace 的字节判据按源 IR 的
   算子类型选择。B 路 pimir 的顶层原语与关键属性还必须与算子类型对应：
   两仓各持一张**闭合的** `op_type → pimir 语义指纹`表，按真实编译实产
   校准（RMSNORM→`pim.normalize` 且带 `rmsNorm`；SILU/GELU→`pim.lut` 且
   激活种类一致；VECTOR_ADD/VECTOR_MUL→`pim.eltwise<add|mul>`；ROPE→带
   `rotateHalf`/`sourceSubBlocks` 的三相 `pim.eltwise` 链，不以发射端的
   `pim.rope` 字符串当判据；PAGED_ATTN→`pim.paged_attn` 且带块循环槽参数；
   视图/搬运与四节点注意力按各自实产登记），未登记的类型直接拒绝。编译器
   共享验证器在严格分支按路径解析一次 pimir 并逐条核对指纹，导出、
   skip-simulation、仿真前复核、仿真 IR 与 trace 五个入口继承同一判定；
   GeneSim 在覆盖校验与 trace 加载时用同一张表，`phase_count` 与字节核对
   集合不得成为跳过语义核对的出口。两仓不互相 import，用跨仓测试钉住
   两张表逐值相等。
   仿真阶段只消费本次 BCD 返回的 sidecar：主链生成临时配置时显式改写
   `compiler_placement_file` 并回读复核，再按本次 `PartitionPlan`/命令行的
   策略名、DPU 数、流水段、Cluster 映射、逐 GEMM 分片数与本地形状核对；
   另一份合法策略的 sidecar 即使结构指纹相同也在仿真启动前失败。运行与结果
   目录按架构与策略派生，禁止跨任务删除共享固定目录。
5. **容量口径**：需求 §2.1 P0-3 的 8 DPU × 8 GiB 是唯一硬约束。编译器侧
   `contracts/op_contract.py::DPU_MRAM_BYTES` 与 `memory.mem_planner.default_hw_budget()`
   是唯一真源；图编译器把一台 DPU 当一个 ClusterPU（8 GiB），编译器放置下模型
   常驻按 ClusterPU 校验，KV/激活仍按单 TensorPU（512 MiB）校验。GeneSim 的
   `max_utilization`/`max_tensor_pu_utilization` 与 `fits` 同源：模型常驻按
   cluster 时分子是 `request_bytes`。`max_resident_bytes` 的 scope 是**单个
   TensorPU 的常驻最大值**（含该 TensorPU 的模型分片与请求期数据），对应
   显式字段 `max_tensor_pu_resident_bytes`；它与单 TensorPU 上限之比单列
   `max_total_resident_ratio`，只作诊断、不参与 `fits`。集群总量只引用
   `max_cluster_resident_bytes`/`max_cluster_utilization`，与 8 GiB 集群
   上限配对，不能与单 TensorPU 读数混用。容量契约是
   不可拆分的三件套：单 DPU 总容量、系统预留（64 MiB）与 MRAM 对齐（64 字节），
   随 sidecar 的 `hw_budget` 回传。`memory.mem_planner.default_site_capacity()`
   从同一三件套同时派生 `HwBudget` 与 `PIMHardwareConfig`，主链导出、vLLM、
   执行期基准与成本精化都走它；编译入口拒绝 `hw.align != hardware.dma_align`，
   B 路生成显式接收本次硬件配置，模块里已有冲突属性直接失败而不是保留；
   只拿到文本、没有本次请求的调用方沿用模块头已写的预算，缺失项才用默认值。
   导出与 GeneSim 加载除核对 sidecar 三字段外，还逐份解析 pim mlir 的
   `pim.dma-align` / `pim.mram-bytes`：自报 64、实产 8 的产物一律拒绝。
   导出、skip-simulation、仿真前复核与直接 GeneSim 用同一份规则校验：自报
   带产物的严格 sidecar 缺任一字段、类型错误或任一值偏离即失败；只有明确
   标识的旧式非严格产物才允许没有该字段。严格性由仿真配置的
   `scheduler.require_compiler_pimir` 与 sidecar 自报取并集：配置显式严格时，
   删掉自报也不能退回兼容分支；只有配置与产物都明确非严格才走旧式兼容。
6. **动态分派**：专家归属沿拓扑序传播，但传播域限定在「本层、本专家」的
   连续子图：层号取模块路径（`nn_module_stack`）或权重名，跨层激活（上一层
   输出、上一层多专家累加状态）一律按中性输入处理；同层多专家汇合点（共享
   累加出口）记为 `(层, 无专家)`，不给 gate。这样分块后的整条专家子图与通信
   都被覆盖，又不会把上一层的状态带进下一层。未命中专家的命令不执行（DPU
   命令写零、主机命令返回零张量），语义与 dense_sim 等价。执行语义是正式编译配置（`compile_llama2(moe_execution=)`，
   缺省跟随 `PIM_MOE_DISPATCH`），由同一入口传给图骨架、放置 sidecar 与
   GeneSim；sidecar 的 `moe_config.execution` 与 IR 独立核对，不一致即失败。
   GeneSim 的 `dynamic_dispatch` 计费必须与执行计划的 any-hit 语义一致：
   编译器用 top-k 选中索引生成的布尔掩码 `selected_mask.any(dim=0)` 判断
   专家是否执行（路由权重可能下溢为 0，不能兼任命中条件），命中后对该专家的
   完整 Tq 张量执行子图，因此每个专家的执行比例是 `1-(1-top_k/E)^Tq`
   （decode 的 Tq=1 退化为 top_k/E），专家通信按同一命中并集计；共享累加
   （`MOE_ACCUM`，含选择与路由加权，成本角色 `shared_reduce`）是主机工作、
   始终执行，不按命中率缩放。依赖边的折算按两端的成本角色判定：专家生产的
   输出（计算或通信）无论流向共享累加、其它设备还是 HOST，都随该专家的命中
   并集折算；共享/HOST 输入到专家时随消费者的命中折算；两端都是共享工作的
   边始终发生，保持全量。放置期通信估算与正式管线、显式系数与旧式依赖共用
   这一判定，不能只看目标角色。
   `dense_sim` 保留全专家计算与驻留，两者共享同一张图、同一份产物。
7. **成本依赖项**：成本项作为配方声明的一部分，由同一描述驱动测量 IR、拟合与
   运行期求值。当前 token 的逐元素、归一化、激活、视图类与专家乘加按 `Tq`
   计费，与 KV 历史 `Tp` 无关；注意力得分/掩码/softmax/上下文/分页注意力按
   `Tq(Tp+Tq)` 计费。ROPE/MASK 等逐头节点按单头形状测量，不再乘 `num_heads`。
   改动成本项即升级成本算法版本，旧精化产物失效。
8. **大规模图修改**：TP 展开与集合通信中转节点的插入必须走批量接口
   （`ModelIR.batch_edit()`）：循环内只增量分配编号并记录增删，退出时一次
   重建索引。禁止在循环里逐条调用 `add_operators`/`remove_operators` 各自
   全量重建——4 层 2581 个算子的展开曾因此耗时 99 秒，48 层放大到 5 小时。

§九 待确认事项：8 GiB 是否仍为 30B 的硬性单 DPU 预算——已由用户确认
（2026-11-10）：8 DPU × 8 GiB 是硬约束，48 层主链超预算的部分作为遗留问题
记录每 DPU 所需/可用字节，不改成 16 GiB。
