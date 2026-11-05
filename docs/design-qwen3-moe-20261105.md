# 技术设计文档：接入 Qwen3 MoE（Qwen3MoeForCausalLM）到 PIM 推理链路
> 文档编号：design-qwen3-moe-20261105
> 创建日期：2026-11-05
> 关联需求文档：request-qwen3-moe-20261105
> 关联项目：flagos-pim-compiler（基线 263fcda）

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
- 不包含范围：genesim MoE 计费、vLLM serve 实测、235B、量化、滑窗 MoE、tied 词嵌入。

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
 │    → softmax·fp32 → topk → renorm(sum) → scatter → routing (T,E)   [全 host★HOST_ONLY]
 └─ for e in experts (静态展开):
      gate/up linear (DPU col, 既有规则) → silu·mul (DPU) → down (DPU row→Partial→reduce)
      → × routing[:,e] (DPU mul, host→DPU scatter 边, 既有) → 累加 add (DPU)
 → 残差 add (DPU, 既有)                          P0 到此为止
 → [P1★] dispatch host_op 读 routing → 置跳过标志 → 未命中专家 launch 直接返回
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
routing = torch.zeros_like(probs).scatter(1, sel, tw.to(probs.dtype))
acc = blk.experts[0](xs) * routing[:, 0:1].to(xs.dtype)      # 首专家直接做初值，省 zeros 节点
for e in range(1, blk.num_experts):
    acc = acc + blk.experts[e](xs) * routing[:, e:e+1].to(xs.dtype)
return acc.view(b, s, h), logits
```
**解决思路**：非选中专家权重为 0、乘加恒等，按专家号升序累加与 HF index_add
顺序一致，已实测位级相等（`norm_topk_prob` 两态均验）。

### 4.2 注册 qwen3_moe 与 MoE 策略入口（需求 P0-1｜`backend/architectures.py:48`｜修改）
**功能点**：`ARCHITECTURES` 加 `"qwen3_moe"`；新增 `_moe_strategy` 把
`moe_intermediate_size`（而非遗留的 `intermediate_size`）传给策略枚举。
**原因**：30B 的 dense `intermediate_size=6144` 是配置残留，真正要按 TP 切的 FFN
是专家的 768；沿用 `_dense_strategy` 会切错宽度并误判容量。
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
`/mnt/sda/butj/fjg/model/Qwen3-30B-A3B`；测试走注册表加载 → tp4_pp2 编译 →
贪心 6 token 对拍 HF（形态对齐 `tests/test_qwen3_8b_real.py`）。
**原因**：真实几何（48 层×128 专家、4 KV 头、vocab 151936）是小模型覆盖不了的
验收面；tp4_pp2 是 8 DPU 下首个可行策略（实测枚举）。
**代码对比**：
```python
# 目标（与 8B 用例的唯一差异：策略与预算）
strategy = ARCHITECTURES["qwen3_moe"].strategy(8, loaded.config)   # → tp4_pp2
hw = HwBudget(mram_bytes=16 * 2**30, align=64, sys_reserve_bytes=64 * 2**20)
compiled = compile_llama2(loaded, strategy, prefill_seq_len=8, max_seq=32, ...)
```
**解决思路**：每 DPU 权重 ≈7.7 GiB（54/8 专家 + 注意力/嵌入/头），测试预算取
16 GiB 给 KV/激活/内核 tile 留余量；KV 槽池 1 请求×32 token 忽略不计。
风险：~4.7 万节点图的 spec_prop/exec_plan_gen 为纯 Python 遍历，编译耗时若超
分钟级属预期，用例不设时限。

### 4.8 P1 动态分派原语（需求 P1-1｜`runtime/exec_plan_gen.py`、`backend/hal_numpy.py`｜修改）
**功能点**：每个 MoE 层的专家 launch 命令带 `gate` 字段指向本层 dispatch host_op；
dispatch 读 routing 结果，把「本步无任何 token 命中」的专家写入跳过集，被跳过的
launch 直接返回。
**原因**：P0 稠密模拟对 128 专家全算，decode（T=1）时仅 8 专家非零，16× 计算浪费；
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
验收：小模型 token 与 P0 完全一致 + 记录执行时间对比（期望 decode 显著下降）。

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
| tests/test_moe_dispatch.py 等 | 新增 | P1 两项验证 | — | P1 |

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
每阶段结束跑 `python -m pytest tests/ -q -k "llama2_7b"`（基线 48 passed）。

## 七、验证与测试方案
### 7.1 单元测试清单
- `test_architectures.py::test_qwen3_moe_generate_matches_hf`：8 专家小模型，
  `run_decode_loop` 与 `model.generate` 逐 token 相等。
- `test_architectures.py::test_qwen3_moe_config_passes`（原 rejected 翻转）：
  `model_type="qwen3_moe"` 通过校验。
- `test_qwen3_30b_real.py::test_qwen3_30b_generate_matches_hf`：真实 30B 贪心
  6 token 与 HF 一致（"The capital of France is" 类提示词）。
- P1：`test_moe_dispatch::test_dispatch_skips_unselected_experts`（decode 步实际
  launch 的专家数 = top_k）、`::test_dispatch_tokens_match_dense`（与 P0 逐 token
  一致）、`test_expert_parallel::test_16_dpu_tokens_match_tp`（16 DPU EP 与 8 DPU
  TP 生成一致）、`::test_expert_balance`（每 DPU 专家数差 ≤1）。

### 7.2 集成验证
小模型 + 30B 各跑一次 `compile→load_weights→run_decode_loop` 全链；
`scripts/status.sh` 确认无未预期脏文件。

### 7.3 验收标准
需求文档第五节逐条：回归基线 `-k llama2_7b` 48 passed 不变；小模型与 30B 逐
token 一致；P1 记录加速比；Llama/Qwen3-8B 既有用例结果不变。

## 八、对需求文档的修正记录
1. 需求 2.1 P0-2 提到「topk/gather/index_select 图规则」——实测静态改写后图里
   **不出现 gather/index_select/one_hot**（scatter 替代），规则面缩小为 5 个
   HOST_ONLY 条目 + 1 条权重规则 + 至多 1 条 mul 规则。
2. 需求 P0-4「MoE 执行路径（内核）」——实测 host 通用回退已覆盖门控与聚合，
   **无需任何新 DPU 内核**，该条收敛为布局规则打通（4.5）。

## 九、待确认事项
1. `_rule_weighted_partial` 是否必要：若既有 mul.Tensor 规则已覆盖
   Partial×Replicate，则删除（P0-a 编码时以小模型实测定）。
2. 30B 编译耗时若超出可接受范围（>30 分钟），是否引入「专家子图复用」优化
   ——默认不做，先测再说。
3. P1 gate 原语是否同步暴露给 `hal_vendor`（当前为 stub，暂不需要）。
