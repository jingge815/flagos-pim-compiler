# vLLM × PIM 接入 TODO

> 范围：`flagos-pim-compiler` 的 vLLM V1 后端（`backend/vllm_worker.py`、`backend/vllm_executor.py`、`scripts/serve_pim.py`）及其依赖的运行时/KV/编译模块。
> 基线环境：vLLM **0.14.0**（`flagOS-installed/pytorch`），Python 3.10，Llama-2 架构，`NumpyBackend` 模拟。
> 本文只拆分任务与验收标准；实现细节按仓库约定写进 `docs/<module>-<日期>.md`。
>
> **2026-11-02 重写**：A–F 已完成任务压缩成「台账」，删掉逐条子任务与旧「当前问题」；新增 G 段
> （并行与内存的下一阶段），每项给出可执行的验收依据。

---

## 0. 规则

### 0.1 状态图例

| 标记 | 含义 |
|---|---|
| ⬜ | 未开始 |
| 🟡 | 进行中 |
| 🔍 | 待评估（方案未定或依赖外部结论） |
| ✅ | 已完成（验收全部通过且证据已记录） |
| ⛔ | 阻塞（写明阻塞原因、解除条件、绕过方式） |

### 0.2 完成定义（DoD）

- [ ] 代码 + 单测在同一次改动内完成，不留「测试稍后补」。
- [ ] 验收命令全部通过，输出粘到任务下的「验收证据」表。
- [ ] 行为对拍：token 与基线逐字一致，或写明允许的数值误差。
- [ ] 新接口写进 `docs/<module>-<日期>.md`；启动参数同步更新 `README.md` 与 `serve_pim.py --help`。
- [ ] 不写防御性兜底；不支持的配置显式报错，消息含「参数、当前值、期望值、怎么办」。
- [ ] 报告净增删行数（如 `+120 / -35`）。

### 0.3 证据格式

```text
| 验收项 | 命令 | 结果 | 日期 |
|---|---|---|---|
| 单测 | python -m pytest tests/test_x.py -x -q | 8 passed | 2026-11-xx |
| token 对拍 | ... | 与基线逐字一致 | 2026-11-xx |
```

---

## 1. 项目状态评估（2026-11-02）

### 1.1 三个目标的完成度

| 目标 | 完成度 | 现状 | 缺口 |
|---|---|---|---|
| vLLM 接入 | 高（单进程 / Llama 族 / fp16） | A–F 全部完成；采样、块表、分块 prefill、前缀缓存、抢占、预检、指标、版本锁定都有单测与端到端 | 只支持 `llama`、fp16、单进程、vLLM 层并行度必须为 1 |
| 内存动态分配 | 中 | **编译期**按剩余 MRAM 自动定容（槽池、`P`、块表窗口）；运行期块号由 vLLM 分配、PIM 跟随 | **运行期**没有分配器：激活窗口、`P`、槽池在编译时定死，少用/多用都不伸缩 |
| 并行执行 | 功能完成，收益未证实 | P=1→2 实测有收益；P=4 相对 P=2 只 +0.2% | 主机侧全局锁、NumpyBackend 线程语义、真硬件未验证 |

### 1.2 关键实测数据

| 指标 | 数值 | 来源 / 日期 |
|---|---|---|
| 全量快测 | **1568 passed, 2 skipped, 4 deselected**（17:04） | `pytest tests/ -x -q --ignore=三个 7B 文件`，2026-11-02 |
| 测试规模 | 102 个文件 / 1589 条用例（4 条 `slow` 默认不跑） | `--collect-only`，2026-11-02 |
| vLLM 并行效率（P=1 → P=2） | 预填充 1.00 → 2.00，解码 1.00 → 1.98；最大并发 1 → 2；窗口冲突 0 | `scripts/profile_execution.py vllm`，2026-11-02 |
| 同一组墙钟（P=1 → P=2） | 3 207 ms → 3 769 ms（**变慢**） | 同上；NumpyBackend 单机线程争用 |
| 多 DPU 并行吞吐 | P=1 27 557 ms / 60.6%；P=2 21 403 ms / 71.2%；P=4 21 360 ms / 73.8% | 4 DPU / 4 层 / hidden 1024 / 4 请求 × 8 步，2026-11-02 |
| 自动定容 | `max_model_len=32`、块 16、`max_num_seqs=2` → 每请求 2 块，`pool=5=2×2+1`，`P=2` | `test_auto_capacity.py`，2026-11-02 |
| 分桶收益 | 100 token prompt：7 675.0 → 3 102.7 ms（−59.6%） | `docs/vllm-prefill-buckets-20261101.md` |
| 融合 prefill | 3 102.7 → 2 846.6 ms（−8.3%）；比强制主机快 56.8% | `docs/vllm-fused-prefill-20261101.md` |
| 指标开销 | 3.2 µs / 步（远低于 2% 门槛） | `docs/vllm-metrics-20261102.md` |
| 融合 vs 四节点 attention（GeneSim） | 196 s → 78 s；总时间 −5.1%、TTFT −17.2%、TPOT +25% | `genesim/results*.20261102/`，2026-11-02 |

### 1.3 方案评估

**站得住的部分**

- 「静态图 + 运行期数据动态」这条路对 dense decoder 是成立的：形状用桶覆盖，KV 长度/块号/
  请求调度全在运行期，图本身不变。
- 块号以 vLLM 为唯一真源、抢占与前缀缓存跟着 vLLM 走，是避免双记账出错的关键决定。
- 自动定容把「装不下」从运行期崩溃提前成启动期报错，7B 上也能给出可用的 `P`。

**需要修正或警惕的假设**

1. **「并行有收益」只在 4 DPU 的编译产物上验证过**，且 P>2 的收益被主机全局锁吃掉。
   在真硬件语义确认前，所有并发收益应视为**上界**。
2. **NumpyBackend 的墙钟不能当性能**：同一配置下 P=2 的墙钟反而比 P=1 长 18%，
   线程争 CPU。后续所有性能结论要么用「忙碌/并行效率」，要么用多 DPU 的相对比值。
3. **内存随 `P` 线性放大**：激活区与块表 ×P。7B 上自动定容可能把 `P` 压到 1，
   并行直接消失；当前只有「压到 1」的路径，没有「压到 1 损失多少」的量化。
4. **7B 级并发只有一条 slow 用例**，且本机 3 个 7B 测试文件会挂起；大规模回归依赖
   缩小模型，存在「小模型过、大模型不过」的风险。
5. **服务形态未定**：只跑单进程。多进程/Ray 会把 HAL 的线程语义换成进程语义，
   需要重新验证泳道折叠的前提。

---

## 2. 已完成台账（A–F）

逐条子任务与旧「当前问题」已删除；细节、设计取舍和完整验收表见对应 `docs/`。

| 任务 | 结果 | 文档 | 测试 | 关键证据 |
|---|---|---|---|---|
| A1 采样贯通 + logprobs | vLLM `SamplingParams` 全量生效，不支持项显式报错 | `vllm-sampling-20261101.md` | `test_vllm_sampling.py`（16） | 温度/seed 复现、logprob 误差 < 1e-3 |
| A2 启动期能力矩阵 | TP/PP/LoRA/投机/量化/dtype/架构逐项点名报错 | `vllm-capabilities-20261101.md` | `test_vllm_capabilities.py`（18） | 报错含参数名与建议 |
| A3 `serve_pim` 预检与 CLI | 槽池、MRAM、图长、分块预算一次性列全 | `vllm-serving-preflight-20261101.md` | `test_serve_pim_preflight.py`（11） | 多个问题一次列全 |
| B1 prefill 分桶 | 按 prompt 长度选桶，短 prompt 不再付最长图代价 | `vllm-prefill-buckets-20261101.md` | `test_prefill_buckets.py`（21） | 7 675 → 3 102.7 ms（−59.6%） |
| B2 prefill 走融合内核 | 标准因果掩码走 `pim.paged_attn`，自定义掩码回退 | `vllm-fused-prefill-20261101.md` | `test_kernels.py`（21） | hit=384 / fallback=384；−8.3% |
| C1 vLLM 块表唯一真源 | 新请求采用、cached 追加、resumed 替换 | `vllm-block-table-20261101.md` | `test_vllm_block_table.py`（7） | 外部块号不改变 token |
| D1 分块 prefill | 任意 `pos` 起跑一段，按段长选桶 | `vllm-chunked-prefill-20261101.md` | `test_chunked_prefill.py`（12，2 slow） | 3000 token 的 64/512 切法与整段逐 token 一致 |
| D2 前缀缓存 | 命中段只读，从命中长度续写 | `vllm-prefix-cache-20261101.md` | `test_prefix_cache.py`（4） | 共享块字节不变；命中前后 token 一致 |
| D3 抢占与重算 | `resumed_req_ids` + `num_computed_tokens` 决定整段/部分重放 | `vllm-preemption-20261102.md` | `test_preemption.py`（6） | 3 块池强制抢占后 token 一致 |
| E1 自动定容 | 按剩余 MRAM 反推槽池，默认 `P=min(2, max_num_seqs)` | `kv-auto-capacity-20261102.md` | `test_auto_capacity.py`（5） | `pool=5=2×2+1`；P 扫描见 1.2 |
| E2 可观测性 | 每步请求/wave/token/槽池/attention 路由 | `vllm-metrics-20261102.md` | `test_step_stats.py`（4） | 3.2 µs/步 |
| E3 架构白名单 | `ARCHITECTURES` 表；GQA 可用，`rope_scaling`/`sliding_window` 拒绝 | `model-architectures-20261102.md` | `test_architectures.py`（5） | 4 Q 头 / 2 KV 头 / 4 DPU 编译通过 |
| F1 版本锁定 | 只收 0.14.x，启动即校验 | `vllm-031-gap-20261102.md` | `test_vllm_compat.py`（2） | 0.31 报错含两个版本 |
| F3 测试与文档加固 | 翻译层单测、文档索引、冒烟入口 | `vllm-integration-20261102.md` | `test_vllm_translate.py`（7） | `verify.sh --vllm` 27 passed |

---

## 3. 下一阶段 G：并行与内存

| 编号 | 任务 | 优先级 | 状态 | 一句话目标 |
|---|---|---|---|---|
| G1 | 主机侧串行瓶颈 | P0 | ⬜ | 让 P≥3 的并行真正带来吞吐 |
| G2 | 运行期窗口与内存 | P0 | 🔍 | 在飞请求数变化时不再固定 `P` 与窗口占用 |
| G3 | 调度超参基准 | P0 | ⬜ | 给出 `max_num_seqs`/`block_size` 等的可复现影响表 |
| G4 | 真硬件语义与多进程（原 F2） | P0 | ⛔ | 验证并发结论在真 HAL 上成立 |
| G5 | 模型覆盖扩展 | P1 | 🔍 | 从 Llama 扩到滑窗/GQA 以外的架构 |
| G6 | 服务形态 | P1 | ⛔ | 多进程 / API server |
| G7 | 性能回归门限 | P2 | ⬜ | 把 G1/G3 的基线固化成 CI 门限 |

里程碑：**M1 = G1 + G2**（并行收益与窗口伸缩）；**M2 = G3**（超参选型）；**M3 = G4**（真硬件）。

---

### G1 主机侧串行瓶颈 ⬜

**目标**：P=4 相对 P=2 有实测吞吐收益，而不是 +0.2%。

**现状**：`backend/hal_numpy.py:145` 的 `_host_lock` 是全局锁，所有主机命令串行；
P=2 已经把主机产能吃满。P=4 的忙碌率只多 2.6 个百分点（71.2% → 73.8%）。

**拆分**

- [ ] G1.1 基准脚本入库：`scripts/bench_parallel.py`，参数化 DPU 数/层数/请求数/步数，
      跑 P=1/2/4，输出墙钟、忙碌率、逐 DPU 忙碌分布、token 摘要；替代 `/tmp/bench_parallel_e1.py`。
- [ ] G1.2 把 `_host_lock` 拆成按 DPU（或按泳道）的锁：同一个 DPU 仍严格串行，
      不同 DPU 的主机命令可并行。`tests/test_hal_numpy.py` 增加同/异 DPU 并发用例。
- [ ] G1.3 量化主机段占比：用 `profile_backend` 输出每 DPU 的忙碌/等待，列出仍串行的命令类。
- [ ] G1.4 只对 G1.3 证明是瓶颈的命令类做下沉或合并，不预先优化。

**验收**

- [ ] `scripts/bench_parallel.py` 三个 P 的输出 token 与该请求的串行基线逐 token 相同
      （脚本内置断言，失败即退出非零）。
- [ ] P=4 墙钟相对 P=2 ≤ −10%；首次实测后把该值写成回归基线。达不到时在证据表里写清
      「host 段占比 X%、锁拆分的上限是 Y」并退回设计，不硬凑指标。
- [ ] `pytest tests/test_hal_numpy.py tests/test_parallel_overlap.py tests/test_parallel_requests.py -q` 全过
      （同 DPU 串行、跨请求交错、窗口不重叠三条语义不变）。
- [ ] 全量快测通过。

**依赖**：无。**风险**：如果主机段占比低于 20%，分片锁收益有限，应转做命令合并或接受 P=2，
并把结论写进 `docs/`，不要为指标改架构。

---

### G2 运行期窗口与内存 🔍

**目标**：在飞请求数变化时，激活窗口与并行窗口按需取用，不再固定为编译期的 `P`。

**现状**：`act_stride`、`num_parallel_slots`、`block_table_slots` 都在编译期定死；
NumpyBackend 的 MRAM 是一次性预分配的字节数组，**运行期没有分配器**。
目前能做的「动态」只有：在编译期生成多套窗口变体，运行期选一套。

**拆分**

- [ ] G2.1 编译 `P=1..Pmax` 的 plan 变体（`bind_plan` 已能按窗口平移，需把变体列表暴露到
      `CompiledModel` 并由 worker 选择）。
- [ ] G2.2 worker 按本步在飞请求数选最小的够用 `P`；`StepStats` 增加「选中 P」一列。
- [ ] G2.3 设备端分配器接口设计（HAL：alloc/free/水位/碎片），只写设计文档，**不实现**；
      实现依赖 G4 的真 HAL 语义。

**验收**

- [ ] 混合负载（1 个长请求 + 突发 4 个短请求）的 token 与固定 `P=2` 逐 token 一致。
- [ ] 只有 1 个请求在飞时，选择 `P=1` 的步占比 ≥ 90%，且单步墙钟不高于 `P=2` 的 105%。
- [ ] 内存账本：日志或 `trace` 能打印「本步选中的窗口 + 实际占用字节」；未使用窗口不产生写操作
      （对 MRAM 字节快照断言）。
- [ ] 编译时间与产物体积增加 ≤ Pmax 倍；`Pmax` 上限 4，超出显式报错。
- [ ] 全量快测通过。

**依赖**：G1（没有并行收益时，选 `P` 没有意义）。**风险**：多套 plan 成倍放大编译时间与缓存；
如果收益不足，应把 G2 降级为「固定 P + 更小的 act_stride」。

---

### G3 调度超参基准 ⬜

**目标**：用一张可复现的表回答「`max_num_seqs`、`max_num_batched_tokens`、`block_size`、
chunked prefill、前缀缓存怎么选」。

**现状**：只有一组 P=1 vs P=2 的并行效率数据。`scripts/profile_execution.py vllm` 在 D1 之后
一度不可用（预检要求预算 ≤ 最长桶），已修：关掉分块 prefill、预算改为 `max_model_len`
（未提交）。

**拆分**

- [ ] G3.1 入库 `scripts/bench_vllm_scheduling.py`：固定微型模型与 prompt，按配置矩阵各跑
      一次，输出步数、每步请求数、token 数、并行效率、墙钟。
- [ ] G3.2 矩阵至少覆盖：`max_num_seqs ∈ {1,2,4}`、`max_num_batched_tokens ∈ {4,8,16}`、
      `block_size ∈ {8,16}`、chunked prefill 开/关、前缀缓存开/关（可正交拆分，不必全组合）。
- [ ] G3.3 结果写入 `docs/vllm-scheduling-2026xx.md`，给出默认值建议与反例。

**验收**

- [ ] 脚本 5 分钟内跑完，直接输出 Markdown 表。
- [ ] 每个配置有可复现命令；同配置重复两次，步数与 token 数完全相同。
- [ ] 文档至少给出 3 条「参数 → 指标」结论，并标出哪些是 NumpyBackend 的伪影（GIL/墙钟），
      哪些是可迁移到真硬件的（步数、每步请求数、并行效率、槽池占用）。

**依赖**：无。

---

### G4 真硬件语义与多进程（原 F2）⛔

**阻塞原因**：`backend/hal_vendor.py` 仍是桩；当前并发结论建立在 NumpyBackend 的线程语义上
（同 DPU 提交序、依赖由 worker 等待）。

**解除条件**：真硬件可执行至少一层 Llama，且 HAL 提供确定的完成/依赖语义与跨进程可用的句柄。

**拆分（解除后）**

- [ ] G4.1 `hal_vendor` 与 `hal_numpy` 同接口，跑通小模型单请求。
- [ ] G4.2 校验泳道/依赖语义在真硬件上成立（同 DPU 串行、跨请求交错、窗口不重叠）。
- [ ] G4.3 评估 `MultiProcExecutor` / Ray 的接入方式（G6 的前置）。

**验收**

- [ ] 真硬件单请求 token 与 `NumpyBackend` 逐 token 一致。
- [ ] 两请求并发不串数据，且 `P=2` 相对 `P=1` 的墙钟确实下降（不再是「效率高但更慢」）。
- [ ] 文档记录硬件约束与进程模型。

---

### G5 模型覆盖扩展 🔍

**目标**：从「只有 Llama」扩到第二类真实架构，并明确哪些拒绝项必须先修。

**现状**：`backend/architectures.py` 只有 `llama`；GQA 可用；`rope_scaling`、`sliding_window`
显式拒绝；量化、LoRA、投机解码未做。

**拆分**

- [ ] G5.1 选型：优先 Qwen2/Mistral 中「不改掩码语义」的一类；滑窗模型单独评估所需改动
      （KV 可见性、块循环边界）。
- [ ] G5.2 接入：权重加载 + 切分策略进 `ARCHITECTURES` 表，复用现有编译路径。
- [ ] G5.3 拒绝项保持不变式：不支持的模型仍启动即报错，不允许静默降级。

**验收**

- [ ] 新架构小模型（≤4 层）的生成 token 与 HF `generate` 逐 token 一致（贪心）。
- [ ] GQA/滑窗的 KV 几何与 `q_heads_by_kv` 单测通过。
- [ ] 未支持的 `model_type`/`rope_scaling`/`sliding_window` 仍显式报错（现有测试不回归）。
- [ ] 全量快测通过。

**风险**：滑窗会改变 attention 的可见性判定，与 `paged_attn` 的「标准因果掩码」开关冲突，
必须先定「谁声明掩码语义」。

---

### G6 服务形态（多进程 / API server）⛔

**阻塞原因**：同 G4——HAL 目前是进程内线程模型；跨进程后命令提交、依赖等待、MRAM 共享
都要重新定义。

**解除条件**：G4 完成且 `hal_vendor` 支持跨进程句柄。

**验收**

- [ ] `vllm serve` 形态下两个并发请求 token 与单进程一致。
- [ ] 进程崩溃/重启后引擎能恢复（至少不串数据）。
- [ ] 文档记录进程模型与限制。

---

### G7 性能回归门限 ⬜

**目标**：把 G1/G3 的基线固化成可执行的回归，而不是文档里的数字。

**拆分**

- [ ] G7.1 `scripts/verify.sh --bench`：跑 `bench_parallel.py` 与 `bench_vllm_scheduling.py` 的
      固定子集，与 `tests/benchmarks.json` 里记录的基线比，超出阈值即失败。
- [ ] G7.2 基线文件随每次合并更新，改动必须附「为什么预期变化」。

**验收**

- [ ] 故意把 `P` 下调或引入一次回归时，`--bench` 能失败并指出指标与偏差。
- [ ] 默认 `verify.sh` 不跑它（保持快测），只有 `--bench` 才跑。

---

## 4. 明确不做的事

- 不做运行时捕获/重编译式的动态图；控制流走「host 分派子图」或「融合内核内变长循环」。
- 不做设备端通用分配器（G2.3 只出设计），直到 G4 的真 HAL 语义确定。
- 不重写 vLLM 调度器；请求生命周期、连续批处理、抢占继续由 vLLM 负责。
- 不把 vLLM 的 CUDA attention backend / CUDA Graph 接到 PIM。
- 不在不支持的能力上静默降级；一律显式报错。

---

## 5. 每周检查清单

- [ ] 总表状态与实际代码一致；🟡 任务有一行「当前进展」。
- [ ] 新完成的 ✅ 任务有验收证据表。
- [ ] 回归：`pytest tests/test_vllm_executor.py tests/test_parallel_requests.py tests/test_kv_slots.py -x -q`。
- [ ] 基线 token 对拍仍逐字一致（1.2 的微型模型基线）。
- [ ] `scripts/verify.sh --vllm` 通过。
