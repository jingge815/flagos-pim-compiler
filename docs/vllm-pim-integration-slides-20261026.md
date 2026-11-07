---
marp: true
theme: default
paginate: true
size: 16:9
header: 'vLLM × PIM'
footer: 'flagos-pim-compiler · 2026-10-26'
---

<style>
section { font-size: 25px; }
section pre code { font-size: 17px; line-height: 1.35; }
section code { font-size: 22px; }
section table { font-size: 22px; }
section h1 { font-size: 40px; }
section h2 { font-size: 30px; }
</style>

<!-- _class: lead -->
# 把 vLLM 接入 PIM

## 从「跑通链路」到「跨请求并行」

组会汇报 · 2026-10-26

---

## 大纲

1. 背景：为什么接 vLLM，核心约束是什么
2. 总体架构与一次 `execute_model` 的数据流
3. 四个关键提交的演进
4. KV 方案：固定槽 + 块表间接寻址
5. 算子层：paged attention 全链路（MLIR → 设备 C）
6. 跨请求并行：五个机制
7. 实测与验证
8. 当前进展、不足与下一步

---

## 背景与目标

- **vLLM 负责**：HTTP 服务、请求调度、连续批处理（continuous batching）
- **PIM 负责**：把每步的调度结果翻译成 DPU 上的静态执行计划

目标：用 `scripts/serve_pim.py` 起一个 **OpenAI 兼容服务**，底层跑 PIM 编译器
产出的静态蓝图。

分工边界很清晰：

```text
vLLM V1 引擎 ──SchedulerOutput──▶ PIMWorker ──▶ step_request ──▶ 执行计划 ──▶ DPU
   (宿主 CPU)                      (适配层)        (逐请求)        (编译产物)   (NumpyBackend)
```

---

## 澄清：我们做的就是「分页 KV」

**MRAM 没有 MMU / 虚拟地址层，但这不阻碍 PagedAttention。**

> PagedAttention 的本质是**软件块表间接寻址**：逻辑块映射到物理块，
> 内核按块表 gather。GPU 上它同样是对裸设备指针做间接寻址，不依赖 MMU。

从 `67a26e6` 起我们做的就是同一套机制：

```text
逻辑块 block ──块表(bt_base 处 int32 数组)──▶ 物理槽 slot
                                              │
             sdpa_kernel: base = k_off + slot * slot_bytes
```

- 固定大小物理槽（page），槽内按 `layer → kv_head → k/v` 排布
- 每个在飞窗口一份块表，`write_block_table` 每步整段覆盖
- 设备内核读表、逐物理槽算 QK/softmax/PV，不拼接 KV 历史
- 物理槽由全局池按需 `alloc/free`、复用

**结论：核心机制与 PagedAttention 一致，不是「做不了」。**

---

## 与 vLLM PagedAttention 的真实差异

| 维度 | vLLM PagedAttention | 我们的实现 |
| --- | --- | --- |
| 块表 → 物理块间接寻址 | ✓ | ✓ |
| 设备内核按块 gather | ✓ | ✓ `sdpa_kernel` 读 `bt_base` |
| 物理块动态分配/复用 | ✓ | ✓ 全局槽池 |
| 块大小 / 槽数 / 并发窗口数 | 运行期配置 | **编译期常量，改需重编译** |
| 每请求块表数量 | 任意多 | 固定 `P` 份窗口，按 wave 复用 |
| 前缀共享 / Copy-on-Write | ✓ | ✗ `enable_prefix_caching=False` |
| 并发上限 | 显存决定 | `max_num_seqs`（host 记账） |

关键澄清：受限于**静态编译蓝图**，块几何与窗口数是编译期常量；
但**运行期的「逻辑块→物理槽」映射是动态的**，这正是分页的关键。

---

## 总体分层

```text
┌──────────────────────────────────────────────┐
│ vLLM V1 Engine   HTTP / Scheduler / Batch     │
├──────────────────────────────────────────────┤
│ PIMExecutor (继承 UniProcExecutor)            │  backend/vllm_executor.py
│   └─ worker_cls = PIMWorker                   │
├──────────────────────────────────────────────┤
│ PIMWorker (继承 WorkerBase)                   │  backend/vllm_worker.py
│   execute_model / load_model / KV 记账        │
├──────────────────────────────────────────────┤
│ runtime/executor.py                           │
│   RequestTable / KVSlotPool / step_request    │
│   run_one_step（wave 并发）                    │
├──────────────────────────────────────────────┤
│ runtime/compile.py + binding.py               │
│   两张图 × P 个窗口变体 + 内存/通信蓝图         │
├──────────────────────────────────────────────┤
│ NumpyBackend：每 (请求上下文, DPU) 一条泳道      │
└──────────────────────────────────────────────┘
```

---

## 一次 execute_model 的数据流

```text
SchedulerOutput
  ├─ finished_req_ids        → table.free_slot()
  ├─ scheduled_new_reqs      → table.add_request()
  ├─ scheduled_cached_reqs   → _sync_preempted()（被抢占则重放 prefill）
  └─ num_scheduled_tokens    → 每请求一步
        │
        ├─ 校验：prefill 长度 / 不支持分块 prefill / decode 只调度 1 token
        │
        ▼
   按 num_parallel_slots 分 wave
        │  wave 内 i 号请求用第 i 号激活/块表窗口
        ▼
   step_request(slot_index=i) ─▶ execute_plan(窗口计划, ctx)
        │
        ▼
   sampled_token_ids 按 req_ids 顺序回填 → ModelRunnerOutput
```

---

## 四个关键提交

| 提交 | 日期 | 主题 | 规模 |
| --- | --- | --- | --- |
| `1a44d83` | 10-23 | 接入 vLLM 后端，静态槽分区 | +1084 / -19 |
| `67a26e6` | 10-24 | KV 固定槽 + 页表间接寻址 | +1011 / -769 |
| `d6894f5` | 10-25 | 全局块表缓冲 + 全局物理槽池 | +483 / -254 |
| `a0739c7` | 10-26 | 运行时跨请求并行 | +1250 / -89 |

一句话概括演进：**先接通 → 重构 KV 地址 → 收敛全局槽池 → 真正并发**。

---

## Stage 1（`1a44d83`）打通链路

新增文件：

- `backend/vllm_worker.py`：`PIMWorker(WorkerBase)`，把每步调度结果翻译成逐请求执行
- `backend/vllm_executor.py`：仅 13 行，把 `worker_cls` 换成 `PIMWorker`
- `scripts/serve_pim.py`：OpenAI 兼容服务入口

```python
class PIMExecutor(UniProcExecutor):
    def _init_executor(self) -> None:
        self.vllm_config.parallel_config.worker_cls = "backend.vllm_worker.PIMWorker"
        super()._init_executor()
```

初版 KV：编译期预留 `num_slots` 个请求槽，每槽独占一段连续 MRAM。
并发上限 = `max_num_seqs`。

---

## Stage 2（`67a26e6`）固定槽 + 块表

不再为每个请求预留整段 `max_seq` 连续区，改成**按需占用多个固定大小物理槽**：

- 一个物理槽容纳 `slot_tokens` 个 token 的全部本地 KV 平面
- 槽位置由运行期**块表**映射（逻辑块 → 物理槽）
- 预填充/解码按逻辑块写 K/V，按块 gather 历史
- 块表项未分配（= `-1`）时直接报错，暴露上游 bug

```python
def physical_slot(owner: int, block: int) -> int:
    slot = int(block_tables[owner][block])
    if slot < 0:
        raise RuntimeError(f"dpu{owner} 块表项 block={block} 未分配")
    return slot
```

---

## Stage 3（`d6894f5`）全局块表 + 全局槽池

- **每台 DPU 一份共享块表缓冲**，host 每步 `write_block_table` 整段覆盖
- **物理槽池全局一份**：`kv_pool_slots`，请求之间共享、复用
- vLLM 看到的 block 只是账面数字：`num_gpu_blocks = kv_pool_slots`

```python
def _write_block_table(compiled, hal, slot, slot_index=0) -> None:
    for spec in compiled.kv_specs.values():
        write_block_table(spec, hal, slot.blocks.slots, slot_index)
```

块表缓冲必须**整段覆盖**：短请求不能继承上一个长请求留在表尾的块号。

---

## KV 区地址布局（每台 DPU）

```text
MRAM 单 DPU 地址空间
├── 权重区        [0, kv_base)
├── KV 区         [kv_base, act_base)
│   ├── 块表窗口 0   blocks_per_request × int32（对齐）
│   ├── 块表窗口 1
│   ├── ...          共 P 份，并行请求各写一份
│   ├── 物理槽池     [slots_base, ...)  pool_slots × slot_bytes
│   └── 槽内平面     layer → kv_head → k/v
└── 激活区        [act_base, act_base + P*S)   P 个窗口，统一 stride S
```

关键几何（`memory/kv_layout.py`）：

```python
slot_bytes          = 2 * len(layers) * len(kv_heads) * align_up(slot_tokens*head_dim*bytes, align)
blocks_per_request  = ceil(max_seq / slot_tokens)
block_table_bytes   = align_up(blocks_per_request * 4, align)
slots_base          = kv_base + P * block_table_bytes
kv_allocated_bytes  = P * block_table_bytes + pool_slots * slot_bytes
```

---

## 块表间接寻址：设备内核怎么读 KV

`runtime/kernels.py::sdpa_kernel` 编译期把 KV 几何烘进命令 payload
（`bt_base` / `slot_bytes` / `max_blocks` / `k_off` / `v_off`）。

```python
# 每个 owner 读一次块表，后续按逻辑块查物理槽
block_tables = {
    owner: hal.read_local(owner, geo["bt_base"], (geo["max_blocks"],), np.int32)
    for owner, geo in geometry.items()
}
...
base = entry["k_off"] + slot * geo["slot_bytes"]   # 物理槽定位
```

要点：

- 块表地址直接进内核，**查表在设备上完成**
- 单遍在线 softmax，逐物理槽算 QK / softmax / PV，不拼接 KV 历史
- 内核编不出来时才回退主机两遍（数值不完全等，≤2.4e-4）

---

## 算子层全链路：从编译请求到设备 C

```text
OpCompileRequest(op="paged_attn", slot_tokens, slot_elems, scale)
   │  opcompiler_bridge/driver.py        形状 / 几何校验
   ▼
paged_attn_kernel()                     拼 MLIR
   %y = pim.paged_attn %q, %c, %bt, %ko, %vo, %pos, %nb, %nt {...}
   │  FlagTree: PIMOps.td 定义 + Ops.cpp::verify()
   ▼
LowerPIMToEmitC.cpp                     → 单遍在线 softmax 的设备 C
   │  compile_op() + load_kernel()
   ▼
fn(q*, cache*, bt*, kOff, vOff, pos, numBlocks, numTokens, out*)
   │  runtime/kernels._compiled_paged_attn   按几何缓存
   ▼
_paged_attention()   ← 每层每头调同一个 .so，只换 kOff / vOff
```

一条算子跨越 4 个仓：编译器（bridge）→ FlagTree 方言 → 降级 → 运行时。

---

## 算子定义：哪些是常量，哪些随请求变

```mlir
%y = pim.paged_attn %q, %cache, %bt, %ko, %vo, %pos, %nb, %nt
    {slotTokens = 16 : i64, slotElems = 65536 : i64,
     scale = 0.08838834764831845 : f64}
    : tensor<1x128xf16>, !pim.memdesc<16777216xf16, #pim.mram>,
      tensor<256xi32>, i32, i32, i32, i32, i32 -> tensor<1x128xf16>
```

| 类别 | 字段 | 理由 |
| --- | --- | --- |
| 编译期属性 | `slotTokens` / `slotElems` / `scale` | 同 DPU 所有层/头一致，可提前展开 |
| 按值操作数（i32） | `kOff` `vOff` `pos` `numBlocks` `numTokens` | 随请求、随步变；写死会每步重编 |
| 指针 | `q` / `cache`(MRAM) / `blockTable`(i32) | 槽池与块表直接交给内核查表 |

`OpCompileRequest` 新增三字段：`slot_tokens` / `slot_elems` / `scale`。
校验：`slotElems % slotTokens == 0`、一行装整数个 `head_dim`、
`cache_elems % slot_elems == 0`、`cache_elems ≤ 2³⁰`（i32 寻址）。

---

## 设备内核：块表 gather + 单遍在线 softmax

```c
poolSlots = cache.count / slotElems;           // 槽池槽数（编译期）
for (bi = 0; bi < numBlocks; ++bi) {
  slot = bt[bi];                               // 逻辑块 → 物理槽
  safe = slot < poolSlots ? slot : 0;          // 表尾 -1 夹到槽 0，不出界
  kBase = kOff + safe * slotElems;             // 本块 K 平面起点
  for (j = blockStart; j < min(blockEnd, hist); ++j) {
    dot = Σ_d q[t,d] * cache[kBase + row*headDim + d];   // f32 累加
    s = roundF16(dot) * scale;
    logits[t,row] = (j <= pos + t) ? s : -1e30;          // 因果掩码
  }
  alpha = exp(mOld - mNew);                    // 在线 softmax 重标定
  runningSum[t] = runningSum[t]*alpha + blockSum;
  acc[t,:]      = acc[t,:]*alpha + Σ_r prob*V;
}
out[t,d] = roundF16(acc[t,d] / runningSum[t]); // 收尾一次除法
```

K/V 各读一次，分数缓冲只占一个块。与主机两遍只差 f16 取整顺序（≤2.4e-4）。

---

## 运行时选路：融合内核 vs 主机兜底

```python
fn = None
if np_dtype == np.float16 and not uses_graph_mask:
    fn = _compiled_paged_attn(q_shape, cache_elems, max_blocks,
                              slot_tokens, slot_elems, scale, ctx=ctx)
if fn is not None:
    base = hal.raw_mram_ptr(owner)
    fn(q_ptr, base + geo["slots_base"], base + geo["bt_base"],
       (entry["k_off"] - slots_base) // dtype_bytes,   # 相对槽池的元素偏移
       (entry["v_off"] - slots_base) // dtype_bytes,
       pos, num_blocks, num_tokens, out_ptr)
else:
    out = _host_paged_attention(...)              # 逐块两遍，allclose 1e-3
```

- 内核缓存键 `(ctx, q_shape, cache_elems, max_blocks, slot_tokens, slot_elems, scale)`
  → **所有层、所有头、所有步共用一个 `.so`**，只换 `kOff` / `vOff`
- 图带自定义掩码或 f32 KV（仅测试）→ 主机路径，语义不变

---

## Stage 4（`a0739c7`）为什么需要跨请求并行

泳道折叠之后，单请求单步利用率停在 ~58%，剩下的等待边都是**跨泳道真实屏障**
（DMA、归约），单请求内再怎么优化也换不来更多重叠。

思路：**让多个独立请求同时在飞**，用一个请求的 DPU 计算去填另一个请求等待
主机重分布的空泡。

| 量 | Phase 0a 后 | Phase 1 后 |
| --- | --- | --- |
| 单请求单步 | 548.9 ms / 58.0% | 不变（单请求语义保持） |
| 多请求 | 串行逐槽 | 按窗口并发 |

---

## 五个机制（缺一不可）

1. **每请求激活窗口**（`memory/mem_planner.py`）
   激活区复制 `P` 份，统一 stride `S`
2. **每请求块表窗口**（`memory/kv_layout.py`）
   KV 区放 `P` 份块表，各写各的
3. **编译期计划变体**（`runtime/binding.py`）
   为窗口 `k` 生成平移后的计划
4. **执行上下文**（`backend/hal_numpy.py`）
   输入 / 位置 / 令牌数 / 事件表按请求隔离
5. **按 (上下文, DPU) 分泳道**（`backend/hal_numpy.py`）
   同请求 FIFO，不同请求不排队

---

## 激活窗口：为什么必须统一 stride

```text
S = align_up(max_d(act_extent_d), hw.align)
total = act_base + P * S
窗口 k 的激活基址 = act_base + k*S
```

**统一步长的原因**：`comm/plan.py` 的批量 `push_from / push_to /
broadcast_to` 用 `first.src_addr` **一个基址扇出到所有 DPU**。若各 DPU 步长
不同，同一批量的偏移就错位。

统一成全局最大值后，任何 DPU 的窗口 `k` 都在 `act_base + k*S`，
批量扇出的单一偏移对所有 DPU 同时成立。

---

## 计划变体：bind_plan 只平移激活与块表

`runtime/binding.py`

```python
def bind_plan(plan, *, act_base_by_dpu, act_shift, block_slot, bt_stride):
    if act_shift == 0 and block_slot == 0:
        return plan                       # 0 号窗口：零拷贝，行为不变
    commands = [
        replace(
            cmd,
            payload=_bind_payload(cmd.payload, ...),   # sdpa.bt_base += k*bt_stride
            reads=[_shift_access(a, ...) for a in cmd.reads],
            writes=[_shift_access(a, ...) for a in cmd.writes],
        )
        for cmd in plan.commands
    ]
    return ExecutionPlan(commands=commands)
```

- **只平移**激活区地址（`offset >= act_base[dpu]`）和 `sdpa.bt_base`
- **不动**权重、KV 区，也**不动** `payload["mram_offset"]`
- `mram_offset` 是 `_ctx_of` 的编译缓存键 → 所有窗口共享同一份已编译内核

---

## 执行上下文隔离

`backend/hal_numpy.py`

```python
@dataclass
class ExecutionContext:
    lane_key: int = 0
    values: dict[str, object] = field(default_factory=dict)
    pos: int | None = None
    num_tokens: int | None = None
    events: dict[int, Event] = field(default_factory=dict)
```

- `submit(cmd, ctx)` 从 `ctx.events` 解等待、写回 `ctx.events`
  → 命令 id 每计划从 0 开始也不冲突
- worker 线程 thread-local 安装 `ctx`：
  `bound_value / bound_pos / bound_num_tokens / result_of` 优先读本上下文
- **不传 ctx 时**全部回退旧全局路径，单请求行为不变

---

## 泳道与 wave 调度

泳道键从 `dpu_id` 改成 `(lane_key, dpu_id)`：

```python
def _submit_on_dpu(self, dpu_id, lane_key, deps, ctx, fn, *args):
    key = (lane_key, dpu_id)
    with self._stream_locks[key]:
        prev = self._stream_tail.get(key)
        def run():
            if prev: prev.result()
            for dep in deps: dep.result()
            with self._install_context(ctx):
                return fn(*args)
```

调度（`runtime/executor.py::run_one_step` / `vllm_worker.execute_model`）：

```text
请求集合 [A B C D E]         P = 2
   wave 0: [A(win0) B(win1)]  ← 并发
   wave 1: [C(win0) D(win1)]  ← 并发
   wave 2: [E(win0)]          ← 并发
```

---

## 泳道模型示意

```text
同一 DPU 上两条请求泳道（lane_key 0 / 1），各自 FIFO：

lane0: [k0]──▶[dma]──▶[k1]────────▶[k2]        (请求 A，窗口 0)
lane1:       [k0]──▶[dma]──▶[k1]──▶[k2]        (请求 B，窗口 1)
                 ▲ 允许交错：不同上下文互不排队
                 │
     同请求内部严格按提交序（泳道折叠前提不变）
```

- host 命令仍由 `_host_lock` 串行（`xfer_buffers` 是机器级全局）
- 加速来自 **DPU 填 host 空泡**，不是 host 并行
- 线程池尺寸 `min(64, num_dpus*P + P + 1)`

---

## 性能实测

配置：4 DPU / 4 层 / hidden 1024 / **4 请求 × 8 步 decode**，
同一份 `num_parallel_slots=4` 蓝图，只改 `parallel_slots`：

| 组合 | 8 步 wall | 每步 wall | DPU 利用率 |
| --- | --- | --- | --- |
| 串行（`parallel_slots=1`） | 27 758 ms | 3 470 ms | 61.2% |
| 并发（`parallel_slots=4`） | 19 913 ms | 2 489 ms | **76.2%** |

- 总工作量相同，吞吐 **×1.39**，利用率 **+15 个百分点**
- 两组每 DPU 忙碌合计接近（~16.8s vs ~15.2s）→ 提速来自**填空泡**

---

## 正确性验证（测试矩阵）

| 测试文件 | 覆盖点 |
| --- | --- |
| `test_vllm_executor.py` | vLLM 引擎经 `PIMExecutor` 跑通两并发请求 |
| `test_parallel_requests.py` | 4 请求 token 逐字等于单跑；混合 prefill/decode；5 次确定性 |
| `test_parallel_overlap.py` | 同 DPU 两窗口 launch 真重叠 + **越级** + 内核/host 重叠 |
| `test_activation_windows.py` | 激活窗口几何、统一步长、超预算报错 |
| `test_exec_binding.py` | bind_plan 只平移激活/块表；`mram_offset` 不变；不别名 |
| `test_hal_numpy.py` | 上下文复用命令 id、各读各的绑定值、同 DPU 不同泳道并发 |
| `test_kv_slots/slot_isolation` | 槽几何、块表乱序、槽隔离 |

---

## 越级判据：证明「真交错」而非「假相邻」

```text
order[0, c0] < order[1, c1]   且   start(lane1,c1) < end(lane0,c0)   且同 DPU
```

后提交的窗口 1 命令，要在**更早提交**的窗口 0 命令结束前就在同一 DPU 上开始。

这排除了「两条流各自跑、恰好时间相邻」的假象，证明共享线程池配合每泳道独立
FIFO 确实产生了交错。

- 小模型：重叠 / 越级 / 内核与 host 重叠 全通过
- **真实 7B / 8 DPU**：两请求 token 与单跑一致 + 跨窗口重叠，
  本地 **344 s 通过**（slow 用例）

---

## 当前进展总结

- **链路打通**：vLLM 引擎 → `PIMExecutor` → `PIMWorker` → `step_request` →
  执行计划 → `NumpyBackend` 全链路可跑，HTTP 冒烟通过
- **KV 定型**：固定槽 + 块表间接寻址 + 多块表窗口 + 全局槽池
- **算子下沉**：`pim.paged_attn` 融合内核（单遍在线 softmax，内核查块表），
  K/V 各读一次，一个 `.so` 服务所有层/头；主机两遍仅作兜底
- **并发落地**：多请求同一步真正并行，token 与单跑一致、7B 规模已验证
- **回归**：快测 1439 passed；4 请求吞吐 ×1.39；7B 慢测 1 passed
- **兼容**：`parallel_slots=1` / `PIM_PARALLEL_SLOTS=1` 逐字节回到 Phase 0a

---

## 已知不足

- 只支持贪心采样（`temperature / top_p` 无效）
- prompt 超 `PIM_PREFILL_SEQ_LEN` 或 `max_model_len` 无法运行；不支持分块 prefill
- 不支持前缀缓存、LoRA、投机解码；PIM 内部有 TP，但不支持多卡
- 改并发数 / 槽几何必须**重新编译**（窗口是编译期几何）
- 全部跑在 `NumpyBackend` 模拟上，真硬件 `hal_vendor` 仍是桩
- 同 DPU 两请求命令并发是**乐观建模**（数值安全：窗口/块表/KV 均不相交）

---

## 下一步

- 真硬件后端 `hal_vendor`：把「同 DPU 命令串行」从实现细节升级为 HAL 硬约束
- 更忠实的同 DPU 调度：前瞻合并，替代当前乐观并发模型
- 前缀缓存 / 分块 prefill（需突破静态图固定长度）
- 继续剪跨泳道 WAR 边，减少主机重分布

---

## 附录：启动方式与环境变量

```bash
python scripts/serve_pim.py --model /path/to/Llama-2-7b-hf \
    --max-model-len 512 --max-num-seqs 8 --max-num-batched-tokens 4096
```

| 变量 | 默认值 | 作用 |
| --- | --- | --- |
| `PIM_PREFILL_SEQ_LEN` | `max_model_len` | prefill 图固定长度，prompt 补齐 |
| `PIM_NUM_DPUS` | 4 | PIM 内部张量并行的 DPU 数 |
| `PIM_MRAM_BYTES` | 4 GiB | 单 DPU 的 MRAM 总量 |
| `PIM_SYS_RESERVE_BYTES` | 64 MiB | 系统预留，不参与三区规划 |
| `PIM_KV_POOL_SLOTS` | `max_num_seqs × 每请求块数 + 1` | 全局物理槽池大小（+1 给 vLLM 的 null 块） |
| `PIM_PARALLEL_SLOTS` | 2 | 同一步并发窗口数（受 `max_num_seqs` 限制） |

---

## 附录：关键文件索引

| 文件 | 职责 |
| --- | --- |
| `backend/vllm_worker.py` | `PIMWorker`，调度结果 → 逐请求执行 |
| `backend/vllm_executor.py` | `PIMExecutor`，单进程驱动 worker |
| `scripts/serve_pim.py` | OpenAI 兼容服务入口 |
| `runtime/executor.py` | `RequestTable` / `KVSlotPool` / `step_request` / `run_one_step` |
| `runtime/binding.py` | `bind_plan`，窗口平移 |
| `runtime/compile.py` | 两图编译 + 窗口变体 |
| `runtime/kernels.py` | `_paged_attention` 选路；`_compiled_paged_attn` 内核缓存 |
| `memory/kv_layout.py` | KV 布局 + 块表写入 |
| `memory/mem_planner.py` | 三区规划 + `fit_activation_windows` |
| `backend/hal_numpy.py` | `ExecutionContext` + `(lane,dpu)` 泳道 |
| `opcompiler_bridge/driver.py` | `paged_attn` 契约校验 + MLIR 拼装 |
| `opcompiler_bridge/oplevel_kernel.py` | `paged_attn_kernel()` 生成 `pim.paged_attn` |
| FlagTree `PIMOps.td` / `Ops.cpp` | `pim.paged_attn` 方言定义 + verifier |
| FlagTree `LowerPIMToEmitC.cpp` | 单遍在线 softmax 设备 C 降级 |
| `docs/vllm-backend-20261023.md` | vLLM 接入设计 |
| `docs/executor-parallel-phase1-20261026.md` | 跨请求并行设计 |
