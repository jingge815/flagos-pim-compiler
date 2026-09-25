# genesim 桥接：14 个算子级原语全部接上（20260924）

## 一、这次要解决的问题

方案表 1.2.4 列了 14 个设备侧原语（助记符），`genesim_bridge/op_classify.py` 里
`MNEMONICS` 也早就写全了，但 GeneSim 那条路实际上只用到 5 个：

- 整模型（`models/llama2_7b.ir`）的骨架只产 `GEMM / GEMV_SCORE / SOFTMAX /
  GEMV_CONTEXT / RMSNORM / SILU / VECTOR_ADD / VECTOR_MUL / MODEL_INPUT /
  MODEL_OUTPUT`，注意力区被抽象成两次 GEMV；
- `build_recipes()` 只认出 8 个 op_type，映射到 `pim.normalize / pim.matmul /
  pim.softmax / pim.lut / pim.eltwise` 5 个助记符。

于是 `--ir-level oplevel` 一跑，打印出来的 mnemonic 列表只有 5 个。这次把骨架里
缺的算子补上，桥接表跟着补齐，14 个全部能在整模型上量到成本。

注意 `src/pim/pim_compiler.py` 的 `COMPILE_FUNCS` 和 ISA 本来就是齐的，一个字没改；
`genesim_bridge/ir_cost.py` 的正则口径也没动（A 路成本不能失真）。

## 二、改了什么

### 1. GeneSim 仓 `src/ir/model_ir.py`（`build_from_hf_config`）

按解码块真实的数据流补节点，每个新算子都挂在原有数据流上，没有悬空节点：

| 新算子 | 位置 | 作用 |
| --- | --- | --- |
| `GATHER` | 层 0 最前面，一次 | 词嵌入查表，token id 换成隐藏态 |
| `QUANT` | 每层 RMSNORM 之前 | 隐藏态动态量化（统计/求倒数/定标/定点化） |
| `RESHAPE` → `TRANSPOSE` → `SPLIT` | q_proj 输出 | 拆头前的改形与换轴，再按头切开 |
| `SPLIT` | k_proj / v_proj 输出 | 按头切开 |
| `ROPE` ×2 | 每个头内，Q 路和 K 路各一个 | 只转本头那一段 |
| `MEM_COPY` ×2 | 每个头内 | K/V 写进缓存（沿用已有 op_type，没造新的） |
| `MASK` | GEMV_SCORE 与 SOFTMAX 之间 | 因果掩码 |
| `CONCAT` | 各头上下文合回 o_proj 之前 | 按头拼回去 |

形状沿用已有的符号写法（`("Tq", hidden_size)`、`("Tp+Tq", head_dim)`），
`flops_coeffs` / `data_bytes_coeffs` 也用同名符号。视图类算子只算搬运不算 flops。

每层补完后的链路：

```
输入 → QUANT → RMSNORM → q/k/v 三个 GEMM
                              │
        ┌─────────────────────┼─────────────────────────┐
        │ q: RESHAPE→TRANSPOSE→SPLIT          k/v: SPLIT │
        ▼                                              ▼
   逐头：ROPE(Q) + ROPE(K) + MEM_COPY(K/V) → GEMV_SCORE → MASK → SOFTMAX
        → GEMV_CONTEXT
        │
        ▼ 所有头 CONCAT → o_proj → 残差 → RMSNORM → gate/SILU/up → 逐元素乘
        → down → 残差
```

### 2. GeneSim 仓 `src/scheduler/gene_sim_scheduler.py`

视图类算子作用在整层上，没有单个头的归属，而 TP 展开原先只会按 `q_head_id`
找分片。改两处：

- 新增模块级常量 `ATTENTION_OP_TYPES`（`ROPE / GEMV_SCORE / MASK / SOFTMAX /
  GEMV_CONTEXT`），替换原先两处写死的同义集合，放置单元按它切「注意力之前 /
  注意力 / 注意力之后」；
- `_expand_attention_tensor_parallel_group` 里把 `shard_of_head` 换成
  `shards_of`：逐头节点仍然只落在一个分片，整层作用的视图算子
  （`SPLIT / CONCAT / TRANSPOSE / RESHAPE`）横跨所有分片，每个分片各拿
  `1/N` 字节。

### 3. GeneSim 仓 `src/predictor/collect.py`

`BENCHMARK_LLAMA7B_SIGNATURE` 跟着新骨架更新（6852 算子 / 14375 依赖 /
13214687232 参数字节，以及新的 op_type 计数表）。

### 4. 本仓 `genesim_bridge/op_classify.py`

- `UNCOVERED_OP_TYPES` 补上 `GATHER / QUANT / ROPE / MASK / MEM_COPY /
  TRANSPOSE / RESHAPE / SPLIT / CONCAT`，并写清它**只挡 A 路**：B 路
  （`--ir-level oplevel`）量的是本仓自己发的整算子 IR，不掺 FlagGems，所以这些
  op_type 在 B 路上照常桥接；
- 新增 `bpath(mnemonic, builder)`：只给 B 路配方的工厂，`source_name` 就是
  mnemonic，`build` 留空（A 路保留模板成本），`pimir` 直接发
  `opcompiler_bridge/oplevel_kernel` 里该 mnemonic 的整算子 IR，形状取它自己的
  代表点——**成本公式不在 op_classify 里再抄一份**；
- `build_recipes()` 的返回表里把 `RMSNORM / SILU` 换成 `bpath(...)`，并新增
  `GATHER / QUANT / ROPE / MASK / MEM_COPY / TRANSPOSE / RESHAPE / SPLIT /
  CONCAT` 九条。

`genesim_bridge/cost_extractor.py` 不用改：它本来就有
`bridged_here = ir_level == "oplevel" and op_type in recipe_types`，
未覆盖类型在 B 路上照样走桥接。

## 三、关键结构体 / 常量

| 名字 | 位置 | 说明 |
| --- | --- | --- |
| `MNEMONICS` | `op_classify.py` | 表 1.2.4 的 14 个助记符，计算 10 + 视图 4 |
| `_OPLEVEL_IR` | `op_classify.py` | mnemonic → 代表形状的整算子 IR |
| `OpRecipe(source_name, build, pimir, mnemonic)` | `op_classify.py` | 一个 op_type 的两路配方；`build` 为空表示只有 B 路 |
| `ATTENTION_OP_TYPES` | `gene_sim_scheduler.py` | 注意力区算子类型，放置单元按它切分 |
| `shards_of(op_id)` | `gene_sim_scheduler.py` | 一条边的对端覆盖哪几个 TP 分片 |

## 四、测试

| 范围 | 命令 | 结果 |
| --- | --- | --- |
| GeneSim 调度与建模 | `cd /media/disk/fengjingge/src/genesim && python -m pytest tests/sim -q` | 668 passed, 1 skipped |
| GeneSim 全仓 | `python -m pytest tests -q` | 746 passed, 9 failed, 1 skipped（9 个是缺 `torch_geometric`，改动前就这样） |
| 本仓桥接 | `python -m pytest tests/test_genesim_bridge.py tests/test_gml_coverage.py -q` | 50 passed, 3 skipped（3 条跳过是因为 A 路派生产物被删，见「不足」） |

改动同步更新的测试：

- `tests/sim/test_model_ir.py`：最小骨架的 `model.input` 引用数改 2，新增
  `layer.0.embedding.output` 计数；GQA 那条用例按新链路逐步取后继算子
  （`successor_of`），字节系数改成 `{"Tq": 64*2}` / `{"Tq(Tp+Tq)": 2}` 这类；
- `tests/sim/test_compiler_placement.py`：原「每个头读自己分片」的用例换成
  `test_layer_wide_view_ops_split_activation_bytes_across_shards`，断言整层视图
  算子的入边恰好 n 条、字节和等于整张激活，o_proj 每条分片入边是 `1/n`；
- `tests/sim/test_partition_compute_graph_with_runtime.py`：按 `semantic_role`
  取 Q 投影，打分与 Q 投影之间补上 MASK；
- `tests/predictor/test_collect.py`：GPT-2 demo 的 op_type 集合补新类型；
- 本仓 `tests/test_genesim_bridge.py`：整模型覆盖那条断言改成
  `covered == sorted(MNEMONICS)`，骨架再缺节点就会当场变红。

## 五、验收实测（2026-09-24 20:14 当场跑）

命令：

```bash
cd /media/disk/fengjingge/src/genesim
python scripts/refine_ir_with_flagtree.py --ir models/llama2_7b.ir \
    --out-ir /tmp/op.ir --sidecar /tmp/op_sc.json --seq-len 16 --ir-level oplevel
```

**1）mnemonic 列表由 5 个变 14 个**

改前（旧骨架 `/tmp/llama2_7b_before.ir`）：

```
[oplevel] mnemonic: ['pim.eltwise', 'pim.lut', 'pim.matmul', 'pim.normalize', 'pim.softmax']
Bridged 3489 operators from oplevel; 2 kept template costs
```

改后（新骨架 `models/llama2_7b.ir`）：

```
[oplevel] mnemonic: ['pim.concat', 'pim.dynamic_quant', 'pim.eltwise', 'pim.gather',
 'pim.kv_cache', 'pim.lut', 'pim.mask', 'pim.matmul', 'pim.normalize', 'pim.reshape',
 'pim.rope', 'pim.softmax', 'pim.split_heads', 'pim.transpose']
Bridged 6850 operators from oplevel; 2 kept template costs
```

**2）新算子在产物里有正计数、成本非零**

新 op_type 在 `/tmp/op.ir` 里的计数（整图 6852 算子 / 19 种 op_type）：

| op_type | 计数 |
| --- | --- |
| GATHER | 1 |
| QUANT | 32 |
| ROPE | 2048 |
| MASK | 1024 |
| MEM_COPY | 64 |
| TRANSPOSE | 32 |
| RESHAPE | 32 |
| SPLIT | 96 |
| CONCAT | 32 |

sidecar 里各 `source_name` 的成本（prefill / decode 取大者）：

| source_name | 算子数 | flops | data_bytes |
| --- | --- | --- | --- |
| `pim.dynamic_quant` | 32 | 132096 | 262144 |
| `pim.mask` | 1024 | 8192 | 1024 |
| `pim.rope` | 2048 | 196608 | 8192 |
| `pim.gather` | 1 | 0 | 131104 |
| `pim.kv_cache` | 64 | 0 | 262144 |
| `pim.transpose` | 32 | 0 | 262144 |
| `pim.reshape` | 32 | 0 | 262144 |
| `pim.split_heads` | 96 | 0 | 135168 |
| `pim.concat` | 32 | 0 | 262144 |

九个新助记符的 `data_bytes` 全大于 0；计算类三个（量化、掩码、旋转）flops 也大于
0，视图类四个加查表、缓存搬运本来就只有搬运量。

**3）测试**（同上表）。

## 六、当前存在的不足

- **A 路产物这轮没能重生成**：`--ir-level ttir/pimir` 要 GPU 加 `/dev/shm`，
  而 `/dev/shm` 现在仍是 504G 全满（被别的进程占着，`df` 显示可用 0），A 路跑不动。
  `triton-opt` 期间还被并行任务重建成 0 字节，20:14 才恢复成可执行。
  旧的 `models/llama2_7b_flagtree.ir` / `llama2_7b_pimir.ir` 是按旧骨架量的派生
  产物，留着会和新骨架对不上，所以先删了；本仓依赖它们的三条用例因此显示
  「SKIPPED: 需要先跑 refine_ir_with_flagtree.py 生成产物」，等 `/dev/shm` 空出来
  重跑 A 路即可恢复。A 路的口径本身没动。
- 视图类算子放在**层一级**，不是逐头一份。这样和参考 GML 的每层计数
  （Split 3 / Transpose 4 / Reshape 2 / Concat 1）一致，代价是 TP 展开时按
  `1/N` 分摊字节，而不是每个分片各算一份完整的搬运量。
- `MEM_COPY` 只建模了 K/V 写入；PIM 侧读回缓存的显式搬运还没单独建节点。
- `pim.gather` 在整模型上只出现一次（词嵌入），decode 阶段反复查表的那条路径
  （每步一次 GATHER）还没建模。
