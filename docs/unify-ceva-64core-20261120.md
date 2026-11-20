# 统一 CEVA 64 核配置贯穿全链路（2026-11-20）

## 问题

用户命令链（model_parser → trace → refine_ir_with_flagtree → ./run.sh 仿真）评测时只用了 8 个核，
而 genesim 配置声明 64 核。根因不在 model IR（图骨架硬件无关），在图编译器侧两处写死的默认值：

- `scripts/export_pp_placement.py` 写死 `NUM_DPUS = 8`；
- `scripts/run_full_pipeline.py` 默认 `--num-dpus 8`。

导出的放置 sidecar `llama2_7b_tp2_pp4_placement.json` 的 `dpu_to_cluster` 只有 8 项（cluster 0-7），
genesim 调度照单执行；且两侧校验（一对一、范围、字节预算）全部合法，唯独没人核「用满」，矛盾静默通过。

## 修改内容

### flagos-pim-compiler

| 文件 | 改动 |
| --- | --- |
| `scripts/export_pp_placement.py` | 新增 `--profile`（默认 `contracts/profiles/ceva_64core_4e.json`）；`--num-dpus`/`--mram-bytes`/`--sys-reserve-bytes` 缺省从画像推导；WRAM/DMA 对齐/编译目标与画像同源；删 `NUM_DPUS`/`DEFAULT_MRAM_BYTES` 常量 |
| `scripts/run_full_pipeline.py` | `--num-dpus` 默认 None → 从画像推导（64）；默认 `--num-stages 4` → 默认策略 tp16_pp4 |
| `scripts/refine_ir_with_flagtree.py` | 本仓入口补 `--profile` 参数，测量前经 `resolve_hardware` 核对画像与放置契约（与 genesim 入口对齐） |
| `genesim_bridge/placement_export.py` | 修复 B 路并发编译竞争：不同形状 key 可能编出相同 text 落到同一内容寻址缓存文件，固定名 tmp 会被并发 rename 吃掉；tmp 名改带 `pid-线程id` |

### genesim

| 文件 | 改动 |
| --- | --- |
| `scripts/export_fixed_pu_mapping.py` | CEVA 整核一对一分支：`num_dpus != 总核数` 直接拒绝（原来只拒超过） |
| `src/scheduler/gene_sim_scheduler.py` | `_validate_ceva_dpu_mapping` 新增核数一致性：正式核数（1/64/128）配置下放置必须映射满全部核，否则报错；非正式核数的测试配置不受影响 |
| `scripts/refine_ir_with_flagtree.py` | 示例命令补 `--profile`/`--placement` |
| `conf/ceva_64core_4e.yaml` | `compiler_placement_file` 指向新的 64 核 sidecar（tp16_pp4） |

### 链路统一后的数据流

```
contracts/profiles/ceva_64core_4e.json（唯一真源：64 核、每核 8e9 B、wram 64KB、align 64）
  ├─ export_fixed_pu_mapping.py  → PartitionPlan（64 DPU 一对一映射 cluster 0-63）
  ├─ export_pp_placement.py      → 放置 sidecar（策略 tp16_pp4，硬件契约与画像同源）
  ├─ refine_ir_with_flagtree.py --profile --placement → PIM MLIR 六元组从画像核对下发
  └─ conf/ceva_64core_4e.yaml + scheduler 校验（字节预算 + 核数用满双向核对）
```

## 关键决策

1. **「用满」校验放在 plan/scheduler 层，不放进画像层**。`HardwareProfile.validate_dpus` 保持
   「2 的幂且 ≤ 总核数」：单 DPU 测量（refine 不带 --placement 的路径）是合法用途，不该被拒。
2. **正式核数 gate**：scheduler 校验只在 `clusters_per_device ∈ CEVA_CORE_COUNTS(1/64/128)` 时强制
   用满。测试自建的非标准核数配置（如 4 核）只验证调度逻辑，不受约束——沿用 config_loader 两层
   校验（integer / count）的既有分层。
3. **64 核默认策略 tp16_pp4**：tp_width ≤ num_heads=32 且须整除 heads/kv_heads/intermediate/vocab；
   64 核下可行组合有 tp32_pp2、tp16_pp4、tp8_pp8 等，保持 `--num-stages 4` 默认即得 tp16_pp4。

## 已实施的测试

- pim-compiler：`tests/test_export_pp_placement_budget.py` 重写默认值用例（画像推导 64 核 + 三项覆盖），
  `tests/test_placement_export.py` 全量（71 通过）。
- genesim：`tests/sim/test_export_fixed_pu_mapping.py`（部分核拒绝用例 + 用满组合表修正，24 通过）；
  `tests/sim/test_compiler_placement.py` 新增正式核数正反两用例（266 通过）。

## 产物重生成与端到端验证

按新链路重生成全套产物并跑完仿真（同 trace、同画像、同 10 请求）：

```
export_fixed_pu_mapping.py --num-stages 4 --num-dpus 64   → models/llama2_7b_tp16_pp4_plan.json
export_pp_placement.py --partition-plan ... --measure-kernel-tiles
    → models/llama2_7b_tp16_pp4_placement.json（dpu_to_cluster 64 项、6850 算子全带 pimir、
      compiler.compile_profile_sha256 非空）
refine_ir_with_flagtree.py --ir-level pimir --profile ... --placement ... --fusion-from ...
    → models/llama2_7b_pimir.ir（224 个算子按 tp16 本地分片形状测量，3296 个桥接）
./run.sh（conf/ceva_64core_4e.yaml）
    → 日志：65 available resources, 65 used resources（64 ClusterPU + host 全用上）；
      run_manifest 绑定新 sidecar（sha e20c5cd8…）
```

新旧仿真对比（8 核 tp2_pp4 → 64 核 tp16_pp4）：

| 指标 | 8 核（旧） | 64 核（新） | 变化 |
| --- | --- | --- | --- |
| 总时延 | 5551.5 s | 1457.7 s | **3.8×** |
| 平均 TTFT | 2664.4 s | 654.6 s | 4.1× |
| 平均 TPOT | 0.766 s | 0.233 s | 3.3× |
| 吞吐 | 1.456 tok/s | 5.546 tok/s | 3.8× |
| MFU | 3.19% | 1.48% | 分母变为 64 核峰值，数字降属预期 |

注意：绝对数值仍建立在画像里 origin=assumption 的参数上（时钟、互连、cycle costs），
本次修复只解决「核数统一与用满」，不改变参数可信度。

## 追加修复（当日回归暴露的两个失败）

`python -m pytest tests/ -q -k "not llama2_7b"` 出现两个失败，均已修复：

1. `test_pimir_sidecar_agrees_with_ttir_on_flops`（两连败，两层问题）：
   - `KeyError: 'pim_options'`——带画像精化的 sidecar 有意不写 `pim_options`（改写
     `hardware` 五元组，见 `cost_extractor.py`）。测试改为双口径断言。
   - 修后继续挂在 `op 4 flops 两层不一致`（比值恰 16=tp_width）：pimir 按 tp16 本地
     形状测量、ttir 仍是全局形状，口径分叉。修复：`run_all.sh` 的 ttir 档也带
     `--profile/--placement/--fusion-from`，两级同口径成对（「PIM pass 不改 flops」
     的对照前提本来就是同形状），并重生成 ttir 产物。
2. `test_a_path_fuses_silu_into_tt_dot` / `test_dynamic_quant_group_size_reaches_the_kernel`
   （概率性，两轮失败的测试不同）：32 路 xdist 下多 worker 同时 `compile_op(force=True)`
   同一缓存键。锁内 `_invalidate_cache` 先删 so 再重编，worker A 拿到结果正要
   `dlopen` 时 worker B 删掉了 so → ENOENT。发布本来就全靠原子 `os.replace`，
   先删没有必要：删掉 `_invalidate_cache` 调用与定义（`opcompiler_bridge/driver.py`）。

修复后全量回归两遍 `2220 passed, 9 skipped`，零失败。

## 当前不足

- refine 的 `--profile` 仍是显式传参：两条入口都没把 CEVA 画像设为默认（GPT2 等非 CEVA 链路共用
  同一入口，不能强加）；漏传时退回全局 `pim_options()`（num-dpus=1 的单 DPU 口径），与放置不同源。
- 「用满」只约束 CEVA 整核模式；UPMEM 式多 TensorPU 模式（tensors_per_cluster>1）维持旧语义。
