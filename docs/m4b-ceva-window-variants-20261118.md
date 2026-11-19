# M4b 首片：CEVA 窗口变体静态派生 + exact 降级防线（2026-11-18）

依据 M0 探路结论（`docs/m0-ceva-probe-20261118.md`）两条裁决：①厂商不支持
运行期行级 gather，CEVA 侧 MoE 只允许 dense_sim / dynamic_dispatch；②
mram_offset 不烧进内核，窗口变体图共享同一份内核。本片只做导出侧与契约，
runner 侧并发改造在下一片。

## 调查结论（决定设计的事实）

- **GML/bin 不携带任何 DDR 地址**。缓冲只按 `.bin` 文件名引用；边界地址由
  runner 在每次 `executeSubgraph` 的 `io_map` 按名字绑
  （`/data/fjg/ceva/src/llama_runner/src/llama2_npm.cpp` 的 `run_decode`）；
  中间缓冲由厂商 SDK 摆进 `p_memory`（outputs ini 的 `intermediate` 区）。
  因此「窗口 w 变体」不落任何 GML 字段——派生只复制文件（M0-② 保证逐字节
  共享内核），窗口平移记录进 bundle.json 供 runner 绑 io_map。
- **runner 并发模型**：一次仿真本来就同时装载全部注册表（prompt 32 + decode
  32 + lm_head，TP2 下更多），多图实例无硬阻碍；`executeSubgraph` 按图阻塞
  原子执行。但整条链是单请求的：每层只有一套 KV cache（`key_cache_<n>`）、
  一个 `intermediate` 暂存区、一组 hidden 进/出乒乓缓冲；唯一的"每请求行"
  先例是 `line_idx`（mask/sin/cos 按 8KiB 行偏移）。DDR 布局
  （`llama2_ddr_manager.py` 的 `MODEL_SIZES`）与驱动循环都没有请求维度。
- **exact 此前的行为**：`run_full_pipeline.py --backend ceva` 解析
  `--moe-execution` 但 `run_ceva` 从不传给 `export_gml.py`，导出侧也没有
  moe_execution 概念——静默忽略，无任何拦截。

## 改动

| 文件 | 内容 |
| --- | --- |
| `gml_bridge/ceva.py` | `require_supported_moe_execution`（exact 拒绝，引用 M0 文档）；`CevaWindowGeometry` / `window_geometry`（从 I/O 契约推步长）；`derive_window_variants`（池深派生 + 变体记录）；`export_ceva_bundle` 加 `moe_execution` / `window_pool` 参数，每图写 `windows` 字段，清单记 `moe_execution` |
| `scripts/export_gml.py` | `--moe-execution`（choices 取 `MOE_EXECUTION_MODES` 单一真源）与 `--window-pool`；非 CEVA 模式给这两参直接报参数错误；`_main_ceva` 在加载模型前先拦 exact |
| `scripts/run_full_pipeline.py` | `run_ceva` 把 `--moe-execution` 传给导出步骤，不再静默丢弃 |
| `tests/test_ceva_export.py` | 假 `load_ceva_spec` 从元组换成真 `CevaGraphSpec` 桩（导出循环现在要读 I/O 契约），假导出补写 GML |
| `tests/test_ceva_windows.py`（新） | 9 个用例，见下 |

## 关键结构

`CevaWindowGeometry`：`pool`（变体份数 = 池深）、`act_stride`（相邻窗口边界
激活 hidden 进/出对的步长，4K 对齐，与 ddr_manager 的 `align_to_x` 一致）、
`bt_stride`（CEVA 无块表——KV 行索引是内容级——恒 0，占位对齐 M1 几何）。
平移语义与主链 `runtime/materialize.py` 的 `WindowSpec` 一致：窗口 w 相对
窗口 0 只平移激活地址，内核身份不动。

bundle.json 每图新增 `windows` 字段：

```json
{"pool": 2, "act_stride": 16384, "bt_stride": 0,
 "variants": [{"window": 0, "directory": "decode/model_layers_0_decode_v2/...",
               "act_shift": 0, "bt_shift": 0, "gml_sha256": "..."},
              {"window": 1, "directory": "decode/model_layers_0_decode_v2_w1/...",
               "act_shift": 16384, "bt_shift": 0, "gml_sha256": "..."}]}
```

变体目录用 `_w<N>` 后缀（与 TP2 分片的 `_attn/_mlp` 同一约定），窗口 0 用
原图；`gml_sha256` 各变体相同，供 runner 核对同源。绝对基址由 DDR 布局定，
清单只记相对平移。

## 测试

- `tests/test_ceva_windows.py`：exact 防线（含"不碰模型不落文件""CLI 在加载
  模型前拦截"）、几何推导（decode 契约 8KiB×2 → 16384）、变体派生（文件逐
  字节共享、记录间差异恰为窗口平移、`gml_structure_check.py` 五条规则对
  窗口 0 与派生变体都通过、重跑覆盖不留旧字节）、bundle 接线（monkeypatch
  假导出后核 windows 字段与变体落盘）、CLI 参数与模式守卫。
- 回归：`pytest tests/ -q -k "ceva or gml or genesim_bridge or placement"`
  549 passed / 4 skipped；另跑触及 `run_full_pipeline` 的五个测试文件
  382 passed / 4 skipped。真实 CLI 冒烟：`--ceva-reference <不存在>
  --moe-execution exact_dispatch` 在加载模型前抛 ValueError（引用 M0 文档）
  且不产任何文件。

## 当前不足（下一片的工作量）

1. **runner 侧多请求缺口**：DDR 布局每窗口需要独立的 KV cache 一套
   （llama2-7b 约 128MB/窗口）、hidden 乒乓缓冲与暂存区份数，驱动循环需要
   请求间交错调度（现在的 decode 循环是 argmax→LUT→embedding 串行反馈）。
   本片池深默认 1，`--window-pool >1` 只完成导出侧派生。
2. **磁盘代价**：整包变体按图根整目录复制（现网一个 65 图包约 55G），池深
   翻倍即翻盘。runner 侧应接 GC 的 weight sharing（`merge_manifest.json` 已
   支持多图共享 weight registry）后再开池深 >1。
3. MoE 图走 CEVA 的真实导出（qwen3_moe 的 dense/dynamic_dispatch 降级）还
   没有实现，本片只把 exact 拦在了边界上。
