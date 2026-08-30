# 技术方案索引

`docs/spec.md` 是完整技术方案（2494 行），**不要整篇读**。按下表用 `Read(offset, limit)` 只读需要的一节。

| 章节 | 起始行 | 内容 | 对应目录 |
| --- | --- | --- | --- |
| 三、整体方案 | 61 | 总体流程、三大组件职责、接口契约、架构图 | 全仓 |
| 3.3 接口契约 | 85 | 图层↔算子编译器双向接口、图层→编排器单向蓝图 | `contracts/` |
| 问题 1 图拆分 | 159 | 白名单 + 连通分组、device/part_id 打标 | `graph/partition.py` |
| 问题 2 切分传播 | 275 | DeviceMesh、placement 推导、redistribute 边识别 | `graph/spec_prop.py` |
| 切分策略调优 | — | TP/PP/混合的表达与遍历（**方案外扩展**，见 `docs/strategy.md`） | `graph/strategy.py`、`runtime/compile.py` |
| 问题 3 redistribute 下沉 | 605 | 编译期通信计划表 + 运行时 DMA 序列 | `comm/` |
| 问题 4 GeneSim 接入 | 829 | 从 TTIR / pim mlir 抽 flops、data_bytes | `genesim_bridge/` |
| 问题 5 算子实现 | 1079 | ttir→pim mlir（**在 FlagTree 仓，不在本仓**） | — |
| 问题 6 主机编排 | 1248 | ExecutionPlan 生成 + 解释 + 解码循环 | `runtime/` |
| 问题 7 KV cache | 1660 | build_kv_layout（编译期）+ update/read_tile/mask（运行时） | `memory/kv_layout.py` |
| 问题 8 内存管理 | 1919 | 权重/KV/激活三区 offset 规划、容量校验 | `memory/mem_planner.py` |
| 六、实施阶段 | 2157 | 三阶段总览、并行分组、实施步骤与验证标准、里程碑 | 排期 |
| 附录 A 布局推演 | 2358 | Shard/Replicate/Partial 手推数值 —— **问题 2 的验收基准** | `tests/` |
| 附录 B redistribute 实例 | 2429 | 从编译期计划表到运行时 DMA 的完整下降实例 —— **问题 3 的验收基准** | `tests/` |

## 阶段范围

**第 1 阶段（当前）**：固定 shape 全链路打通，Llama‑2‑7B(HuggingFace获取静态模型)，NumpyBackend 上跑通 prefill + 完整 decode，与单卡 PyTorch 逐元素对齐。

序列变长、代价模型、自动切分、激活区紧凑复用、算子融合、异步 dispatch 属第 2/3 阶段，**当前不实现、不预留抽象**。

## 实施步骤（`docs/spec.md:2204` 展开）

| 步 | 内容 | 硬件依赖 |
| --- | --- | --- |
| 0 ∥ | NumpyBackend 假后端 + DMA 三件套 | 无 |
| 1 | 编译期链：问题 1 → 2 → 8 | 无 |
| 1-D ∥ | GeneSim TTIR 接入（不依赖其他组件） | 无 |
| 2 | 问题 3 redistribute→DMA + 问题 7 编译期 KV 布局 | 无 |
| 3 | 问题 6 编排器总装，prefill 一步对齐 | 无 |
| 4 | 问题 7 KV runtime，完整 decode 循环对齐 | 无 |
| 5 ∥ | 问题 5 上真硬件，hal_numpy→hal_vendor | 需 SDK |
