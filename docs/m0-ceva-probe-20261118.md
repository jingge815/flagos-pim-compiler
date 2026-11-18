# M0 CEVA 探路结论（2026-11-18，两问全答）

M0 的使命：回答 D4b 两条前置确认，定 M4b（CEVA 接线）路线。当日完成。

## 问② mram_offset 是否烧死在编译内核里 → **不烧死**

实验 `/tmp/probe_m0_mram.py`：eltwise 与 linear 各按同请求仅变
`mram_offset`（0 vs 0x1000）编一次，对照组同请求 force 重编两次。

| 证据 | 结果 |
| --- | --- |
| 对照组（同请求重编） | .so 逐字节相同——构建确定性，排除时间戳类干扰 |
| 跨偏移对比 | 差异**全部是 ASCII 元数据**（内嵌缓存键哈希串 + 计数），机器码零差异 |

**推论**：一份编译内核服务全部摆放；M4b 的窗口物化**无需按窗口派生内核变体**。
附带发现（未处理）：缓存键含 mram_offset，同算子 N 个摆放会存 N 份机器码
相同、仅元数据不同的二进制——可去重。

## 问① 厂商 Gather 能否编译为设备端算子 → **不支持**

方法：文档词表核查 + 最小探针实测（模板=厂商 llama_lm_head 4 节点图，
op_type 替换，`scripts/compile_gml.py` 走 gc_w10 槽）。

| 变体 | 结果 |
| --- | --- |
| ctrl（不改） | 编译 OK（9s，Registry 产出） |
| GatherND / GatherElements | **编译失败**：`TestApp analyze() failed`，无 Registry |
| 裸 Gather | 编译"成功"——**假阳性**：L2A 摄入时 `pin_gather` 把节点删除、入口重钉为 `nprm_0_i0after_gather_0`（带 `gather_dimensions` 元数据），设备端不执行 gather，由 host 取好行后按图输入喂入 |

文档佐证：厂商 130 张 Llama2 GML + ResNet 参考零 gather 节点；TVM 路径
`pin_gather` 删除 Gather、CDNN 路径把静态 Gather 改标 Reshape；GatherND/
GatherElements 仅存在于 L1 成本模型存根（无 lowering、不在算子词表）；
VPU 内核按数字 ID 派发。唯一运行期索引原语 = KV_Cache_DMA（单行写、
int16 运行期索引，只写不读）。**动态形状同样不支持**：65 图全定长，
运行期可变仅内容级（KV 行索引、host 行偏移、量化 scale）。

完整证据与产物：`/tmp/m0_gather_findings.md`（file:line 引用）、
`/data/fjg/ceva/runs/m0g_gather_probe/notes.md`（四变体与日志）、
复现 `bash /data/fjg/ceva/scripts/m0g_probe.sh`（约 1.3G 产物）。

## 对 M4b 的路线裁决

按计划风险 #4 原案：

1. **CEVA 侧 MoE 维持 dense/dynamic_dispatch**——exact_dispatch 的设备行级
   gather 落不了 CEVA，exact 交付口径 = numpy 数值真源 + GeneSim 计费
   （M4a），CEVA 边界登记为降级项。
2. **窗口物化走静态派生**：CEVA 每步可变的只有内容级（块表行、KV 索引），
   与我方「运行期信息写缓冲不改命令」的路线天然对齐；问② 保证派生图
   共享内核。混批并发在 CEVA 的表达 = 导出期窗口变体（池深份），内核侧
   无额外代价。
3. NOT-VERIFIED（如 M4b 需要）：裸 Gather 的 host 喂入模式只有结构验证；
   分钟级运行验证可用 `scripts/standalone_layer.sh` 单图跑，无需 4.5h 全仿真。

## 共享环境记录

当日环境由用户确认可用；探针全程 gc_w10 槽 + `m0g_` 前缀隔离，未触碰
`/home/jgfeng/ceva` 原始目录（只读引用）。探测期间另一任务占用 gc_w12
（smoke_fixed），未冲突。
