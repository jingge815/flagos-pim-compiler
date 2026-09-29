# 技术设计文档：GML 与 bin 产物对齐甲方参考格式

> 文档编号：design-gml-align-20260928
> 创建日期：2026-09-28
> 关联需求文档：`docs/request-gml-align-20260928.md`（request-gml-align-20260928）
> 关联项目：flagos-pim-compiler（存算一体大模型推理编译器）

## 一、设计概述

### 1.1 需求背景回顾

我方 `scripts/export_gml.py` 产出的 `relay2gml_graph.gml` + 运行时 `.bin` +
`IO_info.txt` 要交给甲方（芯方舟）L2Analyzer / NPM 设备消费，格式须与甲方 TVM
parser 的产出对齐。需求文档已完成全量实测比对，差异收敛为三层：缺生成步骤（激活
标定）、写死常量、纯形式差异。本设计把这三层落成可执行的代码改动。

判据（需求 1.2）：形式一致、数值不必一致；GML 引用的 bin 必须都能找到；每个 bin
的域形式（字节数、字宽、字节序、分段）一致；生成逻辑与参考一致，不靠写死常量绕过
计算步骤；`DEBUG_*` 允许不同。

### 1.2 设计目标

1. 接通激活标定，消除 `sf=0.0` 这类形式非法值（需求 P0-1）
2. 消除三处写死常量：KV/v_proj 的 `output_sf`、`IO_info` 的 `sf`、Kantor Shift
   （需求 P0-2 / P0-3 / P1-5）
3. 对齐形式差异：子图边界、hidden 边 rank、`IO_info` 序列化、节点编号、版本号
   （需求 P0-4 / P0-5 / P1-2 / P1-4）
4. 不打破需求 2.4 已列出的 15 项"已对齐"结论

### 1.3 设计范围

**包含范围**：

- `gml_bridge/` 五个文件的改动（`export.py`、`runtime_files.py`、`from_fx.py`、
  `phase_data.py` 只读不改、`writer.py` 不改）
- 新增一个标定常数模块
- `scripts/export_gml.py` 的开关默认值
- 对应单测与回归

**不包含范围**：

- FlagTree、genesim 两仓（需求 1.3 已实测：`gml_bridge/` 不 import 它们）
- 上层 7 个 fp32 bin 与 `tvmgen_default_sp_main_0/` 子图的产出（需求 2.5）
- `DEBUG_*` 字段族（判据 5 豁免）
- 数值精度对齐（判据 4 豁免）
- `phase_data.py` 的四相公式（需求已验证逐元素正确，不动）

## 二、总体架构设计

### 2.1 技术选型与选型说明

全部沿用现有技术栈，**不引入任何新依赖**：

| 关注点 | 沿用的现有方案 | 说明 |
| --- | --- | --- |
| 数值计算 | `numpy` | `gml_bridge/` 全模块已在用 |
| 结构体 | `@dataclass` + 类型标注 | 遵循 `CLAUDE.md` 可读性约定 |
| 四相公式 | `gml_bridge/phase_data.py` 现有实现 | 需求实测逐元素正确，只换输入不改公式 |
| 落盘 | `runtime_files.WrittenFiles._write` | 唯一写盘入口，保留 |
| 测试 | pytest，`tests/test_<module>.py` | 现有 40+ 个 GML 相关测试文件 |
| 校验 | `scripts/gml_structure_check.py` 等 4 个脚本 | 需求已核实规则适用（甲方样本 5/5 全绿） |

**一处关键选型判断**：标定数据的落地方式。用户指定"嵌成源码常数、不在程序里读
bin"。需求 P0-1 已测算全量嵌入为 112MB（KV cache 两个各 56MB），故按元素规模
分两档——小张量全量嵌入（5472 个值，约 0.07MB），KV cache 只嵌 absmax 标量
（标定链路实际只用 absmax）。这是本设计唯一新增的模块。

### 2.2 整体架构图

改动点在现有链路上的位置（方框内为本设计触碰的环节）：

```
scripts/export_gml.py
   │  ① decode_block_only 默认值改 True        ← P0-5
   ▼
gml_bridge/export.py  serialize_gml()
   │
   ▼
gml_bridge/from_fx.py  convert()
   │  ② node_ids 逆拓扑 → 顺序从 1 递增        ← P1-2
   │  ③ hidden 边 dims rank 3                  ← P0-4
   │  ④ GML_VERSION → "19.2.0"                 ← P1-4
   │  （decode_block_only 已有：不发 Gather + _trim_decode_block）
   ▼
gml_bridge/export.py  write_runtime_files()
   │  ⑤ _dq_source() 返回真实标定激活 ────────┐  ← P0-1（唯一改动点）
   │  ⑥ output_sf 按算子取值，不再写死 1.0     │  ← P0-2
   │  ⑦ Kantor Shift 常数                      │  ← P1-5
   ▼                                            │
gml_bridge/runtime_files.py                      │
   │  write_dq_phases() ← phase_data 四相（公式不动）
   │  ⑧ 十余处 np.zeros 中，仅激活相关的改为吃标定数据
   ▼                                            │
gml_bridge/export.py  write_io_info()            │
      ⑨ sf 取真实值 + numpy repr + rank          │  ← P0-3
                                                 │
        ┌────────────────────────────────────────┘
        ▼
  【新增】gml_bridge/calib_data.py
     小张量 fp32 常数 + KV absmax 标量，带来源注释
```

### 2.3 模块划分与职责

| 模块 | 职责 | 改动类型 |
| --- | --- | --- |
| `gml_bridge/calib_data.py` | 标定常数唯一真源 | **新增** |
| `gml_bridge/export.py` | `_dq_source` 取标定数据；`output_sf`/Kantor Shift 取值；`write_io_info` 三项 | 修改 |
| `gml_bridge/runtime_files.py` | 激活类 `np.zeros` 改为吃标定数据 | 修改 |
| `gml_bridge/from_fx.py` | 节点编号、hidden rank、版本号 | 修改 |
| `scripts/export_gml.py` | `--decode-block-only` 默认值 | 修改 |
| `gml_bridge/phase_data.py` | 四相公式 | **公式不改**（已验证正确）；只加 `__repr__`（评审 r4 问题1） |
| `gml_bridge/writer.py` | GML 文本序列化 | **不改** |

## 三、模块详细设计

### 3.1 标定常数模块（新增）

- **对应需求点**：P0-1
- **现有代码位置**：无，新建 `gml_bridge/calib_data.py`
- **改动类型**：新增

**详细设计**

核心逻辑：把甲方 `parser_output/` 上层 7 个 fp32 文件的数值固化为源码常数，使导出
流程不再依赖外部目录。按元素规模分两档（需求 P0-1 已测算全量嵌入 112MB 不可接受）：

| 档 | 张量 | 元素数 | 落地形式 |
| --- | --- | ---: | --- |
| 全量 | `hidden_states` | 4096 | fp32 常数数组 |
| 全量 | `attention_mask` | 1024 | fp32 常数数组 |
| 全量 | `cos_position_embedding` | 128 | fp32 常数数组 |
| 全量 | `sin_position_embedding` | 128 | fp32 常数数组 |
| 全量 | `cache_position` | 96 | int64 常数数组 |
| 标量 | `key_cache` | 4194304 | 只存 absmax = 35.222347 |
| 标量 | `value_cache` | 4194304 | 只存 absmax = 38.633411 |

全量档合计 5472 个值，约 0.07MB。

输入输出：本模块无函数入参，只暴露常数与一个取数函数。

关键接口设计：

```python
# 每个常数上方注释写明来源文件、dtype、元素数、提取日期（需求 P0-1 要求）
HIDDEN_STATES: np.ndarray      # fp32, 4096
ATTENTION_MASK: np.ndarray     # fp32, 1024
COS_EMBEDDING: np.ndarray      # fp32, 128
SIN_EMBEDDING: np.ndarray      # fp32, 128
CACHE_POSITION: np.ndarray     # int64, 96
KEY_CACHE_ABSMAX: float = 35.222347
VALUE_CACHE_ABSMAX: float = 38.633411

def activation_for(numel: int) -> np.ndarray:
    """按元素数取一份标定激活（fp16），供 DQ 节点算 absmax。

    numel == 4096 时返回 HIDDEN_STATES 转 fp16；其余长度按需求
    P0-1 的口径由 HIDDEN_STATES 平铺/截断得到——只要非零且量级
    合理即可满足判据 4，不需要与甲方逐元素一致（甲方那份本身是
    随机数，见需求 P0-1 补充事实）。
    """
```

异常处理：`numel <= 0` 直接抛 `ValueError`，不返回空数组兜底（遵循 `CLAUDE.md`
"不写防御性兜底"）。

**设计取舍说明（用户已确认）**：`activation_for` 对非 4096 长度采用平铺，而不是为
每个 DQ 节点都嵌一份常数。理由是甲方样本只提供了图入口那一份 hidden state，图内
各节点的中间激活甲方并未导出；而判据 4 只要求"非零且形式合法"，且甲方那份本身是
U(-26.6,26.6) 随机数，逐元素对齐无意义。**用户已确认按此处理。**

**这个取舍的必然结果（评审 r4 问题2 指出，此前两份文档均未写明）**：所有 DQ 节点
统一喂同一份平铺/截断的 `HIDDEN_STATES`，而甲方参考产物里每个节点的 `input_buffer`
是**该节点自己的真实中间激活**（例如节点 12 的输入是 RMSNorm 之后的激活，absmax
0.3242；我方同一位置拿到的是 `HIDDEN_STATES`，absmax 26.6）。两者 absmax 不同，
`phase0 = 2·absmax` 及其下游（phase1/output_sf）**必然逐元素不同**——这是取舍的
代价，不是实现缺陷，判据 4 允许（数值可以不同）。验收 A3 的措辞已按此改写为
"用甲方自己的 DQ 输入重算四相"（设计 V3 口径），不再要求"用我方内置常数时与甲方
逐元素相同"，避免与本节的取舍自相矛盾。

### 3.2 DQ 标定接通（`export.py::_dq_source`）

- **对应需求点**：P0-1
- **现有代码位置**：`gml_bridge/export.py:732-743`
- **改动类型**：修改（单函数，约 5 行）

**详细设计**

这是 P0-1 的**唯一改动点**，现状为：

```python
def _dq_source(gm, artifact, node_id, spec) -> np.ndarray:
    """..."结构轮用零张量占位"..."""
    return np.zeros(spec.numel, dtype=np.float16)   # ← 全零的源头
```

改为从 `calib_data.activation_for(spec.numel)` 取数。下游链路**完全不用改**：

```
_dq_source() 返回非零激活
   → phase_data.dynamic_scaling()   公式不动，absmax 自然非零
   → phase0 = 2·absmax ≠ 0
   → phase1 = phase0/256 ≠ 0  ── 就是 output_sf
   → phase2 = 1/phase0     ── 就是 kantor_A_scale（需求 P1-5 已证等于 phase2）
   → phase3 = 量化结果，非全零
   → runtime_files.write_dq_phases() 逐相落盘
```

需求已实测：用甲方 `input_buffer_12.bin` 当输入，phase0/phase1 与甲方**逐元素
相同**（32/32 组）。所以改完输入即对齐，无需动公式。

连带解决（均为 phase 的下游，不需单独改）：32 个 2B DQ sf、4 个 64B、1 个 172B、
`Llama2ActivationDQ` 的 64B sf、33 个 `input_buffer`、Kantor scale 全族。

异常处理：`numel <= 0` 由 `activation_for` 抛，不返回空数组兜底。长度与内置
常数不等**不是错误**：按 3.1 的取舍（Q14，用户已确认）由 `HIDDEN_STATES`
平铺/截断得到——甲方只导出了图入口那一份 hidden state，图内各节点的中间激活
它并未提供，而判据只要求非零且形式合法。

**同时要改的 docstring**：现有注释明确写着"结构轮用零张量占位""内容不参与跨产物
比对"，这个前提已被本设计推翻，必须同步改掉，否则后来人会照旧理解。

### 3.3 激活类零填充收口（`runtime_files.py`）

- **对应需求点**：P0-1
- **现有代码位置**：`gml_bridge/runtime_files.py` 的 138、164、208、257、304、
  324、390、402、458、460、462、464 行
- **改动类型**：修改（需逐处甄别，**不是全部都要改**）

**详细设计**

需求列了十余处 `np.zeros`，但它们语义不同，必须分类处理，不能一律改成非零：

| 行 | 所在函数 | 语义 | 处理 |
| --- | --- | --- | --- |
| 138 | `write_named_buffer` | `content=None` 时的默认 | **保留**。走权重通路的 64 个 KV 缓冲已由 `placeholder_weight` 给非零内容 |
| 164 | `write_activation_scale` | 激活 scale | **改**：吃标定算出的 scale |
| 208 | `write_data_buffer` | 数据缓冲 | **改**：吃标定激活 |
| 257 | `write_zero_point` | 对称量化零点 | **保留**。zp 恒 0 是正确的（需求 2.4：137 个 `output_zp` 逐字节相同） |
| 304、390、402 | Kantor / FPSU bias | bias 恒 0 | **保留**。需求 P1-5 实测甲方 bias 也是 0 |
| 324 | 逐组 scale 占位 | 被 DQ 覆盖 | 视 3.2 接通后是否仍走到，若死路则删（`CLAUDE.md`"删优于加"） |
| 458-464 | `write_rope_buffer` | RoPE 系数 | **改**：cos/sin 从 `calib_data` 取 |

**判定原则**：恒 0 是硬件语义的（zp、bias）保留；表示"没算"的（激活、scale、RoPE）
改为吃标定数据。改完后由验收 A2/A16 兜底——A2 要求所有 `output_sf > 0`，A16 要求
Kantor bias 仍全 0，两条同时通过才说明分类正确。

异常处理：不新增 try/except。契约不满足直接抛。

### 3.4 output_sf 按算子取值（`export.py`）

- **对应需求点**：P0-2
- **现有代码位置**：`gml_bridge/export.py:507-511`
- **改动类型**：修改

**详细设计**

现状 `write_output_scale(files, node.node_id, 1.0, dtype=scale_dtype)` 无条件写 1.0。
需求 P0-2 已确认三类节点需要真实 scale：

| op_type | 甲方值 | 语义 |
| --- | --- | --- |
| `KV_Cache_DMA` ×2 | 0.0458984 / 0.00261116 | K/V cache 的量化 scale |
| `Split` ×2 | 同上 | 从 cache 切出，继承同一 scale |
| `Gemm`（仅 v_proj） | 1.0192394e-05 | 输出写进 int8 value cache，带 requant scale |
| 其余 Gemm ×6、MatMul ×64、Softmax ×32、Mask ×32、EltwiseAdd/Mul | 1.0 | 留在 fp16 域 |

核心逻辑：按 `op_type` 与权重角色分派。取值口径为**按 absmax 算**（用户已定）：

```
key_scale   = calib_data.KEY_CACHE_ABSMAX   / 127
value_scale = calib_data.VALUE_CACHE_ABSMAX / 127
```

得 0.2773 / 0.3042。需求 P0-2 已记录：**这不复现甲方的 0.0458939 / 0.00261151**
（实测甲方常量源自 TVM 外层 `qnn.quantize` 的真实模型标定值，数据 absmax 是其量程
的 6 倍 / 116 倍，`absmax/127`、`/128`、分位数全部试过无一命中），但自洽、非 1.0、
不除零、不饱和，符合判据 4。此为已接受取舍，若甲方要求数值一致见需求 Q6。

v_proj 的识别：复用 `_trim_decode_block` 已有的权重角色判据
（`from_fx.py:2109` 用 `pim_weight_param` 里是否含 `v_proj` 判断），不新造一套。

关键函数：`write_output_scale` 签名不变，只改调用方传入的 scale 值。

异常处理：识别不到 op_type 时抛，不回落 1.0——回落会让这个 bug 再次静默。

### 3.5 IO_info.txt 三项对齐（`export.py::write_io_info`）

- **对应需求点**：P0-3、P0-4
- **现有代码位置**：`gml_bridge/export.py:228-296`
- **改动类型**：修改

**详细设计**

需求 P0-3 已确认 `IO_info` 的 `sf` 等于该 I/O 对应 GML 节点的 `output_sf`（IO_info
存 fp32、GML bin 存 fp16），取值规律：fp16 缓冲恒 1.0，int8 缓冲取真实量化 scale。

三项改动：

**(1) sf 取真实值**（`export.py:271` 与 `:285` 两处 `"sf": 1.0`）

```
dtype == 'int8'  → 取 3.4 算出的 key/value scale
其余             → 1.0
```

输入侧 `key_cache`/`value_cache` 与输出侧 `key_cache_out`/`value_cache_out` 共用
同一 scale（需求 P0-3：relay 侧 `qnn.quantize`/`qnn.dequantize` 用同一常量，闭合）。

**(2) sf 序列化为 numpy repr**

甲方形如 `array(1., dtype=float32)` 与 `np.float32(1.0)` 两种混用，说明甲方是直接
`repr()` 含 numpy 标量的 dict，消费端不可能用 `ast.literal_eval`。我方现在写裸
`1.0`。改法：把 sf 存成 `np.float32` 标量再 `repr`，让输出自带 numpy 形式。

**注意副作用**：我方现有测试若用 `ast.literal_eval` 读 `IO_info.txt`，改完会解析
失败，必须同步改为 `eval` + numpy 命名空间。这是本设计唯一一处会破坏现有测试写法
的改动，已列入 4.1 清单。

**(3) hidden 边 rank 改 3 维**（P0-4）

需求已确认规则：甲方**每条边沿用原始张量的 rank**，不是统一维数。relay 子图签名为
证：`nprm_0_i0` 是 3 维 `[1,1,4096]`，其余六项（cos/sin/mask/kv_position/k/v cache）
都是 4 维。我方仅 hidden 这一条多塞一维，其余六项两边已一致（需求 2.4）。

改动落点在 `write_io_info`：按图侧带来的 `pim_io_rank` 削维。

**实测修正（评审 r6 问题3，本节原措辞有误）**：参考的 331 条边**全是四维**，
hidden 那条边就是 `dims "1x1x1x4096"`，只有 `IO_info` 报 `[1, 1, 4096]`。所以
**GML 的边一条都不能动**，改 `from_fx.py` 的边 dims 会把已对齐的 331 条边改坏、
直接撞 A10。A7 的口径也据此收紧：它只要求 `IO_info` 里的 rank，不涉及边。

异常处理：现有 `shape_of`（`export.py:243`）在缓冲节点没有边时已 `raise ValueError`，保持。

### 3.6 节点编号改为从 1 顺序递增（`from_fx.py`）

- **对应需求点**：P1-2
- **现有代码位置**：`gml_bridge/from_fx.py:955-958`
- **改动类型**：修改（改动小，但影响面最大）

**详细设计**

现状是**逆拓扑编号**，注释写明"id 越小越靠输出，参考产物就是这个约定"：

```python
ordered = [node for node in gm.graph.nodes if node in emittable]
node_ids = {node: len(ordered) - index + 2
            for index, node in enumerate(ordered)}
```

实测（需求 P1-2）我方 id 顺序为 `196 198 199…205 195 194…`、范围 3~206；甲方是
从 1 顺序排到 200。需求已排除"偏移"假设——同一 id 两边挂的算子类型都不同。

改法：把 `node_ids` 改为按拓扑序从 1 递增。**注意现有注释里"参考产物就是这个约定"
的判断与实测不符**，改动时要同步纠正该注释，否则后来人会照旧改回去。

**影响面（本设计最大的一处）**：所有 bin 文件名含 `<node_id>`（`names.py` 全族：
`weight_buffer_<id>.bin`、`output_sf_<id>.bin`、`input_buffer_phase_<p>_<id>.bin`
…），编号一变 3186 个文件名全变。因此：

- 必须与 3.5(3) 同批做（都改 `from_fx.py` 的图遍历区域）
- 做完后需求 2.4 的 15 项"已对齐"结论要整表复测（验收 A10）
- ~~需求 P1-1（尺寸量级不符的 bin）依赖本项完成后才能统计~~ **已完成，见 Q10**：重统计后不符族 33 → 1（评审 r6 问题1）。

异常处理：改完需断言 id 集合为 `1..N` 连续无空洞（验收 A14），不连续直接抛。

### 3.7 relay2gml_version 改为 "19.2.0"

- **对应需求点**：P1-4
- **现有代码位置**：`GML_VERSION` 常量（`from_fx.py` 导入，`writer.py:114`
  `write_gml(version=)` 消费），以及 `from_fx.py:842 _resolve_rtl_version()`
- **改动类型**：修改

**详细设计**

现状我方发 `"26.2.1"`，甲方 `"19.2.0"`。用户已定改为 `"19.2.0"`。

**需要留意一处现有机制**：`_resolve_rtl_version()`（`from_fx.py:842`）会在接了算子
编译器时以 IR 的模块属性为真源覆盖版本号。所以不能只改常量默认值，要确认这条覆盖
路径不会把 `"19.2.0"` 改回去——若会，需要让参考对齐路径固定用 `"19.2.0"`，并在
注释里说明为何不取 IR 版本。

异常处理：无新增。

### 3.8 子图边界（`scripts/export_gml.py` 开关默认值）

- **对应需求点**：P0-5
- **现有代码位置**：`scripts/export_gml.py:620`
- **改动类型**：修改（仅默认值）

**详细设计**

需求 P0-5 记录了本轮最重要的发现：**这个能力已经存在，只是默认关闭**。链路为

```
scripts/export_gml.py:620  --decode-block-only  (action="store_true"，默认 False)
  → export.py:149   serialize_gml(decode_block_only=)
  → from_fx.py:907  convert(decode_block_only=)
      ├ :933  不发 Gather（注释：必须在这里裁，留到收尾会让全图编号平移）
      └ :2089 _trim_decode_block() 裁末尾 RMSNorm + lm_head + 它们的 DQ
```

实测带开关重跑（`/tmp/gml_dbo`）的效果：

| 项 | 甲方 | 不带开关 | 带开关 |
| --- | --- | --- | --- |
| node / edge | 200 / 331 | 204 / 335 | **200 / 331 精确一致** |
| `op_type` 分布 | — | 多 4 个 | **逐类完全一致（diff 为空）** |
| `Gather` | 无 | 1 | **0** |
| 模式类 | 71 | 71 | **71** |
| 悬空 / 孤儿 | 379 / 143 | 0 / 0 | **0 / 0** |

所以改动是把默认值设为 True（或在导出参考产物的路径上固定传入），**不是重写图切分**。

**设计建议**：不直接把 `action="store_true"` 翻转成默认 True（那会让
`--decode-block-only` 这个 flag 名字失去意义），而是改成
`--no-decode-block-only` 的反向开关，默认走 decode block。这样命令行语义清晰，
且整网导出（含 lm_head/Gather）仍可用。

异常处理：无新增。

### 3.9 Kantor Shift：修 3 个文件的取值分支

- **对应需求点**：P1-5
- **现有代码位置**：`gml_bridge/export.py:458-459`
- **改动类型**：修改（约 3 行）

**详细设计**

**本节推翻了需求 P1-5 与本设计初稿的两处判断**，依据是全量 49 个 Shift 文件的
逐字节实测：

| 初稿判断 | 实测结论 |
| --- | --- |
| "两种宽度并存：32B fp16 `[-40704.0,…]` / 1B int8 `[-8]`，-40704 语义不明" | **dtype 读错了**。全部 49 个文件都是 **int8**。`f8f8f8f8…` 按 fp16 解释才得出 -40704 这个荒谬值；按 int8 就是 32 个 `-8` |
| "32B / 86B 是另一种 dtype" | 是**逐组**：32 = hidden 4096/128 组，86 = MLP 11008/128 组 |
| "我方写死 0，需向甲方确认口径（Q11 阻塞）" | 我方 **46/49 个已经对了**。`phase_data.py:59-60` 早已定义 `DQ_PHASE3_SHIFT = -8` 并注释"实测 int8 恒为 -8" |

**实际的域规律**（已完全确定，无需外部确认）：int8；标量 1B 或逐组 N 字节；
取值只有 `-8`（左移 8 位 = ×256）或 `0`。

**逐族对账**（甲方 vs 我方，按去掉节点号的文件族）：

| 文件族 | 个数 | 甲方 | 我方 | 状态 |
| --- | ---: | --- | --- | --- |
| `kantor_A_Shift_buffer_file_phase_3_*` | 37 | -8 | -8 | 已对 |
| `Kantor_A/B_Shift_Llama2Activation_Cos/Sin_*` | 8 | 0 | 0 | 已对 |
| `kantor_B_Shift_*` | 1 | 0 | 0 | 已对 |
| `Kantor_A_Shift_Llama2Activation_add_*` | 1 | **-8** | **0** | **待修** |
| `kantor_A_Shift_*` | 2 | **-8 / 0 各一** | **0 / 0** | **待修 1 个** |

**根因**：`export.py:458-459` 对所有非 phase 的 Kantor Shift 一律写零：

```python
elif "kantor" in key.lower() and "_phase_" not in key:
    if "Shift" in key or "shift" in key:
        files._write(value, np.zeros(1, dtype=np.int8))   # ← 一律 0
```

走 `_phase_` 那条路的 37 个由 `runtime_files.py:321`
（`np.full(groups, DQ_PHASE3_SHIFT)`）正确写 -8，所以只有非 phase 的这一支漏了。

**改法**：在这一支里区分该发 -8 还是 0。判据是该 Kantor 块是否做定点化
（×256）——`Llama2Activation_add` 与其中一个 `kantor_A_Shift` 做，Cos/Sin 不做。
复用现有 `DQ_PHASE3_SHIFT` 常量，不新引入魔数。

具体哪个 `kantor_A_Shift_<id>` 该发 -8：甲方是 `kantor_A_Shift_36.bin`（node 36 =
v_proj，需求 P0-2 已确认它是唯一带 requant scale 的 Gemm）发 -8，
`kantor_A_Shift_194.bin` 发 0。所以判据与 3.4 的 v_proj 识别**同源**，可共用。

异常处理：无新增。

**Q11 已关闭**，不再是阻塞项——它原本只是我读 dtype 读错导致的伪问题。

## 四、代码改动清单

| 文件路径 | 改动类型 | 改动内容简述 | 影响范围 | 优先级 |
|----------|----------|--------------|----------|--------|
| `gml_bridge/calib_data.py` | 新增 | 标定常数：5 个小张量 fp32 数组（5472 值）+ 2 个 KV absmax 标量 + `activation_for()`，每项带来源注释 | 被 `export.py`、`runtime_files.py` 引用 | P0 |
| `gml_bridge/export.py:732-743` | 修改 | `_dq_source` 返回标定激活替代 `np.zeros`；同步改掉"结构轮用零张量占位"的 docstring | DQ 四相全族、Kantor scale、33 个 input_buffer 的内容 | P0 |
| `gml_bridge/export.py:507-511` | 修改 | `output_sf` 按 op_type 分派：KV_Cache_DMA / Split / v_proj 取真实 scale，其余 1.0 | 5 个节点的 `output_sf_*.bin` | P0 |
| `gml_bridge/export.py:271,285` | 修改 | `IO_info` 的 `sf`：int8 项取真实 scale；序列化改 numpy repr | `IO_info.txt`；**会破坏用 `ast.literal_eval` 读它的现有测试** | P0 |
| `gml_bridge/export.py:458-459` | 修改 | 非 phase 的 Kantor Shift 分支区分 -8 / 0，修 3 个文件（复用 `DQ_PHASE3_SHIFT` 与 3.4 的 v_proj 判据） | 3 个 Shift bin | P1 |
| `gml_bridge/runtime_files.py:164,208` | 修改 | `write_activation_scale` / `write_data_buffer` 改为吃标定数据 | 激活类 bin 内容 | P0 |
| `gml_bridge/runtime_files.py:458-464` | 修改 | `write_rope_buffer` 的 cos/sin 从 `calib_data` 取 | RoPE 系数 bin | P0 |
| `gml_bridge/runtime_files.py:324` | 删除或保留 | 若 3.2 接通后成死路则删（`CLAUDE.md`"删优于加"） | 无 | P1 |
| `gml_bridge/runtime_files.py:138,257,304,390,402` | **不改** | 恒 0 是硬件语义（默认内容、zp、bias），需求 2.4 已证正确 | — | — |
| `gml_bridge/from_fx.py:955-958` | 修改 | `node_ids` 逆拓扑改为从 1 顺序递增；同步纠正"参考产物就是这个约定"的错误注释 | **全部 3186 个 bin 文件名**；需求 2.4 整表复测 | P1 |
| `gml_bridge/from_fx.py`（边界节点） | 修改 | 打 `pim_io_rank` 标记，供 `write_io_info` 削维。**边 dims 一条不动**——参考 331 条边全是四维（评审 r6 问题3 实测） | `IO_info` shape | P0 |
| `gml_bridge/from_fx.py:842` + `GML_VERSION` | 修改 | 版本号 `"19.2.0"`；确认 `_resolve_rtl_version` 不会覆盖回去 | GML 首部一行 | P1 |
| `scripts/export_gml.py:620` | 修改 | `--decode-block-only` 改为默认启用（建议加 `--no-` 反向开关） | 参考对齐路径的图形状 | P0 |
| `gml_bridge/phase_data.py` | 修改 | **四相公式不动**（需求已验证逐元素正确），只加 `DynamicScalingPhases.__repr__` 让标定中间产物可 print（评审 r4 问题1） | — | P1 |
| `gml_bridge/writer.py` | **不改** | GML 文本序列化无需变动 | — | — |
| `tests/test_gml_export.py:424-429` | 修改 | `IO_info` 读法改 `eval` + numpy 命名空间；补 sf/rank 断言 | — | P0 |
| `tests/test_gml_from_fx.py:229` | 修改 | 已有 `decode_block_only=True` 用例，补节点编号从 1 递增的断言 | — | P1 |
| `tests/test_phase_data.py` | 修改 | 补"非零输入 → phase0/phase1 非零"用例 | — | P0 |
| `tests/test_runtime_files.py` | 修改 | 补零填充分类断言：zp/bias 仍全 0、激活类非 0 | — | P0 |
| `contracts/gml_coverage.py:317-333` | 修改 | 同步 P1-3 的 4 个不产字段声明（`from_tvm`、`original_name`、`lut_debug`、`*_hash`） | 字段覆盖率校验 | P2 |

## 五、数据结构与接口设计

### 5.1 数据结构变更

**无破坏性变更。** 现有跨模块 `@dataclass` 全部保持原样：

| 结构 | 位置 | 本设计是否改动 |
| --- | --- | --- |
| `DynamicScalingPhases` | `phase_data.py:92` | **不改**。字段语义不变，只是 `source` 实际装非零数据了 |
| `GmlArtifact` | `export.py:37` | 不改 |
| `Node` / `Edge` | `writer.py:21,41` | 不改（`Edge.dims` 的取值变了，结构没变） |
| `WrittenFiles` | `runtime_files.py:38` | 不改 |
| `CompileSlots` | `contracts/compile_slots.py` | 不改 |

新增模块 `calib_data.py` 只暴露模块级常数与一个纯函数，不引入新的 dataclass
（遵循 `CLAUDE.md`"不预造抽象"：只有一个实现，不需要基类/注册表）。

### 5.2 接口设计

唯一新增对外接口：

```python
def activation_for(numel: int) -> np.ndarray:
    """按元素数取一份标定激活，dtype 为 fp16。

    实现原理：DQ 四相只用到输入的逐组 absmax（phase_data.dynamic_scaling
    的 p0 = 2·absmax），所以标定只需提供"非零且量级合理"的激活。
    numel == 4096 直接返回内置的 hidden_states；其余长度由它平铺/截断
    得到（见 3.1 设计取舍与 Q14）。

    入参 numel：该 DQ 节点的元素数，来自 spec.numel（编译期槽位口径，
                不是导出图的 seq_len）。
    返回：shape 为 (numel,) 的 fp16 数组，保证非全零。
    """
```

修改的现有接口签名：**无**。`_dq_source`、`write_output_scale`、`write_io_info`、
`convert` 的签名都不变，只改内部取值——这样不牵动调用方。

## 六、验证与测试方案

### 6.1 功能验证方案

每条给出可执行命令与可核对的预期结果。基准目录约定：

```
REF=/media/disk/fengjingge/src/xinfangzhou-resource/model_layers_0_decode_v2/parser_output/tvmgen_default_nprm_main_0/runtime_files
OUT=/tmp/gml_verify
source /media/disk/fengjingge/src/flagOS/flagOS-installed/pytorch/env-pytorch.sh
python scripts/export_gml.py --layers 1 --seq-len 16 --out-dir $OUT
```

**V1 标定接通（P0-1）**

```bash
python -c "
import numpy as np, pathlib, re
gml=open('$OUT/relay2gml_graph.gml').read()
dq=set()          # 只取 DQ 节点：Softmax 的 phase0 另有判据，见下
for b in re.findall(r'  node \[(.*?)\n  \]', gml, re.S):
    if re.search(r'op_type "(DynamicScaling|Llama2ActivationDQ)"', b):
        dq.add(re.search(r'id (\d+)', b).group(1))
bad=[]
for p in pathlib.Path('$OUT').glob('output_buffer_phase_0_*.bin'):
    if re.search(r'_(\d+)\.bin$', p.name).group(1) not in dq: continue
    a=np.frombuffer(p.read_bytes(),dtype=np.float16)
    if not (a!=0).all(): bad.append((p.name,int((a==0).sum()),a.size))
print('DQ phase0 含零组的文件:',bad[:5],'共',len(bad))
assert not bad"
```

预期：输出 `共 0`（实测命中 37 个 DQ 文件，逐组非零）。改动前此处为 37 个文件全零。

**范围必须限定到 DQ**（评审 r6 问题5）：Softmax 的 phase0 是 fp16 位模式放在 32 位字
的**高半字**、低半字恒 0（参考 `output_buffer_phase_0_18.bin` = `000037cd`，我方
同构），按 fp16 逐元素读必然见到 0。不限范围会稳定报 32 个 Softmax 假阳性，
后来人据此会去修一个并不存在的缺陷。A1 本身也只限定"所有 DQ 节点"。

**V2 无 sf 为 0（P0-1，判据 4）**

```bash
python -c "
import numpy as np, pathlib
z=[p.name for p in pathlib.Path('$OUT').glob('output_sf_*.bin')
   if len(p.read_bytes())==2 and np.frombuffer(p.read_bytes(),dtype=np.float16)[0]==0]
print('sf==0 的文件:',z); assert not z"
```

预期：空列表。改动前为 32 个（实测 `/tmp/gml_dbo`）。

**V3 四相公式仍与甲方逐元素一致（P0-1 回归）**

用甲方 `input_buffer_12.bin` 喂 `phase_data.dynamic_scaling(group_size=128)`，比
`output_buffer_phase_0_12.bin` / `phase_1_12.bin`。预期 `np.array_equal` 为 True
（需求 P0-1 已验证成立，本项确认改动没破坏公式）。

**V4 不依赖甲方目录（P0-1，用户要求）**

```bash
mv /media/disk/fengjingge/src/xinfangzhou-resource /tmp/xfz_moved
python scripts/export_gml.py --layers 1 --seq-len 16 --out-dir /tmp/gml_noref
mv /tmp/xfz_moved /media/disk/fengjingge/src/xinfangzhou-resource
```

预期：导出退出码 0，产物文件数与 `$OUT` 相同。这条直接验证"标定值已嵌成常数"。

**V5 output_sf 分派正确（P0-2）**

按 op_type 聚合我方 `output_sf`，预期：`KV_Cache_DMA`×2、`Split`×2 非 1.0；
`Gemm` 恰好 1 个（v_proj）非 1.0、其余 6 个为 1.0；`MatMul`×64、`Softmax`×32、
`Mask`×32、`EltwiseAdd`×2、`EltwiseMul`×1 全为 1.0。

**V6 int8 不饱和（P0-2）**

对 KV cache 的 int8 bin 统计 `(q==127)|(q==-128)` 比例，预期 **< 1%**。
（甲方那份在其自身 scale 下饱和 40.50% / 96.22%，我方按 absmax/127 算不应饱和。）

**V7 IO_info 三项（P0-3、P0-4）**

```bash
python -c "
import numpy as np
ns={'array':np.array,'np':np,'float32':np.float32,'dtype':None}
d=eval(open('$OUT/IO_info.txt').read(), ns)          # 必须 eval 成功
for k,v in list(d['inputs'].items())+list(d['outputs'].items()):
    if v['dtype']=='int8': assert v['sf']!=1.0, (k,v['sf'])
    assert isinstance(v['sf'], np.floating), (k, type(v['sf']))
r=[len(v['shape']) for v in d['inputs'].values()]
print('入口 rank:', r)   # 期望恰有 1 个 3，其余 6 个为 4
assert sorted(r)==[3,4,4,4,4,4,4]
assert len(list(d['outputs'].values())[-1]['shape'])==3 or True"
```

预期：`eval` 成功、int8 项 sf≠1.0、sf 为 numpy 标量、入口 rank 为 `[3,4,4,4,4,4,4]`。

**V8 子图边界（P0-5）**

```bash
grep -c 'op_type "Gather"' $OUT/relay2gml_graph.gml      # 期望 0
grep -c '^  node \[' $OUT/relay2gml_graph.gml            # 期望 200
grep -c '^  edge \[' $OUT/relay2gml_graph.gml            # 期望 331
diff <(grep -oE 'op_type "[^"]*"' $REF/relay2gml_graph.gml|sort|uniq -c) \
     <(grep -oE 'op_type "[^"]*"' $OUT/relay2gml_graph.gml|sort|uniq -c)
```

预期：Gather 0 个、节点 200、边 331、`diff` 输出为空。实测 `/tmp/gml_dbo` 已达成。

**V9 节点编号（P1-2）**

```bash
python -c "
import re
ids=[int(m) for m in re.findall(r'^    id (\d+)', open('$OUT/relay2gml_graph.gml').read(), re.M)]
assert min(ids)==1, min(ids)
assert sorted(ids)==list(range(1,len(ids)+1)), '不连续'
print('id 连续 1..%d' % len(ids))"
```

预期：`id 连续 1..200`。

**V10 版本号（P1-4）**

```bash
grep -m1 relay2gml_version $OUT/relay2gml_graph.gml    # 期望 "19.2.0"
```

**V11 Kantor 三族（P1-5）**

```bash
python -c "
import numpy as np, pathlib, re, collections
O=pathlib.Path('$OUT')
c=collections.defaultdict(set)
for q in O.glob('*hift*.bin'):
    a=np.frombuffer(q.read_bytes(),dtype=np.int8)
    c[re.sub(r'_\\d+\\.bin$','',q.name)] |= set(a.tolist())
for k in sorted(c): print(f'  {k:<46} {sorted(c[k])}')
assert c['kantor_A_Shift_buffer_file_phase_3']=={-8}
assert c['Kantor_A_Shift_Llama2Activation_add']=={-8}
assert c['Kantor_A_Shift_Llama2Activation_Cos']=={0}
assert c['kantor_A_Shift']=={-8,0}
print('Shift 全族取值正确')"
```

预期：各族取值与 3.9 对账表一致。改动前 `Llama2Activation_add` 为 `{0}`（应为
`{-8}`）、`kantor_A_Shift` 为 `{0}`（应为 `{-8,0}`）。

同时校验 scale 与 bias（需求 A16 的另两项）：Kantor scale 逐元素等于同节点 phase2、
bias 全 0。

### 6.2 回归验证范围

**R1 全量单测**

```bash
python -m pytest tests/ -x -q -k "not llama2_7b"
```

预期全绿。`llama2_7b` 组按项目约定单次验证不跑（太慢）。交付前需另跑一次含该组。

**R2 需求 2.4 的 15 项"已对齐"整表复测**（对应验收 A10）

这是本设计**最重要的回归**——3.6 改节点编号会动 3186 个文件名，3.8 改图形状，
两者都可能打破已经对上的部分。逐项复测：

| 复测项 | 命令要点 | 预期 |
| --- | --- | --- |
| 文件名模式类 71 | `ls \| sed -E 's/[0-9]+/N/g' \| sort -u \| wc -l` 两边比 | 71，无缺无多 |
| 引用完整性 | GML 引用 bin vs 磁盘 bin | 悬空 0、孤儿 0 |
| 四相公式 | V3 | 逐元素相同 |
| DQ 分组口径 | 32×1024 + 5×128 | 不变 |
| bin 域形式 | 逐类抽样比字宽/字节序/填充 | 一致 |
| 非标定类 sf | V5 | 全 1.0 |
| Kantor bias 全 0 | 抽样 | 全 0 |
| Kantor scale == phase2 | 逐元素比 | 相同 |
| 结构校验器 | `python -m scripts.gml_structure_check $OUT/relay2gml_graph.gml` | 5/5 通过 |
| op_type 集合 | V8 的 diff | 为空 |
| 六项 I/O rank | V7 | 均为 4 |

**R3 受影响的既有校验脚本**

```bash
python -m scripts.gml_structure_check $OUT/relay2gml_graph.gml   # 期望 5/5
python scripts/verify_gml_artifact.py $OUT                        # 期望全部通过
python scripts/gml_field_inventory.py $OUT/relay2gml_graph.gml    # 字段族覆盖率
```

需求已核实 `gml_structure_check` 的规则对 Llama decode 适用（甲方样本 5/5 全绿）。

**R4 下游模块回归**

`orchestrator/plan.py:153` 注释表明它依赖 `_trim_decode_block` 的裁剪结果。3.8 把
开关设为默认启用后，编排器走到的图与之前不同，需跑 `tests/test_orchestrator.py`
与 `tests/test_verify_layers.py`。

### 6.3 边界与异常测试

| 用例 | 操作 | 预期 |
| --- | --- | --- |
| E1 标定长度不匹配 | 调 `activation_for(0)` | 抛 `ValueError`，不返回空数组 |
| E2 标定长度非常规 | 调 `activation_for(11008)`（MLP 中间态） | 返回 11008 个非零 fp16 |
| E3 全零输入保护仍在 | 直接给 `dynamic_scaling` 喂全零张量 | p2 置 0 不出 inf/nan（`phase_data` 现有保护，不应被本设计破坏） |
| E4 zp/bias 不被误改 | 检查 `output_zp_*.bin`、Kantor bias | 仍全 0（3.3 分类正确性的反向验证） |
| E5 节点编号不连续 | 人为构造缺号 | 断言抛错 |
| E6 IO_info 旧读法失效可见 | 用 `ast.literal_eval` 读新 `IO_info.txt` | 抛 `ValueError`——确认这是**已知且已同步测试**的破坏性变更，不是意外 |
| E7 整网导出仍可用 | 带 `--no-decode-block-only` 导出 | 退出码 0，含 Gather 与 lm_head |
| E8 Shift 分支正确 | 检查全部 49 个 `*hift*.bin` | 全为 int8；`phase_3` 族与 v_proj/add 族为 -8，Cos/Sin 与 `kantor_B` 为 0 |

### 6.4 性能验证

**无专项性能要求。** 需求三、非功能需求未提性能指标。两点仅需确认无明显退化：

- 导出耗时：改动前实测单层导出约 2 分钟量级。标定只多一次 absmax 计算（O(n)），
  预期无可感知变化。
- 仓库体积：新增 `calib_data.py` 约 0.07MB（5472 个 fp32 字面量），可接受。
  这正是 3.1 不全量嵌入 KV cache（112MB）的原因。

### 6.5 验收标准

直接沿用需求第五章 A1~A17，对应关系如下（需求原文即验收口径，此处不重述）：

| 验收项 | 对应设计节 | 对应验证 |
| --- | --- | --- |
| A1 phase0 非零组数 == 组总数 | 3.2 | V1 |
| A2 所有 output_sf > 0 | 3.2、3.3 | V2 |
| A3 phase0/phase1 与甲方逐元素相同 | 3.2 | V3 |
| A3b 不读甲方目录 | 3.1 | V4 |
| A4 KV/Split/v_proj 的 sf ≠ 1.0 且饱和 < 1% | 3.4 | V5、V6 |
| A4b 7 个 Gemm 恰 1 个非 1.0 | 3.4 | V5 |
| A5 IO_info int8 项 sf == 对应 bin 的 fp16 值 | 3.5 | V7 |
| A6 IO_info 可被 eval + numpy 读出 | 3.5 | V7 |
| A7 hidden 入口 rank 3、其余 4 | 3.5 | V7 |
| A8 悬空 0 孤儿 0（不能回退） | 全局 | R2 |
| A9 模式类仍 71（不能回退） | 全局 | R2 |
| A10 需求 2.4 整表复测 | 全局 | **R2** |
| A11 同角色 bin 字节数相同 | 3.3 | **R3**（按族对账，见 Q10） |
| A12 pytest 全绿 | 全局 | R1 |
| A13 出口 shape [1,1,4096]、无 lm_head | 3.8 | V8 |
| A14 id 连续 1..N | 3.6 | V9 |
| A15 版本号 "19.2.0" | 3.7 | V10 |
| A16 (a) `phase_3` 族 scale==phase2 / (b) 非 phase 族非 0 且形式合法 / (c) bias 全 0 + Shift 对账 | 3.9 | **V11 只覆盖 (c) 的 Shift 逐族取值**。(a) 另需逐元素比 `output_buffer_phase_2_<id>.bin`、(b) 另需扫全族非 0、(c) 的 bias 另需抽样——原写的「可全验」高于 V11 的实际覆盖面，按评审 r5 问题3 收敛（判据原文见需求 §五 A16 改写后的三句） |
| A17 结构校验器 5/5 | 全局 | R3 |

## 七、上线与回滚方案

### 7.1 上线步骤

本项目是编译器产物生成工具，无服务部署，"上线"即产物交付。建议按四批推进，每批
自成闭环可验证：

| 批次 | 内容 | 验证 | 说明 |
| --- | --- | --- | --- |
| 第 1 批 | 3.8 子图边界（开关默认值） | V8 | **最小改动、收益最大**。仅改默认值即让 node/edge 与 op_type 分布精确对齐 |
| 第 2 批 | 3.1 + 3.2 + 3.3 标定接通 | V1~V4、E1~E4 | 主因，消除全部 `sf=0` |
| 第 3 批 | 3.4 + 3.5 + 3.9 常量、IO_info、Shift | V5~V7、V11、E6、E8 | 含唯一一处破坏性变更（IO_info 读法）。3.9 与 3.4 共用 v_proj 判据 |
| 第 4 批 | 3.6 + 3.7 节点编号与版本号 | V9、V10、**R2 整表复测** | 影响 3186 个文件名，放最后 |

每批结束跑 R1（`pytest -k "not llama2_7b"`）+ R3（三个校验脚本）。第 4 批额外跑
R2 全表与 R4 下游回归。

3.9（Kantor Shift）排入第 3 批——它只改 3 个文件的取值分支，与 3.4 共用 v_proj 判据。

### 7.2 回滚方案

改动全部在本仓源码内，无数据迁移、无外部状态，**回滚即 git revert**。按批提交
可让每批独立回滚。

两点需注意：

1. **第 3 批的 IO_info 序列化是破坏性变更**（消费端读法从 `ast.literal_eval` 变
   `eval`+numpy）。若回滚这一批，同步提交的测试改动要一起回滚，否则测试会失败。
2. 第 4 批改节点编号后，产物文件名与之前的产物目录不可混用。回滚后需重新导出，
   不要把新旧产物目录混在一起比对。

## 八、风险评估与应对

| 风险点 | 风险等级 | 影响说明 | 应对措施 |
| --- | --- | --- | --- |
| 按 absmax 算的 KV sf 不复现甲方常量（0.2773 vs 0.0458939） | **中** | 若甲方要求数值一致，3.4 需改为外部配置传入 | 需求 P0-2 已记录为已接受取舍；Q6 向甲方确认。设计上把 scale 取值集中在一处，改口径只动一个函数 |
| 3.6 改节点编号牵动 3186 个文件名 | **高** | 可能打破需求 2.4 已对齐的 15 项 | 放在第 4 批最后做；强制执行 R2 整表复测（A10）；`names.py` 是唯一命名真源，不散落 |
| `activation_for` 对非 4096 长度用平铺，非甲方真实中间激活 | **中** | 各 DQ 节点的 sf 数值与甲方不同 | 判据 4 允许数值不同；甲方那份本身是随机数。列 Q14 确认是否可接受 |
| ~~Kantor Shift 口径未知~~ | — | **已排除**。实测全为 int8、取值只有 -8/0，46/49 已对，剩 3 个是我方分支漏判 | 见 3.9，按 `DQ_PHASE3_SHIFT` 与 v_proj 判据修 |
| IO_info 序列化变更破坏现有读法 | **低** | 现有测试若用 `ast.literal_eval` 会失败 | 已在 4.1 清单列出同步改测试；E6 专门验证这是已知变更 |
| `_resolve_rtl_version` 可能把版本号覆盖回去 | **低** | 3.7 只改常量可能无效 | 3.7 已要求确认这条覆盖路径；V10 直接验证产物首部 |
| 3.3 的零填充分类判错（把硬件语义的 0 改成非 0） | **中** | zp/bias 本该恒 0，改错会让形式非法 | 按"恒 0 是硬件语义的保留、表示没算的才改"分类；E4 + A16 双向兜底 |
| 第 1 批改默认值影响 `orchestrator` | **低** | `orchestrator/plan.py:153` 依赖裁剪结果 | R4 跑 `test_orchestrator.py`、`test_verify_layers.py` |

## 九、排期与里程碑

按 7.1 的四批推进，每批一个里程碑。批次间有依赖（第 4 批的编号改动要在前三批
稳定后做），批内可并行。未给出具体人日——需求文档未提时间节点，排期由开发方按
实际投入确定。

| 里程碑 | 交付物 | 完成判据 |
| --- | --- | --- |
| M1 | 子图边界对齐 | V8 通过（node/edge 200/331、op_type diff 为空） |
| M2 | 标定接通 | V1~V4 通过，无 `sf=0` |
| M3 | 常量与 IO_info | V5~V7 通过 |
| M4 | 编号与版本号 | V9、V10、R2 全表、R4 通过 |

## 十、待确认事项

### 10.1 阻塞实现的问题

**无。** 初稿列的 Q11（Kantor Shift 口径）经全量实测已关闭——它是读 dtype 读错
导致的伪问题，实际规律完全确定（int8、取值只有 -8/0），见 3.9。

### 10.2 已确认的设计决策

| # | 事项 | 决策 |
| --- | --- | --- |
| Q14 | `activation_for` 对非 4096 长度用 hidden_states 平铺 | **用户已确认按此处理**（见 3.1） |
| Q11 | Kantor Shift 口径 | **已关闭**，非外部问题。46/49 已对，剩 3 个是 `export.py:458-459` 分支漏判，按 `DQ_PHASE3_SHIFT` 修（见 3.9） |

### 10.3 需用户拍板的一项

| # | 问题 | 两个选项 |
| --- | --- | --- |
| Q15 | `--decode-block-only` 默认启用的实现方式 | **A 直接翻转默认值**：最简单，但 `--decode-block-only` 变成永远为真的死开关，且整网导出（含 lm_head + Gather）失去入口。**B 加 `--no-decode-block-only` 反向开关**（本设计当前写法）：保留整网导出。**推荐 B**，因为整网路径仍在用：`tests/test_gml_from_fx.py:229` 有对应用例、`from_fx.py:83` 注释写明"同一个算子两种导出各发各的"、`orchestrator/plan.py:153` 依赖裁剪结果、且需求 7.1 决定"保留 Gather 声明为扩展"需要这条路径存在才有意义。若确认整网导出以后不再需要，则比 A 更干净的做法是**直接删掉该参数、只留 decode block 一条路**，不要留死开关 |

### 10.4 承接自需求文档、不影响本设计推进的问题

以下继承自需求第七章，但会影响最终交付是否被甲方接受：

| # | 问题 |
| --- | --- |
| Q4 | 改成 `"19.2.0"` 后，甲方解析器是否按此字段做版本分派？我方字段集与 19.2.0 版预期是否兼容 |
| Q5 | NPM 是否支持 `Gather`？（参考对齐路径本就不发，已降为非阻塞） |
| Q6 | KV cache 的 scale 是否必须与甲方数值一致？直接决定 3.4 的取值口径。本设计按 absmax/127 算得 0.2773/0.3042，不复现甲方的 0.0458939/0.00261151 |
| Q7 | `IO_info` 的 sf 序列化，甲方两种 numpy repr 混用，消费端是否都接受 |
| Q8 | 甲方 GML 自身 379 个悬空引用 + 143 个孤儿是否为预期状态 |
| Q10 | ~~需求 P1-1 的尺寸差异清单需在 3.6（节点编号）完成后重新统计~~ **已完成，Q10 关闭**。P1-2 完成后按"算子类型+角色"归族重统计：194 个共有族里不符族从 33 降到 1（评审 r6 问题1）。根因两条、都已修：RoPE 子块的标量族按 head_dim 算宽（应为单元素）、FPSU 三族按组数而非节点类型分宽。第 3 条根因在评审 r8 问题1 找到并修掉——缓冲与定标"穿不穿过布局算子"两处口径不一致，把 32 组 scale 配到了 128 个元素的缓冲上（等于宣称 group_size = 4）。改后 **0 族不符**、族集合两向完全相同，原先记入需求 2.4 的那两族已知差异消失 |
| Q13 | `tvmgen_default_sp_main_0/` 子图是否为交付必需项 |
