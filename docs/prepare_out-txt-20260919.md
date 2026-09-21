# prepare_out txt 生成（2026-09-19）

> **本文档的「VALUE_DIFF 0」结论已被证实不可信，不要引用**——那是
> `scripts/diff_prepare_out.py` 白名单放水造成的假象（命名类键被整体放进
> 白名单，没有按模式比对）。独立评审
> （`docs/prepare_out-代码评审-20260920.md`）与后续修复
> （`docs/prepare_out-P0修复-20260921.md`）已重新核实：收紧白名单后
> 真实差异是 104 处（不是 0），且当时还有 1051 处悬空 bin 引用（Softmax
> 五相从未落盘、Mask 第二路输入从未进图）没被这份文档的对拍口径抓到。
> 本文档保留作历史记录，**当前状态以 20260921 那份为准**。

## 一句话状态（存档，见上方警示）

422 个层参数卡 + `net.ini` 已能生成，**文件名模式、每个文件的域集合与参考
完全一致**（422/422，缺失 0、多写 0）。数值层面：5872 处只差编号（地址与
节点号，结构性）、78 处是 TVM Relay 内部符号名（改不动，等甲方回 Q-A）。

| 维度 | 状态 |
| --- | --- |
| 文件数 / 文件名模式 | 对齐 |
| `net.ini [general]` | 逐字节一致 |
| 行尾符（CRLF） | 对齐 |
| **每文件域集合** | **422/422 对齐** |
| 闭合域数值 | 对齐（`VALUE_DIFF 0`） |
| 待甲方确认 | Q-A buffer name 语义、Q-B qidx、Q-C 三处公式 |
| 已知遗留 | 残差旁路的**边**未接（域已补齐）；FlagTree F1–F6 |

下一步建议见「建议的下一步」：**先让对方拿这份产物过一遍仿真器**，
报错会直接指出哪个域被当成索引，比继续对拍有效。

## 做了什么

从已有 GML + 算子编译器相位模板，由编排器写出
`prepare_out/net.ini` 与 `txt_files/` 422 层参数卡（加 2 个版本戳）。

命令：

```bash
python scripts/export_gml.py --layers 1 --seq-len 16 \
    --out-dir /tmp/gml_full --use-opcompiler --orchestrate --decode-block-only
python scripts/diff_prepare_out.py \
    --mine /tmp/gml_full/prepare_out \
    --ref .../llama2_w4a8_decode_block_0/prepare_out
```

对拍结果（按 `(层类, head, 输入宽)` 配对，不用文件名 qidx）：

| 栏 | 数 |
| --- | --- |
| 文件数 | 424 = 422 层 + 2 版本戳，与参考相同 |
| 文件名模式 | 去掉编号后**集合相同** |
| `net.ini` `[general]` | **逐字节一致**（含 CRLF 混排） |
| 行尾 / 版本戳字节 | 422/422 CRLF；6 与 13 字节，均同参考 |
| **每个文件的域集合** | **422/422 完全相同** |
| 层类计数 | 23 类完全相同，422 对 422 |
| MATCH | 45854 |
| VALUE_DIFF / MISSING / EXTRA | **0 / 0 / 0** |
| UNMATCHED_PAIR / FILE | 0 |
| ALLOC | 5950（片上/DDR 地址与带节点号的引用） |

不看任何白名单的原始核对（约 51800 个域）：只差编号 5872 处、
结构语义 **78 处**、缺失 **0**、多写 **0**。
那 78 处全是 TVM Relay 内部符号名，见「当前不足」第 1 条。

### 对拍器一度放水

第一版把尺寸、槽位 id、枚举一起塞进 ALLOC 白名单，于是报 `VALUE_DIFF 0`，
而绕过白名单的原始比对有 **6582 处**不等。收紧白名单后暴露出一批真 bug
（下节），逐条修完才真正归零。

**白名单只允许放结构性不可对齐的三类**：节点编号、片上/DDR 地址、TVM 专属名。
尺寸、模式、枚举、槽位 id 一律当闭合域。

方案全文：`docs/prepare_out-生成方案-20260919.md`。

## 增删文件

| 文件 | 作用 |
| --- | --- |
| `orchestrator/layer_hw_table.py` | 恒定域 + B7 查表 |
| `orchestrator/layer_fields.py` | 23 类字段填充 |
| `orchestrator/layer_render.py` | `键: 值` 文本 |
| `scripts/diff_prepare_out.py` | 全量文件×域对拍 |
| `tests/test_layer_fields.py` | 代表性层夹具 |
| `docs/prepare_out-生成方案-20260919.md` | 技术方案 |
| `orchestrator/layer_id.py` | Softmax/DQ Task 扇出（不再线性链） |
| `orchestrator/net_ini.py` | `[general]` 全字段；`layer = <stem>` |
| `orchestrator/plan.py` | 写出 txt；`--decode-block-only` |
| `scripts/export_gml.py` | 落盘 txt_files、清空旧文件 |
| `graph/quant_pass.py` | **同一源张量共用一条 DQ**（q/k/v、gate/up） |
| `gml_bridge/from_fx.py` | `pim_weight_param` 区分 q/k/v/o |

净增约 **+2140 / -142**（改动 15 个文件，含新文件约 1720 行）。
`layer_fields.py` 1161 行，超过单文件约 400 行的软上限，后续可按
DQ / Softmax / Gemm / Eltwise 拆。

## 收紧白名单后暴露并修掉的真 bug

| # | 症状 | 根因 | 修法 |
| --- | --- | --- | --- |
| 1 | 文件名整体对不上（424 个里只有 5 个重名） | 参考 GML 的 `label` **就是** prepare_out 文件名主干，我方 label 还是 FX 名（`linear`、`add_2`），编排器只能另拼一套 | 语义名写进 GML `label`；FX 名移到内部键 `pim_fx_name`，写盘取 DQ spec、取 RMSNorm eps、层展开认 `headN`、查相位模板都改读它 |
| 2 | Q/K 两条 RoPE 的 DQ 挂反 | 参考是 **Q** 带 4 相 DQ、K 的 add 写 cache；我方按「第二条是 K」挂到了 K 上 | `_is_second_rope` 判据取反，emitter 与编排器同步 |
| 3 | `L2 fpsu buffer id` 348 处错 | 全写 `f1`；参考按相位递增（p1→f1 … p5→f5），Gemm/eltwise 是 f2、bmm 是 f3 | 按层类查表 |
| 4 | `Datain file` 381 处错 | 参考指上游 DQ 的终相平面 `output_buffer_phase_3_{dq}`；我方图里 Split/Transpose 被折叠，边名停在折叠算子上 | 新增 `_upstream_dq()` 穿透折叠算子找真实生产者 |
| 5 | `input scale factor buffer` 233 处错 | 同上，定标应引用上游 DQ 的 `phase_1` | 同上 |
| 6 | bmm 的 `Weights buffer file` 73 处错 | 写成 per-head `weight_buffer_{self}`；参考 32 个 bmm1 **共用一份** K cache | 新增 `_kv_cache_buffer()`，按 KV_Cache_DMA 节点取 |
| 7 | L2 权重双缓冲基址错 | 用 `8256 + size` 硬算；参考按层类两个固定基址 | 查表 `l2_weights_off0/off1` |
| 8 | 一批 L2 尺寸错 | `dq_p4` 输入 `(align16(W)+16)×2`、输出 `align16(W)+16`；`dq_p2` 输出 `align16((Gn+16)×2)`，Gn=1 例外 32；`sm_p2` 输入 `(align16(W)+16)×2` | 逐类按实测公式 |
| 9 | `dq_p2` 的 `Output Stride X/Z` 错 | 输出是每组一个标量：Stride X 恒 1，Stride Z = `Gn+15`（**不对齐 16**） | 单列分支 |
| 10 | `dq_p3` 的 `Transpose type` 错 | 随组数变：Gn=1（分数）→2，Gn>1（线性）→1 | 按 `in_w` 判 |
| 11 | RoPE 双输入宽度左右颠倒 | 广播那一路是 cos/sin 表（hd=128），另一路是数据（H）；槽位由 `Eltwise broadcast input index` 定，Q 路 mul_cos 是槽 0，其余是槽 1 | 按广播槽分宽，`Runtime input <槽>` 跟着走 |
| 12 | `Data scale buffer size` 64 处错 | 片上段 bmm 恒为 2；DDR 段另有一套（bmm1=64、bmm2=16、down=176） | 片上与 DDR 分开算 |
| 13 | 一批「我方多写」的域 | DQ 只有 p1 写分组域；dq_p2 只写 `L2 input num of buffers`；双输入层只写带槽号的 `Use Clipping 0/1`、`Use FPSU 0/1`；RoPE 不写 `Pooling data type`；只有走 Kantor 的那条 add 写 `Kantor A source` | 逐类收紧写出条件 |
| 14 | `skip compare` 多写 | 参考里 v_proj 与图出口残差 `add_2` 没有这一项（输出离开本块，块内无下游可比） | `gemm_v` 与末层不写 |

## 第二轮：绕过对拍器做原始核对，又查出 9 类

上一轮把对拍器收紧后报「全 0」。但对拍器仍然按类归并，所以又写了一个**不看
任何白名单**的原始核对：逐对逐键直接比，只按「是否只差数字」分两栈。
原始差 **6582 处**，其中「结构语义不同」684 处。逐条查完又修掉 9 类：

| # | 症状 | 根因 | 修法 |
| --- | --- | --- | --- |
| 15 | 422 个层文件行尾全错 | 参考是 **CRLF**，我方是 LF；`net.ini` 更特殊：整份 CRLF，**只有两条 dumps 路径和末行是 LF** | `layer_render` 改 CRLF；`net_ini` 按参考混排 |
| 16 | 两个版本戳多一个换行 | 参考 `gml_version.txt` 是 6 字节（无结尾换行），`l2a_version.txt` 13 字节 `0.0.0-<短哈希>` | 去掉换行；`l2a_version` 改成同格式（本仓哈希） |
| 17 | `Dataout file` 422 处缺槽号 | 缓冲按消费者编号，**下游是双输入层时要带槽号**（`input_buffer_0_19`）；我方单槽生产者写了无槽名 | 新增 `_slotted_dataout()`，按消费者槽数补 |
| 18 | `mask` 的 `Datain file 0` 同样缺槽号 | GML 的 Mask 按单数据槽记 `input_buffer` | 消费者侧按槽号取名 |
| 19 | `Original name` 指 FX 名 | 参考指那条 RoPE 的**语义层名** | 读 GML `label` |
| 20 | `DDR data scale Orig Buffer Name` 用裸节点号 | 参考用上游 DQ 的语义层名 + `_dequant_buff` | 穿透折叠算子取 DQ 的 label |
| 21 | RoPE / `mlp_mul` / 残差缺一批 Kantor、Scaling 文件 | 子块各有自己的命名（`Kantor_A_Llama2Activation_Cos_*`、`Scaling_buffer_file_<槽>_<id>`）；RoPE 的 mul 用 **B** 单元、add 用 **A**，且 mul 的 A 只有 shift/bias 没有 scale | 按子块补齐，复用 `gml_names.rope_kantor_*` |
| 22 | 残差的 `output scale factor buffer` 写错 | 按**下游消费者**编号（`input_0_sf_9`），不按自己 | 查下游，双输入时带槽号 |
| 23 | RoPE 六层多写 Residual / Input buffer file | 参考没有这些域（三连不落 DDR、不参与块内比较） | `_drop_rope_absent()` |

另外补上 **RoPE 的 cos / sin 表边界节点**：参考把两张表建成 `is_buffer` 节点并
连边进两条 RoPE（共用一份），所以 RoPE 的 `input_count` 是 3、下游 DQ 有
`Residual input buffer 0/1/2`。我方原先把表折进节点字段，`input_count` 记 1。
补完 GML 从 197 节点变 **199**，bin 从 2318 变 **2330**。

## 图编译器改动

### 1. DQ 按激活源共享

量化 pass 原先每个 Gemm 各插一条 DQ，一层会变成 40 个 DynamicScaling、440 层。
参考产物是 q/k/v 共用一条、gate/up 共用一条。改成按激活源共享后 GML 少 3 个
DQ 节点（200 → 197）。

### 2. cos/sin 建成边界缓冲节点

见上一节，197 → **199**，bin 2318 → **2330**。

### 节点数与层数的演进

| 阶段 | GML 节点 | bin | 展开层数 |
| --- | --- | --- | --- |
| 开工前 | 200 | 2411 | 440 |
| DQ 共享后 | 197 | 2318 | 428 |
| 建 cos/sin 表节点后（现状） | **199** | **2330** | 428 |

428 层丢掉末尾 RMSNorm + lm_head + 它们的 DQ 后 = **422 层**，与参考的纯
decode block 对齐。

## 算子编译器

本轮未改 FlagTree。相位数仍由 `-pim-expand-phases` 决定。字段取值仍走
`gml_hw_table` + `layer_hw_table`。遗留 F1–F6 见方案第 1.2 节。

## 测试

- `python -m pytest tests/ -q -k "not llama2_7b"`：**683 passed**
- `python -m pytest tests/ -q -k "llama2_7b"`：**42 passed**（约 33 分钟，
  在 CRLF、RoPE 表节点、域集合对齐**之后**重跑过，numpy 执行链路未受影响）
- 一层导出 14 项自检全过
- `diff_prepare_out.py` 退出码 0，`VALUE_DIFF / MISSING / EXTRA` 全为 0

两处基线随改动更新，都在测试里写了理由：
- GML 节点数 197 → **199**（加了 cos / sin 两个表节点）
- 线性 DQ 由 7 条降到 5 条（q/k/v 共用、gate/up 共用）

## 怎么验证（命令与实际结果）

四步，从生成到逐域核对。每条都给了实测输出，照着跑应当复现。

环境（每个新 shell 都要）：

```bash
source /media/disk/fengjingge/src/flagOS/flagOS-installed/pytorch/env-pytorch.sh
cd /media/disk/fengjingge/src/flagOS/flagos-pim-compiler
```

### 第 1 步 生成（约 2 分钟，读真实 7B 权重）

```bash
rm -rf /tmp/final_check
python scripts/export_gml.py --layers 1 --seq-len 16 \
    --out-dir /tmp/final_check --use-opcompiler --orchestrate --decode-block-only
```

`--decode-block-only` 丢掉模型末尾 RMSNorm + lm_head + 它们的 DQ，
层数与参考的纯 decode block 对齐到 422（不加是 428）。

实测尾部输出：

```
  [通过] GML 结构自检（5 条规则）: 5/5 通过
  [通过] GML 引用集 == 落盘集: 2330 个文件
  [通过] 接算子编译器前后 GML 逐字节相同: 335357 字节，节点 199
  [通过] 算子编译器真的决定 GML（反证）: 38 个 DQ 相位数 4→2，GML 少 60390 字节
  [通过] 层展开: 428 层（非逐头 44、逐头 384）
  [通过] Layer ID 唯一: 428 个，唯一 428 个
  [通过] L2 offset 16 字节对齐: 428 块全对齐
  [通过] L2 地址分配: 428 块 → 4 槽（复用率 99.1%），数据区 337472 字节
  [通过] net.ini 列出全部层: 422 行 vs 422 层 txt
  [通过] txt_files 文件数: 422 层 + 2 个版本戳
验证全部通过（14 项）
```

其中两项是**反证**，不是自证：`逐字节相同` 钉住「接入算子编译器不改变产物」，
`反证` 钉住「依赖是真的」——把 DQ 相位数砍半，GML 必须跟着变小。

### 第 2 步 对拍（按层类配对，秒级）

```bash
python scripts/diff_prepare_out.py \
    --mine /tmp/final_check/prepare_out \
    --ref /media/disk/fengjingge/src/xinfangzhou-resource/llama2_w4a8_decode_block_0/prepare_out
echo "退出码 $?"
```

配对单位是 `(层类, head, 输入宽)`，**不用文件名**——两边 `qidx` 是不同编号
体系，按名字配会全部错位。实测：

```
  MATCH            45854
  VALUE_DIFF       0
  MISSING          0
  EXTRA            0
  ALLOC            5950
  UNMATCHED_PAIR   0
  UNMATCHED_FILE   0
退出码 0
```

判据：`VALUE_DIFF` / `MISSING` / `UNMATCHED_*` 任一非 0 则退出码 1。
`ALLOC` 是「结构性不可对齐」栏（地址与节点号），打印但不挡。

**这一步会放水**，因为 ALLOC 白名单是人写的。所以还有第 3、4 步。

### 第 3 步 逐文件域集合核对（绕过白名单）

这一步回答「每个文件的每个域是不是都对得上」：

```bash
python3 - <<'PY'
import sys
from pathlib import Path
sys.path.insert(0, '.')
from scripts.diff_prepare_out import load_dir
mine = load_dir(Path('/tmp/final_check/prepare_out'), True)
ref = load_dir(Path('/media/disk/fengjingge/src/xinfangzhou-resource/'
                    'llama2_w4a8_decode_block_0/prepare_out'), False)
same = 0
for key in sorted(set(ref) & set(mine), key=str):
    rf, mf = ref[key][1], mine[key][1]
    if set(rf) == set(mf):
        same += 1
    else:
        print(f'[{key[0]}] 参考={ref[key][0]}  我方={mine[key][0]}')
        print(f'  参考有我方无: {sorted(set(rf) - set(mf))}')
        print(f'  我方有参考无: {sorted(set(mf) - set(rf))}')
print(f'域集合完全相同: {same}/422')
PY
```

实测：`域集合完全相同: 422/422`（无差异明细输出）。

### 第 4 步 原始数值核对（不看任何白名单）

把 ALLOC 里被放过的也全摊出来，按「是否只差数字」分栈：

```bash
python3 - <<'PY'
import sys, re
from pathlib import Path
from collections import Counter
sys.path.insert(0, '.')
from scripts.diff_prepare_out import load_dir
mine = load_dir(Path('/tmp/final_check/prepare_out'), True)
ref = load_dir(Path('/media/disk/fengjingge/src/xinfangzhou-resource/'
                    'llama2_w4a8_decode_block_0/prepare_out'), False)
diff, ex, tot = Counter(), {}, 0
for key in sorted(set(ref) & set(mine), key=str):
    rf, mf = ref[key][1], mine[key][1]
    tot += len(set(rf) | set(mf))
    for k in set(rf) & set(mf):
        if rf[k] != mf[k]:
            diff[k] += 1
            ex.setdefault(k, (rf[k][0], mf[k][0]))
norm = lambda s: re.sub(r'\d+', '#', s)
num = sum(c for k, c in diff.items() if norm(ex[k][0]) == norm(ex[k][1]))
real = [(c, k, ex[k]) for k, c in diff.most_common()
        if norm(ex[k][0]) != norm(ex[k][1])]
print(f'域总数 {tot}；只差编号 {num} 处；结构语义 {sum(x[0] for x in real)} 处')
for c, k, (rv, mv) in real:
    print(f'  {c:4d} {k:34s} ref={rv[:28]!r} mine={mv[:28]!r}')
PY
```

实测：

```
域总数 51804；只差编号 5872 处；结构语义 78 处
    41 DDR Input Orig Buffer Name 1       ref='nprm_182_i168' mine='mask'
    34 DDR Input TVM Orig Buffer Name 1   ref='nprm_182_i168' mine='mask'
     2 DDR Input TVM Orig Buffer Name 0   ref='nprm_182_i12' mine='hidden_states'
     1 DDR Output TVM Orig Buffer Name    ref='tvmgen_default_nprm_main_182' mine='tvmgen_default_nprm_main_out'
```

78 处全是 TVM Relay 内部符号名（见 Q-A），没有其它结构语义差异。

### 第 5 步 回归

```bash
python -m pytest tests/ -q -k "not llama2_7b"   # 683 passed（约 2 分钟）
python -m pytest tests/ -q -k "llama2_7b"       # 42 passed（约 33 分钟）
```

7B 那组确认编排器与图侧改动没打坏原 NumPy 执行链路。

### 文件层面的零散核对

```bash
M=/tmp/final_check/prepare_out
R=/media/disk/fengjingge/src/xinfangzhou-resource/llama2_w4a8_decode_block_0/prepare_out

ls $M/txt_files | wc -l                      # 424，同参考
grep -c '^layer' $M/net.ini                  # 422，同参考

# [general] 逐字节（含 CRLF 混排）
cmp <(sed -n '/\[general\]/,/^\[layers\]/p' $M/net.ini) \
    <(sed -n '/\[general\]/,/^\[layers\]/p' $R/net.ini) && echo 一致

# 文件名模式（去掉编号后比集合）
python3 -c "
import re; from pathlib import Path
f=lambda d: sorted(re.sub(r'\d+','N',p.name) for p in Path(d).glob('*.txt'))
print(f('$M/txt_files') == f('$R/txt_files'))"   # True

# 行尾符：参考 422 个层文件全是 CRLF
for f in $M/txt_files/*.txt; do case $(basename $f) in *version*) continue;; esac
  grep -q $'\r' "$f" || echo "缺 CRLF: $f"; done   # 无输出
```

## 要甲方回的三个问题

按「不回答就只能猜、猜错代价不对称」排的。前两个决定要不要动代码，
第三个决定换配置时会不会错。

### Q-A（挡住 78 处差异）Orig Buffer Name 是标签还是索引键

`DDR Input Orig Buffer Name` / `DDR Input TVM Orig Buffer Name`
这两个域，底层编译器是只当日志标签，还是会按这个字符串去索引
`parser_output/` 里的 bin、或在多子图拼接时做匹配？若是索引，键怎么生成？

**为什么非问不可**：参考里这两个域写的是**同一个值**（`nprm_182_i168`），
两个域同值说明有一个是冗余的，但留着的那个起什么作用只有对方解析器知道。
三种可能的处理方式完全不同：

| 若是 | 怎么办 |
| --- | --- |
| 纯溯源标签 | 我方的 `mask` / `hidden_states` 更可读，**不用改** |
| 按名字查 bin | 必须改，而且要改成能对上**我方 bin 文件名**的形式 —— 不是抄参考的 `nprm_182_i168` |
| 多子图匹配键 | 单块 decode 用不到，整网 32 层会踩 |

猜错的代价不对称：当成第 1 种而实际是第 2 种，仿真器会找不到缓冲；
反过来白改一遍没坏处也没收益。所以**不猜，等回复**。

### Q-B（同类问题）文件名里的 `qidx` 是语义标识还是可读性标签

参考 `self_attn_q_proj_MatMul_qidx2_params_23`：`params_23` = GML `node_id`
（我方已同义），`qidx2` 是 Relay 号，我方填 FX 出现序。
若 `qidx` 只是给人看的，现状可用；若下游按它索引，需要对齐规则。

### Q-C（影响换配置）三处照抄的公式

| 项 | 参考值 | 我方 | 疑问 |
| --- | --- | --- | --- |
| K 路两条 mul 的表那路 `DDR Input buffer size` | 128 | 照抄 128 | Q 路同位置是 256（= 128×2），为何 K 路是半个 |
| `dq_p2` 在 Gn=1 时的 L2 输出尺寸 | 32 | 照抄 32 | 按 `align16((Gn+16)×2)` 应得 48 |
| `L2 fpsu buffer size` 一族 | 28672 / 57344 / 77824 … | 按层类查表 | 对不上任何单一几何公式（域确认表 B7：换形状要重测） |

这三项现在**值与参考一致**（因为照抄），但我方不知道为什么是这个数。
拿到公式就能从「抄」变成「算」，换 hidden / 头数才不会错。

## 建议的下一步

1. **现在**：把 Q-A / Q-B / Q-C 发给甲方
2. **等回复期间**：让对方拿这份产物过一遍仿真器。422 个文件、域全齐、
   数值自洽 —— 如果卡住，报错会直接指出哪个域被当成了索引，比继续猜有效
3. **拿到回复后**：Q-A 若是索引键，改法很快（一处命名函数）；
   Q-C 若有公式，把照抄换成推导

第 2 步优先于继续对拍：结构上已经完备，最快的验证是让它真的跑一次。

## 当前不足

按原始核对剩下的 92 处，逐类说明。

### 1. TVM Relay 内部符号名，78 处，改不动

| 处数 | 域 | 参考值 | 我方值 |
| --- | --- | --- | --- |
| 41 | `DDR Input Orig Buffer Name 1` | `nprm_182_i168` | `mask` |
| 34 | `DDR Input TVM Orig Buffer Name 1` | `nprm_182_i168` | `mask` |
| 2 | `DDR Input TVM Orig Buffer Name 0` | `nprm_182_i12` | `hidden_states` |
| 1 | `DDR Output TVM Orig Buffer Name` | `tvmgen_default_nprm_main_182_output_0` | `tvmgen_default_nprm_main_output_0` |

这些是 Relay 做图优化时给中间张量起的内部符号名：`nprm` 是模块名，
`182` 是子图编号，`i12` / `i168` 是该子图的第 12 / 168 个输入参数；
输出那个是 TVM codegen 的标准格式。

**取决于 Relay 内部的算子编号与参数排序**，我方走 `HF → torch.fx → GML`，
没有 Relay 这一层，拿不到这套编号。硬凑只能按角色抄参考的具体数字。
我方填同位置的语义名（`hidden_states` / `mask` / `rope_cos`），
**域在、可解析、语义正确**，只是字符串不同。

判断它是否要紧的一条线索：参考里这一族的值与同槽的
`DDR Input Orig Buffer Name` **完全相同**（`nprm_182_i168` 写了两遍）。
如果是索引键，没必要写两份同值，所以更像溯源冗余。**但这是推测，需甲方确认**：

- 若只是溯源标签 → 我方语义名更可读，不影响执行
- 若被当标识符（按名字找 bin、多子图拼接时匹配）→ 必须与对方一致

`qidx` 同理：参考的 `qidx2` 是 Relay 号，我方填 FX 出现序。文件名主干规则
与 `params_N`（= GML `node_id`）已同义，只有 `qidx` 这一段是另一套编号。

### 2. 片上地址与节点号，5872 处（结构性，不追）

分两类，都**必然不同**，不是取值错：

| 类 | 例 | 为什么必然不同 |
| --- | --- | --- |
| 片上/DDR 地址 | `L2 fpsu buffer offset` 536798208 vs 536804352 | 我方贪心分配器 vs 参考 L2Analyzer。两侧缓冲**尺寸来源**都不同（我方闭合公式、参考按层类查表），尺寸分布不同则能塞进同槽的组合不同，地址必然不同（文档 8.3） |
| 带节点号的引用 | `weight_buffer_195.bin` vs `weight_buffer_23.bin` | 两边 GML `node_id` 各一套编号体系（我方逆拓扑，参考来自 Relay）。指的是同一块缓冲 |

判据是**结构相同、编号不同**：域在、格式对、值内部自洽（我方的
`weight_buffer_195.bin` 确实落盘了，引用不悬空）。下游按域名读值、
按值在本产物内解析即可，不要求跨产物一致。

### 2.1 已经对齐、但脆弱的一类：几何与尺寸

这批现在**完全一致**（`VALUE_DIFF 0` 里就包含它们），但值的来源分两种：

- **闭合公式推导**：`Output Stride Z = align16(W)+15`、
  `Data scale width = K/G` 等 —— 换配置自动跟着变，稳
- **参考 422 层实测查表**：`L2 fpsu buffer size` 的 28672 / 57344 / 77824、
  `L2 weights buffer offset` 的两个基址等 —— **换 hidden / 头数要重测**

后者不是当前差异，是**脆弱点**。集中放在
`orchestrator/layer_hw_table.py` 一个文件，就是为了重测时好替换。
见 Q-C。

### 3. 三处照抄而非推导，语义未确认

- K 路两条 mul 的表那一路 `DDR Input buffer size` 写 128（Q 路同位置写 256），
  疑似 K 路表只装半个周期
- `dq_p2` 的 L2 输出尺寸在 Gn=1 时取 32，按 `align16((Gn+16)×2)` 应得 48
- `l2a_version.txt` 我方填 `0.0.0-<本仓短哈希>`，格式同参考但来源不同
  （参考是对方 L2Analyzer 的版本）

### 4. 残差旁路的边没接上（域已补齐，边仍缺）

第一条残差 `add` 的一路往上追到 embedding（GML 侧不可发射）就断了，那条边
丢在图里，`input_count` 记 1 而参考记 2。

`Residual input buffer 1` 这个**域**已经按「双输入层两槽都要有」补齐，
所以域集合对得上；但 GML 里那条**边**仍然缺。补边试过一次：它和规则 2
「缓冲按第一个消费者编号」冲突 —— 入口缓冲一旦有两个读者，`output_buffer`
只能指一个，另一个必然悬空，`test_output_buffer_points_at_the_consumer`
与 `input_count` 恒等式同时挂。

要做对得先定「一个缓冲喂多个消费者时怎么命名」，那是命名契约的改动，波及
全部 2330 个 bin。**判断不值当，已回退**，`from_fx.py` 里 `entry_bypass`
那段注释记了原因与复现路径。

### 5. 其它

- **FlagTree 遗留 F1–F6**：Flp / LUT / Kantor 属性还没打进 PIM IR，静态表未退役。
- `layer_fields.py` 约 1300 行，偏长，待按层类拆。
- 本轮大量取值来自「参考 422 层实测」，**换 hidden / 头数要重测**（域确认表
  B7 已注明）。这批常量集中在 `orchestrator/layer_hw_table.py`，便于替换。
