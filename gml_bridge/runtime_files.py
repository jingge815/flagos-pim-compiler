"""把量化产物写成 GML 引用的 `.bin` 文件。

文件名一律经 `contracts.gml_names`，不在这里拼字符串——那是跨语言真源，
两侧靠它保持一致（见文档第 10 节）。

写盘用 `tofile`，不做任何字节序转换：实物是小端，本机也是小端，恒等 LUT 的
字节级比对（26.4）已经确认这条路径对了。
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from contracts import gml_names as names
from contracts.gml_lut import (
    synth_exp,
    synth_identity,
    synth_reciprocal,
    synth_silu,
)
from contracts.gml_quant import INT8_MAX, INT8_MIN, lut_identity
from gml_bridge.phase_data import (
    DQ_PHASE0_BIAS,
    DQ_PHASE1_SCALE,
    DQ_PHASE3_SCALE,
    DQ_PHASE3_SHIFT,
    DynamicScalingPhases,
    SoftmaxPhases,
)
from quant.weights import QuantizedWeight


@dataclass
class WrittenFiles:
    """一次写盘的结果，供交叉校验用。

    `names_written` 是实际落盘的文件名集合。它要与 GML 里引用的名字集合完全相等——
    多一个是垃圾文件，少一个是悬空引用，两者都会让底层编译器读不下去。
    """

    directory: Path
    names_written: set[str] = field(default_factory=set)
    total_bytes: int = 0
    # 文件名 -> 内容 sha256（小写 hex）。GML 的 `weight_buffer_hash` 要与权值
    # bin 的字节对上，而这里是字节最终成形、也是唯一成形的地方：在序列化时算
    # 得把量化再做一遍，在写盘之后算又得回头扫盘。
    hashes: dict[str, str] = field(default_factory=dict)

    def _write(self, name: str, data: np.ndarray | bytes) -> None:
        path = self.directory / name
        if isinstance(data, bytes):
            path.write_bytes(data)
            size = len(data)
            payload = data
        else:
            data.tofile(path)
            size = data.nbytes
            payload = data.tobytes()
        digest = hashlib.sha256(payload).hexdigest()
        # 按节点必填：参考 73 个带 weight_buffer 的节点全都写 hash。
        # 全零占位也写——互证检查的是「写出的字节与声明一致」，不是
        # 「内容是真实权值」。KV-as-weight 本轮按形状补零，下游不得
        # 依赖其内容；内容是否两两不同另用测试守。
        self.hashes[name] = digest
        self.names_written.add(name)
        self.total_bytes += size


def write_weight(
    files: WrittenFiles, node_id: int, quantized: QuantizedWeight
) -> None:
    """写一个节点的权重与它的 per-group scale。

    权重按**本节点**编号——它属于节点自己，不属于某条边（见规则 2）。
    """
    files._write(names.weight_buffer(node_id), quantized.values)
    files._write(names.weight_scale(node_id), quantized.scales)


def write_output_scale(
    files: WrittenFiles, node_id: int, scale: float | np.ndarray,
    dtype=np.float16,
) -> None:
    """写节点输出的 requant scale。按**本节点**编号（它属于节点自己）。

    静态算子上它等于下游的 `input_sf`；DQ 节点上它逐元素等于本节点
    `output_buffer_phase_1`（见 phase_data）。RMSNorm 系列是 fp32。
    """
    values = np.atleast_1d(np.asarray(scale, dtype=dtype))
    files._write(names.output_scale(node_id), values)


def write_per_tensor_weight(
    files: WrittenFiles, node_id: int, weight: np.ndarray
) -> None:
    """写 per-tensor 量化的权重（RMSNorm 的一维缩放张量）。

    与 int4 per-group 的路径**完全不同**，实测依据（节点 25 / 197）：

        weight_buffer_25.bin   4096 个 int8      （= hidden_size，一维）
        weight_sf_25.bin       单个 **fp32**     （其余算子是 fp16）

    参考产物里那个 sf 解出 0.007874 = 1/127，且 4096 个值全为 127
    —— 后者是合成数据的痕迹（真实 layernorm 权重量级 0.01~0.8），
    但 **1/127 这个定法是真的**：per-tensor 对称量化、分母取 INT8_MAX。

    注意这里分母用 127 而不是 128：per-tensor 路径没有 DQ 那条
    `×2 再 /256` 的硬件通路，absmax 直接映射到 127。
    """
    flat = np.ascontiguousarray(weight, dtype=np.float32).ravel()
    peak = float(np.abs(flat).max()) if flat.size else 0.0
    scale = np.float32(peak / INT8_MAX) if peak > 0 else np.float32(1.0)

    quantized = np.clip(
        np.rint(flat / scale), INT8_MIN, INT8_MAX).astype(np.int8)
    files._write(names.weight_buffer(node_id), quantized)
    files._write(names.weight_scale(node_id), np.full(1, scale, dtype=np.float32))


def write_named_buffer(
    files: WrittenFiles, name: str, element_count: int, dtype=np.int8,
    content: np.ndarray | None = None,
) -> None:
    """按**给定文件名**写一个数据缓冲。

    命名规则算不出来的那几种走这里：生产者流进下游权重通路时命名为
    `weight_buffer_<消费者>`，名字已由 GML 侧定好，这里照抄即可。

    `content` 给了就写它，否则写全零。走权重通路的那 64 个 KV 缓冲**必须**
    给内容：全零的 bin 与「真的算出 0」在盘上无法区分，读的人会把
    「这个文件没内容」当成「这个权值就是 0」。
    """
    if content is None:
        content = np.zeros(element_count, dtype=dtype)
    files._write(name, content)


def placeholder_weight(node_id: int, element_count: int) -> np.ndarray:
    """KV 缓存走权重通路时的占位内容。

    这 64 个 bin 装的是 KV cache 而不是模型权值，参考产物那一侧同样是
    合成数据（方案 §9.6：结构是权威的，数值不是）。但**不能填全零**。
    按节点号定种子，给一段确定、可复现、非零的 int8。
    """
    rng = np.random.default_rng(node_id)
    return rng.integers(-128, 128, size=element_count, dtype=np.int8)


def write_phase_output_buffer(
    files: WrittenFiles, node_id: int, element_count: int, dtype=np.int8
) -> None:
    """写 phase 型节点**自命名**的 output_buffer。

    与 `write_data_buffer` 的区别只在命名：这个按本节点编号
    （`output_buffer_<self>.bin`），那个按消费者编号
    （`input_buffer_<consumer>.bin`）。装的是本节点量化后的完整输出（int8）。
    """
    files._write(
        names.phase_output_buffer_self(node_id),
        np.zeros(element_count, dtype=dtype))


def write_activation_scale(
    files: WrittenFiles, consumer_id: int, scale: float | np.ndarray,
    slot: int | None = None, dtype=np.float16,
) -> None:
    """写一条边上的激活 scale。

    按**消费者**编号：缓冲区代表边，编号取读它的那个节点。

    `scale` 可以是标量也可以是 per-group 数组——实测 DQ 节点的
    `output_sf` 是数组（hidden 32 个、MLP 86 个），普通边上是标量。

    `dtype` 默认 fp16；**RMSNorm 系列是 fp32**（实测 2 处），传 np.float32。
    """
    values = np.atleast_1d(np.asarray(scale, dtype=dtype))
    files._write(names.scale(consumer_id, slot), values)


def write_activation_zero_point(
    files: WrittenFiles, consumer_id: int, zero_point: int = 0,
    slot: int | None = None,
) -> None:
    """写零点。实物里恒为 0（对称量化），但字段必须在。"""
    files._write(
        names.zero_point(consumer_id, slot),
        np.array([zero_point], dtype=np.int32),
    )


def write_data_buffer(
    files: WrittenFiles, consumer_id: int, element_count: int,
    slot: int | None = None, dtype=np.int8,
) -> None:
    """写一条边上的数据缓冲区。

    按**消费者**编号——缓冲区代表边，编号取读它的那个节点（规则 2）。

    结构层写全零：这些是运行时才有内容的激活缓冲区，实物里它们带的是某次推理的
    真实数据。尺寸必须对，因为对方按形状读取；内容在结构验证阶段不参与。
    """
    files._write(
        names.data_buffer(consumer_id, slot),
        np.zeros(element_count, dtype=dtype),
    )


def write_identity_lut(files: WrittenFiles, node_id: int) -> None:
    """写一张恒等 LUT。

    在采样规则确认之前只能发恒等表——它是实物里的合法值（37 个），
    而且我们生成的与实物字节完全一致（见 26.4）。
    """
    files._write(names.activation_lut(node_id), lut_identity())


def write_scaling(
    files: WrittenFiles, node_id: int, scaling: float,
    post_shift: int = 0, bias: float = 0.0, slot: int | None = None,
) -> None:
    """写 FPSU 定标三族：scale / post-shift / bias。

    三个文件的宽度**各不相同**，对应硬件规范里 FPSU 的三个操作数
    （加 32 位 bias、乘 16 位 scale、round 后右移）：

        Scaling_buffer_file     fp16  2 字节
        Scaling_PS_buffer_file  u8    1 字节
        Bias_buffer_file        fp32  4 字节   <- 是 fp32，不是 int32

    `scaling` 是**算子自身的数学缩放**，不是量化因子——attention 的 `1/√d`
    就写在这里。实测全图只有五个取值：
        1/√head_dim（32 个 matmul1）、1/multiplier（次正规数补偿 2 处）、
        2.0（KV_Cache_DMA）、0.25/0.5（各 1 处）、1.0（其余 55 个）

    `post_shift` 实测除 KV_Cache_DMA 的两个节点（=14）外恒为 0。
    `bias` 在非 phase 节点上实测 79/79 全为 0.0。
    """
    files._write(
        names.fpsu_scale(node_id, slot), np.array([scaling], dtype=np.float16))
    files._write(
        names.fpsu_post_shift(node_id, slot),
        np.array([post_shift], dtype=np.uint8))
    files._write(
        names.fpsu_bias(node_id, slot), np.array([bias], dtype=np.float32))


def write_zero_point(files: WrittenFiles, name: str) -> None:
    """写一个 zero-point 文件：int32 的 0。

    对称量化下全部 381 个 `*_zp_*.bin` 都是 4 字节 int32 的 0，
    但**文件必须存在**——GML 引用了它们，缺一个就是悬空引用。
    """
    files._write(name, np.zeros(1, dtype=np.int32))


def write_dq_phases(
    files: WrittenFiles, node_id: int, phases: DynamicScalingPhases
) -> None:
    """写一个 DynamicScaling 节点的四相：缓冲、定标、LUT、Kantor 系数。

    这些文件按**本节点**自命名，不按消费者——它们是节点内部的中间态，不流经边。

    定标三族的宽度按相取：常量相是标量，逐组相是向量（实测节点 22 是 32 元素
    向量，节点 12/193 是标量）。这里按 phases 的组数决定。
    """
    groups = phases.phase1.size

    # 各相的输入缓冲。**四相各有一个**，尺寸各不相同 —— 每一相的输入就是
    # 上一相的输出，只有 p0 与 p3 吃完整张量。实测节点 12（4096 元素 / 32 组）：
    #
    #   input_buffer_phase_0   8192B  = 4096 fp16   原张量
    #   input_buffer_phase_1     64B  =   32 fp16   p0 的输出（逐组）
    #   input_buffer_phase_2     64B  =   32 fp16   **p0 的输出**（逐组）
    #   input_buffer_phase_3   8192B  = 4096 fp16   原张量（Kantor 吃它做定点化）
    #
    # **p2 吃的是 p0 而不是 p1**：它算 1/p0（倒数表），而 p1 算的是 p0/256。
    # 两相都从 p0 分叉出来，不是串行的 —— 实测 3/3 个节点上
    # `input_buffer_phase_2 == output_buffer_phase_0` 逐字节成立。
    # 写成 p1 会让倒数的输入错 256 倍。
    #
    # 只写 p0 会让另外三个引用悬空。
    phase_inputs = (
        phases.source, phases.phase0, phases.phase0, phases.source)
    for phase, data in enumerate(phase_inputs):
        files._write(names.phase_input_buffer(node_id, phase), data)

    # 各相的输出缓冲。
    for phase, data in enumerate(
            (phases.phase0, phases.phase1, phases.phase2, phases.phase3)):
        files._write(names.phase_output_buffer(node_id, phase), data)

    # phase0 的 bias 是 fp32 常量 2^-63，其余相为 0。
    for phase in range(4):
        bias = DQ_PHASE0_BIAS if phase == 0 else 0.0
        files._write(
            names.phase_fpsu_bias(node_id, phase),
            np.full(1, bias, dtype=np.float32))
        files._write(
            names.phase_fpsu_post_shift(node_id, phase),
            np.zeros(1, dtype=np.uint8))

    # 定标常量：p1 乘 1/256、p3 乘 256，p0/p2 不缩放。
    for phase, scale in enumerate(
            (1.0, DQ_PHASE1_SCALE, 1.0, DQ_PHASE3_SCALE)):
        files._write(
            names.phase_fpsu_scale(node_id, phase),
            np.full(1, scale, dtype=np.float16))

    # p1 走恒等表，p2 走倒数表。两张都自行合成。
    files._write(names.phase_lut(node_id, 1), synth_identity())
    files._write(names.phase_lut(node_id, 2), synth_reciprocal())

    # p3 的 Kantor 系数：scale 等于 p2、shift 恒为 -8、bias 为 0。
    files._write(names.phase_kantor_scale(node_id, 3), phases.kantor_scale)
    files._write(
        names.phase_kantor_shift(node_id, 3),
        np.full(groups, DQ_PHASE3_SHIFT, dtype=np.int8))
    files._write(
        names.phase_kantor_bias(node_id, 3),
        np.zeros(groups, dtype=np.float32))


def write_softmax_phases(
    files: WrittenFiles, node_id: int, phases: SoftmaxPhases
) -> None:
    """写一个 Softmax 节点的五相。

    **两个归约相的 4 字节编码不同**（见 phase_data）：phase0 是 fp16 位模式放
    高 2 字节，phase2 是真 fp32。所以这里取 `phases.*_bytes` 而不是自己打包。

    另外两处是**运行时落点**而非常量：`Bias_buffer_phase_1` 等于 phase0 的输出、
    `Scaling_buffer_phase_4` 等于 phase3 的输出。不要硬编码 -30.75。

    **相位模型是 5 入 + 5 出对称结构**（复核 20260921 纠正：上一版这里的
    docstring 说"phase1 没有 output_buffer_phase_1"是错的——那是只看
    `prepare_out/txt_files` 的层卡字段推出来的，没有去对参考 GML 文本本身。
    实测参考 `relay2gml_graph.gml` 的 node 18 逐相都声明了
    `input_buffer_phase_N` 与 `output_buffer_phase_N`（N=0..4），
    `parser_output` 也确实有全部 10 个文件。逐相数据流（实测逐字节验证）：

        phase0  in=原始分数(1024fp16)         out=[0,-max]（高2字节编码）
        phase1  in=原始分数(1024fp16，同p0)    out=exp 数组(1024fp16)
        phase2  in=exp 数组(读 p1 的输出)       out=Σexp（真 fp32，4B）
        phase3  in=Σexp（fp32，读 p2 的输出）   out=1/Σexp（fp16，2B）
        phase4  in=exp 数组(再读一次 p1 的输出)  out=exp×(1/Σexp)（1024fp16）

    即 phase1/phase4 的 `input_buffer_phase_*` 都是「重复读 phase1 的
    output_buffer」，不是开一条新数据；phase2/phase3 的 `input_buffer_phase_*`
    是「读上一相的 output_buffer」。这里按此写，不再发明
    `input_buffer_phase_3` 是 2 元素的猜测（复核前那版是错的，参考实测是
    1 个 fp32）。
    """
    files._write(names.phase_input_buffer(node_id, 0), phases.source)
    files._write(names.phase_output_buffer(node_id, 0), phases.phase0_bytes)

    # phase1：源数据再读一次（结构上与 phase0 相同的输入），输出是真正的
    # exp 数组。
    files._write(names.phase_input_buffer(node_id, 1), phases.source)
    files._write(names.phase_output_buffer(node_id, 1), phases.phase1)

    # phase2：输入是 phase1 的输出（exp 数组），输出是 fp32 归约和。
    files._write(names.phase_input_buffer(node_id, 2), phases.phase1)
    files._write(names.phase_output_buffer(node_id, 2), phases.phase2_bytes)

    # phase3：输入是 phase2 的输出（fp32 标量，1 个元素，不是 2 个——上一版
    # 猜错了），输出是倒数。
    files._write(
        names.phase_input_buffer(node_id, 3),
        np.full(1, phases.phase2, dtype=np.float32))
    files._write(
        names.phase_output_buffer(node_id, 3),
        np.full(1, phases.phase3, dtype=np.float16))

    # phase4：输入再读一次 phase1 的输出（exp 数组），输出是归一化后的
    # softmax 结果。
    files._write(names.phase_input_buffer(node_id, 4), phases.phase1)
    files._write(names.phase_output_buffer(node_id, 4), phases.phase4)

    for phase in range(5):
        # p1 的 bias 是 phase0 的落点；p4 的 scale 是 phase3 的落点。
        if phase == 1:
            files._write(names.phase_fpsu_bias(node_id, phase), phases.phase1_bias)
        else:
            files._write(
                names.phase_fpsu_bias(node_id, phase),
                np.zeros(1, dtype=np.float32))

        if phase == 4:
            files._write(names.phase_fpsu_scale(node_id, phase), phases.phase4_scale)
        else:
            scale = 0.5 if phase == 1 else 1.0
            files._write(
                names.phase_fpsu_scale(node_id, phase),
                np.full(1, scale, dtype=np.float16))

        files._write(
            names.phase_fpsu_post_shift(node_id, phase),
            np.zeros(1, dtype=np.uint8))

    # p1 过 exp 表（占位，见 synth_exp 的说明——数值路径本来就不经过它）、
    # p3 过倒数表。
    files._write(names.phase_lut(node_id, 1), synth_exp())
    files._write(names.phase_lut(node_id, 3), synth_reciprocal())


def write_rms_norm_epsilon(files: WrittenFiles, node_id: int, epsilon: float) -> None:
    """写 RMSNorm 的 eps 常量（fp32）。取自 config.json 的 rms_norm_eps。"""
    files._write(
        names.rms_norm_epsilon(node_id), np.full(1, epsilon, dtype=np.float32))


def write_fused_silu_lut(files: WrittenFiles, node_id: int) -> None:
    """写融合进 Gemm 的 SiLU 表。"""
    files._write(names.activation_lut(node_id), synth_silu())


def verify_against_graph(files: WrittenFiles, referenced: set[str]) -> None:
    """交叉校验：落盘的文件名与 GML 引用的名字必须完全相等。

    这是架构 C 的那道防线（第 10 节）。不相等就抛，因为悬空引用不会在我们这侧
    报错，要到对方的解析器才炸。
    """
    missing = referenced - files.names_written
    extra = files.names_written - referenced

    problems = []
    if missing:
        problems.append(f"GML 引用了但没写盘: {sorted(missing)[:5]}")
    if extra:
        problems.append(f"写了盘但 GML 没引用: {sorted(extra)[:5]}")
    if problems:
        raise ValueError("；".join(problems))


def write_rope_buffer(
    files: WrittenFiles, key: str, name: str, element_count: int
) -> None:
    """写一个 RoPE 子块的缓冲。

    宽度按族分（实测）：

    | 族 | dtype | 元素数 |
    | --- | --- | --- |
    | `*_sf` / `Scaling_buffer_file_*` / `*_scale_buffer_file` | fp16 | head_dim |
    | `*_zp` | int32 | 1（对称量化恒为 0） |
    | `Scaling_PS_buffer_file_*` / `*_Shift_*` | uint8 | head_dim |
    | `*_bias_buffer_file` | fp32 | head_dim |
    | `cos_mul_output` / `sin_mul_output` | fp16 | 整张中间态 |

    cos/sin 本身的取值域是 [-1, 1]（校验器有这条断言），但这里写零占位 ——
    真实值要跑一次前向标定，见计划 §11.6 的说明。
    """
    if key.endswith("_zp"):
        files._write(name, np.zeros(1, dtype=np.int32))
    elif "_Shift_" in key or key.startswith("Scaling_PS_buffer_file"):
        files._write(name, np.zeros(element_count, dtype=np.uint8))
    elif key.endswith("_bias_buffer_file"):
        files._write(name, np.zeros(element_count, dtype=np.float32))
    else:
        files._write(name, np.zeros(element_count, dtype=np.float16))
