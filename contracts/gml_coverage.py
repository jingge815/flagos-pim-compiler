"""GML 字段族的覆盖状态声明——第二层验证的判据。

参考产物有 118 个字段族（见 scripts/gml_field_inventory.py）。序列化器对每一族
必须表态：已产出、按轮次待产出、或显式声明不适用。**不允许静默遗漏**——
漏一族就是底层编译器少一项配置，而这不会在我们这侧报错。

`tests/test_gml_coverage.py` 拿这份声明与实际产物比对：声明已产出的必须真的在
产物里，声明不适用的必须真的不在。声明与实现不一致就失败。
"""

from __future__ import annotations

# 第 3 轮（结构）已产出的字段族。
EMITTED = frozenset({
    # 身份与拓扑
    "id", "node_id", "label", "name", "input_count",
    "residual_input_buffer", "input0_node_id", "input1_node_id",
    "input2_node_id", "output0_node_id", "output1_node_id",
    "source", "target", "dims", "directed", "relay2gml_version",
    # 算子类型与融合
    "op_type", "activation_op_type",
    # 数据缓冲（名字已定，内容待第 4 轮）
    "input_buffer", "input_buffer_0", "input_buffer_1", "input_buffer_2",
    "output_buffer",
    # 输入量化参数的**名字**（值待第 4 轮）
    "input_sf", "input_0_sf", "input_1_sf", "input_2_sf",
    # 边界缓冲节点。两份参考产物都有（ResNet50 两个，llama2 的 decode block
    # 十个），而且算子节点悬空会被结构校验判为违规。
    "is_buffer", "residual_output_buffer",
    # 权重与它的 per-group scale。量化路径已接通（见文档 27、29 节），
    # int4 布局用实物做过字节级往返验证。
    "weight_buffer", "weight_sf",
    # 权重/scale 的 dtype。两条路径都已接通：
    #   int4 per-group（q/k/v/o/gate/up/down_proj）
    #   int8 per-tensor + **fp32** sf（RMSNorm 的一维缩放张量）
    "weight_buffer_dtype", "weight_sf_dtype",
    "input_sf_dtype", "output_sf_dtype",
    # RMSNorm 走向量单元，配置在 vpu_params 子块里，eps 取自图。
    "RMSNorm_Add_Const", "Use_Scaling", "vpu_params", "Vpu_Axis",
    "input_scale_factor_buffer", "output_scale_factor_buffer",
    "Weights_buffer_file", "weights_scaling_buffer_file", "bias_buffer_file",
    # 硬件单元配置。由 (op_type, phase) 唯一确定，查 contracts/gml_hw_table.py
    # 即得 —— 实测 364 个字段项里 353 项单值，无一需要算子编译器参与。
    "nmu_mode", "fpsu_mode", "fpsu_spc", "fpsu_spc_axis", "fpsu_spg",
    "pooling_dtype", "kantor_mode",
    # 逐头展开产出的 attention 字段。
    "weight_format", "split_channel_number", "MatMul_input_as_weight",
    "group_attention_data_num", "group_attention_weight_num",
    "transpose",
    # 量化参数的零点。对称量化下恒为 0，但**文件必须存在**
    # （实测 381 个 zp 文件全是 4 字节 int32 的 0）。
    "input_zp", "input_0_zp", "input_1_zp", "input_2_zp",
    "weight_zp", "output_zp", "output_sf",
    # FPSU 定标三族。三个文件宽度各不相同（fp16 / u8 / fp32），对应硬件
    # FPSU 的三个操作数：加 32 位 bias、乘 16 位 scale、round 后右移。
    # attention 的 1/√head_dim 就折在 Scaling_buffer_file 里。
    "Scaling_buffer_file", "Scaling_buffer_file_0", "Scaling_buffer_file_1",
    "Scaling_PS_buffer_file", "Scaling_PS_buffer_file_0",
    "Scaling_PS_buffer_file_1",
    "Bias_buffer_file", "Bias_buffer_file_0", "Bias_buffer_file_1",
    # DynamicScaling 的 4 相流水线。phase 在 GML 里不是独立节点，
    # 而是同一节点内的 *_phase_<k> 字段族。
    "use_dynamic_quantization",
    "rtl_version",
    # 数据通道声明。dtype **沿边传播**（上游是 DQ 就吃 int8），
    # `*_data_extensions` 是它的编码（float16→3、int8→1）。
    "input_buffer_dtype", "output_buffer_dtype",
    "input_buffer_0_dtype", "input_buffer_1_dtype",
    "input_data_extensions", "output_data_extension",
    # 被量化张量的形状，以及按 group_size 拆开后的形状。
    "original_shape", "output_shape_by_group",
    # Gemm v_proj / mlp_mul 的 Kantor，gate 的 SiLU LUT，双输入 FPSU 分槽。
    "activation_lut_file",
    "fpsu_mode_0", "fpsu_mode_1",
    "fpsu_0_spc", "fpsu_0_spg", "fpsu_1_spc", "fpsu_1_spg",
    "pooling_dtype_0", "pooling_dtype_1",
    "kantor_A_spc", "kantor_A_spg", "kantor_A_scale_axis",
    "kantor_A_scale_buffer_file", "kantor_A_bias_buffer_file",
    "kantor_A_Shift",
    "activation_mode", "activation_special_operators",
})

# 第 4 轮（静态量化）要补的字段族。量化参数、定标系数、LUT 都依赖校准数据，
# 结构轮拿不到。
PENDING_QUANTIZATION = frozenset({
    # 输入/权重/偏置/输出的量化参数
    "input_0_sf_dtype", "input_1_sf_dtype",
    "bias_buffer", "bias_buffer_dtype", "bias_sf",
    "bias_sf_dtype", "bias_zp",
    # 激活前中间态（只多输入算子有）
    "Relu_input", "Relu_input_dtype", "Relu_input_sf", "Relu_input_sf_dtype",
    "Relu_input_zp",
})

# 卷积/池化几何。llama2 用不到（没有卷积），但字段族在参考产物里存在。
# 接 CV 模型时由 `#pim.window` 直接映射。
PENDING_CONVOLUTION = frozenset({
    "kernel_shape", "strides", "pads", "dilations", "group", "output_padding",
    "axis", "axes",
})

# 显式声明不产出，各有理由。
NOT_APPLICABLE = {
    # PDF 注明由对方 L2Analyzer 自行推导
    "from_tvm": "参考产物来自 TVM 前端的标记，本方案不经 TVM",
    "original_name": "ONNX 原名，本方案从 FX 图导出，没有 ONNX 名",
    "idx": "输入序号，已由 input{i}_node_id 表达",
    # 调试副本：Debug 模式才写，正式产物不需要
    "lut_debug": "调试副本",
    # 哈希校验：可选的完整性校验，非必需
    "weight_buffer_hash": "可选完整性校验",
    "bias_buffer_hash": "可选完整性校验",
    "Relu_input_hash": "可选完整性校验",
    "DEBUG_Relu_input_hash": "调试副本",
    # 特殊算子标记
    "A": "MatMul_input_as_weight，由 pim.matmul 的 bIsActivation 表达",
    "clip_to_relu": "Clip 转 Relu 的标记，由 #pim.act_spec 的 relu_x 表达",
    "contraction": "融合块本身，已按结构产出",
    "subnetwork": "PDF 标 irrelevant, not used by NGC currently",
    "link_node": "PDF 未说明用途，参考产物中不存在",
}


def all_declared() -> set[str]:
    """全部已表态的字段族。"""
    return (set(EMITTED) | set(PENDING_QUANTIZATION)
            | set(PENDING_CONVOLUTION) | set(NOT_APPLICABLE))


def undeclared(reference_families: set[str]) -> set[str]:
    """参考产物里有、但我们没表态的字段族。

    非空即说明声明不完整——这是要修的，不是可忽略的。调试副本（`DEBUG*`）单独
    排除：它们只在 Debug 模式产出，正式流程不涉及。
    """
    return {
        family
        for family in reference_families - all_declared()
        if not family.startswith("DEBUG")
    }
