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
    # 本节点输出挂在消费者的第几个输入端口。参考产物 197 个节点各带一个，
    # 规则是 consumer.input<idx>_node_id == self.node_id。它不是「第几个
    # 输入」——每个节点只有一个，取值 0..31 是 head 序号。
    "idx",
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
    # 矩阵单元 A 侧操作数的节点号。与 `input0_node_id` 同值（实测 71/71
    # 相等），但参考两个名字都写，按 `A` 取值的读者只认这个名字。
    "A",
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
    # `pim.gather` 的两个操作数。这一族只在**全模型**那条导出里出现：图入口是
    # input_ids 时才发 `Gather`；decode 块以隐藏态为入口、词嵌入在块外，不发。
    "table", "indices",
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
    # 权值 bin 的内容指纹，用于跨节点权值去重。参考产物里 73 个带
    # `weight_buffer` 的节点全都写它——**是按节点必填的，不是可选校验**，
    # 所以它曾经被归进 NOT_APPLICABLE 是错的。由 `WrittenFiles._write`
    # 写盘时算出，`fill_weight_hashes` 回填。
    "weight_buffer_hash",
    # llama2 v2 上已经产出、原先只按 ResNet50 建表时漏掉的族。
    "is_mask",
    "axis", "axes",
    "num_heads",
    "dq_contraction",
    "weight_sf_multiplier",
    "updates_sf", "updates_sf_dtype", "updates_zp",
    "use_input_buffer",
    "LUT_phase",
    "flp_min_exp", "flp_max_exp", "flp_mantisa",
    "flp_min_exp_phase", "flp_max_exp_phase", "flp_mantisa_phase",
    "activation_mode_phase", "activation_special_operators_phase",
    "fpsu_mode_phase", "fpsu_spc_phase", "fpsu_spc_axis_phase",
    "fpsu_spg_phase", "fpsu_spg_axis_phase", "fpsu_spg_group_size_phase",
    "pooling_dtype_phase",
    "kantor_mode_phase",
    "nmu_output_type_phase",
    "global_pooling_spc_phase", "global_pooling_spc_axis_phase",
    "global_pooling_spg_phase", "global_pooling_spg_axis_phase",
    "global_pooling_group_size_phase",
    "input_buffer_phase", "input_buffer_dtype_phase",
    "input_data_extensions_phase",
    "output_buffer_phase", "output_buffer_dtype_phase",
    "output_data_extensions_phase",
    "Scaling_buffer_phase", "Scaling_PS_buffer_phase", "Bias_buffer_phase",
    "kantor_A_spc_phase", "kantor_A_spg_phase",
    "kantor_A_spg_axis_phase", "kantor_A_spg_group_size_phase",
    "kantor_A_scale_axis_phase",
    "kantor_A_scale_buffer_file_phase",
    "kantor_A_bias_buffer_file_phase",
    "kantor_A_Shift_buffer_file_phase",
    "kantor_B_spc", "kantor_B_spg", "kantor_B_Shift",
    "kantor_B_scale_axis", "kantor_B_scale_buffer_file",
    "kantor_B_bias_buffer_file",
    "cos_mul_output", "cos_mul_output_dtype",
    "sin_mul_output", "sin_mul_output_dtype",
    "Llama2Activation_Add_Cos_sf", "Llama2Activation_Add_Cos_sf_dtype",
    "Llama2Activation_Add_Cos_zp",
    "Llama2Activation_Add_Sin_sf", "Llama2Activation_Add_Sin_sf_dtype",
    "Llama2Activation_Add_Sin_zp",
    "Llama2Activation_Cos_Broadcast_sf", "Llama2Activation_Cos_Broadcast_sf_dtype",
    "Llama2Activation_Cos_Broadcast_zp",
    "Llama2Activation_Cos_sf", "Llama2Activation_Cos_sf_dtype",
    "Llama2Activation_Cos_zp",
    "Llama2Activation_Sin_Broadcast_sf", "Llama2Activation_Sin_Broadcast_sf_dtype",
    "Llama2Activation_Sin_Broadcast_zp",
    "Llama2Activation_Sin_sf", "Llama2Activation_Sin_sf_dtype",
    "Llama2Activation_Sin_zp",
    "fpsu_mode_Llama2Activation",
    "pooling_dtype_Llama2Activation",
    "Scaling_buffer_file_Llama2Activation",
    "Scaling_PS_buffer_file_Llama2Activation",
    "kantor_mode_Llama2Activation",
    "Kantor_A_spc_Llama2Activation", "Kantor_A_spg_Llama2Activation",
    "Kantor_A_Shift_Llama2Activation",
    "Kantor_A_spg_axis_Llama2Activation",
    "Kantor_A_spg_group_size_Llama2Activation",
    "Kantor_B_spc_Llama2Activation", "Kantor_B_spg_Llama2Activation",
    "Kantor_B_Shift_Llama2Activation",
    "Kantor_B_spg_axis_Llama2Activation",
    "Kantor_B_spg_group_size_Llama2Activation",
    "fpsu_1_scale_axis", "fpsu_2_scale_axis", "fpsu_3_scale_axis",
    "fpsu_4_scale_axis", "fpsu_5_scale_axis", "fpsu_6_scale_axis",
    "input2_node_id",
    "input_0_sf", "input_0_sf_dtype", "input_0_zp",
    "input_buffer_0", "input_buffer_0_dtype",
    "input_1_sf", "input_1_sf_dtype", "input_1_zp",
    "input_buffer_1", "input_buffer_1_dtype",
    "input_2_sf", "input_2_sf_dtype", "input_2_zp",
    "input_buffer_2", "input_buffer_2_dtype",
    "input2_node_id", "output2_node_id",
    "input_3_sf", "input_3_sf_dtype", "input_3_zp",
    "input_buffer_3", "input_buffer_3_dtype",
    "input3_node_id", "output3_node_id",
    "input_4_sf", "input_4_sf_dtype", "input_4_zp",
    "input_buffer_4", "input_buffer_4_dtype",
    "input4_node_id", "output4_node_id",
    "input_5_sf", "input_5_sf_dtype", "input_5_zp",
    "input_buffer_5", "input_buffer_5_dtype",
    "input5_node_id", "output5_node_id",
    "input_6_sf", "input_6_sf_dtype", "input_6_zp",
    "input_buffer_6", "input_buffer_6_dtype",
    "input6_node_id", "output6_node_id",
    "input_7_sf", "input_7_sf_dtype", "input_7_zp",
    "input_buffer_7", "input_buffer_7_dtype",
    "input7_node_id", "output7_node_id",
    "input_8_sf", "input_8_sf_dtype", "input_8_zp",
    "input_buffer_8", "input_buffer_8_dtype",
    "input8_node_id", "output8_node_id",
    "input_9_sf", "input_9_sf_dtype", "input_9_zp",
    "input_buffer_9", "input_buffer_9_dtype",
    "input9_node_id", "output9_node_id",
    "input_10_sf", "input_10_sf_dtype", "input_10_zp",
    "input_buffer_10", "input_buffer_10_dtype",
    "input10_node_id", "output10_node_id",
    "input_11_sf", "input_11_sf_dtype", "input_11_zp",
    "input_buffer_11", "input_buffer_11_dtype",
    "input11_node_id", "output11_node_id",
    "input_12_sf", "input_12_sf_dtype", "input_12_zp",
    "input_buffer_12", "input_buffer_12_dtype",
    "input12_node_id", "output12_node_id",
    "input_13_sf", "input_13_sf_dtype", "input_13_zp",
    "input_buffer_13", "input_buffer_13_dtype",
    "input13_node_id", "output13_node_id",
    "input_14_sf", "input_14_sf_dtype", "input_14_zp",
    "input_buffer_14", "input_buffer_14_dtype",
    "input14_node_id", "output14_node_id",
    "input_15_sf", "input_15_sf_dtype", "input_15_zp",
    "input_buffer_15", "input_buffer_15_dtype",
    "input15_node_id", "output15_node_id",
    "input_16_sf", "input_16_sf_dtype", "input_16_zp",
    "input_buffer_16", "input_buffer_16_dtype",
    "input16_node_id", "output16_node_id",
    "input_17_sf", "input_17_sf_dtype", "input_17_zp",
    "input_buffer_17", "input_buffer_17_dtype",
    "input17_node_id", "output17_node_id",
    "input_18_sf", "input_18_sf_dtype", "input_18_zp",
    "input_buffer_18", "input_buffer_18_dtype",
    "input18_node_id", "output18_node_id",
    "input_19_sf", "input_19_sf_dtype", "input_19_zp",
    "input_buffer_19", "input_buffer_19_dtype",
    "input19_node_id", "output19_node_id",
    "input_20_sf", "input_20_sf_dtype", "input_20_zp",
    "input_buffer_20", "input_buffer_20_dtype",
    "input20_node_id", "output20_node_id",
    "input_21_sf", "input_21_sf_dtype", "input_21_zp",
    "input_buffer_21", "input_buffer_21_dtype",
    "input21_node_id", "output21_node_id",
    "input_22_sf", "input_22_sf_dtype", "input_22_zp",
    "input_buffer_22", "input_buffer_22_dtype",
    "input22_node_id", "output22_node_id",
    "input_23_sf", "input_23_sf_dtype", "input_23_zp",
    "input_buffer_23", "input_buffer_23_dtype",
    "input23_node_id", "output23_node_id",
    "input_24_sf", "input_24_sf_dtype", "input_24_zp",
    "input_buffer_24", "input_buffer_24_dtype",
    "input24_node_id", "output24_node_id",
    "input_25_sf", "input_25_sf_dtype", "input_25_zp",
    "input_buffer_25", "input_buffer_25_dtype",
    "input25_node_id", "output25_node_id",
    "input_26_sf", "input_26_sf_dtype", "input_26_zp",
    "input_buffer_26", "input_buffer_26_dtype",
    "input26_node_id", "output26_node_id",
    "input_27_sf", "input_27_sf_dtype", "input_27_zp",
    "input_buffer_27", "input_buffer_27_dtype",
    "input27_node_id", "output27_node_id",
    "input_28_sf", "input_28_sf_dtype", "input_28_zp",
    "input_buffer_28", "input_buffer_28_dtype",
    "input28_node_id", "output28_node_id",
    "input_29_sf", "input_29_sf_dtype", "input_29_zp",
    "input_buffer_29", "input_buffer_29_dtype",
    "input29_node_id", "output29_node_id",
    "input_30_sf", "input_30_sf_dtype", "input_30_zp",
    "input_buffer_30", "input_buffer_30_dtype",
    "input30_node_id", "output30_node_id",
    "input_31_sf", "input_31_sf_dtype", "input_31_zp",
    "input_buffer_31", "input_buffer_31_dtype",
    "input31_node_id", "output31_node_id",
    "fpsu_1_spc_Llama2Activation_Add_Cos",
    "fpsu_1_spg_Llama2Activation_Add_Cos",
    "fpsu_1_spg_axis_Llama2Activation_Add_Cos",
    "fpsu_1_spg_group_size_Llama2Activation_Add_Cos",
    "fpsu_2_spc_Llama2Activation_Add_Sin",
    "fpsu_2_spg_Llama2Activation_Add_Sin",
    "fpsu_2_spg_axis_Llama2Activation_Add_Sin",
    "fpsu_2_spg_group_size_Llama2Activation_Add_Sin",
    "fpsu_3_spc_Llama2Activation_Sin",
    "fpsu_3_spg_Llama2Activation_Sin",
    "fpsu_3_spg_axis_Llama2Activation_Sin",
    "fpsu_3_spg_group_size_Llama2Activation_Sin",
    "fpsu_4_spc_Llama2Activation_Sin",
    "fpsu_4_spg_Llama2Activation_Sin",
    "fpsu_4_spg_axis_Llama2Activation_Sin",
    "fpsu_4_spg_group_size_Llama2Activation_Sin",
    "fpsu_5_spc_Llama2Activation_Cos",
    "fpsu_5_spg_Llama2Activation_Cos",
    "fpsu_5_spg_axis_Llama2Activation_Cos",
    "fpsu_5_spg_group_size_Llama2Activation_Cos",
    "fpsu_6_spc_Llama2Activation_Cos",
    "fpsu_6_spg_Llama2Activation_Cos",
    "fpsu_6_spg_axis_Llama2Activation_Cos",
    "fpsu_6_spg_group_size_Llama2Activation_Cos",
    "Kantor_A_Llama2Activation_Cos_bias_buffer_file",
    "Kantor_A_Llama2Activation_Sin_bias_buffer_file",
    "Kantor_A_Llama2Activation_add_bias_buffer_file",
    "Kantor_A_Llama2Activation_add_scale_buffer_file",
    "Kantor_B_Llama2Activation_Cos_bias_buffer_file",
    "Kantor_B_Llama2Activation_Cos_scale_buffer_file",
    "Kantor_B_Llama2Activation_Sin_bias_buffer_file",
    "Kantor_B_Llama2Activation_Sin_scale_buffer_file",
})

# 第 4 轮（静态量化）要补的字段族。量化参数、定标系数、LUT 都依赖校准数据，
# 结构轮拿不到。
PENDING_QUANTIZATION = frozenset({
    # 输入/权重/偏置/输出的量化参数
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
})

# 显式声明不产出，各有理由。
NOT_APPLICABLE = {
    # PDF 注明由对方 L2Analyzer 自行推导
    "from_tvm": "参考产物来自 TVM 前端的标记，本方案不经 TVM",
    "original_name": "ONNX 原名，本方案从 FX 图导出，没有 ONNX 名",
    # 输入序号，已由 input{i}_node_id 表达。`idx` 已移入 EMITTED：它不是
    # 输入序号，而是本节点输出挂在消费者的第几个输入端口。
    # 调试副本：Debug 模式才写，正式产物不需要
    "lut_debug": "调试副本",
    # 其余 hash 字段参考产物里没有对应物，仍不产出。
    "bias_buffer_hash": "可选完整性校验",
    "Relu_input_hash": "可选完整性校验",
    "DEBUG_Relu_input_hash": "调试副本",
    # 特殊算子标记

    "clip_to_relu": "Clip 转 Relu 的标记，由 #pim.act_spec 的 relu_x 表达",
    "contraction": "融合块本身，已按结构产出",
    "subnetwork": "PDF 标 irrelevant, not used by NGC currently",
    "link_node": "PDF 未说明用途，参考产物中不存在",
    "cos_mul_output_hash": "参考调试指纹，本方案不经对方哈希口径",
    "sin_mul_output_hash": "参考调试指纹，本方案不经对方哈希口径",
    # 带 DEBUG 前缀但不是数据转储的语义标记。前缀豁免只放行 *_float / *_hash，
    # 这些必须显式表态，否则一个新的语义字段会被前缀悄悄放过。
    "DEBUG_div_value": "参考调试值，本方案不产调试副本",
    "DEBUG_silu_input": "参考调试副本，本方案不产",
    "DEBUG_silu_input_dtype": "参考调试副本，本方案不产",
    "DEBUG_silu_input_sf": "参考调试副本，本方案不产",
    "DEBUG_silu_input_sf_dtype": "参考调试副本，本方案不产",
    "DEBUG_silu_input_zp": "参考调试副本，本方案不产",
    "DEBUG_sub_normal_weights_sf": "次正规标记，语义已由 weight_sf_multiplier 表达",
    "DEBUG_weight_buffer_spc": "权值定标粒度的调试副本，正式字段是 fpsu_spc",
    "DEBUG_weight_buffer_spc_axis": "权值定标粒度的调试副本",
    "DEBUG_weight_buffer_spg": "权值定标粒度的调试副本",
    "DEBUG_weight_buffer_spg_axis": "权值定标粒度的调试副本",
    "DEBUG_weight_buffer_spg_group_size": "权值定标粒度的调试副本",
    "residual_input_buffer_": "键名归一化噪声，真实字段是 residual_input_buffer",
    "residual_output_buffer_": "键名归一化噪声，真实字段是 residual_output_buffer",
}


def all_declared() -> set[str]:
    """全部已表态的字段族。"""
    return (set(EMITTED) | set(PENDING_QUANTIZATION)
            | set(PENDING_CONVOLUTION) | set(NOT_APPLICABLE))


def undeclared(reference_families: set[str]) -> set[str]:
    """参考产物里有、但我们没表态的字段族。

    非空即说明声明不完整——这是要修的，不是可忽略的。`DEBUG` 前缀里只有
    数据转储（`*_float` 浮点副本、`*_hash` 指纹）可以放过；其余即使带
    `DEBUG` 前缀也是语义标记（如 `DEBUG_sub_normal_weights_sf`），必须表态。
    """
    return {
        family
        for family in reference_families - all_declared()
        if not (family.endswith("_float") or family.endswith("_hash"))
    }
