"""Shared keys and values used for FX graph metadata annotations."""

DEVICE_META_KEY = "device"
PART_ID_META_KEY = "part_id"
SPEC_META_KEY = "spec"
REDISTRIBUTE_META_KEY = "redistribute"
# 折进本节点的激活与池化，值是 FusedTail。GML 把它们放进 contraction 块。
FUSED_TAIL_META_KEY = "fused_tail"

DEVICE_DPU = "dpu"
DEVICE_HOST = "host"
