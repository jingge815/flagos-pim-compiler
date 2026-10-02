"""兼容层：键名与取值的定义已收归 `contracts/unified_ir.py`。

12 个既有 import 点不动，这里只做 re-export。新代码请直接从
`contracts.unified_ir` 引入，那里还有四维登记表与阶段协议。
"""

from contracts.unified_ir import (  # noqa: F401
    ABSORBED_META_KEY,
    ATTENTION_SCALE_META_KEY,
    DEVICE_DPU,
    DEVICE_HOST,
    DEVICE_META_KEY,
    DQ_META_KEY,
    FUSED_TAIL_META_KEY,
    HEAD_INDEX_META_KEY,
    HEAD_ROLE_META_KEY,
    KV_DMA_META_KEY,
    PART_ID_META_KEY,
    REDISTRIBUTE_META_KEY,
    RMS_NORM_META_KEY,
    ROPE_META_KEY,
    SPEC_META_KEY,
    SPLIT_META_KEY,
)
