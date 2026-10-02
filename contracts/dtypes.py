"""元素类型名的唯一真源。

本轮不扩充**计算类型**（需求 §2.3）：不引入 bf16、fp8、亚字节打包。

int4 记 1 字节是**实测事实**而非疏漏：weight_buffer 的字节数等于权重元素数，
值域严格落在 [-8,7]，高 4 位是符号扩展（见 `contracts/gml_quant.py`）。
PyTorch 侧这些张量的 dtype 实际是 int8，所以 `element_size()` 也返回 1 ——
两者一致，这是把 `element_size()` 换成 `dtype_bytes()` 的行为等价前提。
"""

# 参与计算与落盘的定点/浮点类型。
ELEMENT_DTYPES = frozenset({"int4", "int8", "int16", "int32", "float16", "float32"})

# 索引张量的类型。它们不参与数值计算，但确实出现在图上（input_ids 一路、
# 位置下标），且**占内存**：内存规划要对它们算字节数。不补进真源的话，
# element_size() 改查 dtype 就会在这几个节点上抛错，不是等价重构。
INDEX_DTYPES = frozenset({"int64"})

_DTYPE_BYTES = {"int4": 1, "int8": 1, "int16": 2,
                "int32": 4, "float16": 2, "float32": 4, "int64": 8}


def validate_dtype(name: str) -> None:
    """不在集合内直接抛错（不写防御性兜底）。"""
    if name not in ELEMENT_DTYPES and name not in INDEX_DTYPES:
        raise ValueError(
            f"未知元素类型 {name!r}，允许 "
            f"{sorted(ELEMENT_DTYPES | INDEX_DTYPES)}")


def dtype_bytes(name: str) -> int:
    """单个元素占几字节。取代散落各处的 `.meta["val"].element_size()`。"""
    validate_dtype(name)
    return _DTYPE_BYTES[name]
