"""算子级 mnemonic 的 numpy 镜像。

编译内核（`opcompiler_bridge.driver.compile_op`）与这里的镜像逐元素对拍：
两边算的是同一个算子，一边是算子编译器降级出来的 C，一边是 numpy。

**公式只认 `gml_bridge/phase_data.py`**，不在本模块里另写一份。那是这些相位的
数值真源（每个常量都在参考产物上逐字节验证过），再抄一份就一定会有两份不一致
的时候，而且分不出谁对。

命名跟 mnemonic 走，与 `runtime/kernels.py` 的 aten 内核分开：那些按 aten 目标
注册给 HAL，这些按算子级名字供对拍。
"""

from __future__ import annotations

import numpy as np

from gml_bridge import phase_data


def softmax(source: np.ndarray) -> np.ndarray:
    """`pim.softmax`：沿最后一维，fp16 存储、fp32 计算。

    逐行算而不是整体算：`phase_data.softmax` 是**一个节点**的相位链，而一个
    节点就是一个头的分数行——压平会让归约跨行。
    """
    array = np.ascontiguousarray(source, dtype=np.float16)
    rows = array.reshape(-1, array.shape[-1])
    return np.stack([phase_data.softmax(row).phase4
                     for row in rows]).reshape(array.shape)


def dynamic_quant(source: np.ndarray, group_size: int) -> np.ndarray:
    """`pim.dynamic_quant`：分组动态量化到 int8，返回量化结果。

    相 1 的分组缩放是这条链的第二个产物（GML 的 `output_buffer_phase_1`），
    由下游节点按缓冲读；本镜像只返回主结果。
    """
    phases = phase_data.dynamic_scaling(source, group_size=group_size)
    return phases.phase3.reshape(np.shape(source))


def gather(table: np.ndarray, indices: np.ndarray) -> np.ndarray:
    """`pim.gather`：按索引取表的行。

    就是 `table[indices]`。这里不做越界钳位——越界索引是上游的错，钳掉它会把
    一个本该暴露的 bug 变成一行看起来正常的数据。
    """
    return np.asarray(table)[np.asarray(indices)]


def rope(source: np.ndarray, cos: np.ndarray, sin: np.ndarray) -> np.ndarray:
    """`pim.rope`：`x*cos + rotate_half(x)*sin`，按三相各自落 fp16。

    **逐相落 fp16 不是可选的**：硬件跑三次引擎遍历（乘 cos / 乘 sin / 相加），
    中间结果真的落进缓冲。全程 f32 算完再截，在 1/3 的元素上会与编译内核差一个
    fp16 ulp——那不是内核错了，是这个镜像少模拟了两次落盘。

    `rotate_half(x) = cat(-后半, 前半)`，负号随被换到前面的那一半。
    """
    src = np.ascontiguousarray(source, dtype=np.float32)
    half = src.shape[-1] // 2
    rotated = np.concatenate((-src[..., half:], src[..., :half]), axis=-1)

    # 相 0 与相 1 各落一次 fp16，相 2 读回它们再相加。
    phase0 = (src * np.asarray(cos, dtype=np.float32)).astype(np.float16)
    phase1 = (rotated * np.asarray(sin, dtype=np.float32)).astype(np.float16)
    return (phase0.astype(np.float32)
            + phase1.astype(np.float32)).astype(np.float16)


def _apply_activation(x: np.ndarray, activation: str | None) -> np.ndarray:
    if not activation:
        return x
    kind = str(activation).lower()
    xf = x.astype(np.float32)
    if kind == "silu":
        # 闭式，与编译内核的 pim_lut_silu 同一条公式。
        # 288 B 分段线性表在 32 层里累积后，整网 logits 与 torch 对不上。
        # |x| 很大时 exp 溢出，fp32 里先夹到闭式已经饱和的位置。
        z = np.clip(-xf, -80.0, 80.0)
        return (xf / (1.0 + np.exp(z))).astype(x.dtype)
    if kind == "relu":
        return np.maximum(xf, 0).astype(x.dtype)
    if kind == "gelu":
        return (0.5 * xf * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (xf + 0.044715 * xf**3)))).astype(x.dtype)
    raise ValueError(f"没有 {kind!r} 的 numpy 镜像")


def matmul(a: np.ndarray, w: np.ndarray, *, group_size: int | None = None,
           scales: np.ndarray | float = 1.0,
           activation: str | None = None) -> np.ndarray:
    """整算子矩阵乘，int4 权值，结果饱和到 int8。

    与 C 侧的 `pim.matmul` 分支同一个函数：整数乘、f32 累加，每 `group_size`
    个 K 把这一组的和按**这一组自己的定标**反量化一次再折进总数，最后按 RNE
    取整并饱和。

    分组时 `scales` 是 `[K/group_size, N]`：一个组、一列一个因子。
    **不能是标量**（评审 20260923 的 P1-4）：标量乘在组边界与乘在整行末尾在
    精确算术里相等，于是「按组反量化」与「整段累加完再反量化」给出同一个数，
    方案 §5.15 那条判据就无论实现对错都通过。逐组不同的因子才让两种顺序算出
    不同的数，这条判据也才真的能失败。

    不分组时 `scales` 是标量：整段 K 一个组，那个因子是权值绑定的存储补偿
    （`sfMultiplier`），不是逐组量。
    """
    x = a.astype(np.float32)
    weight = w.astype(np.float32)
    k = x.shape[1]
    width = k if group_size is None else group_size
    if k % width:
        raise ValueError(f"K={k} 不是组宽 {width} 的整数倍，最后一组会短一截")
    groups = k // width
    grouped = x.reshape(x.shape[0], groups, width)
    wg = weight.reshape(groups, width, weight.shape[1])
    # 每组各自的和：f32 下整数乘加是精确的，所以先求和再乘这一组的因子，
    # 与 C 侧「组内累加、边界反量化」逐位一致。
    per_group = np.einsum("mgk,gkn->gmn", grouped, wg)
    factors = np.asarray(scales, dtype=np.float32)
    if factors.ndim == 0:
        total = (per_group * factors).sum(axis=0)
    else:
        if factors.shape != (groups, weight.shape[1]):
            raise ValueError(
                f"逐组定标要 [K/group_size, N] = "
                f"[{groups}, {weight.shape[1]}]，收到 {factors.shape}")
        # `factors[g, n]` 乘到第 g 组、第 n 列上——这就是「按组反量化」与
        # 「整段反量化」分道的那一步。
        total = (per_group * factors[:, None, :]).sum(axis=0)
    # 激活在 f32 域做，量化只在最后落一次。**不能先取整再激活**：那会多取整
    # 一次，与编出来的 C 分道——C 是 `pim_f32_to_i8(激活(累加值))`，激活落在
    # 取整之前。实测累加值 0.5337 处，先激活得 silu(0.5337)=0.336→0，
    # 先取整则 rint(0.5337)=1、silu(1)=0.731→1，差一个整数。
    if activation:
        total = _apply_activation(total.astype(np.float32), activation)
    return np.clip(np.rint(total), -128, 127).astype(np.int8)


def normalize(source: np.ndarray, gamma: np.ndarray,
              epsilon: float = 1e-5) -> np.ndarray:
    """RMS 归一化：平方、沿末轴求均、加 ε、rsqrt、乘 γ。

    与 C 侧 `pim.normalize` 分支同一个函数。**不减均值**——那是层归一化，
    是另一个算子，多减一次均值不会报错，只会让每个数都偏一点。
    """
    x = source.astype(np.float32)
    mean_sq = np.mean(x * x, axis=-1, keepdims=True)
    inv = (1.0 / np.sqrt(mean_sq + np.float32(epsilon))).astype(np.float32)
    return (x * inv * gamma.astype(np.float32)).astype(source.dtype)


def mask(scores: np.ndarray, mask_values: np.ndarray) -> np.ndarray:
    """加性掩码：分数 + 掩码，按末轴广播。

    掩码在浮点域里是**加性偏置**不是布尔：被掩掉的位置带的是负无穷，所以
    起作用的是这个加法，而不是某个 select。
    """
    return (scores.astype(np.float32) + mask_values.astype(np.float32)).astype(
        scores.dtype)



def transpose(source: np.ndarray, axes: tuple[int, ...]) -> np.ndarray:
    """按轴序重排。与 C 侧同一个结果：走结果坐标、映射回源。"""
    return np.transpose(source, axes)


def reshape(source: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    """改形状，元素顺序不变。"""
    return source.reshape(shape)


def concat(sources: list[np.ndarray], axis: int) -> np.ndarray:
    """沿一根轴把若干张量接起来。"""
    return np.concatenate(sources, axis=axis)


def convert(source: np.ndarray, dtype) -> np.ndarray:
    """只换元素类型，不带缩放、不带偏置。int8 侧同样饱和。"""
    return np.clip(np.rint(source.astype(np.float32)), -128, 127).astype(
        dtype) if dtype == np.int8 else source.astype(dtype)
