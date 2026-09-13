"""让算子编译在没有 GPU 硬件的机器上照常产出 pim mlir。

## 为什么需要这个模块

算子编译这条链的产物是 **pim mlir**（再往下是 EmitC → C → .so），全都在 CPU 上跑；
GPU 从来不参与生成，只是 Triton 前端的既有实现顺手依赖了它：

    driver.py 原来的做法
        torch.empty(..., device="cuda") ×3  →  真实 launch  →  compiled.asm["ttir"]

真实 launch 只是为了拿到 TTIR 文本。而 TTIR 是纯前端产物（AST → IR），不需要设备。

## 无卡时到底卡在哪

卡点不在"编译需要算力"，而在两处**设备探测**：

1. `triton/runtime/driver.py:_create_driver()` 要求恰好一个 active driver，而
   `backends/nvidia/driver.py` 的 `is_active()` 直接返回 `torch.cuda.is_available()`。
   无卡时是 0 个，抛 "0 active drivers"。
2. Triton 的 hint manager 和 FlagGems 的 `triton_driver_helper` 在 **import 期**
   就摸 `driver.active`，于是连 import 都过不去。

两处都只是"问一下当前设备是什么"，没有一处真的要跑 kernel。所以注入一个只回答这几个
问题的 driver 就够了——不是绕过校验，而是把"编译期不需要设备"这件事说清楚。

## 与有卡产物的一致性

`tt.divisibility = 16` 这个属性来自真实张量指针的对齐推断（`torch.empty` 返回的指针
是 16 字节对齐）。无卡时没有真实张量，必须显式声明同样的特化，否则产出的 pim mlir 会
与有卡路径差这一行——而 sidecar 里记的是内容哈希，差一行就对不上，属于会静默偏掉的
那类问题。`make_ttir` 里的 `_DIVISIBILITY_16` 就是补这个。

实测：同一形状（4096→4096）、同一硬件参数下，本模块产出的 pim mlir 与有 GPU 时
`.opcompiler_cache` 里的产物 **sha256 完全相同**。
"""

from __future__ import annotations

import functools
import os
from typing import Any, Dict, Tuple

# 真实 launch 时 torch.empty 的指针是 16 字节对齐，Triton 据此给每个指针参数打
# tt.divisibility=16。无卡路径没有真实张量，显式声明同样的特化。
_DIVISIBILITY_16 = [["tt.divisibility", 16]]

# 前端编译用的目标。计算能力只影响 TTGIR 之后的阶段（我们不走那条路），但
# ASTSource.make_ir 需要一个具体 target 才能选 codegen 实现。sm80 是 PIM pass
# 验证过的取值。
_FRONTEND_ARCH = 80
_FRONTEND_WARP_SIZE = 32


def gpu_hardware_present() -> bool:
    """当前机器上有可用的 GPU 硬件吗。

    仅用于决定"走原生 launch 还是走无卡前端路径"，不作为可用性校验——两条路径产出
    的 pim mlir 是一致的（见模块 docstring）。
    """
    try:
        import torch
    except ImportError:
        return False
    return torch.cuda.is_available()


# 编译期会被问到的设备属性。取值只影响 Triton 的共享内存上限校验等前端检查，
# 不进入 pim mlir——PIM 的分块由 `-pim-tile-to-budget` 按 WRAM 预算独立决定。
# 按 sm80 的真实参数填，避免前端因为"共享内存为 0"而拒绝合法的分块。
_SM80_PROPERTIES = {
    "max_shared_mem": 166912,
    "multiprocessor_count": 108,
    "sm_clock_rate": 1410000,
    "mem_clock_rate": 1215000,
    "mem_bus_width": 5120,
}


class _CpuHostUtils:
    """回答编译期的设备属性查询，不加载 CUDA 运行时。

    真实的 `CudaUtils` 会在构造时编译一个 C 扩展并链接 `libcuda.so.1`。纯 CPU 机器上
    那个库根本不存在（`libcuda.so cannot found!`），所以这里不能复用它——这也是为什么
    本模块不继承 `CudaDriver`：它的 `__init__` 第一行就是 `CudaUtils()`。
    """

    @staticmethod
    def get_device_properties(device: Any = None) -> Dict[str, int]:
        return dict(_SM80_PROPERTIES)

    @staticmethod
    def load_binary(*args: Any, **kwargs: Any):
        raise RuntimeError(
            "纯 CPU 环境不能加载 GPU kernel 二进制。算子编译只需要 TTIR → pim mlir，"
            "不需要这一步；走到这里说明有人试图在无 GPU 的机器上启动 Triton kernel。"
        )


def _cpu_host_driver_class():
    """构造一个"只回答编译期查询、完全不碰 CUDA 运行时"的 driver。

    **不继承 `CudaDriver`**：它的 `__init__` 会构造 `CudaUtils()`，后者要找
    `libcuda.so.1`。纯 CPU 机器上那个库不存在，构造就 assert 失败——这一点用
    `CUDA_VISIBLE_DEVICES=""` 测不出来（那只是让 `cuda.is_available()` 返回 False，
    驱动和 libcuda 仍在），必须在真正没有 GPU 的环境里才会暴露。

    继承 `DriverBase` 并只实现编译期真正会被读到的成员。这份清单是从 Triton 和
    FlagGems 的源码里逐个 grep 出来的：

        driver.active.get_active_torch_device    hint_manager
        driver.active.utils.get_device_properties compiler / flag_gems
        driver.active.get_current_device          compiler
        driver.active.get_current_stream          compiler
        driver.active.get_current_target          compiler / autotuner
        driver.active.get_benchmarker             autotuner
        driver.active.launcher_cls                运行 kernel 时才用
        driver.active.current_arch_id             flag_gems

    `launcher_cls` 和 `get_benchmarker` 只在真正启动 kernel 时才被读，纯编译路径不会
    走到；保留是为了让"误用"能报出清楚的错，而不是 AttributeError。
    """
    from triton.backends.compiler import GPUTarget
    from triton.backends.driver import DriverBase

    class CpuHostDriver(DriverBase):
        """无 GPU 机器上的编译期 driver。"""

        def __init__(self) -> None:
            self.utils = _CpuHostUtils()
            # 与 `make_ttir` 用的前端目标保持一致，否则 autotuner 的缓存键和
            # 编译时的 target 会对不上。
            self._target = GPUTarget("cuda", _FRONTEND_ARCH, _FRONTEND_WARP_SIZE)

        @staticmethod
        def is_active() -> bool:
            # 只通过 set_active 显式注入，不参与 _create_driver 的自动探测——
            # 否则有卡机器上会变成"两个 active driver"。
            return False

        def get_current_target(self) -> Any:
            return self._target

        def get_active_torch_device(self):
            import torch

            return torch.device("cpu")

        def get_current_device(self) -> int:
            return 0

        def set_current_device(self, device: Any) -> None:
            return None

        def get_current_stream(self, device: Any = None) -> int:
            return 0

        def get_device_capability(self, device: Any = None) -> Tuple[int, int]:
            return (_FRONTEND_ARCH // 10, _FRONTEND_ARCH % 10)

        def map_python_to_cpp_type(self, ty: str) -> str:
            # DriverBase 的抽象方法，生成 launcher 的 C 签名时才用到。纯编译路径
            # 不生成 launcher，沿用 nvidia backend 的映射即可（纯字符串表，不碰驱动）。
            from triton.backends.nvidia.driver import ty_to_cpp

            return ty_to_cpp(ty)

        def get_device_interface(self):
            import torch

            return torch.cpu

        @property
        def current_arch_id(self) -> int:
            # FlagGems 用它做 autotune 配置的索引，纯编译路径下取值无实质影响。
            return _FRONTEND_ARCH

        @property
        def launcher_cls(self):
            raise RuntimeError(
                "纯 CPU 环境没有 kernel launcher。算子编译只需要 TTIR → pim mlir；"
                "走到这里说明有人试图在无 GPU 的机器上启动 Triton kernel。"
            )

        def get_benchmarker(self):
            raise RuntimeError(
                "纯 CPU 环境不能 benchmark GPU kernel。autotune 需要真实执行，"
                "无卡机器上不可用。"
            )

    return CpuHostDriver


@functools.lru_cache(maxsize=1)
def ensure_compile_driver() -> bool:
    """无卡时把编译期 driver 注入 Triton；返回是否做了注入。

    必须在 `import flag_gems` 和任何 `triton.compile` 之前调用：那些模块在 import
    期就会读 `driver.active`，一旦读到就会缓存住 `_create_driver()` 的结果（无卡时
    是异常）。`set_active` 是 Triton 的公开接口，正是为替换 driver 准备的。

    同时替 FlagGems 越过它自己那套设备探测：`flag_gems` 在 import 期构造
    `DeviceDetector()`，探测不到任何厂商设备就直接
    `raise RuntimeError("No device were detected on your machine !")`，import 就失败。
    它支持用 `GEMS_VENDOR` 指定厂商跳过探测，这里补上默认值。

    选 `nvidia` 而不是看起来更贴切的 `arm`（后者的 `device_name` 恰好就是 `"cpu"`）：
    `arm` vendor 的数学函数 shim 缺 `asin` 等符号，`import flag_gems` 会在
    `ops/arcsin.py` 挂掉。`nvidia` vendor 走 libdevice，纯编译路径不会真的调用它。

    有卡时什么都不做，保持原生行为。
    """
    if gpu_hardware_present():
        return False

    from triton.runtime import driver as driver_config

    driver_config.set_active(_cpu_host_driver_class()())
    # setdefault：使用者显式指定过就尊重他的选择。
    os.environ.setdefault("GEMS_VENDOR", "nvidia")
    return True


def make_ttir(
    kernel,
    signature: Dict[str, str],
    constexprs: Dict[str, int],
    *,
    num_stages: int,
) -> str:
    """不经过真实 launch，直接把一个 Triton kernel 前端编译成 TTIR 文本。

    产物与有卡路径的 `compiled.asm["ttir"]` 同口径：先 AST → IR，再跑 ttir stage
    的 pass（含 inliner）。少了后半步的话 TTIR 里还留着未内联的 `tt.call`，
    `convert-triton-to-pim` 会因为函数签名与 tasklet 布局不匹配而报错。
    """
    ensure_compile_driver()

    from triton._C.libtriton import ir
    from triton.backends.compiler import GPUTarget
    from triton.backends.nvidia.compiler import CUDABackend
    from triton.compiler import ASTSource

    target = GPUTarget("cuda", _FRONTEND_ARCH, _FRONTEND_WARP_SIZE)
    backend = CUDABackend(target)
    options = backend.parse_options({"num_stages": num_stages})

    context = ir.context()
    ir.load_dialects(context)
    backend.load_dialects(context)

    source = ASTSource(
        fn=kernel,
        signature=signature,
        constexprs=constexprs,
        attrs=_pointer_attrs(signature),
    )
    module = source.make_ir(
        target=target,
        options=options,
        codegen_fns=backend.get_codegen_implementation(options),
        module_map=backend.get_module_map(),
        context=context,
    )
    # 与 Triton 自己的 ttir stage 一致：内联 + 规范化，之后才是 PIM pass 的输入。
    return str(CUDABackend.make_ttir(module, {}, options, target.arch))


def _pointer_attrs(signature: Dict[str, str]) -> Dict[Tuple[int, ...], Any]:
    """给每个指针参数打上 16 字节对齐特化，对齐有卡路径的推断结果。"""
    return {
        (index,): _DIVISIBILITY_16
        for index, kind in enumerate(signature.values())
        if kind.startswith("*")
    }
