"""CEVA 链路的子图契约：I/O 绑定、导入文件绑定与整图规格。

只描述契约，不读文件；解析厂商子图在 `gml_bridge/ceva.py`。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from contracts.compile_slots import CompileSlots


@dataclass(frozen=True)
class CevaIoBinding:
    """runner 按名字绑定地址的一个图级 I/O。"""

    direction: Literal["input", "output"]
    index: int                      # IO_info 中的槽号
    name: str                       # 如 nprm_0_i3、tvmgen_default_nprm_main_0_output_1
    label: str | None               # runner 用的别名，如 key_cache_out；无则 None
    shape: tuple[int, ...]
    dtype: str
    sf: float


@dataclass(frozen=True)
class CevaGraphSpec:
    """一张子图的厂商契约。"""

    stage: Literal["prefill", "decode", "lm_head"]
    layer_id: int | None            # lm_head 为 None
    query_tokens: int               # prefill=1024，decode/head=1
    slots: CompileSlots
    version: str                    # 26.8.0 或 19.2.0，取自厂商同名 GML
    io: tuple[CevaIoBinding, ...]
    runtime_dir: Path               # 厂商同名子图目录，只用于核对结构与 runner 契约

    def io_for(self, name: str) -> CevaIoBinding:
        for binding in self.io:
            if binding.name == name or binding.label == name:
                return binding
        raise KeyError(f"{self.stage} 第 {self.layer_id} 层没有 I/O {name!r}")
