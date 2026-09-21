"""层参数卡文本：有序字典 → `键: 值` 行。

对齐宽度不要求与参考字节级一致。多值键（同一键出现多次）用 list。
RoPE 的 force consecutive 与 skip compare 拆成两行，不复现参考里的粘连。

**行尾用 CRLF**：参考产物 422 个层文件无一例外都是 `\\r\\n`
（两个版本戳文件例外，它们连结尾换行都没有）。对方解析器按行切分时若把
`\\r` 当值的一部分，LF 版本会让每个值末尾多一个字符 —— 所以照参考写。
"""

from __future__ import annotations

from collections import OrderedDict


def _format_value(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        if value == int(value) and value in (0.0, 1.0):
            return "0.0" if value == 0 else "1.0"
        return repr(value)
    return str(value)


def render_layer_txt(fields: OrderedDict[str, object]) -> str:
    """一层 txt。值为 list 时按顺序重复键。

    `force consecutive execution` 后面紧跟 `skip compare` 时**粘成一行**，
    与参考一致：参考的 6 个 RoPE 文件都是
    `force consecutive execution: 1skip compare: 1`（域确认表 Q27）。
    拆成两行会让对方解析器少读一个域 —— 它按行切分，`skip compare` 那半行
    在参考里根本不是独立行，解析器的状态机可能没有处理它的分支。
    照参考写，把「是否容忍拆行」的问题留给甲方确认。
    """
    lines: list[str] = []
    pending_glue = False
    for key, value in fields.items():
        if key == "Dump files list":
            lines.append("Dump files list: ")
            continue
        if pending_glue and key == "skip compare":
            # 粘到上一行末尾，不另起一行。
            lines[-1] += f"{key}: {_format_value(value)}"
            pending_glue = False
            continue
        pending_glue = key == "force consecutive execution"
        if isinstance(value, (list, tuple)):
            for item in value:
                lines.append(f"{key}: { _format_value(item)}")
            continue
        lines.append(f"{key}: {_format_value(value)}")
    return "\r\n".join(lines) + "\r\n"
