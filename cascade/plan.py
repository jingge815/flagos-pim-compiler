"""多核级联蓝图：从 bundle 的 cascade 节与每核 map 推导段序列与搬运表。

段（CascadeSegment）是 runner 一次控制块命令执行的单位；搬运（CascadeTransfer）
是 host 在段间搬运的数据块。蓝图只描述"做什么"，不带执行时刻。
"""

from __future__ import annotations

import configparser
import dataclasses
import json
import struct
from pathlib import Path

# 与 /data/fjg/ceva 段式 runner 的 ControlBlock 逐字段镜像（设计 4.1）。
# 布局：go/cmd/layer_lo/layer_hi(4×u32) + in/out(2×u64) + line_idx/done/
# cycles_lo/cycles_hi/token(5×u32) + pad(3×u32)，共 14 字段 = 64B。
CONTROL_FORMAT = "<IIIIQQIIIIIIII"
CONTROL_SIZE = 64


@dataclasses.dataclass(frozen=True)
class ControlBlock:
    """段式 runner 的 64B 控制块镜像。go=1 启动一段，done=1 表示段完成。"""

    go: int = 0
    cmd: int = 0            # 1=prompt_layers 2=decode_layers 3=lm_head 4=attn 5=mlp 6=head_shard
    layer_lo: int = 0
    layer_hi: int = 0
    in_addr: int = 0        # 本段 hidden 输入 DDR 地址（64 位）
    out_addr: int = 0       # 本段 hidden 输出 DDR 地址（64 位）
    line_idx: int = 0       # decode 步序
    done: int = 0
    cycles_lo: int = 0      # 本段 busy cycles 低 32 位
    cycles_hi: int = 0
    token: int = 0          # cmd=3 的 argmax 结果
    pad0: int = 0
    pad1: int = 0
    pad2: int = 0

    def pack(self) -> bytes:
        return struct.pack(CONTROL_FORMAT, *dataclasses.astuple(self))

    @classmethod
    def unpack(cls, raw: bytes) -> "ControlBlock":
        return cls(*struct.unpack(CONTROL_FORMAT, raw[:CONTROL_SIZE]))

    def with_go(self) -> "ControlBlock":
        return dataclasses.replace(self, go=1)

    @property
    def cycles(self) -> int:
        return self.cycles_hi << 32 | self.cycles_lo


@dataclasses.dataclass(frozen=True)
class CascadeSegment:
    """一次控制块命令：某核跑一段层范围的子图。"""

    seg_id: str
    core_id: int
    kind: str               # prompt_layers/decode_layers/lm_head/attn_shard/mlp_shard/head_shard
    layer_lo: int
    layer_hi: int           # 开区间上界
    in_addrs: dict
    out_addrs: dict


@dataclasses.dataclass(frozen=True)
class CascadeTransfer:
    """段间一次数据搬运。src/dst 为 None 表示 host 侧。"""

    xfer_id: str
    src_core: int | None
    dst_core: int | None
    nbytes: int
    src_addr: int
    dst_addr: int
    kind: str               # hidden_handoff/all_reduce/logits_gather/embedding/token
    after_seg: str


@dataclasses.dataclass(frozen=True)
class CascadeBlueprint:
    """整次级联运行的段序列与搬运表。"""

    strategy: str
    num_cores: int
    core_base_addrs: dict
    segments: list
    transfers: list
    control_addrs: dict

    @property
    def decode_steps(self) -> int:
        """蓝图覆盖的 decode 步数，PP2/TP2 统一按段名 decode:{k} 前缀取。

        评估的基线折算与 token 计数都以它对齐范围；按段 kind（decode_layers）
        推导会把 TP2 蓝图截断成 0 步。"""
        return len({int(s.seg_id.split(":")[1]) for s in self.segments
                    if s.seg_id.startswith("decode:")})


def _read_ini(path: Path) -> list[dict]:
    parser = configparser.ConfigParser()
    parser.read(path)
    return [dict(parser[s]) for s in parser.sections()]


def _addr(value: str) -> int:
    return int(value, 16) if value.strip().lower().startswith("0x") else int(value)


def _core_io(run_dir: Path, core_id: int) -> dict:
    """读一个核的 inputs/outputs/control 三份 map。"""
    d = run_dir / f"ddr_core{core_id}"
    inputs = {e["input_name"]: (_addr(e["address"]), int(e["size"]))
              for e in _read_ini(d / "inputs_map.ini")}
    outputs = {e["output_name"]: (_addr(e["address"]), int(e["size"]))
               for e in _read_ini(d / "outputs_map.ini")}
    control = _read_ini(d / "control_map.ini")[0]
    if int(control["core_id"]) != core_id:
        raise ValueError(f"{d}/control_map.ini 的 core_id 与目录不符")
    return {"inputs": inputs, "outputs": outputs,
            "control": _addr(control["address"])}


def _seg(seg_id, core, kind, lo, hi, in_addrs, out_addrs):
    return CascadeSegment(seg_id, core, kind, lo, hi, in_addrs, out_addrs)


def _xfer(xfer_id, src, dst, nbytes, src_addr, dst_addr, kind, after_seg):
    return CascadeTransfer(xfer_id, src, dst, nbytes, src_addr, dst_addr,
                           kind, after_seg)


def from_bundle(bundle_path: Path, run_dir: Path, *, steps: int) -> CascadeBlueprint:
    """从 bundle.json 的 cascade 节与每核 ddr 产物构造蓝图。"""
    manifest = json.loads(Path(bundle_path).read_text())
    cascade = manifest.get("cascade")
    if not cascade:
        raise ValueError(f"{bundle_path} 没有 cascade 节，不能构造级联蓝图")
    num_cores = int(cascade["num_cores"])
    cores = [_core_io(Path(run_dir), i) for i in range(num_cores)]
    control_addrs = {i: c["control"] for i, c in enumerate(cores)}
    base = {i: c["control"] - 0x1000 for i, c in enumerate(cores)}
    strategy = cascade["strategy"]
    if strategy == "pp2":
        segments, transfers = _pp2_segments(cascade, cores, steps)
    elif strategy == "tp2":
        segments, transfers = _tp2_segments(cascade, cores, steps)
    else:
        raise ValueError(f"未知级联策略 {strategy!r}")
    return CascadeBlueprint(strategy, num_cores, base, segments, transfers,
                            control_addrs)


def _pp2_segments(cascade, cores, steps):
    c0, c1 = cores[0], cores[1]
    (lo0, hi0), (lo1, hi1) = cascade["core_layers"]
    head_core = cascade["lm_head_core"]
    prompt_n = c0["inputs"]["hidden_states"][1]
    decode_n = c0["inputs"]["embedding_row"][1]
    segments, transfers = [], []
    n = [0]

    def add_xfer(src, dst, nbytes, src_addr, dst_addr, kind, after_seg):
        n[0] += 1
        transfers.append(_xfer(f"x{n[0]}", src, dst, nbytes, src_addr,
                               dst_addr, kind, after_seg))

    def one_step(tag, kind, in_c0, hidden_n):
        segments.append(_seg(f"{tag}:c0", 0, kind, lo0, hi0,
                             {"hidden": in_c0}, {"hidden": c0["outputs"]["output"][0]}))
        segments.append(_seg(f"{tag}:c1", 1, kind, lo1, hi1,
                             {"hidden": c1["inputs"]["hidden_states"][0]},
                             {"hidden": c1["outputs"]["output"][0]}))
        add_xfer(0, 1, hidden_n, c0["outputs"]["output"][0],
                 c1["inputs"]["hidden_states"][0], "hidden_handoff", f"{tag}:c0")
        # token 序：prompt 段产出第 0 个，decode:{k} 产出第 k+1 个。
        step_no = "0" if tag == "prompt" else str(int(tag.split(":")[1]) + 1)
        head = _seg(f"head:{step_no}", head_core, "lm_head", lo1, hi1,
                    {"hidden": cores[head_core]["outputs"]["output"][0]},
                    {"lm_output": cores[head_core]["outputs"]["lm_output"][0]})
        segments.append(head)
        add_xfer(head_core, None, 2, 0, 0, "token", head.seg_id)
        add_xfer(None, 0, decode_n, 0, c0["inputs"]["embedding_row"][0],
                 "embedding", head.seg_id)

    one_step("prompt", "prompt_layers", c0["inputs"]["hidden_states"][0], prompt_n)
    for step in range(steps):
        one_step(f"decode:{step}", "decode_layers",
                 c0["inputs"]["embedding_row"][0], decode_n)
    return segments, transfers


def _tp2_segments(cascade, cores, steps):
    (lo, hi) = cascade["core_layers"][0]
    prompt_n = cores[0]["inputs"]["hidden_states"][1]
    decode_n = cores[0]["inputs"]["embedding_row"][1]
    logits_n = cores[0]["outputs"]["lm_output"][1]
    segments, transfers = [], []
    n = [0]

    def add_xfer(src, dst, nbytes, src_addr, dst_addr, kind, after_seg):
        n[0] += 1
        transfers.append(_xfer(f"x{n[0]}", src, dst, nbytes, src_addr,
                               dst_addr, kind, after_seg))

    def shard_pair(tag, kind, layer, in_addrs):
        for c in (0, 1):
            segments.append(_seg(f"{tag}:c{c}", c, kind, layer, layer + 1,
                                 dict(in_addrs[c]),
                                 {"hidden": cores[c]["outputs"]["output"][0],
                                  "lm_output": cores[c]["outputs"]["lm_output"][0]}))

    def one_step(tag, hidden_n, first_in):
        in_addrs = [{}, {}]
        for layer in range(lo, hi):
            in_addrs[0]["hidden"] = first_in[0] if layer == lo else cores[0]["inputs"]["hidden_states"][0]
            in_addrs[1]["hidden"] = first_in[1] if layer == lo else cores[1]["inputs"]["hidden_states"][0]
            shard_pair(f"{tag}:L{layer}:attn", "attn_shard", layer, in_addrs)
            # 归约 = 读两核部分和 + 写回两核（4 份 hidden）
            add_xfer(None, None, 4 * hidden_n, 0, 0, "all_reduce",
                     f"{tag}:L{layer}:attn:c1")
            shard_pair(f"{tag}:L{layer}:mlp", "mlp_shard", layer,
                       [{"hidden": cores[0]["inputs"]["hidden_states"][0]},
                        {"hidden": cores[1]["inputs"]["hidden_states"][0]}])
            add_xfer(None, None, 4 * hidden_n, 0, 0, "all_reduce",
                     f"{tag}:L{layer}:mlp:c1")
        # token 序：prompt 相产出第 0 个，decode:{k} 相产出第 k+1 个（与 PP2 同口径）。
        head_no = "0" if tag == "prompt" else str(int(tag.split(":")[1]) + 1)
        for c in (0, 1):
            segments.append(_seg(f"head:{head_no}:c{c}",
                                 c, "head_shard", lo, hi,
                                 {"hidden": cores[c]["inputs"]["hidden_states"][0]},
                                 {"lm_output": cores[c]["outputs"]["lm_output"][0]}))
        for c in (0, 1):
            add_xfer(c, None, logits_n, cores[c]["outputs"]["lm_output"][0], 0,
                     "logits_gather", f"head:{head_no}:c{c}")
        add_xfer(None, None, 2, 0, 0, "token", f"head:{head_no}:c1")
        for c in (0, 1):
            add_xfer(None, c, decode_n, 0, cores[c]["inputs"]["embedding_row"][0],
                     "embedding", f"head:{head_no}:c{c}")

    one_step("prompt", prompt_n,
             (cores[0]["inputs"]["hidden_states"][0],
              cores[1]["inputs"]["hidden_states"][0]))
    for step in range(steps):
        one_step(f"decode:{step}", decode_n,
                 (cores[0]["inputs"]["embedding_row"][0],
                  cores[1]["inputs"]["embedding_row"][0]))
    return segments, transfers


def plan_text(plan: CascadeBlueprint) -> str:
    """蓝图的可读文本（设计 3.1：关键中间产物要能 print）。"""
    lines = [f"级联蓝图 strategy={plan.strategy} cores={plan.num_cores}",
             "控制块: " + " ".join(f"c{c}=0x{a:x}" for c, a in sorted(plan.control_addrs.items())),
             f"段 {len(plan.segments)} 个，搬运 {len(plan.transfers)} 条",
             "段:", "  seg_id            core kind            layers   in/out"]
    for s in plan.segments:
        lines.append(f"  {s.seg_id:<18}{s.core_id:<5}{s.kind:<16}"
                     f"{s.layer_lo}-{s.layer_hi:<5}")
    lines.append("搬运:")
    for t in plan.transfers:
        lines.append(f"  {t.xfer_id:<5}{t.kind:<16}{t.nbytes:>10d}B  "
                     f"after {t.after_seg}  c{t.src_core}->c{t.dst_core}")
    return "\n".join(lines)


def canonical_seg_id(name: str) -> str:
    """早期工作区脚本把 lm_head 段记作 prompt:lm / decode:k:lm，统一成蓝图命名。"""
    if name == "prompt:lm":
        return "head:0"
    if name.startswith("decode:") and name.endswith(":lm"):
        return f"head:{int(name.split(':')[1]) + 1}"
    return name


def authoritative_timing(rows: list) -> list:
    """timing 日志的权威投影：按归一化段名取每个逻辑段的最后一次记录。

    timing 是追加式执行日志，断点续跑会留下同物异名的重复记录（decode:k:lm
    与 head:k+1）；counters、时间线与 host 汇总一律以本投影为准，
    历史记录保留在日志里供审计。
    """
    last: dict = {}
    for row in rows:
        last[canonical_seg_id(row["seg"])] = row
    return list(last.values())
