"""全量对拍 prepare_out：每个文件、每个域。

配对单位是 (层类, phase, head)，不用文件名字符串（两边 qidx 不同）。
闭合域 VALUE_DIFF / MISSING / UNMATCHED_* 非 0 退出。
ALLOC / PENDING_DIFF / EXTRA 打印但不挡。
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

# 23 类在参考产物里的计数。
REFERENCE_COUNTS = {
    "dq_p1": 37, "dq_p2": 37, "dq_p3": 37, "dq_p4": 37,
    "sm_p1": 32, "sm_p2": 32, "sm_p3": 32, "sm_p4": 32, "sm_p5": 32,
    "bmm1": 32, "bmm2": 32, "mask": 32,
    "gemm_qko": 3, "gemm_v": 1, "gemm_gate": 1, "gemm_up": 1, "gemm_down": 1,
    "rope_mul_cos": 2, "rope_mul_sin": 2, "rope_add": 2,
    "rmsnorm": 2, "residual": 2, "mlp_mul": 1,
}

# 片上地址：我方贪心分配器 vs 参考 L2Analyzer，尺寸来源不同（文档 8.3）。
# **只放地址，不放尺寸、不放槽位 id、不放枚举** —— 那些是闭合域。
ALLOC_KEYS = {
    "L2 qman buffer offset",
    "L2 input buffer offset 0", "L2 input buffer offset 1",
    "L2 output buffer offset",
    "L2 fpsu buffer offset", "L2 fpsu buffer offset 0", "L2 fpsu buffer offset 1",
    "L2 weights buffer offset", "L2 weights buffer offset 0",
    "L2 weights buffer offset 1", "L2 weight scale buffer offset 0",
    "L2 data scale buffer offset engine 0",
    "DDR Input buffer offset 0", "DDR Input buffer offset 1",
    "DDR Output buffer offset", "DDR Weight buffer offset",
    "DDR data scale buffer offset",
}




def _is_alloc_key(k: str) -> bool:
    """只放过「不可能对齐」的地址域，其余一律当闭合域查。

    放过的判据必须是**结构性**的，不是「我方暂时算不对」：

    - 片上地址：我方是贪心分配器，参考是 L2Analyzer，尺寸来源不同（文档 8.3）。

    **注意**：值本身的语义（尺寸、枚举、模式、槽位 id）不在此列，命名类键
    也不在此列（见 `_is_naming_key`——那些按模式比对，不整体放过）。
    早前把 `L2 fpsu buffer id`、`L2 * buffer size`、`Transpose type` 一起
    放进来过，结果 348 处 `f1` vs `f3`、70 处尺寸错值被判成通过；
    `Datain file`/`Dataout file` 等命名类键也曾整体放过过，评审
    （docs/prepare_out-代码评审-20260920.md §3.3）测出真实差异有 173 处
    被这样静默吃掉。
    """
    return k in ALLOC_KEYS


# 带节点号 / 头号的命名类键：文件名主干规则已对齐，数字是节点号，两边各自
# 一套编号体系（我方逆拓扑，参考来自 Relay），不能要求数字相等；但除数字
# 外的部分（前缀、后缀、“这条边接了几个槽”这类结构）必须相等，否则就
# 是真实的命名差异，不能整体放过——参见 `_normalise_naming_value`。
_NAMING_PREFIXES = (
    "Virtual ", "Residual ", "DDR Input TVM", "DDR Output TVM",
)
_NAMING_KEYS = {
    # 文件名：主干规则已对齐，数字是节点号。
    "Datain file", "Datain file 0", "Datain file 1",
    "Dataout file", "Input buffer file 0", "Input buffer file 1",
    "Weights buffer file", "Bias buffer file", "Scaling buffer file",
    "Scaling PS buffer file", "Activation LUT file",
    "input scale factor buffer", "output scale factor buffer",
    "weights scaling buffer file", "Original name", "Original cache file",
    "Scaling buffer file 0", "Scaling buffer file 1",
    "Scaling PS buffer file 0", "Scaling PS buffer file 1",
    "Kantor A scale buffer file", "Kantor A bias buffer file",
    "Kantor A scale shift buffer file",
    "Kantor B scale buffer file", "Kantor B bias buffer file",
    "Kantor B scale shift buffer file",
    "kantor B scale shift buffer file",
    "bias buffer file", "Bias Buffer File",
    "Weights Buffer File", "Weights Scaling Buffer File",
    "Input Scale Factor Buffer", "Output Scale Factor Buffer",
    # DDR 段的 Orig 名带节点号 / TVM 名。
    "DDR Input Orig Buffer Name 0", "DDR Input Orig Buffer Name 1",
    "DDR Output Orig Buffer Name", "DDR Weight Orig Buffer Name",
    "DDR data scale Orig Buffer Name",
}

# 纯结构性、跟数字无关，两边编号体系不同导致永远不等的键——整体放过。
_STRUCTURAL_KEYS = {
    "Layer ID",
    # 参考产物自己把两个域挤在一行：
    #   `force consecutive execution: 1skip compare: 1`
    # （域确认表 Q27 已记录，6 个 RoPE 文件都是这样）。我方拆成两行，
    # 所以这一项永远"不等"，而 `skip compare` 在 RoPE 层会被算成我方
    # 多写。比的是参考的排版瑕疵，不是我方取值。
    "force consecutive execution",
    "skip compare",
}


def _is_structural_key(k: str) -> bool:
    if k in _STRUCTURAL_KEYS:
        return True
    return k.startswith(_NAMING_PREFIXES)


def _is_naming_key(k: str) -> bool:
    return k in _NAMING_KEYS


# 按键 schema 归一：节点号 / qidx / params 通配，槽位 / 相位 / 段号 / map<头号>
# 原样保留。不能按「数字 ≤8」切——那会把头号 map5 当成纯数字（漏判），
# 把节点号 8 当成槽位（误判）。见评审 4 §3.3。
_KEEP = (
    re.compile(r"(?<=_phase_)\d+"),
    re.compile(r"(?<=map)\d+"),
    re.compile(r"(?<=input_buffer_)\d+(?=_)"),
    re.compile(r"(?<=output_buffer_)\d+(?=_)"),
    re.compile(r"(?<=Scaling_buffer_file_)\d+(?=_)"),
    re.compile(r"(?<=Scaling_PS_buffer_file_)\d+(?=_)"),
    re.compile(r"(?<=Bias_buffer_file_)\d+(?=_)"),
    re.compile(r"(?<=LUT_phase_)\d+"),
)


def _normalise_naming_value(value: str) -> str:
    """命名类值：节点号通配成 `#`，槽位/相位/段号/头号原样保留。

    `input_buffer_0_18.bin` 与 `input_buffer_0_25.bin` → `input_buffer_0_#.bin`
    `input_buffer_18.bin` 与 `input_buffer_0_18.bin` 骨架不同
    `Scaling_buffer_file_5_Cos_22` 与 `_6_Cos_184` → 段号 5 vs 6
    `buffer19_map0` 与 `buffer19_map5` → map 头号保留，不相等
    `buffer8` 与 `buffer193` → 都是 `buffer#`（节点号，不论位数）
    """
    kept: list[tuple[int, int, str]] = []
    for pattern in _KEEP:
        for match in pattern.finditer(value):
            kept.append((match.start(), match.end(), match.group(0)))
    kept.sort()
    out: list[str] = []
    cursor = 0
    for start, end, token in kept:
        if start < cursor:
            continue
        out.append(re.sub(r"\d+", "#", value[cursor:start]))
        out.append(token)
        cursor = end
    out.append(re.sub(r"\d+", "#", value[cursor:]))
    return "".join(out)


def parse_txt(path: Path) -> dict[str, list[str]]:
    """键 -> 值列表（多值键保留全部）。"""
    fields: dict[str, list[str]] = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.rstrip("\r\n")
        if not line.strip() or line.startswith("Dump files"):
            continue
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.rstrip()
        if key.startswith("Virtual "):
            key = "Virtual Input/Output"
        fields.setdefault(key, []).append(value.strip())
    return fields


def _head_of(name: str, fields: dict[str, list[str]]) -> int | None:
    m = re.search(r"head(\d+)", name)
    if m:
        return int(m.group(1))
    split = (fields.get("Split Head Index") or [""])[0]
    if split.isdigit():
        return int(split)
    return None


def classify_ref(name: str, fields: dict[str, list[str]]) -> tuple[str, int | None]:
    """参考文件名 → (层类, head)。"""
    head = _head_of(name, fields)
    if "gp_dq_phase1" in name:
        return "dq_p1", head
    if "act_dq_phase2" in name:
        return "dq_p2", head
    if "act_dq_phase3" in name:
        return "dq_p3", head
    if "act_dq_phase4" in name:
        return "dq_p4", head
    if "gp_sm_phase1" in name:
        return "sm_p1", head
    if "act_sm_phase2" in name:
        return "sm_p2", head
    if "gp_sm_phase3" in name:
        return "sm_p3", head
    if "act_sm_phase4" in name:
        return "sm_p4", head
    if "act_sm_phase5" in name:
        return "sm_p5", head
    if "mul_cos" in name:
        return "rope_mul_cos", head
    if "mul_sin" in name:
        return "rope_mul_sin", head
    if "_add_params_" in name and "Reshape" in name:
        return "rope_add", head
    if "batch_matmul1" in name:
        return "bmm1", head
    if "batch_matmul2" in name:
        return "bmm2", head
    if "mha_masking" in name:
        return "mask", head
    if name.startswith("RMSNorm"):
        return "rmsnorm", head
    if "v_proj" in name:
        return "gemm_v", head
    if "gate_proj" in name:
        return "gemm_gate", head
    if "up_proj" in name:
        return "gemm_up", head
    if "down_proj" in name:
        return "gemm_down", head
    if "q_proj" in name or "k_proj" in name or "o_proj" in name:
        return "gemm_qko", head
    if name.startswith("add_"):
        return "residual", head
    if "mlp_mul" in name:
        return "mlp_mul", head
    return "OTHER", head


def classify_mine(name: str, fields: dict[str, list[str]]) -> tuple[str, int | None]:
    """我方文件：优先读字段，再退回文件名。"""
    def one(key: str) -> str:
        values = fields.get(key) or []
        return values[0] if values else ""

    head = _head_of(name, fields)
    split = one("Split Head Index")
    if split.isdigit():
        head = int(split)

    dq = one("dynamic quantization phase")
    sm = one("softmax phase")
    layer_type = one("layer type")
    elt = one("Eltwise mode")
    llama = one("Llama2Activation")
    fmt = one("Weight Format")
    act = one("Activation Type")
    out_w = one("Output Width")
    in_w = one("Input Width")
    kantor = one("Kantor mode")
    sub = one("sublayer type")

    if dq:
        return f"dq_p{dq}", head
    if sm:
        return f"sm_p{sm}", head
    if "mul_cos" in name:
        return "rope_mul_cos", head
    if "mul_sin" in name:
        return "rope_mul_sin", head
    if llama == "True" and elt == "0":
        return "rope_add", head
    if layer_type == "vpu" or sub == "rmsnorm":
        return "rmsnorm", head
    if "masking" in name or (elt == "2"):
        return "mask", head
    if layer_type == "matmul" or fmt in ("2", "3"):
        if fmt == "3" or "matmul1" in name:
            return "bmm1", head
        return "bmm2", head
    if act == "13" and layer_type == "gemm":
        return "gemm_gate", head
    if kantor == "3" and layer_type == "gemm":
        return "gemm_v", head
    if layer_type == "gemm":
        if out_w == "11008":
            return "gemm_up", head
        if in_w == "11008":
            return "gemm_down", head
        return "gemm_qko", head
    if elt == "1" and "Llama2" not in name:
        return "mlp_mul", head
    if elt == "0":
        return "residual", head
    return classify_ref(name, fields)


def net_ini_order(root: Path) -> list[str]:
    path = root / "net.ini"
    if not path.is_file():
        return []
    names = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line.startswith("layer = "):
            names.append(line.split("=", 1)[1].strip())
        elif line.endswith(".txt") and not line.startswith("#"):
            names.append(line)
    return names


def load_dir(root: Path, mine: bool) -> dict[tuple, tuple[str, dict]]:
    """(kind, index_in_kind) -> (filename, fields)，按 net.ini 执行序。"""
    txt = root / "txt_files"
    if not txt.is_dir():
        txt = root
    by_name = {}
    for path in txt.glob("*.txt"):
        if path.name in ("gml_version.txt", "l2a_version.txt"):
            continue
        by_name[path.stem] = path
        by_name[path.name] = path

    order = net_ini_order(root)
    if not order:
        order = sorted(p.stem for p in txt.glob("*.txt")
                       if p.name not in ("gml_version.txt", "l2a_version.txt"))

    buckets: dict[tuple, list] = defaultdict(list)
    seen = set()
    for stem in order:
        path = by_name.get(stem) or by_name.get(stem + ".txt")
        if path is None or path in seen:
            continue
        seen.add(path)
        fields = parse_txt(path)
        kind, head = (classify_mine if mine else classify_ref)(path.name, fields)
        width = (fields.get("Input Width") or ["0"])[0]
        buckets[(kind, head, width)].append((path.name, fields))
    keyed = {}
    for key, items in buckets.items():
        for index, item in enumerate(items):
            keyed[(key[0], key[1], key[2], index)] = item
    return keyed


def first(values: list[str] | None) -> str:
    return values[0] if values else ""


def compare(mine_root: Path, ref_root: Path) -> int:
    mine = load_dir(mine_root, mine=True)
    ref = load_dir(ref_root, mine=False)

    counts_ref = Counter(k[0] for k in ref)
    counts_mine = Counter(k[0] for k in mine)
    print("参考层类计数:", dict(sorted(counts_ref.items())))
    print("我方层类计数:", dict(sorted(counts_mine.items())))

    unmatched_file = 0
    for kind, want in REFERENCE_COUNTS.items():
        got = counts_ref.get(kind, 0)
        if got != want:
            print(f"UNMATCHED_FILE 参考 {kind}: {got} 期望 {want}")
            unmatched_file += 1
        # 我方计数只打印、不比对，等于「422/422」这个数字从来没被验证过——
        # 评审 §3.3 第 3 点指出这正是让「基本对齐」显得比实际更可信的口径
        # 漏洞。这里补上：我方每一类的计数也必须匹配参考。
        got_mine = counts_mine.get(kind, 0)
        if got_mine != want:
            print(f"UNMATCHED_FILE 我方 {kind}: {got_mine} 期望 {want}")
            unmatched_file += 1

    stats = Counter()
    diffs: list[str] = []
    keys_ref = set(ref)
    keys_mine = set(mine)
    for key in sorted(keys_ref - keys_mine, key=str):
        stats["UNMATCHED_PAIR"] += 1
        diffs.append(f"UNMATCHED_PAIR 参考有我方无 {key} {ref[key][0]}")
    for key in sorted(keys_mine - keys_ref, key=str):
        stats["UNMATCHED_PAIR"] += 1
        diffs.append(f"UNMATCHED_PAIR 我方有参考无 {key} {mine[key][0]}")

    for key in sorted(keys_ref & keys_mine, key=str):
        ref_name, ref_f = ref[key]
        mine_name, mine_f = mine[key]
        ref_keys = set(ref_f)
        mine_keys = set(mine_f)
        for k in sorted(ref_keys - mine_keys):
            if _is_alloc_key(k) or _is_structural_key(k):
                stats["ALLOC"] += 1
            else:
                stats["MISSING"] += 1
                diffs.append(f"MISSING {key} {k} 参考={first(ref_f[k])}")
        for k in sorted(mine_keys - ref_keys):
            if _is_alloc_key(k) or _is_structural_key(k):
                stats["ALLOC"] += 1
            else:
                stats["EXTRA"] += 1
        for k in sorted(ref_keys & mine_keys):
            rv, mv = ref_f[k], mine_f[k]
            if rv == mv:
                stats["MATCH"] += 1
                continue
            if _is_alloc_key(k) or _is_structural_key(k):
                stats["ALLOC"] += 1
                continue
            if _is_naming_key(k):
                # 按模式比对，不整体放过：数字（节点号）换成 `#` 再比，
                # 归一后仍不同才是真实的命名差异（见评审 §3.3）。
                rv_norm = [_normalise_naming_value(v) for v in rv]
                mv_norm = [_normalise_naming_value(v) for v in mv]
                if rv_norm == mv_norm:
                    stats["ALLOC"] += 1
                    continue
            stats["VALUE_DIFF"] += 1
            diffs.append(
                f"VALUE_DIFF {key[0]}[{key[1]}] {k} "
                f"参考={rv[:2]} 我方={mv[:2]}")

    print()
    for name in ("MATCH", "VALUE_DIFF", "MISSING", "EXTRA", "ALLOC",
                 "UNMATCHED_PAIR"):
        print(f"  {name:16s} {stats[name]}")
    print(f"  UNMATCHED_FILE   {unmatched_file}")
    print()
    keys_vd = Counter()
    keys_miss = Counter()
    for line in diffs:
        if line.startswith("VALUE_DIFF"):
            keys_vd[line.split(" 参考=")[0].split(" ", 2)[2]] += 1
        elif line.startswith("MISSING"):
            keys_miss[line.split(" 参考=")[0].split(") ", 1)[-1]] += 1
    print("VALUE_DIFF 按键:")
    for k, c in keys_vd.most_common():
        print(f"  {c:4d}  {k}")
    print("MISSING 按键:")
    for k, c in keys_miss.most_common():
        print(f"  {c:4d}  {k}")
    print()
    for line in diffs[:40]:
        print(line)
    if len(diffs) > 40:
        print(f"... 另有 {len(diffs) - 40} 条")

    bad = stats["VALUE_DIFF"] + stats["MISSING"] + stats["UNMATCHED_PAIR"] + unmatched_file
    return 1 if bad else 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mine", type=Path, required=True)
    parser.add_argument("--ref", type=Path, required=True)
    args = parser.parse_args()
    return compare(args.mine, args.ref)


if __name__ == "__main__":
    sys.exit(main())
