"""级联运行时驱动：按蓝图逐段拉起 TCS 进程，host 搬运/归约/LUT，产物统一落盘。

一段 = 一个新 TCS 进程（M0 实测：app return 才是真正停核）。跨段数据走输入
文件（装载会覆盖 DDR 预写），KV 由段式 runner 在进程内快照到 kv_core{N}。
PP2 按蓝图段序串行推进；TP2 同层两核分片段并行下发，host 归约与拼接。
"""

from __future__ import annotations

import hashlib
import json
import re
import struct
import sys
import threading
import time
from pathlib import Path

import numpy as np

from cascade.plan import ControlBlock, authoritative_timing

# 控制块命令号（设计 4.1，与段式 runner 的分发一致）。
CMD = {"prompt_layers": 1, "decode_layers": 2, "lm_head": 3,
       "attn_shard": 4, "mlp_shard": 5, "head_shard": 6}

_EOS_TOKEN = 2                    # 厂商 runner 的 DELIMITER

# 与段式 app 的 kv_fingerprint 同算法（每 1MB 取 8 字节 FNV 折叠），做粗粒度存活检查。
FNV_OFFSET = 1469598103934665603
FNV_PRIME = 1099511628211


def _embed_row(embed_path: Path, token: int, nbytes: int) -> bytes:
    with embed_path.open("rb") as handle:
        handle.seek(token * nbytes)
        return handle.read(nbytes)


def _kv_fp_file(path: Path) -> int:
    """KV 快照文件的采样指纹（与 app 侧同算法）；文件缺失返回 0。"""
    fp = FNV_OFFSET
    if not Path(path).exists():
        return 0
    with open(path, "rb") as f:
        off = 0
        while True:
            f.seek(off)
            chunk = f.read(8)
            if len(chunk) < 8:
                break
            fp = ((fp ^ struct.unpack("<Q", chunk)[0]) * FNV_PRIME) \
                & 0xFFFFFFFFFFFFFFFF
            off += 1 << 20
    return fp


def _kv_diff(prev: Path, cur: Path) -> tuple[int, list[int]]:
    """逐 4MB 张量统计差异字节：decode 一步只应改每个张量里的一行。"""
    a = np.fromfile(prev, dtype=np.uint8)
    b = np.fromfile(cur, dtype=np.uint8)
    n = min(a.size, b.size)
    d = (a[:n] != b[:n]).reshape(-1, 4 << 20)
    return int(np.count_nonzero(d)), [int(x) for x in np.count_nonzero(d, axis=1)]


def _parse_app_log(path: Path) -> dict:
    """从 app 日志抽段内自检信息：KV 指纹、输入首字、位置行地址。"""
    try:
        text = Path(path).read_text(errors="ignore")
    except OSError:
        return {}
    info: dict = {}
    patterns = {
        "kv_restore_fp": r"kv restore .*?fp=0x([0-9a-f]+)",
        "kv_exec_fp": r"kv-exec fp=0x([0-9a-f]+)",
        "in_head": r"in_head=0x([0-9a-f]+) line_idx=(-?\d+)",
        "pos": r"pos mask=0x([0-9a-f]+) sin=0x([0-9a-f]+) cos=0x([0-9a-f]+)",
    }
    for key, pat in patterns.items():
        found = re.findall(pat, text)
        if found:
            info[key] = found[-1]
    return info


def _top_tokens(data: bytes, k: int = 5) -> list:
    """logits 前 k 名（下标 + 分值），出错时用来看分布是接近还是乱掉。"""
    v = np.frombuffer(data, dtype=np.float16).astype(np.float32)
    idx = np.argsort(-v)[:k]
    return [[int(i), round(float(v[i]), 4)] for i in idx]


_LAST_KV: dict = {}          # (run_dir, core) -> 上一段留档文件名，串起 diff 链


def _append_diagnostics(run_dir: Path, seg_id: str, core: int, data: bytes,
                        log_path: Path, *, wall: float, exec_cycles: int,
                        clock: int, kv_before_fp: int = 0,
                        kv_name: str = "kv.bin") -> None:
    """段后自检（可注入替换）：KV 指纹链 + diff + 输出摘要写 diagnostics.jsonl。

    `kv_name`：PP2 是整区 kv.bin（跨段 diff 链）；TP2 attn 分片是按层
    kv_L{n}.bin（层间内容不可比，只记指纹与留档，不进 diff 链）。"""
    run_dir = Path(run_dir)
    kv_path = run_dir / f"kv_core{core}" / kv_name
    diag = {"seg": seg_id, "core": core, "wall_s": round(wall, 1),
            "exec_cycles": exec_cycles, "device_cycles": clock,
            "kv_before_fp": f"{kv_before_fp:016x}",
            "kv_after_fp": f"{_kv_fp_file(kv_path):016x}",
            "app": _parse_app_log(log_path),
            "out_md5": hashlib.md5(data).hexdigest()}
    if re.fullmatch(r"head:\d+(:c\d+)?|prompt:lm|decode:\d+:lm", seg_id):
        diag["top5"] = _top_tokens(data)
    hist = run_dir / "hist"
    if kv_path.exists():
        hist.mkdir(exist_ok=True)
        name = f"{seg_id.replace(':', '_')}.bin"
        cur = hist / name
        cur.write_bytes(kv_path.read_bytes())       # 留档，出错时可回卷
        if kv_name == "kv.bin":
            prev_name = _LAST_KV.get((str(run_dir), core))
            if prev_name and (hist / prev_name).exists():
                total, per = _kv_diff(hist / prev_name, cur)
                diag["kv_prev"] = prev_name
                diag["kv_diff_bytes"] = total
                diag["kv_diff_per_tensor"] = per
            _LAST_KV[(str(run_dir), core)] = name
    with (run_dir / "diagnostics.jsonl").open("a") as f:
        f.write(json.dumps(diag, ensure_ascii=False) + "\n")


class SegmentProcessRunner:
    """一段 = 一个新 TCS 进程（M0 实测：app return 才是真正停核）。

    跨段数据走输入文件（loadBinMapToMemory 会用文件内容覆盖 DDR 预写）；
    KV 由段式 runner 在进程内快照到 kv_core{N}/kv.bin，decode 段启动时恢复。
    每核独立工作目录：并行段进程的 stepwise_output/CycleEstimation 不互抢。
    """

    def __init__(self, workspace: Path, run_dir: Path, elf: Path, *,
                 num_engines: int = 4):
        self.workspace = Path(workspace)
        self.run_dir = Path(run_dir)
        self.elf = Path(elf)
        self.num_engines = num_engines
        self._port = 40000 + int(time.time()) % 10000
        self._port_lock = threading.Lock()
        self._io: dict[int, dict] = {}

    def core_io(self, core: int) -> dict:
        """读某核 inputs/outputs 地址表（含输入文件路径），并缓存。"""
        if core not in self._io:
            import configparser
            maps = {}
            for name in ("inputs", "outputs"):
                parser = configparser.ConfigParser()
                parser.read(self.run_dir / f"ddr_core{core}" / f"{name}_map.ini")
                maps[name] = {}
                for section in parser.sections():
                    entry = dict(parser[section])
                    key = entry.get("input_name") or entry.get("output_name")
                    maps[name][key] = (int(entry["address"], 16), int(entry["size"]))
                    if entry.get("input_file"):
                        maps[name][f"{key}_file"] = entry["input_file"]
            self._io[core] = maps
        return self._io[core]

    def write_input_file(self, core: int, name: str, data: bytes) -> int:
        """写本核输入文件；下次进程装载时进入 DDR，并记一条搬运日志。"""
        path = self.core_io(core)["inputs"].get(f"{name}_file")
        if path is None:
            raise KeyError(f"core{core} 无 {name} 输入文件")
        t0 = time.time()
        Path(path).write_bytes(data)
        _log_transfer(self.run_dir, name, len(data), time.time() - t0)
        return len(data)

    def run(self, core: int, cmd: int, in_addr: int, out_addr: int,
            read_size: int, line_idx: int = 0, kv_dir: str = "",
            stage: str = "", layer: int | None = None) -> tuple:
        """拉起一个 TCS 进程跑一段，返回 (wall 秒, 输出字节, 时钟, exec_cycles, 日志路径)。

        `stage`（prompt/decode/head）与 `layer` 是 TP2 分片段的注册表选择器；
        PP2 的 cmd=1/2/3 不需要。attn/mlp 段必须带 layer：argv[14]="None"
        会让 app 静默跑第 0 层图，这里直接抛错让契约违例显式失败。"""
        import subprocess
        import os
        work = self.run_dir / f"work_c{core}"
        work.mkdir(parents=True, exist_ok=True)
        with self._port_lock:
            self._port += 1
            port = self._port
        cmd_parts = [str(self.run_dir / f"ddr_core{core}" / f"{n}_map.ini")
                     for n in ("prompt", "decode", "lm_head", "inputs", "outputs")]
        cmd_parts += ["1", hex(plan_control_addr(self.run_dir, core)), str(cmd),
                      hex(in_addr), hex(out_addr), str(line_idx), kv_dir]
        if stage:
            cmd_parts.append(stage)
            if cmd in (CMD["attn_shard"], CMD["mlp_shard"]):
                if layer is None:
                    raise ValueError(
                        f"core{core} cmd={cmd} 分片段必须传 layer（stage={stage}）")
                cmd_parts.append(str(layer))
        argv = [sys.executable, "-u", str(self.workspace / "src" / "automation" / "run_tools_step.py"),
                "-tcs", os.environ["TCS_CFG"], "-vpu", os.environ["VPU_ELF"],
                "-n", str(self.num_engines), "-elf", str(self.elf),
                "-cmd", " ".join(cmd_parts),
                "-log", str(self.run_dir / f"app_{port}.log"), "-port", str(port)]
        outfile = work / "stepwise_output.bin"
        outfile.unlink(missing_ok=True)
        proc = subprocess.Popen(argv, stdin=subprocess.DEVNULL, cwd=str(work),
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                start_new_session=True)
        session = None
        for _ in range(900):
            try:
                session = StepProxySession(proc, port)
                break
            except OSError:
                time.sleep(1)
        if session is None:
            _kill_group(proc)
            raise RuntimeError(f"core{core} 端口 {port} 未就绪")
        t0 = time.time()
        session.resume_sync()
        wall = time.time() - t0
        data = outfile.read_bytes()[:read_size] if outfile.exists() else b""
        outfile.unlink(missing_ok=True)
        log_path = self.run_dir / f"app_{port}.log"
        clock = _end_clock(log_path)
        cycles_file = work / "stepwise_cycles.txt"
        exec_cycles = int(cycles_file.read_text().strip()) if cycles_file.exists() else 0
        cycles_file.unlink(missing_ok=True)
        _stop_session(session, proc)
        return wall, data, clock, exec_cycles, log_path


def _kill_group(proc) -> None:
    """杀掉包装进程所在的整个进程组（run_tools_step 与它的 TCS 子进程）。"""
    import os
    import signal
    pid = getattr(proc, "pid", 0)
    if pid:
        try:
            os.killpg(os.getpgid(pid), signal.SIGKILL)
            return
        except OSError:
            pass
    proc.kill()


def _stop_session(session, proc, timeout_s: int = 300) -> None:
    """先关协议通道让 run_tools_step 清理 TCS，超时才整组强杀。"""
    import subprocess
    try:
        session.close()
        proc.wait(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        _kill_group(proc)
    except OSError:
        _kill_group(proc)


def _log_transfer(run_dir: Path, kind: str, nbytes: int, dt: float) -> None:
    path = Path(run_dir) / "run" / "transfer_log.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps({"kind": kind, "bytes": nbytes, "t_host": dt}) + "\n")


def _end_clock(log_path: Path) -> int:
    """app 日志里的 EndOfExecution 全局模拟时钟（与单核基线 TIMER 同口径）。

    CycleEstimation 尾值不含控制器开销，会低估设备时间约 40%（prompt 段），
    评估必须用这个时钟。
    """
    try:
        text = Path(log_path).read_text(errors="ignore")
    except OSError:
        return 0
    found = re.findall(r"clock (\d+): Branch to self", text)
    return int(found[-1]) if found else 0


def _load_state(state_path: Path) -> dict:
    if state_path.exists():
        return json.loads(state_path.read_text())
    return {"done": {}, "timing": []}


def run_pp2_processes(plan, runner: SegmentProcessRunner, steps: int, *,
                      run_dir: Path, embed_path: Path, prompt_len: int,
                      eos_token: int = _EOS_TOKEN,
                      fresh: bool = False,
                      diagnostics=None) -> list[int]:
    """PP2 编排：按蓝图段序驱动两核，段间写输入文件交接，支持断点续跑。

    段产物与进度落在 `results/state.json`，重跑时已完成段直接复用，
    因此任何中断都能从中断点继续（KV 快照本身就是每核最后一段的状态）。
    `diagnostics` 是段后自检 hook（默认关闭；真机运行注入 _append_diagnostics）。
    """
    run_dir = Path(run_dir)
    results = run_dir / "results"
    state_path = results / "state.json"
    if fresh:
        state_path.unlink(missing_ok=True)
    # KV 快照目录必须先存在（app 用 semihosting 直接写文件，不建目录）。
    for core in range(plan.num_cores):
        kv = run_dir / f"kv_core{core}"
        kv.mkdir(parents=True, exist_ok=True)
        if fresh:
            for stale in kv.glob("*.bin"):
                stale.unlink()
    results.mkdir(parents=True, exist_ok=True)
    state = _load_state(state_path)

    def save():
        state_path.write_text(json.dumps(state, indent=1, ensure_ascii=False))

    def execute(seg, read_size, in_addr=None):
        """执行或复用一段；返回 (输出字节, 是否实跑)。"""
        if seg.seg_id in state["done"]:
            entry = state["done"][seg.seg_id]
            return Path(entry["out"]).read_bytes(), False
        line_idx = 0
        if seg.kind == "decode_layers":
            line_idx = prompt_len + 1 + int(seg.seg_id.split(":")[1])
        in_addr = seg.in_addrs["hidden"] if in_addr is None else in_addr
        out_addr = seg.out_addrs.get("hidden", seg.out_addrs.get("lm_output"))
        kv_path = run_dir / f"kv_core{seg.core_id}" / "kv.bin"
        before_fp = _kv_fp_file(kv_path)
        wall, data, clock, exec_cycles, log_path = runner.run(
            seg.core_id, CMD[seg.kind], in_addr, out_addr, read_size,
            line_idx=line_idx, kv_dir=str(run_dir / f"kv_core{seg.core_id}"))
        if not data:
            raise RuntimeError(f"{seg.seg_id} 无输出（stepwise_output.bin 缺失）")
        path = results / f"seg_{seg.seg_id.replace(':', '_')}.bin"
        path.write_bytes(data)
        state["done"][seg.seg_id] = {"out": str(path), "bytes": len(data),
                                     "wall_s": wall, "cycles": 0,
                                     "device_cycles": clock,
                                     "exec_cycles": exec_cycles}
        state["timing"].append({"seg": seg.seg_id, "core": seg.core_id,
                                "wall_s": wall, "cycles": 0,
                                "device_cycles": clock,
                                "exec_cycles": exec_cycles})
        save()
        if diagnostics is not None:
            diagnostics(run_dir, seg.seg_id, seg.core_id, data, log_path,
                        wall=wall, exec_cycles=exec_cycles, clock=clock,
                        kv_before_fp=before_fp)
        return data, True

    segs = {s.seg_id: s for s in plan.segments}
    # prompt 交接=首核整块输出；decode 行宽=embedding 搬运字节数（单 token hidden）。
    prompt_n = max(t.nbytes for t in plan.transfers if t.kind == "hidden_handoff")
    decode_n = next(t.nbytes for t in plan.transfers if t.kind == "embedding")

    # ===== Prompt =====
    c0_out, _ = execute(segs["prompt:c0"], prompt_n)
    runner.write_input_file(segs["prompt:c1"].core_id, "hidden_states", c0_out)
    c1_out, _ = execute(segs["prompt:c1"], prompt_n)

    def run_head(seg_id, hidden):
        seg = segs[seg_id]
        full = bytearray(prompt_n)
        full[0:len(hidden)] = hidden
        io = runner.core_io(seg.core_id)
        runner.write_input_file(seg.core_id, "hidden_states", bytes(full))
        logits, ran = execute(seg, io["outputs"]["lm_output"][1],
                              in_addr=io["inputs"]["hidden_states"][0])
        if ran:
            state["done"][seg_id]["token"] = int(np.argmax(
                np.frombuffer(logits, dtype=np.float16)))
            save()
        return state["done"][seg_id]["token"]

    def head_input(hidden_bytes):
        # prompt hidden 按 seq=1024 行铺开，取最后一个 prompt token 所在行。
        row = prompt_n // 1024
        return hidden_bytes[prompt_len * row:(prompt_len + 1) * row]

    tokens = [run_head("head:0", head_input(c1_out))]

    # ===== Decode =====
    for step in range(steps if tokens[-1] != eos_token else 0):
        row = _embed_row(embed_path, tokens[-1], decode_n)
        runner.write_input_file(0, "embedding_row", row)
        c0_dec, _ = execute(segs[f"decode:{step}:c0"], decode_n)
        # 输入文件必须写满本核 hidden_states 尺寸（装载按 ini 的 size 整块读）。
        full = bytearray(prompt_n)
        full[0:len(c0_dec)] = c0_dec
        runner.write_input_file(1, "hidden_states", bytes(full))
        c1_dec, _ = execute(segs[f"decode:{step}:c1"], decode_n)
        tokens.append(run_head(f"head:{step + 1}", c1_dec))
        if tokens[-1] == eos_token:
            break
    save()
    _write_cascade_artifacts(run_dir, state, tokens)
    return tokens


def _write_cascade_artifacts(run_dir: Path, state: dict, tokens: list[int]) -> None:
    """落盘 tokens 与每核分段计数/时间线（判定与评估的唯一输入）。

    计数与时间线取 timing 的权威投影：重跑段的同物异名记录只保留最后一条，
    历史仍在 state.json 里供审计。cycles 口径 exec（TIMER）> device > CSV。"""
    out = Path(run_dir) / "run"
    out.mkdir(parents=True, exist_ok=True)
    (out / "argmax_results.bin").write_bytes(
        struct.pack(f"<{len(tokens)}H", *tokens))
    rows = authoritative_timing(state["timing"])
    cores = sorted({int(t["core"]) for t in rows})
    for core in cores:
        mine = [t for t in rows if int(t["core"]) == core]
        (out / f"performance_counters_core{core}.txt").write_text(
            "".join(f"{t['seg']}: {t.get('exec_cycles') or t.get('device_cycles') or t['cycles']} cycles\n"
                    for t in mine))
        with (out / f"segment_timeline_core{core}.jsonl").open("w") as handle:
            for row in mine:
                handle.write(json.dumps(row) + "\n")


def _overlap_summary(rows: list[dict]) -> dict:
    """两核同层段时间线重叠判定（R5 的并行证据）。rows 是同层的两条记录。"""
    if len(rows) != 2:
        return {"overlap_s": 0.0, "ratio": 0.0}
    start = max(r["t_submit"] for r in rows)
    end = min(r["t_done"] for r in rows)
    overlap = max(0.0, end - start)
    shorter = min(r["t_done"] - r["t_submit"] for r in rows)
    return {"overlap_s": overlap, "ratio": overlap / shorter if shorter > 0 else 0.0}


def parallel_overlap_summary(state: dict) -> dict:
    """全量配对统计：每对同层两核段是否执行区间重叠。"""
    rows = authoritative_timing(state["timing"])
    groups: dict[str, list[dict]] = {}
    for row in rows:
        seg = row["seg"]
        if re.match(r"(prompt|decode:\d+):L\d+:(attn|mlp):c[01]", seg) \
                or re.match(r"head:\d+:c[01]", seg):
            groups.setdefault(re.sub(r":c[01]$", "", seg), []).append(row)
    pairs = [v for v in groups.values() if len(v) == 2 and
             all("t_submit" in r for r in v)]
    summary = [_overlap_summary(p) for p in pairs]
    return {"pairs": len(pairs),
            "overlapping_pairs": sum(1 for s in summary if s["ratio"] > 0.5),
            "min_overlap_ratio": min((s["ratio"] for s in summary), default=0.0)}


def run_tp2_processes(plan, runner: SegmentProcessRunner, steps: int, *,
                      run_dir: Path, embed_path: Path, prompt_len: int,
                      eos_token: int = _EOS_TOKEN,
                      fresh: bool = False,
                      diagnostics=None) -> list[int]:
    """TP2 编排（设计 4.7 的段式进程版）：两核同层分片并行，host 归约+残差。

    每层 attn/mlp 两段各拉起两核进程并行执行（时间线重叠见 tp2_overlap.json），
    部分和相加再加残差得下一层输入；head 列切拼接后 argmax。KV 由 app 分片段
    自己按层快照/恢复（kv_core{N}/kv_L{layer}.bin）。"""
    run_dir = Path(run_dir)
    results = run_dir / "results"
    state_path = results / "state.json"
    prompt_snap = results / "prompt_input_core0.bin"
    if fresh:
        state_path.unlink(missing_ok=True)
        prompt_snap.unlink(missing_ok=True)
    for core in range(plan.num_cores):
        (run_dir / f"kv_core{core}").mkdir(parents=True, exist_ok=True)
    results.mkdir(parents=True, exist_ok=True)
    state = _load_state(state_path)

    def save():
        state_path.write_text(json.dumps(state, indent=1, ensure_ascii=False))

    segs = {s.seg_id: s for s in plan.segments}
    io = [runner.core_io(c) for c in range(plan.num_cores)]
    prompt_n = io[0]["inputs"]["hidden_states"][1]
    decode_n = io[0]["inputs"]["embedding_row"][1]
    logits_n = io[0]["outputs"]["lm_output"][1]     # tp2 每核半宽
    layers = range(plan.segments[0].layer_lo,
                   max(s.layer_hi for s in plan.segments))

    def out_addr_of(seg):
        return seg.out_addrs.get("hidden", seg.out_addrs.get("lm_output"))

    def execute_pair(pair, cmd, stage, read_n, line_idx=0, kv_layer=None):
        """并行下发两核成对段；先查断点，缺的段两核同时拉起。

        返回按 core 序的输出字节列表。`kv_layer` 给 attn 分片段按层快照指纹；
        attn/mlp 段按 seg.layer_lo 传层号（argv[14]，app 据此定位分片注册表）。"""
        outs: list = [None] * len(pair)
        pending = []
        for core, seg in enumerate(pair):
            if seg.seg_id in state["done"]:
                outs[core] = Path(state["done"][seg.seg_id]["out"]).read_bytes()
            else:
                pending.append((core, seg))
        finished: list[dict] = []
        kv_name = f"kv_L{kv_layer}.bin" if kv_layer is not None else "kv.bin"
        layer = pair[0].layer_lo if cmd in (CMD["attn_shard"],
                                            CMD["mlp_shard"]) else None

        def work(core, seg):
            kv_path = run_dir / f"kv_core{core}" / kv_name
            before_fp = _kv_fp_file(kv_path)
            t_submit = time.time()
            log_stub = run_dir / "app.log"
            wall, data, clock, exec_cycles, log_path = runner.run(
                core, cmd, seg.in_addrs["hidden"], out_addr_of(seg), read_n,
                line_idx=line_idx, kv_dir=str(run_dir / f"kv_core{core}"),
                stage=stage, layer=layer)
            finished.append({"seg": seg.seg_id, "core": core, "wall_s": wall,
                             "cycles": 0, "device_cycles": clock,
                             "exec_cycles": exec_cycles,
                             "t_submit": t_submit, "t_done": time.time(),
                             "data": data, "log": str(log_path),
                             "kv_before_fp": before_fp})
            outs[core] = data

        threads = [threading.Thread(target=work, args=(c, s))
                   for c, s in pending]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        if not all(outs):
            # 空输出先于提交拦截：失败段不能进 state，否则续跑会复用坏结果
            raise RuntimeError(f"{pair[0].seg_id} 所在段对有核无输出")
        for row in finished:
            path = results / f"seg_{row['seg'].replace(':', '_')}.bin"
            path.write_bytes(row["data"])
            seg = segs[row["seg"]]
            state["done"][row["seg"]] = {
                "out": str(path), "bytes": len(row["data"]),
                "wall_s": row["wall_s"], "cycles": 0,
                "device_cycles": row["device_cycles"],
                "exec_cycles": row["exec_cycles"],
                "t_submit": row["t_submit"], "t_done": row["t_done"]}
            state["timing"].append({k: v for k, v in row.items()
                                    if k not in ("data", "log", "kv_before_fp")})
            if diagnostics is not None:
                diagnostics(run_dir, row["seg"], row["core"], row["data"],
                            Path(row["log"]), wall=row["wall_s"],
                            exec_cycles=row["exec_cycles"],
                            clock=row["device_cycles"],
                            kv_before_fp=row["kv_before_fp"],
                            kv_name=kv_name)
        if finished:
            save()
        return outs

    def reduce(x: np.ndarray, partials: list[bytes]) -> np.ndarray:
        # host 归约 + 残差：fp16 口径，顺序 = 部分和相加后再加残差。
        p0 = np.frombuffer(partials[0], dtype=np.float16)
        p1 = np.frombuffer(partials[1], dtype=np.float16)
        return (x + (p0 + p1)).astype(np.float16)

    def fanout(x: np.ndarray):
        # 两核下一段都吃同一份归约结果；输入文件必须写满 hidden_states 尺寸。
        raw = x.tobytes()
        full = raw + bytes(prompt_n - len(raw))
        for core in (0, 1):
            runner.write_input_file(core, "hidden_states", full)

    def run_head(head_no, phase, x):
        pair = [segs[f"head:{head_no}:c{c}"] for c in (0, 1)]
        if phase == "prompt":
            # head 只吃最后一个 prompt 位置的行（与 PP2 的 head_input 同口径）
            row_elems = decode_n // 2
            row = x[prompt_len * row_elems:(prompt_len + 1) * row_elems].tobytes()
            full = row + bytes(prompt_n - len(row))
            for core in (0, 1):
                runner.write_input_file(core, "hidden_states", full)
        outs = execute_pair(pair, CMD["head_shard"], "head", logits_n)
        logits = np.concatenate(
            [np.frombuffer(o, dtype=np.float16) for o in outs])
        token = int(np.argmax(logits))
        state["done"][f"head:{head_no}:c0"]["token"] = token
        save()
        return token

    tokens: list[int] = []
    for phase, line_idx_of in [("prompt", lambda: 0)] + \
            [(f"decode:{k}", lambda k=k: prompt_len + 1 + k) for k in range(steps)]:
        if phase == "prompt":
            # 原始 prompt 输入快照：fanout 每层覆盖两核 hidden_states 输入文件，
            # 断点重启必须从快照取 x，否则归约链从被覆盖的中间态起算（静默算错）。
            if not prompt_snap.is_file():
                prompt_snap.write_bytes(
                    Path(io[0]["inputs"]["hidden_states_file"]).read_bytes())
            x = np.frombuffer(prompt_snap.read_bytes(), dtype=np.float16)
            read_n, stage = prompt_n, "prompt"
        else:
            if tokens and tokens[-1] == eos_token:
                break
            row = _embed_row(embed_path, tokens[-1], decode_n)
            for core in (0, 1):
                runner.write_input_file(core, "embedding_row", row)
            x = np.frombuffer(row, dtype=np.float16)
            read_n, stage = decode_n, "decode"
        for layer in layers:
            attn = [segs[f"{phase}:L{layer}:attn:c{c}"] for c in (0, 1)]
            mlp = [segs[f"{phase}:L{layer}:mlp:c{c}"] for c in (0, 1)]
            outs = execute_pair(attn, CMD["attn_shard"], stage, read_n,
                                line_idx=line_idx_of(), kv_layer=layer)
            x = reduce(x, outs)
            fanout(x)
            outs = execute_pair(mlp, CMD["mlp_shard"], stage, read_n,
                                line_idx=line_idx_of())
            x = reduce(x, outs)
            fanout(x)
        head_no = 0 if phase == "prompt" else int(phase.split(":")[1]) + 1
        tokens.append(run_head(head_no, phase, x))
    _write_cascade_artifacts(run_dir, state, tokens)
    (results / "tp2_overlap.json").write_text(
        json.dumps(parallel_overlap_summary(state), indent=1, ensure_ascii=False))
    return tokens


class StepProxySession:
    """run_tools_step.py 子进程的会话代理：JSON 行协议走本地 socket
    （fd1/fd2 让给 budi 的 app 日志桥接并落盘，协议通道完全独立）。"""

    def __init__(self, process, port):
        import socket
        self.process = process
        self.sock = socket.create_connection(("127.0.0.1", port), timeout=86400)
        self.reader = self.sock.makefile("r")
        self.writer = self.sock.makefile("w")

    def _ask(self, request: dict):
        import json as _json
        self.writer.write(_json.dumps(request) + "\n")
        self.writer.flush()
        return _json.loads(self.reader.readline())

    def write(self, space, addr, data):
        self._ask({"op": "write", "space": space, "addr": addr,
                   "hex": data.hex()})

    def read(self, space, addr, size):
        return bytes.fromhex(self._ask({"op": "read", "space": space,
                                        "addr": addr, "size": size})["hex"])

    def write_file(self, space, addr, path):
        self._ask({"op": "write_file", "space": space, "addr": addr, "path": str(path)})

    def read_file(self, space, addr, size, path):
        self._ask({"op": "read_file", "space": space, "addr": addr,
                   "size": size, "path": str(path)})

    def set_symbol_bp(self, symbol):
        self._ask({"op": "set_symbol_bp", "symbol": symbol})

    def resume_sync(self):
        self._ask({"op": "resume_sync"})

    def close(self):
        """关协议通道；run_tools_step 读到 EOF 后走 finally 清理并停掉 TCS。

        直接 kill 包装进程会让 TCS 变孤儿常驻（v20 慢段的根因），必须优雅关。
        """
        try:
            self.writer.close()
        finally:
            self.reader.close()
            self.sock.close()


def plan_control_addr(run_dir: Path, core: int) -> int:
    """从本核 control_map.ini 读控制块地址（spawn 时要传给 app 的 argv[7]）。"""
    import configparser

    parser = configparser.ConfigParser()
    parser.read(Path(run_dir) / f"ddr_core{core}" / "control_map.ini")
    return int(dict(parser["control"])["address"], 16)


def main() -> int:
    """命令行入口：run_cascade.sh 第 5 步调用。"""
    import argparse
    import os

    from cascade.plan import from_bundle

    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--strategy", choices=("pp2", "tp2"), required=True)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--elf", default=None, help="段式 runner ELF（缺省取 STEP_ELF）")
    parser.add_argument("--embed", default=None, help="embedding 权重文件")
    parser.add_argument("--fresh", action="store_true", help="忽略断点状态重新跑")
    args = parser.parse_args()
    run_dir = Path(args.run_dir)
    plan = from_bundle(args.bundle, run_dir, steps=args.steps)
    workspace = Path(args.workspace)
    elf = Path(args.elf or os.environ.get(
        "STEP_ELF", workspace / "src/llama_runner/build/Llama2StepRunner.elf"))
    runner = SegmentProcessRunner(workspace, run_dir, elf)
    embed = Path(args.embed) if args.embed else \
        run_dir / "inputs" / "outputs" / "embed_weights.bin"
    index_file = runner.core_io(0)["inputs"]["index_file"]
    prompt_len = struct.unpack("<h", Path(index_file).read_bytes()[:2])[0]
    if args.strategy == "tp2":
        tokens = run_tp2_processes(plan, runner, args.steps, run_dir=run_dir,
                                   embed_path=embed, prompt_len=prompt_len,
                                   fresh=args.fresh,
                                   diagnostics=_append_diagnostics)
    else:
        tokens = run_pp2_processes(plan, runner, args.steps, run_dir=run_dir,
                                   embed_path=embed, prompt_len=prompt_len,
                                   fresh=args.fresh,
                                   diagnostics=_append_diagnostics)
    print(f"tokens: {tokens}")
    return 0 if tokens else 1


if __name__ == "__main__":
    raise SystemExit(main())
