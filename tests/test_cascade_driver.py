"""cascade/driver.py 的 PP2 段式进程编排与 TP2 归约测试（设计 4.6/4.7、7.2）。

PP2 用可注入的 FakeRunner 替掉 TCS 子进程：只验证段序、段间交接、
断点续跑与产物落盘；真实进程由工作区 run_tools_step.py 提供。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from cascade.driver import SegmentProcessRunner, run_pp2_processes
from cascade.plan import from_bundle

# prompt hidden 按 seq=1024 行铺开：每行 4 个 fp16 = 8B，整块 8192B。
PROMPT_HIDDEN = 1024 * 4 * 2
DECODE_HIDDEN = 4 * 2
LOGITS = 64          # 32 个 fp16 logits，够放 token=13


def _write_ini(path: Path, sections: list[dict]) -> None:
    lines = []
    for i, kv in enumerate(sections):
        lines.append(f"[{i}]")
        lines += [f"{k}={v}" for k, v in kv.items()]
    path.write_text("\n".join(lines) + "\n")


def _maps(run_dir: Path, core: int, *, layers: int,
          prompt_hidden: int = PROMPT_HIDDEN) -> None:
    d = run_dir / f"ddr_core{core}"
    d.mkdir(parents=True)
    base = 0x100000000 + core * (1 << 33)
    _write_ini(d / "inputs_map.ini", [
        {"input_file": str(run_dir / f"h{core}.bin"), "input_name": "hidden_states",
         "address": hex(base + 0x2000), "size": prompt_hidden},
        {"input_file": str(run_dir / f"row{core}.bin"), "input_name": "embedding_row",
         "address": hex(base + 0x5000), "size": DECODE_HIDDEN},
        {"input_file": str(run_dir / f"idx{core}.bin"), "input_name": "index",
         "address": hex(base + 0x6000), "size": 2}])
    outputs = [{"output_name": "output", "address": hex(base + 0x10000),
                "size": prompt_hidden}]
    outputs += [{"output_name": f"key_cache_{i}",
                 "address": hex(base + 0x20000 + i * 64), "size": 64}
                for i in range(layers)]
    outputs.append({"output_name": "lm_output", "address": hex(base + 0x30000),
                    "size": LOGITS})
    _write_ini(d / "outputs_map.ini", outputs)
    (d / "control_map.ini").write_text(
        f"[control]\ncore_id={core}\naddress={hex(base + 0x1000)}\n")


def _pp2_plan(tmp_path: Path, steps: int = 3):
    bundle = tmp_path / "bundle.json"
    bundle.write_text(json.dumps({
        "graphs": [], "cascade": {"strategy": "pp2", "num_cores": 2,
                                  "core_layers": [[0, 1], [1, 2]],
                                  "lm_head_core": 1}}))
    _maps(tmp_path, 0, layers=1)
    _maps(tmp_path, 1, layers=1)
    return from_bundle(bundle, tmp_path, steps=steps)


class FakeRunner(SegmentProcessRunner):
    """只替掉 TCS 子进程：core_io/write_input_file 用真实实现（落到 tmp 文件）。"""

    def __init__(self, run_dir: Path, token: int = 13):
        super().__init__(workspace=Path("/nonexistent"), run_dir=run_dir,
                         elf=Path("/nonexistent"))
        self.token = token
        self.calls: list[tuple] = []
        self.input_writes: list[tuple] = []

    def write_input_file(self, core, name, data):
        self.input_writes.append((core, name, bytes(data)))
        return super().write_input_file(core, name, data)

    def run(self, core, cmd, in_addr, out_addr, read_size, line_idx=0, kv_dir="",
            stage="", layer=None):
        self.calls.append((core, cmd, read_size, line_idx, kv_dir, in_addr,
                           out_addr, stage, layer))
        if cmd == 3:
            logits = np.zeros(read_size // 2, dtype=np.float16)
            logits[self.token] = 1.0
            return 1.0, logits.tobytes(), 900, 880, "/no/log"
        return 2.0, bytes([core + 1]) * read_size, 800 + len(self.calls), \
            780 + len(self.calls), "/no/log"


def _embed_file(tmp_path: Path) -> Path:
    embed = tmp_path / "embed_weights.bin"
    embed.write_bytes(bytes(range(256)) * 64)
    return embed


def test_pp2_processes_flow_and_artifacts(tmp_path) -> None:
    """PP2 段序、交接文件、每段 cycles 与断点状态齐全。"""
    plan = _pp2_plan(tmp_path, steps=2)
    runner = FakeRunner(tmp_path)
    tokens = run_pp2_processes(plan, runner, 2, run_dir=tmp_path,
                               embed_path=_embed_file(tmp_path), prompt_len=1)
    assert tokens == [13] * 3
    # 段序：prompt c0/c1 → head:0 → 每步 c0/c1 → head
    assert [(c, cmd) for c, cmd, *_ in runner.calls] == [
        (0, 1), (1, 1), (1, 3), (0, 2), (1, 2), (1, 3), (0, 2), (1, 2), (1, 3)]
    # decode 行号 = prompt_len + 1 + 步号
    assert [line for _, cmd, _, line, *_ in runner.calls if cmd == 2] == [2, 2, 3, 3]
    # 交接：core1 的 hidden_states 先收到 core0 的整块 prompt 输出；
    # decode 的 8B 行也必须补满输入文件（装载按 ini 的 size 整块读）。
    first_handoff = runner.input_writes[0]
    assert first_handoff == (1, "hidden_states", b"\x01" * PROMPT_HIDDEN)
    hidden_writes = [d for c, n, d in runner.input_writes if n == "hidden_states"]
    assert all(len(d) == PROMPT_HIDDEN for d in hidden_writes)
    # 写出顺序：prompt 交接 → prompt head 入参 → decode 交接 → head 入参……
    assert hidden_writes[0][:DECODE_HIDDEN] == b"\x01" * DECODE_HIDDEN
    assert hidden_writes[1][:DECODE_HIDDEN] == b"\x02" * DECODE_HIDDEN
    assert hidden_writes[2][:DECODE_HIDDEN] == b"\x01" * DECODE_HIDDEN
    # embedding 行来自文件第 13 行（行宽 8B）
    emb_writes = [d for c, n, d in runner.input_writes if n == "embedding_row"]
    assert emb_writes and emb_writes[0] == \
        (tmp_path / "embed_weights.bin").read_bytes()[13 * DECODE_HIDDEN:14 * DECODE_HIDDEN]
    state = json.loads((tmp_path / "results" / "state.json").read_text())
    assert len(state["done"]) == 9
    assert state["done"]["head:1"]["token"] == 13
    # 每段记录 EndOfExecution 设备时钟（评估用），不只是 CSV 尾值
    assert all(t["device_cycles"] > 0 for t in state["timing"])
    # 判定/评估产物：tokens 二进制 + 每核分段计数与时间线
    import struct
    raw = (tmp_path / "run" / "argmax_results.bin").read_bytes()
    assert struct.unpack(f"<{len(raw) // 2}H", raw) == (13, 13, 13)
    assert "prompt:c0" in (tmp_path / "run" / "performance_counters_core0.txt").read_text()
    assert (tmp_path / "run" / "segment_timeline_core1.jsonl").read_text().count("\n") == 6
    # KV 快照目录由编排器负责创建（app 不建目录）
    assert (tmp_path / "kv_core0").is_dir() and (tmp_path / "kv_core1").is_dir()
    assert [t["wall_s"] for t in state["timing"]] == [2.0, 2.0, 1.0] * 3
    transfers = [json.loads(line) for line in
                 (tmp_path / "run" / "transfer_log.jsonl").read_text().splitlines()]
    assert transfers and all(t["kind"] in ("hidden_states", "embedding_row")
                             for t in transfers)


def test_stop_session_closes_then_waits() -> None:
    """段末清理：先关协议通道让 TCS 自行退出，超时才强杀包装进程。"""
    import subprocess

    from cascade.driver import _stop_session

    class FakeSession:
        closed = False

        def close(self):
            self.closed = True

    class FakeProc:
        def __init__(self):
            self.waited = False
            self.killed = False

        def wait(self, timeout):
            self.waited = True

        def kill(self):
            self.killed = True

    session, proc = FakeSession(), FakeProc()
    _stop_session(session, proc)
    assert session.closed and proc.waited and not proc.killed

    class HangingProc(FakeProc):
        def wait(self, timeout):
            raise subprocess.TimeoutExpired("runner", timeout)

    hanging = HangingProc()
    _stop_session(FakeSession(), hanging)
    assert hanging.killed


def test_pp2_processes_eos_stops(tmp_path) -> None:
    """prompt 首 token 命中 EOS(2)：不再进 decode。"""
    plan = _pp2_plan(tmp_path, steps=3)
    runner = FakeRunner(tmp_path, token=2)
    tokens = run_pp2_processes(plan, runner, 3, run_dir=tmp_path,
                               embed_path=_embed_file(tmp_path), prompt_len=1)
    assert tokens == [2]
    assert [(c, cmd) for c, cmd, *_ in runner.calls] == [(0, 1), (1, 1), (1, 3)]


def test_pp2_processes_resume_from_state(tmp_path) -> None:
    """同样的 state 再跑一次：所有段复用落盘输出，不再启动仿真。"""
    plan = _pp2_plan(tmp_path, steps=1)
    runner = FakeRunner(tmp_path)
    embed = _embed_file(tmp_path)
    first = run_pp2_processes(plan, runner, 1, run_dir=tmp_path,
                              embed_path=embed, prompt_len=1)
    runner.calls.clear()
    second = run_pp2_processes(plan, runner, 1, run_dir=tmp_path,
                               embed_path=embed, prompt_len=1)
    assert second == first == [13, 13]
    assert runner.calls == []


def test_tp2_reduce_and_argmax(tmp_path) -> None:
    """TP2 段式进程版：两核同层并行、host 归约+残差、logits 拼接 argmax。

    蓝图 2 层：每段 layer 必须随层递增传给 runner（argv[14] 契约），
    否则 app 侧会静默跑第 0 层图（round02 问题1）。"""
    from cascade.driver import run_tp2_processes

    bundle = tmp_path / "bundle.json"
    bundle.write_text(json.dumps({
        "graphs": [], "cascade": {"strategy": "tp2", "num_cores": 2, "tp_width": 2,
                                  "core_layers": [[0, 2], [0, 2]],
                                  "lm_head_core": None}}))
    _maps(tmp_path, 0, layers=2, prompt_hidden=32)
    _maps(tmp_path, 1, layers=2, prompt_hidden=32)
    plan = from_bundle(bundle, tmp_path, steps=1)

    class Tp2FakeRunner(FakeRunner):
        """部分和按核区分：归约结果可手推；记录并发数验证真并行下发。"""

        def __init__(self, run_dir, token=13):
            super().__init__(run_dir, token)
            self.inflight = 0
            self.max_inflight = 0

        def run(self, core, cmd, in_addr, out_addr, read_size, line_idx=0,
                kv_dir="", stage="", layer=None):
            import threading
            self.calls.append((core, cmd, read_size, line_idx, kv_dir,
                               in_addr, out_addr, stage, layer))
            with threading.Lock():
                self.inflight += 1
                self.max_inflight = max(self.max_inflight, self.inflight)
            try:
                import time
                time.sleep(0.05)
                if cmd == 6:                     # head 分片：每核半宽 logits
                    logits = np.zeros(read_size // 2, dtype=np.float16)
                    if core == 0:
                        logits[self.token] = 1.0   # token 落在前半段
                    return 1.0, logits.tobytes(), 900, 880, "/no/log"
                delta = 1.0 + core                # c0=+1，c1=+2 的部分和
                part = np.full(read_size // 2, delta, dtype=np.float16)
                return 2.0, part.tobytes(), 800, 780, "/no/log"
            finally:
                with threading.Lock():
                    self.inflight -= 1

    # 行间不同：head 输入必须是最后 prompt 位置（第 1 行）的归约结果
    rows = np.array([[1] * 4, [2] * 4, [3] * 4, [4] * 4], dtype=np.float16)
    (tmp_path / "h0.bin").write_bytes(rows.tobytes())
    (tmp_path / "h1.bin").write_bytes(rows.tobytes())
    runner = Tp2FakeRunner(tmp_path)
    tokens = run_tp2_processes(plan, runner, 1, run_dir=tmp_path,
                               embed_path=_embed_file(tmp_path), prompt_len=1)
    assert tokens == [13, 13]
    # 段序：每相逐层 attn→mlp，末尾 head，两核成对（并行下发，对内完成序不保证）
    pairs = [sorted((c, cmd) for c, cmd, *_ in runner.calls[i:i + 2])
             for i in range(0, len(runner.calls), 2)]
    assert pairs == [[(0, 4), (1, 4)], [(0, 5), (1, 5)]] * 2 \
        + [[(0, 6), (1, 6)]] + [[(0, 4), (1, 4)], [(0, 5), (1, 5)]] * 2 \
        + [[(0, 6), (1, 6)]]
    # layer 接线：attn/mlp 段逐层递增（0,1），head 段不传层
    shard_layers = [(cmd, layer) for _, cmd, _, _, _, _, _, _, layer
                    in runner.calls if cmd in (4, 5)]
    assert [layer for _, layer in shard_layers] == [0, 0, 0, 0, 1, 1, 1, 1] * 2
    assert all(layer is None for _, cmd, *_ , stage, layer in runner.calls
               if cmd == 6)
    # 真并行：同层两核段进程同时在场
    assert runner.max_inflight == 2
    # decode 行号 = prompt_len + 1 + 步号（decode attn 段带 stage）
    decode_attn = [r for r in runner.calls if r[1] == 4
                   and r[2] == DECODE_HIDDEN]
    assert all(line == 2 for *_, line, kv, _in, _out, _s, _l in decode_attn)
    # host 归约 + 残差：每层 x += (1+2)。初值行 [1,2,3,4] ⇒ L0 后 [4,5,6,7]，
    # L1 后 [10,11,12,13]（attn/mlp 各加一次 3）
    hidden_writes = [d for c, n, d in runner.input_writes if n == "hidden_states"]
    assert len(hidden_writes) >= 10               # 4 次 fanout×2 核×2 相 + head×2
    assert list(np.frombuffer(hidden_writes[0][:8], dtype=np.float16)) == [4.0] * 4
    assert list(np.frombuffer(hidden_writes[1][:8], dtype=np.float16)) == [4.0] * 4
    assert list(np.frombuffer(hidden_writes[2][:8], dtype=np.float16)) == [7.0] * 4
    assert list(np.frombuffer(hidden_writes[3][:8], dtype=np.float16)) == [7.0] * 4
    assert list(np.frombuffer(hidden_writes[4][:8], dtype=np.float16)) == [10.0] * 4
    assert list(np.frombuffer(hidden_writes[6][:8], dtype=np.float16)) == [13.0] * 4
    # prompt 相的 head 入参 = 最后一个 prompt 位置（第 1 行 = 14），不是整块首行
    assert list(np.frombuffer(hidden_writes[8][:8], dtype=np.float16)) == [14.0] * 4
    assert list(np.frombuffer(hidden_writes[9][:8], dtype=np.float16)) == [14.0] * 4
    # 断点与产物
    state = json.loads((tmp_path / "results" / "state.json").read_text())
    assert state["done"]["head:0:c0"]["token"] == 13
    import struct
    raw = (tmp_path / "run" / "argmax_results.bin").read_bytes()
    assert struct.unpack(f"<{len(raw) // 2}H", raw) == (13, 13)
    overlap = json.loads((tmp_path / "results" / "tp2_overlap.json").read_text())
    assert overlap["pairs"] == 10 and overlap["overlapping_pairs"] == 10


def test_tp2_processes_eos_and_resume(tmp_path) -> None:
    """TP2：EOS 提前停 + 断点续跑（复用段不再启动进程）。"""
    from cascade.driver import run_tp2_processes

    bundle = tmp_path / "bundle.json"
    bundle.write_text(json.dumps({
        "graphs": [], "cascade": {"strategy": "tp2", "num_cores": 2, "tp_width": 2,
                                  "core_layers": [[0, 1], [0, 1]],
                                  "lm_head_core": None}}))
    _maps(tmp_path, 0, layers=1, prompt_hidden=32)
    _maps(tmp_path, 1, layers=1, prompt_hidden=32)
    plan = from_bundle(bundle, tmp_path, steps=2)
    (tmp_path / "h0.bin").write_bytes(np.zeros(16, dtype=np.float16).tobytes())
    (tmp_path / "h1.bin").write_bytes(np.zeros(16, dtype=np.float16).tobytes())

    class Tp2FakeRunner(FakeRunner):
        def run(self, core, cmd, in_addr, out_addr, read_size, line_idx=0,
                kv_dir="", stage="", layer=None):
            self.calls.append((core, cmd, read_size, line_idx, kv_dir,
                               in_addr, out_addr, stage, layer))
            if cmd == 6:
                logits = np.zeros(read_size // 2, dtype=np.float16)
                logits[self.token] = 1.0
                return 1.0, logits.tobytes(), 900, 880, "/no/log"
            return 2.0, bytes([core + 1]) * read_size, 800, 780, "/no/log"

    runner = Tp2FakeRunner(tmp_path, token=2)      # 首 token 即 EOS
    embed = _embed_file(tmp_path)
    tokens = run_tp2_processes(plan, runner, 2, run_dir=tmp_path,
                               embed_path=embed, prompt_len=1)
    assert tokens == [2]
    pairs = [sorted((c, cmd) for c, cmd, *_ in runner.calls[i:i + 2])
             for i in range(0, len(runner.calls), 2)]
    assert pairs == [[(0, 4), (1, 4)], [(0, 5), (1, 5)], [(0, 6), (1, 6)]]
    # 续跑：全部段已在 state，零进程启动，tokens 一致
    runner.calls.clear()
    again = run_tp2_processes(plan, runner, 2, run_dir=tmp_path,
                              embed_path=embed, prompt_len=1)
    assert again == [2] and runner.calls == []


def test_tp2_prompt_interrupt_resume_keeps_input(tmp_path) -> None:
    """TP2 prompt 相中断续跑：x 必须取原始输入快照，不能读被 fanout 覆盖的文件。

    删去中间层的 done 记录后重启，pending 段读到的输入文件须与首跑逐字节一致
    （round02 问题2：原始 prompt 输入文件同时承担装载源与重启推导源）。"""
    from cascade.driver import run_tp2_processes

    bundle = tmp_path / "bundle.json"
    bundle.write_text(json.dumps({
        "graphs": [], "cascade": {"strategy": "tp2", "num_cores": 2, "tp_width": 2,
                                  "core_layers": [[0, 2], [0, 2]],
                                  "lm_head_core": None}}))
    _maps(tmp_path, 0, layers=2, prompt_hidden=32)
    _maps(tmp_path, 1, layers=2, prompt_hidden=32)
    plan = from_bundle(bundle, tmp_path, steps=0)

    class RecordingRunner(FakeRunner):
        """记录每次 run 时两核 hidden_states 输入文件的字节（= 段的真实入参）。"""

        def __init__(self, run_dir):
            super().__init__(run_dir)
            self.inputs_at_run: list[dict] = []

        def run(self, core, cmd, in_addr, out_addr, read_size, line_idx=0,
                kv_dir="", stage="", layer=None):
            self.calls.append((core, cmd, read_size, line_idx, kv_dir,
                               in_addr, out_addr, stage, layer))
            if stage == "prompt" and cmd in (4, 5):
                self.inputs_at_run.append({
                    "layer": layer, "cmd": cmd,
                    "h0": (tmp_path / "h0.bin").read_bytes(),
                    "h1": (tmp_path / "h1.bin").read_bytes()})
            if cmd == 6:
                logits = np.zeros(read_size // 2, dtype=np.float16)
                logits[self.token] = 1.0
                return 1.0, logits.tobytes(), 900, 880, "/no/log"
            return 2.0, bytes([core + 1]) * read_size, 800, 780, "/no/log"

    rows = np.array([[1] * 4, [2] * 4, [3] * 4, [4] * 4], dtype=np.float16)
    (tmp_path / "h0.bin").write_bytes(rows.tobytes())
    (tmp_path / "h1.bin").write_bytes(rows.tobytes())
    runner = RecordingRunner(tmp_path)
    run_tp2_processes(plan, runner, 0, run_dir=tmp_path,
                      embed_path=_embed_file(tmp_path), prompt_len=1)
    first = {(r["layer"], r["cmd"]): (r["h0"], r["h1"])
             for r in runner.inputs_at_run}
    assert set(first) == {(0, 4), (0, 5), (1, 4), (1, 5)}
    # 中断模拟：首跑后原始输入文件已被最后一步 fanout 覆盖（问题本体）
    assert (tmp_path / "h0.bin").read_bytes() != rows.tobytes()

    state = json.loads((tmp_path / "results" / "state.json").read_text())
    for seg_id in list(state["done"]):            # 删去 L1 与 head 的完成记录
        if ":L1:" in seg_id or seg_id.startswith("head:"):
            del state["done"][seg_id]
    (tmp_path / "results" / "state.json").write_text(json.dumps(state))

    runner2 = RecordingRunner(tmp_path)
    run_tp2_processes(plan, runner2, 0, run_dir=tmp_path,
                      embed_path=_embed_file(tmp_path), prompt_len=1)
    second = {(r["layer"], r["cmd"]): (r["h0"], r["h1"])
              for r in runner2.inputs_at_run}
    # pending 段（L1）的输入文件与首跑逐字节一致；L0 复用不再出现
    assert set(second) == {(1, 4), (1, 5)}
    assert second[(1, 4)] == first[(1, 4)] and second[(1, 5)] == first[(1, 5)]


def test_cascade_artifacts_use_authoritative_projection(tmp_path) -> None:
    """落盘产物按权威投影：同物异名的重跑段只留最后一条，不重复计数。"""
    from cascade.driver import _write_cascade_artifacts

    plan = _pp2_plan(tmp_path, steps=1)
    timing = [{"seg": s.seg_id, "core": s.core_id, "wall_s": 5.0, "cycles": 100,
               "device_cycles": 900, "exec_cycles": 0} for s in plan.segments]
    timing.append({"seg": "decode:0:lm", "core": 1, "wall_s": 1.0, "cycles": 9,
                   "device_cycles": 90, "exec_cycles": 0})   # 旧脚本的重复记录
    timing.append({"seg": "head:1", "core": 1, "wall_s": 2.0, "cycles": 7,
                   "device_cycles": 70, "exec_cycles": 50})  # driver 重跑在后
    _write_cascade_artifacts(tmp_path, {"timing": timing}, [13, 13])
    text = (tmp_path / "run" / "performance_counters_core1.txt").read_text()
    lines = [l for l in text.splitlines() if l.strip()]
    assert len(lines) == len([s for s in plan.segments if s.core_id == 1])
    # head:1 取最后记录（exec_cycles=50），且 decode:0:lm 不再另占一行
    assert "head:1: 50 cycles" in text and "decode:0:lm" not in text
    # 无 TIMER 的段回退到 device_cycles（旧段口径），再回退 CSV
    assert "prompt:c1: 900 cycles" in text


def test_segment_diagnostics_hook(tmp_path) -> None:
    """段后自检 hook：KV 指纹、diff、top5 落 diagnostics.jsonl；可注入替换。"""
    from cascade.driver import _append_diagnostics

    kv = tmp_path / "kv_core0" / "kv.bin"
    kv.parent.mkdir()
    kv.write_bytes(b"\x01" * 4096)
    log = tmp_path / "app_1.log"
    log.write_text("[STEPWISE] kv restore 6 rows from /x fp=0xabc\n"
                   "[STEPWISE] in_head=0x11 line_idx=6\n")
    _append_diagnostics(tmp_path, "decode:0:c0", 0, b"\x02" * 16, log,
                        wall=1.5, exec_cycles=100, clock=200,
                        kv_before_fp=0)
    rows = [json.loads(l) for l in
            (tmp_path / "diagnostics.jsonl").read_text().splitlines()]
    assert rows[-1]["seg"] == "decode:0:c0"
    assert rows[-1]["kv_before_fp"] == "0000000000000000"
    assert int(rows[-1]["kv_after_fp"], 16) > 0
    assert rows[-1]["app"]["in_head"] == ["11", "6"]
    # head 段记 top5
    logits = np.zeros(32, dtype=np.float16)
    logits[7] = 3.0
    _append_diagnostics(tmp_path, "head:0", 1, logits.tobytes(), log,
                        wall=0.1, exec_cycles=10, clock=20, kv_before_fp=0)
    rows = [json.loads(l) for l in
            (tmp_path / "diagnostics.jsonl").read_text().splitlines()]
    assert rows[-1]["top5"][0][0] == 7


def test_segment_diagnostics_tp2(tmp_path) -> None:
    """TP2 段自检等价生效：kv_L{n}.bin 指纹与 head:{n}:c{m} 的 top5（round02 问题7）。"""
    from cascade.driver import _append_diagnostics

    (tmp_path / "kv_core0").mkdir()
    (tmp_path / "kv_core0" / "kv_L3.bin").write_bytes(b"\x03" * 4096)
    log = tmp_path / "app_2.log"
    log.write_text("[STEPWISE] kv restore L3 from /x\n")
    _append_diagnostics(tmp_path, "prompt:L3:attn:c0", 0, b"\x02" * 16, log,
                        wall=1.0, exec_cycles=50, clock=90, kv_before_fp=0,
                        kv_name="kv_L3.bin")
    rows = [json.loads(l) for l in
            (tmp_path / "diagnostics.jsonl").read_text().splitlines()]
    assert rows[-1]["seg"] == "prompt:L3:attn:c0"
    assert int(rows[-1]["kv_after_fp"], 16) > 0
    assert (tmp_path / "hist" / "prompt_L3_attn_c0.bin").is_file()
    # head 分片（head:{n}:c{m} 命名）也记 top5
    logits = np.zeros(32, dtype=np.float16)
    logits[13] = 2.0
    _append_diagnostics(tmp_path, "head:1:c1", 1, logits.tobytes(), log,
                        wall=0.1, exec_cycles=10, clock=20, kv_before_fp=0)
    rows = [json.loads(l) for l in
            (tmp_path / "diagnostics.jsonl").read_text().splitlines()]
    assert rows[-1]["top5"][0][0] == 13
