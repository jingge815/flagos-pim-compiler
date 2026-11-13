#!/usr/bin/env python3
"""PIM 编译调优统一入口。

    python scripts/tune_pipeline.py --config examples/tune/quick.yaml
    python scripts/tune_pipeline.py --config examples/tune/quick.yaml --resume
    python scripts/tune_pipeline.py --evaluate-candidate <best.json> --cache-mode strict

配置是唯一状态：命令行高频覆盖先合并进生效配置并落盘，再开始编译。
`--dry-run` 只检查输入和候选计数，不加载 7B 权重、不编译也不仿真。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tune.candidate import CandidateSpace, model_dims
from tune.config import TuneConfig, cli_overrides, load_config, write_effective_config
from tune.inputs import (
    file_sha256,
    prepare_inputs,
    publish_inputs,
    validate_dry_inputs,
    verify_input_identity,
)
from tune.loop import run_tune, verify_resume_config, verify_resume_identity
from tune.report import (
    RoundRecord,
    compare_scores,
    read_rounds,
    write_best,
    write_summary,
)
from tune.steps import (
    CandidateArtifacts,
    compile_key,
    evaluate_candidate,
    sim_key,
    verify_candidate_identity,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=None, help="调优配置 YAML")
    parser.add_argument("--max-evaluations", type=int, default=None,
                        help="外层候选评估数上限")
    parser.add_argument("--max-time-s", type=float, default=None,
                        help="外层墙钟预算（秒）；只限制发起新评估")
    parser.add_argument("--algorithm", default=None,
                        help="外层搜索算法：enumerate/simulated_annealing/"
                             "genetic_algorithm/tabu_search")
    parser.add_argument("--profile", default=None, help="硬件画像 JSON 路径")
    parser.add_argument("--cache-mode", default=None, choices=("fast", "strict"))
    parser.add_argument("--dry-run", action="store_true",
                        help="只列合法候选与拒绝原因，不编译不仿真")
    parser.add_argument("--resume", action="store_true",
                        help="沿用输出目录里已有的响应日志继续")
    parser.add_argument("--output-dir", default=None, help="输出目录")
    parser.add_argument("--evaluate-candidate", default=None,
                        help="严格重编并重跑一个 best.json 记录的候选")
    parser.add_argument("--replay-rounds", action="store_true",
                        help="按轮次日志的真实评估顺序复跑并比较时延")
    args = parser.parse_args()
    if args.evaluate_candidate and (args.config or args.replay_rounds):
        parser.error("--evaluate-candidate 与 --config / --replay-rounds 互斥")
    if args.evaluate_candidate and args.dry_run:
        parser.error("--evaluate-candidate 不能与 --dry-run 同时使用")
    if not args.evaluate_candidate and not args.replay_rounds and not args.config:
        parser.error("必须给 --config，或使用 --evaluate-candidate / --replay-rounds")
    if (args.replay_rounds and not args.config
            and not args.output_dir):
        parser.error("--replay-rounds 需要 --config 或 --output-dir 定位生效配置")
    return args


def main() -> int:
    args = parse_args()
    if args.evaluate_candidate:
        return _evaluate_candidate(Path(args.evaluate_candidate), args)
    if args.replay_rounds:
        try:
            cfg = _replay_config(args)
        except (FileNotFoundError, ValueError) as exc:
            print(str(exc), file=sys.stderr)
            return 2
        return _replay_rounds(cfg)
    cfg = load_config(args.config, cli_overrides(args))
    space = CandidateSpace(cfg, model_dims(cfg.model.model_dir))
    if args.dry_run:
        return _dry_run(cfg, space)
    journal = cfg.output.dir / "responses.jsonl"
    # 先按已有日志判定续跑，再写生效配置：异配置不能覆盖原配置证据。
    if journal.exists() and not args.resume:
        print(
            f"输出目录已有调优日志：{journal}\n"
            "要继续请加 --resume，或换一个 --output-dir。", file=sys.stderr)
        return 2
    if journal.exists() and args.resume:
        try:
            verify_resume_config(cfg, journal)
        except ValueError as exc:
            print(str(exc), file=sys.stderr)
            return 3
    _prepare_environment()
    # 输入身份必须在写任何输出目录文件之前核对：异输入续跑不能先改写证据。
    inputs = prepare_inputs(cfg)
    if journal.exists() and args.resume:
        try:
            verify_resume_identity(cfg, inputs, journal)
        except ValueError as exc:
            print(f"续跑被拒绝：{exc}", file=sys.stderr)
            return 3
    inputs = publish_inputs(cfg, inputs)
    effective = write_effective_config(cfg)
    try:
        result = run_tune(cfg, inputs, space)
    except ValueError as exc:
        print(f"续跑被拒绝：{exc}", file=sys.stderr)
        return 3
    best_path = write_best(cfg.output.dir, list(result.records), extra={
        "effective_config": str(effective),
        # 扫描按生成时内容摘要核对历史生效配置，路径本身不是身份。
        "effective_config_sha256": file_sha256(effective),
        # 扫描必须从同一基线画像派生；硬件层搜索结果不改变这个基线。
        "baseline_profile_sha256": cfg.profile.profile_sha256(),
        "inputs": inputs.to_json(),
        "result_maturity": result.best.result_maturity if result.best else None,
        "search_coverage": result.coverage.to_json(),
        "reproduce": (
            f"python scripts/tune_pipeline.py --evaluate-candidate "
            f"{cfg.output.dir / 'best.json'} --cache-mode strict"),
    })
    coverage = result.coverage
    coverage_line = (
        f"真实搜索覆盖：算法={coverage.algorithm}，"
        f"新 pipeline 评估={coverage.new_pipeline_evaluations}，"
        f"编译缓存命中={coverage.compile_cache_hits}，"
        f"仿真缓存命中={coverage.sim_cache_hits}，"
        f"新后端执行={coverage.new_backend_executions}，"
        f"最终映射重放={coverage.final_mapping_replays}，"
        f"真实搜索={'是' if coverage.real_search else '否'}")
    summary = write_summary(cfg.output.dir, cfg.run_name, list(result.records),
                            extra_lines=[
                                f"停止原因：{result.stop_reason}",
                                f"逻辑评估：{result.logical_calls}，"
                                f"实际执行：{result.actual_calls}，"
                                f"续跑回放：{result.resumed_calls}",
                                coverage_line,
                            ])
    print(f"停止原因: {result.stop_reason}")
    print(f"逻辑评估 {result.logical_calls} 次，实际执行 {result.actual_calls} 次，"
          f"回放 {result.resumed_calls} 次")
    print(coverage_line)
    print(f"摘要: {summary}")
    if best_path is None:
        print("没有成功候选。", file=sys.stderr)
        return 1
    print(f"最优候选: {result.best.candidate_id} total_time_s={result.best.score}")
    print(f"best: {best_path}")
    return 0


def _dry_run(cfg, space) -> int:
    trace = validate_dry_inputs(cfg)
    report = space.plan()
    print(f"配置: {cfg.run_name}（{cfg.config_dir}）")
    print(f"轨迹: {len(trace.requests)} 个请求，最长提示 {trace.max_prompt}，"
          f"上下文上界 {trace.max_context}")
    print(f"画像: {cfg.profile.name} 编译摘要 "
          f"{cfg.profile.compile_profile_sha256()[:16]}")
    print(f"合法候选 {report['legal']} 个：")
    for state in space.states:
        optimization = state.resolved_optimization(cfg)
        flags = "".join(
            "1" if value else "0"
            for value in optimization.to_payload().values())
        print(f"  tp{state.strategy.tp_width}_pp{state.strategy.num_stages} "
              f"dpu={state.strategy.num_dpus} tasklets={state.op.num_tasklets} "
              f"opt={flags} clock={state.profile.clock_period_ns}")
    if report["rejected"]:
        print(f"拒绝 {len(report['rejected'])} 条：")
        for label, reason in report["rejected"]:
            print(f"  {label}: {reason}")
    print(f"外层预算: {cfg.search.algorithm} "
          f"max_evaluations={cfg.search.max_evaluations} "
          f"max_time_s={cfg.search.max_time_s}")
    if cfg.placement.enabled:
        print(f"内层放置: {cfg.placement.algorithm} "
              f"max_evaluations={cfg.placement.max_evaluations}")
    # dry-run 只做校验；已有输出目录里的响应与生效配置是复现证据，不能改写。
    if (cfg.output.dir / "responses.jsonl").exists() or (
            cfg.output.dir / "effective_config.yaml").exists():
        print("输出目录已有运行状态，dry-run 不改写已生效配置；"
              "如只需预览请换一个 --output-dir。")
    else:
        write_effective_config(cfg)
    return 0


def _replay_precheck(cfg, inputs, record: RoundRecord, *,
                     require_placement: bool) -> Path | None:
    """单候选与轮次重放共用：逐项重算内容键并核对产物与最终映射。

    任何不符都在发布输入或开始编译之前抛出 ValueError；返回应重放的
    最终放置路径。重放附加映射的哈希另计入本次仿真键，与搜索时记录的
    原始 sim_key 分开核对。
    """
    if record.state is None:
        raise ValueError("记录缺少候选状态，无法重算内容键")
    expected_compile_key = compile_key(cfg, inputs, record.state)
    if record.compile_key:
        if record.compile_key != expected_compile_key:
            raise ValueError(
                "记录的编译键与当前输入/候选不一致："
                f"{record.compile_key!r} vs {expected_compile_key!r}")
    elif record.successful:
        raise ValueError("成功轮次缺少编译键，无法证明复现的是同一产物")
    expected_sim_key = sim_key(
        cfg, inputs, record.state, expected_compile_key)
    if record.sim_key:
        if record.sim_key != expected_sim_key:
            raise ValueError(
                "记录的仿真键与当前配置/输入不一致："
                f"{record.sim_key!r} vs {expected_sim_key!r}")
    elif record.successful:
        raise ValueError("成功轮次缺少仿真键，无法证明复现的是同一配置")
    placement = (record.paths or {}).get("effective_placement")
    replay_placement = (_read_replay_placement(Path(placement))
                        if placement else None)
    if not record.successful:
        return replay_placement
    if record.artifacts is None:
        raise ValueError("成功轮次缺少产物清单，无法核对编译证据")
    if record.artifacts.compile_key != record.compile_key:
        raise ValueError("产物清单的编译键与轮次记录不一致")
    compile_dir = (record.paths or {}).get("compile_dir")
    if not compile_dir:
        raise ValueError("成功轮次缺少编译目录，无法核对产物证据")
    compile_path = Path(compile_dir)
    if not compile_path.is_dir():
        raise ValueError(f"成功轮次的编译目录不存在：{compile_path}")
    # 与正式候选、快速缓存共用同一身份判据：重算键、清单输入角色、
    # 五个产物路径与服务端内容、精化 IR 的派生身份必须属于当前输入。
    verify_candidate_identity(
        cfg, inputs, record.state,
        CandidateArtifacts(
            compile_key=record.compile_key,
            compile_dir=compile_path,
            placement_sidecar=compile_path / "placement.json",
            partition_plan=compile_path / "partition_plan.json",
            refined_ir=compile_path / "refined.ir",
            cost_sidecar=compile_path / "cost.json",
            manifest=record.artifacts,
            cache_hit=False),
        expected_key=expected_compile_key)
    if replay_placement is None and require_placement:
        raise ValueError("首次记录没有最终放置，无法严格重放")
    return replay_placement


def _read_replay_placement(path: Path) -> Path:
    """最终放置必须带版本、干净展开图标记与可复核的图摘要。"""
    try:
        payload = json.loads(Path(path).read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"最终放置无法读取：{path}：{exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"最终放置必须是 JSON 对象：{path}")
    if payload.get("schema_version") != 1:
        raise ValueError(f"最终放置版本 {payload.get('schema_version')!r} 不受支持")
    if payload.get("targets_include_relay") is not False:
        raise ValueError("最终放置包含合成 relay 目标，需用新版本重新生成")
    targets = payload.get("targets")
    if not isinstance(targets, list) or not targets:
        raise ValueError("最终放置没有执行图目标")
    for key in ("inputs_sha256", "execution_graph_sha256", "final_graph_sha256"):
        value = payload.get(key)
        if not isinstance(value, str) or len(value) != 64:
            raise ValueError(f"最终放置缺少 {key}，无法核对归属")
    return Path(path)


def _evaluate_candidate(path: Path, args) -> int:
    best = json.loads(path.read_text())
    config_path = best.get("effective_config")
    if not config_path:
        print(f"{path} 没有 effective_config，无法重建生效配置", file=sys.stderr)
        return 2
    cfg = load_config(
        config_path,
        {"cache": {"mode": args.cache_mode}} if args.cache_mode else None)
    _prepare_environment()
    # 严格复跑先核对输入、内容键与产物证据，失败时不能改写任何旧证据。
    inputs = prepare_inputs(cfg)
    try:
        record = RoundRecord.from_json(best)
        verify_input_identity(inputs, best.get("inputs") or {})
        replay_placement = _replay_precheck(
            cfg, inputs, record, require_placement=True)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"严格复跑被拒绝：{exc}", file=sys.stderr)
        return 3
    inputs = publish_inputs(cfg, inputs)
    replayed = evaluate_candidate(
        cfg, inputs, record.state, force_compile=True,
        replay_placement=replay_placement)
    if replayed.status != "success":
        print(f"严格复跑失败：{replayed.failure}", file=sys.stderr)
        return 1
    compare_scores(float(best["score"]), replayed.score)
    print(f"严格复跑一致：{replayed.candidate_id} "
          f"total_time_s={replayed.score}（相对容差 1e-9）")
    print(f"产物: {replayed.paths.get('sim_dir')}")
    return 0


def _replay_config(args) -> TuneConfig:
    """轮次重放必须沿用原生效配置：显式 YAML 或输出目录里的落盘副本。"""
    if args.config:
        return load_config(args.config, cli_overrides(args))
    base_dir = Path(args.output_dir).resolve()
    effective = base_dir / "effective_config.yaml"
    if not effective.is_file():
        raise FileNotFoundError(
            f"{base_dir} 里没有 effective_config.yaml，无法确定原生效配置")
    return load_config(effective, {"output": {"dir": str(base_dir)}})


def _replay_rounds(cfg) -> int:
    """按首次真实评估顺序复跑，失败项也记录状态变化。"""
    rounds = read_rounds(cfg.output.dir / "rounds.jsonl")
    if not rounds:
        print("没有轮次日志可重放", file=sys.stderr)
        return 2
    _prepare_environment()
    inputs = prepare_inputs(cfg)
    recorded_inputs = cfg.output.dir / "inputs" / "inputs.json"
    if recorded_inputs.is_file():
        try:
            verify_input_identity(
                inputs, json.loads(recorded_inputs.read_text()))
        except (ValueError, OSError) as exc:
            print(f"轮次重放被拒绝：{exc}", file=sys.stderr)
            return 3
    first_by_candidate = {}
    for record in rounds:
        if record.state is not None and record.candidate_id not in first_by_candidate:
            first_by_candidate[record.candidate_id] = record
    # 逐轮重算内容键并核对产物证据；任何不符都在发布输入与评估前拒绝。
    plans = []
    failures = 0
    for candidate, expected in first_by_candidate.items():
        try:
            placement = _replay_precheck(
                cfg, inputs, expected, require_placement=False)
        except (OSError, ValueError) as exc:
            print(f"{candidate}: 轮次重放被拒绝：{exc}", file=sys.stderr)
            return 3
        if expected.successful and placement is None:
            print(f"{candidate}: 首次记录没有最终放置，无法严格重放",
                  file=sys.stderr)
            failures += 1
            continue
        plans.append((candidate, expected, placement))
    inputs = publish_inputs(cfg, inputs)
    for candidate, expected, placement in plans:
        replay = evaluate_candidate(
            cfg, inputs, expected.state,
            force_compile=cfg.cache.mode == "strict",
            replay_placement=placement)
        if expected.successful:
            if replay.status != "success":
                print(f"{candidate}: 重放失败 {replay.failure}", file=sys.stderr)
                failures += 1
                continue
            try:
                compare_scores(expected.score, replay.score)
            except AssertionError as exc:
                print(f"{candidate}: {exc}", file=sys.stderr)
                failures += 1
                continue
            print(f"{candidate}: total_time_s={replay.score}（一致）")
        else:
            print(f"{candidate}: 首次失败，重放状态 {replay.status}")
    return 1 if failures else 0


def _prepare_environment() -> None:
    """pimir 成本提取需要带 PIM pass 的 triton 环境。"""
    from tune.steps import refine_level

    if refine_level() == "pimir":
        from genesim_bridge import assert_pim_passes_available, prepare_triton_env

        prepare_triton_env(pim=True)
        assert_pim_passes_available()


if __name__ == "__main__":
    sys.exit(main())
