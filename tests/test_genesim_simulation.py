"""genesim 端到端仿真的慢速冒烟。

仿真本身要十几分钟，不进快速回归（`-m "not slow"` 排除）。这里真的调用
流水线的仿真步骤，而不是只读上一次的产物——只读产物的话，仿真彻底坏掉时
只要六小时内有人手工跑过一次，测试照样是绿的。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from genesim_bridge.paths import genesim_root
from scripts.run_full_pipeline import limited_config, step_e_simulate

# 与流水线默认配置一致：tp2_pp4，按算子编译器实测的分块算成本。
_CONFIG = "sim_llama2_7b_pp_tp2pp4_globalcost.yaml"


@pytest.mark.slow
def test_simulation_produces_a_summary(tmp_path: Path) -> None:
    """跑一次仿真，核对产物里的请求数与输出 token。

    `step_e_simulate` 会清掉 pim_traces 重编，所以这次运行的每个算子都
    重新走过 trace 生成；它内部还会核对 pim mlir 的来源（见流水线的
    `verify_trace_provenance`），模板产物不会蒙混过关。
    """
    genesim = genesim_root()
    summary = step_e_simulate(
        genesim, _CONFIG, tmp_path, label="pytest",
        results_dir=tmp_path / "results",
    )

    summary_path = genesim / "results" / "summary.json"
    assert summary_path.is_file(), f"仿真没产出 {summary_path}"
    assert json.loads(summary_path.read_text()) == summary

    assert summary["completed_requests"] > 0, summary["completed_requests"]
    assert summary["processed_tokens"] > 0, summary["processed_tokens"]
    assert summary["total_time_s"] > 0
    assert summary["throughput_tokens_per_s"] > 0
    # processed_tokens 是 prompt + generated，只处理了 prompt 也会 > 0。
    # 需求 P1 要的是输出 token，钉输出侧的吞吐。
    assert summary["throughput_output_tokens_per_s"] > 0, summary


def test_default_request_count_follows_the_config() -> None:
    """不带参数跑时，仿真请求数跟随配置，不另设默认上限。

    评审 r2 问题 2：`--max-requests` 默认 1 把端到端仿真的默认覆盖从配置里的
    10 个请求降到 1 个，而文档记录的数字是 10 个请求。默认应跟随配置，
    需要收短时显式传参。
    """
    import sys

    from scripts.run_full_pipeline import parse_args

    saved = sys.argv
    sys.argv = ["run_full_pipeline.py"]
    try:
        args = parse_args()
    finally:
        sys.argv = saved
    assert args.max_requests is None, args.max_requests


def test_limited_config_caps_the_request_count(tmp_path: Path) -> None:
    """默认只仿 1 个请求，覆盖 prefill 与 decode 即可。

    10 个请求共 8084 个 token，仿真要二十多分钟。第 1 个请求是 726 个输入
    token、69 个输出 token，两种阶段都有。原配置不能改，genesim 仓的 yaml
    还要能复现完整数字。
    """
    genesim = genesim_root()
    source = genesim / "conf" / _CONFIG
    out = limited_config(source, tmp_path / "conf", max_requests=1)
    import yaml
    loaded = yaml.safe_load(out.read_text())
    assert loaded["trace"]["num_requests"] == 1
    assert yaml.safe_load(source.read_text())["trace"]["num_requests"] == 10
