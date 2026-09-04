"""GeneSim 给定切分方案 → 图编译器的输入契约。

这是四段接口里方向相反的那一段。其余三段都是「编译器算出什么、告诉 GeneSim」，
这一段是「GeneSim 定下 PU 映射、约束编译器」。

## 为什么不直接读 Model IR 的 per-op 字段

Model IR 表达的是 `op → (node_id, vpu_id, attached_pu_id)`，一个算子一个 PU。
而图编译器需要的是另一组量：多少台 DPU 参与、几个流水段、每层归属哪几台、每个
权重按哪一维切。从 per-op 的 `vpu_id` 反推这些需要一堆假设（认出哪些 op 属于同
一层、同一段，TP 宽度是几），而 `ShardStrategy` 本来就以这些概念为单位。

所以用一份显式的方案文件，由 GeneSim 侧写、编译器侧读，两边都不必猜对方的内部
表示。GeneSim 想换搜索算法、想把状态空间从「每个算子一个 PU」变粗成「层块 → PU
集合」，只要产出这份文件，编译器无需改动。

## 与 placement sidecar 的区别

方向相反，不要混用：

| | `PartitionPlan`（本文件） | placement sidecar |
| --- | --- | --- |
| 方向 | GeneSim → 编译器 | 编译器 → GeneSim |
| 内容 | 切分方案（约束） | 切分结果（本地形状、分块、放置） |
| 时机 | 图编译**之前** | 图编译**之后** |
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Literal

# 当前只认这一个版本；字段含义变了必须提版本号，让旧文件明确报错而不是被误读。
PLAN_VERSION = 1

_VALID_MODES = ("col", "row")


@dataclass(frozen=True)
class PartitionPlan:
    """一份切分方案：流水段数 + 段内张量并行宽度 + 权重切分规则。

    字段刻意与 `ShardStrategy` 的概念对齐（见 graph/strategy.py），转换是直接的，
    不需要推断。
    """

    num_dpus: int
    num_stages: int
    # {权重名片段: "col" | "row"}。省略则用 LLAMA_WEIGHT_RULES 的默认规则。
    weight_rules: tuple[tuple[str, Literal["col", "row"]], ...] = ()
    # 参与切分的物理 DPU 编号；省略则为 range(num_dpus)。
    dpu_ids: tuple[int, ...] | None = None
    # 每台逻辑 DPU 对应哪个 GeneSim ClusterPU，按 dpu_id 顺序给出 (device_id, cluster_id)。
    #
    # 这是两套编号空间之间的桥：图编译器内部用逻辑编号 0..num_dpus-1（它的执行链路
    # 假设编号连续，见 docs/split-pp-20260830.md 第 9 条），而 GeneSim 的 PU 编号是
    # 另一套数值空间。粒度也不同——图编译器的「一台 DPU」对应 GeneSim 的一个
    # ClusterPU（默认 16 个 TensorPU、8 GiB），而不是单个 TensorPU（512 MiB）：
    # 一台 DPU 要放下整段权重（llama2-7b 每段约 1544 MiB），钉到单个 TensorPU 必然
    # 触发容量超限，所以段内算子要在该 Cluster 的 TensorPU 之间轮转分摊。
    #
    # 省略时 GeneSim 侧退回原有的 `dpu_id % len(cluster_keys)` 取模换算（向后兼容）。
    # 显式给出的好处：不靠取模碰巧落到哪几个 Cluster，且能让同段的 DPU 落在同一个
    # Cluster 内走快链路（PIM_Intra_Cluster 512 GB/s vs PIM_Inter_Cluster 128 GB/s）。
    dpu_to_cluster: tuple[tuple[int, int], ...] | None = None
    # 方案来源，仅供人读与排错（如 "genesim_autotune" / "fixed_tp2_pp4"）。
    #
    # 当前只产出固定映射，但格式对将来的自动调优结果是同一份：调优侧只要能给出
    # num_stages 和 dpu_to_cluster，填进来编译器侧读法完全不变。
    source: str = ""
    # 产出这份方案时 GeneSim 的目标模型，用于核对没张冠李戴。
    model_id: str = ""

    def __post_init__(self) -> None:
        if self.num_dpus <= 0:
            raise ValueError(f"num_dpus 必须为正，收到 {self.num_dpus}")
        if self.num_stages <= 0:
            raise ValueError(f"num_stages 必须为正，收到 {self.num_stages}")
        if self.num_dpus % self.num_stages:
            raise ValueError(
                f"num_stages={self.num_stages} 不能整除 num_dpus={self.num_dpus}"
                "（每个流水段必须持有同样多的 DPU）"
            )
        tp_width = self.num_dpus // self.num_stages
        if tp_width & (tp_width - 1):
            raise ValueError(f"tp_width={tp_width} 不是 2 的整数次幂（切分契约 5）")
        for pattern, mode in self.weight_rules:
            if mode not in _VALID_MODES:
                raise ValueError(
                    f"权重 {pattern!r} 的切分方向 {mode!r} 不合法，只能是 {_VALID_MODES}"
                )
        if self.dpu_ids is not None:
            ids = tuple(self.dpu_ids)
            if len(ids) != self.num_dpus or len(set(ids)) != len(ids):
                raise ValueError(
                    f"dpu_ids 必须是 {self.num_dpus} 个互不相同的编号，收到 {ids}"
                )
            if any(dpu_id < 0 for dpu_id in ids):
                raise ValueError(f"dpu_ids 不能为负：{ids}")
        if self.dpu_to_cluster is not None:
            mapping = tuple(self.dpu_to_cluster)
            if len(mapping) != self.num_dpus:
                raise ValueError(
                    f"dpu_to_cluster 必须给全 {self.num_dpus} 台 DPU，收到 {len(mapping)} 项"
                )
            for entry in mapping:
                if len(entry) != 2:
                    raise ValueError(
                        f"dpu_to_cluster 的每项必须是 (device_id, cluster_id)，收到 {entry!r}"
                    )
                if any(value < 0 for value in entry):
                    raise ValueError(f"dpu_to_cluster 的编号不能为负：{entry!r}")

    def clusters_of_stage(self, stage: int) -> tuple[tuple[int, int], ...]:
        """返回某个流水段的 DPU 各自落在哪个 ClusterPU。

        用于核对"同段是否共用一个 Cluster"——共用则段内通信走
        PIM_Intra_Cluster（512 GB/s），否则走 PIM_Inter_Cluster（128 GB/s）。
        """
        if self.dpu_to_cluster is None:
            raise ValueError("这份方案没有 dpu_to_cluster，无法回答段内 Cluster 归属")
        if not 0 <= stage < self.num_stages:
            raise ValueError(f"stage={stage} 越界 [0,{self.num_stages})")
        width = self.tp_width
        return tuple(self.dpu_to_cluster[stage * width : (stage + 1) * width])

    @property
    def tp_width(self) -> int:
        """段内张量并行宽度（1 = 纯流水）。"""
        return self.num_dpus // self.num_stages

    @property
    def name(self) -> str:
        return f"tp{self.tp_width}_pp{self.num_stages}"

    def to_payload(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "version": PLAN_VERSION,
            "num_dpus": self.num_dpus,
            "num_stages": self.num_stages,
        }
        if self.weight_rules:
            payload["weight_rules"] = [list(rule) for rule in self.weight_rules]
        if self.dpu_ids is not None:
            payload["dpu_ids"] = list(self.dpu_ids)
        if self.dpu_to_cluster is not None:
            payload["dpu_to_cluster"] = [list(entry) for entry in self.dpu_to_cluster]
        if self.source:
            payload["source"] = self.source
        if self.model_id:
            payload["model_id"] = self.model_id
        return payload

    @classmethod
    def from_payload(cls, payload: object) -> "PartitionPlan":
        if not isinstance(payload, dict):
            raise ValueError(f"切分方案必须是 dict，收到 {type(payload).__name__}")
        version = payload.get("version")
        if version != PLAN_VERSION:
            raise ValueError(
                f"切分方案版本 {version!r} 不受支持（当前只认 {PLAN_VERSION}）。"
                "字段含义可能已变，请用当前版本的导出工具重新生成。"
            )
        for key in ("num_dpus", "num_stages"):
            if key not in payload:
                raise ValueError(f"切分方案缺少必需字段 {key!r}")
        rules = payload.get("weight_rules") or ()
        weight_rules = tuple(
            (str(pattern), str(mode)) for pattern, mode in rules  # type: ignore[misc]
        )
        dpu_ids = payload.get("dpu_ids")
        mapping = payload.get("dpu_to_cluster")
        return cls(
            num_dpus=int(payload["num_dpus"]),
            num_stages=int(payload["num_stages"]),
            weight_rules=weight_rules,  # type: ignore[arg-type]
            dpu_ids=tuple(int(v) for v in dpu_ids) if dpu_ids is not None else None,
            dpu_to_cluster=(
                tuple((int(entry[0]), int(entry[1])) for entry in mapping)
                if mapping is not None
                else None
            ),
            source=str(payload.get("source", "")),
            model_id=str(payload.get("model_id", "")),
        )

    def write(self, path: Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(self.to_payload(), indent=2, ensure_ascii=False))

    @classmethod
    def read(cls, path: Path) -> "PartitionPlan":
        return cls.from_payload(json.loads(Path(path).read_text()))
