"""编排器：GML 之后的那一段。

链路里的位置是**串行的第三段**，不回填 GML：

    图编译器 + 算子编译器 → GML（200 节点）→ 编排器 → 层参数 + net.ini

为什么不回填：把参考产物 `relay2gml_graph.gml` 的全部 616 个键名提出来，搜
`virt|task|order|layer|offset|addr|prev|next|sched|l2` —— 零命中。GML 的字段
分类表也只有「图编译器 / 算子编译器 / 常量表」三类，没有编排器这一栏。更直接
的证据是 `scripts/gml_field_inventory.py` 的 `NOT_REQUIRED` 表，逐条抄了对方
PDF 的原话：`in_virtual` / `out_virtual` 是 "L2A deducts it on its own"，
`prev_task` / `next_task` 是 "L2A will determine the execution flow"。

所以编排器管的是 GML **表达不了**的四件事：

| 事 | 为什么 GML 管不了 |
| --- | --- |
| 200 节点 → 422 层展开 | GML 一个节点就是一个节点，相位是节点内字段 |
| Layer ID / Task 图 | GML 是纯 DAG，不带执行序 |
| L2 offset（实测 96.4% 复用） | 要跨整张图做 liveness，单节点看不到 |
| net.ini 执行序 | 同上 |

**不管切分放置**——那是图编译器的事（`graph/strategy.py`）。切分决定「算什么、
多大」（进 GML 的形状），编排决定「片上怎么放、按什么序走」。
"""
