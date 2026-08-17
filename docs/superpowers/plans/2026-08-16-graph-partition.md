# Graph Partition Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the problem-1 graph pass that labels an exported GPT-2 FX graph as DPU or host and returns deterministic DPU-only connected partitions.

**Architecture:** `partition_graph` keeps the input `GraphModule` intact and writes only `device` and `part_id` metadata. It computes union-find components over direct DPU-to-DPU FX data edges, so host nodes are hard boundaries and partitions are ordered by FX topology.

**Tech Stack:** Python 3.10, PyTorch FX and `torch.export`, Transformers 4.57.6, pytest.

## Global Constraints

- Implement only problem 1; do not add capacity checks, DPU mapping, placement propagation, communication planning, QKV rewriting, or runtime work.
- The phase-1 whitelist is exactly `aten.addmm.default`, `aten.linear.default`, `aten.add.Tensor`, `aten.mul.Tensor`, and `aten.tanh.default`.
- Every FX node receives `node.meta["device"]`; only DPU nodes receive `node.meta["part_id"]`.
- A host node never bridges DPU components.
- Preserve unrelated metadata and make repeated execution deterministic.
- Tests run on CPU without model downloads, CUDA, FlagGems, or FlagTree APIs.
- The current worktree is `main` by explicit user authorization; do not create a Git commit unless explicitly requested.

---

## File Structure

- Modify: `graph/partition.py` - public types, whitelist, annotation, and direct-edge grouping pass.
- Modify: `contracts/graph_meta.py` - shared device and partition metadata names and values.
- Create: `tests/test_partition.py` - unit and random GPT-2 strict-export integration tests.
- Create: `docs/partition.md` - public contract and design decision.

### Task 1: Write the Partition Contract Tests

**Files:**
- Create: `tests/test_partition.py`
- Modify: `graph/partition.py`

**Interfaces:**
- Produces: `DPU_LOWERABLE: frozenset`, `Partition(part_id: int, nodes: list[Node])`, and `partition_graph(gm: GraphModule) -> list[Partition]`.

- [ ] **Step 1: Add failing tests for the public contract**

Create `tests/test_partition.py`. Build graphs with actual ATen overloads and assert:

```python
def test_partition_marks_whitelist_and_host_breaks_components() -> None:
    gm, nodes = _module_with_host_break()
    partitions = partition_graph(gm)

    assert nodes["add"].meta["device"] == "dpu"
    assert nodes["relu"].meta["device"] == "host"
    assert [(part.part_id, part.nodes) for part in partitions] == [
        (0, [nodes["add"]]),
        (1, [nodes["mul"]]),
    ]


def test_partition_keeps_direct_dpu_fork_and_join_together() -> None:
    # add -> {mul, tanh} -> add must be one direct DPU component.
    ...


def test_partition_numbers_disconnected_components_by_fx_order() -> None:
    # Two independent DPU nodes are numbered in graph order.
    ...


def test_partition_replaces_stale_metadata_without_touching_other_metadata() -> None:
    # A stale host part_id disappears while another metadata key remains.
    ...
```

Add an integration test that strictly exports a CPU-only random GPT-2 with
`n_layer=4`, `n_head=8`, `n_embd=512`, and `max_seq=128`. The logits-only
wrapper takes a precomputed 4D causal mask and calls GPT-2 with
`use_cache=False`; this bypasses the installed Transformers dynamic-mask path
that cannot be exported strictly. Assert every node is marked, DPU and host
nodes both exist, returned partitions cover DPU nodes exactly once, and all
part IDs are consecutive.

- [ ] **Step 2: Run the new test file and observe its expected collection failure**

Run:

```bash
/media/disk/fengjingge/src/flagOS/flagOS-installed/flagTree/python/bin/python \
  -m pytest tests/test_partition.py -q
```

Expected: import or collection failure because `graph.partition` does not yet
define the requested public symbols.

### Task 2: Implement Direct-Edge DPU Grouping

**Files:**
- Modify: `graph/partition.py`
- Modify: `contracts/graph_meta.py`
- Test: `tests/test_partition.py`

**Interfaces:**
- Consumes: an arbitrary valid FX `GraphModule`.
- Produces: metadata annotations on the same nodes plus ordered `Partition` entries.

- [ ] **Step 1: Implement the minimal pass**

Define `DEVICE_META_KEY`, `PART_ID_META_KEY`, `DEVICE_DPU`, and `DEVICE_HOST`
in `contracts/graph_meta.py`, then write a module-level immutable whitelist and a `Partition` dataclass. In
`partition_graph`:

```python
for node in nodes:
    node.meta["device"] = "dpu" if _is_dpu_node(node) else "host"
    node.meta.pop("part_id", None)
    if node.meta["device"] == "dpu":
        parent[node] = node

for node in parent:
    for input_node in node.all_input_nodes:
        if input_node in parent:
            union(node, input_node)
```

Collect each root's nodes, sort nodes by original graph order, sort components
by their first node, enumerate them from zero, and write each DPU node's
`part_id`. Do not use `CapabilityBasedPartitioner`, because its horizontal
fusion can join DPU consumers across a host node.

- [ ] **Step 2: Run focused tests**

Run:

```bash
/media/disk/fengjingge/src/flagOS/flagOS-installed/flagTree/python/bin/python \
  -m pytest tests/test_partition.py -q
```

Expected: all unit and strict-export integration tests pass.

### Task 3: Document and Verify the Module

**Files:**
- Create: `docs/partition.md`
- Modify: `graph/partition.py`
- Test: `tests/test_partition.py`

**Interfaces:**
- Consumes: the Task 2 public module API.
- Produces: concise user-facing documentation.

- [ ] **Step 1: Document the contract**

Document `partition_graph`, `device`/`part_id` metadata, the five lowerable
ATen targets, direct-DPU-edge grouping, host hard boundaries, deterministic FX
ordering, and the reference `docs/spec.md:159-273`.

- [ ] **Step 2: Run full verification**

Run:

```bash
/media/disk/fengjingge/src/flagOS/flagOS-installed/flagTree/python/bin/python \
  -m pytest tests/ -x -q
git diff --check
git diff --stat
```

Expected: all current and new tests pass, no whitespace errors occur, and the
diff is limited to the graph pass, its tests, and documentation.

- [ ] **Step 3: Leave changes uncommitted**

Report the verification output and net line count. Do not create a Git commit.
