# vLLM 调度超参基准（2026-11-02）

## 目的

回答「`max_num_seqs`、`max_num_batched_tokens`、`block_size`、分块预填充、前缀缓存
怎么影响每一步的调度和指标」，并给出一组可复现的默认值建议。

之前只有 `PIM_PARALLEL_SLOTS` 1 vs 2 一组对照，看不出超参的作用。

## 方法

新增 `scripts/bench_vllm_scheduling.py`：固定一个微型 Llama（hidden 64、4 层、上下文 64），
两条 20 token 的 prompt（前 16 token 相同，用于前缀缓存），贪心解码 2 token。

每组配置先跑一条无关 prompt 预热，把首次执行的编译挡在计时之外，再跑正式生成；
脚本记录每步的阶段、请求数、并行效率与墙钟，并断言各组生成 token 完全一致。

```bash
source "$(python3 -c 'import json; print(json.load(open("paths.local.json"))["pytorch_env_script"])')"
python scripts/bench_vllm_scheduling.py --json /tmp/sched.json
python scripts/bench_vllm_scheduling.py --only baseline,prefix
```

默认 6 组配置，全部跑完约 1 分钟。

## 结果（2026-11-02）

| 配置 | 步数 | 最大并发 | 解码效率 | 墙钟 ms | 每步请求数 |
|---|---|---|---|---|---|
| baseline（seqs=2、块 16、无缓存） | 3 | 2 | 1.00 | 2 909 | prefill:1; prefill:2; decode:1 |
| seqs1（`max_num_seqs=1`） | 4 | 1 | 1.00 | 2 848 | prefill:1; decode:1; prefill:1; decode:1 |
| block8（`block_size=8`） | 3 | 2 | 1.00 | 3 456 | prefill:1; prefill:2; decode:1 |
| prefix（前缀缓存开） | **2** | 2 | **1.98** | 2 537 | prefill:2; decode:2 |
| seqs4（`max_num_seqs=4`） | 3 | 2 | 1.00 | 2 895 | prefill:1; prefill:2; decode:1 |
| budget40（预算 40） | **2** | 2 | **1.94** | 2 855 | prefill:2; decode:2 |
| chunked（分块预填充，预算 16） | 4 | 2 | 1.00 | 3 667 | prefill:1; prefill:2; prefill:2; decode:1 |

各组的生成 token 完全相同：`[[217, 238], [250, 42]]`。

## 结论

1. **真正卡住同批预填充的是 token 预算，不是 `max_num_seqs`。**
   baseline 与 `seqs4` 的调度完全一样：两条 20 token 的 prompt 合起来 40 > 预算 32，
   vLLM 只能先做一条，下一条等下一步。把预算提到 40（`budget40`）后，
   两条 prompt 在**同一步**完成预填充，总步数 3 → 2，解码效率 1.94，直接验证了这条结论。

2. **前缀缓存是这批配置里唯一能减步数的开关。**
   两条 prompt 共享 16 token，命中后每条只剩 4 token 要算，8 ≤ 32，于是
   「预填充 2 条 + 解码 2 条」只用 2 步，解码并行效率 1.98（两条真正重叠）。
   其余配置的 1.00 表示一步里只有一个请求在算，不是串行惩罚。

3. **`max_num_seqs=1` 会把请求完全串起来。**
   调度变成「预填充一条、解码一条、再预填充另一条」，总步数 4 对 3；
   微型模型上墙钟差不多（2 848 vs 2 909 ms），因为步数代价被固定开销淹没，
   在真实模型上这个差值会随每步计算量放大。

4. **`block_size=8` 在模拟后端上更慢**（3 456 vs 2 909 ms，+19%）。
   槽变小意味着同样上下文要用两倍的 KV 块，块表与跨 DPU 通信的条数都变多。
   真实硬件上要结合显存碎片再权衡，但这里没有看到小块的好处。

5. **分块预填充把 20 token 的 prompt 拆成 16+4 两段，多出一步**（4 步 vs 3 步）。
   它的价值在长 prompt 与短请求混批，短 prompt 场景反而增加调度开销。

## 指标解读

| 指标 | 是否可迁移到真硬件 | 说明 |
|---|---|---|
| 步数、每步请求数、最大并发、解码效率 | 是 | 由调度器决定，与后端无关 |
| 墙钟 ms | **否** | NumpyBackend 受 GIL 限制（见 `executor-parallel-bottleneck-20261102.md`），只能横向比较同一批配置 |
| 生成 token | 是 | 各组必须完全一致，脚本内置断言 |

## 已实施的测试

脚本内置两条约束：每组生成 token 与 baseline 逐 token 相同；配置名拼错直接返回 2。
与 `profile_execution.py` 一样没有单独单测。

## 当前不足

- 只有两条请求、贪心解码；没有覆盖 `temperature>0` 与结构化输出。
- `max_num_batched_tokens` 只试了 16 与 32 两档；没有做 40 以上的同批预填充对照，
  这一组是下一步要补的（能直接验证结论 1）。
- 前缀缓存的命中块数没有从 worker 统计里取，只能从步数间接看。
