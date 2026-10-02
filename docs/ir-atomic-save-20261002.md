# 图骨架 IR 并发读写导致流水线崩溃的修复（2026-10-02）

## 现象

`python scripts/run_full_pipeline.py --num-stages 4` 跑到 [B+C+D] 步失败：

```
File "genesim_bridge/placement_export.py", line 340, in export_placement_to_genesim
    ir = json.loads(Path(ir_path).read_text())
json.decoder.JSONDecodeError: Expecting value: line 1 column 1 (char 0)
```

这个报错的意思是「读到的字符串第一个字符就不是 JSON」——文件要么是空的，
要么只剩 BOM。而磁盘上的 `models/llama2_7b.ir` 是好的：9844226 字节、
合法 JSON、6852 个算子、七种投影身份齐全。

## 排查

先排除「IR 内容变了」和「本仓未提交改动把代码改坏了」两种可能：

- 单独重跑同一条命令（`scripts/export_pp_placement.py --partition-plan ...
  --measure-kernel-tiles`）**能跑通**，224 个 GEMM 全部放置成功。不是确定性失败。
- 崩溃前后几秒的文件时间戳，指向「读的那一刻文件正被另一个进程截断重写」：

| 时刻 | 事件 | 依据 |
| --- | --- | --- |
| 17:55:19.95 | 流水线 [A] 步写完图骨架 IR | `a_model_parser.log` mtime |
| 17:55:24.47 | [0] 步读完 IR、写完切分方案 | `0_pu_mapping.log` mtime |
| 17:55:50.29 | [B+C+D] 抛出 JSONDecodeError | `bcd_export.log` mtime |
| 17:55:50.74 | `models/llama2_7b.ir` 被重写完成 | 文件 mtime |
| 17:56:18.64 | `llama2_7b_pimir.ir` 出现 | 文件 mtime |

也就是说：读到空文件发生在 17:55:50.28 前后，而同一份 IR 在 0.45 秒后
被另一个进程写完整。谁写的日志不在本流水线的日志目录里，是**并发的另一个进程**
（手动重跑 `model_parser.py` 或精化流程）。

- `models/` 目录的 mtime 停在 9 月 26 日，说明这些写都是**原地截断重写**，
  而不是「写临时文件再改名」——改名会改动目录 mtime。

## 根因

GeneSim 的 `ModelIR.save()` 是截断式写入：

```python
with open(file_path, "w") as f:     # 先截断成 0 字节
    json.dump(self.to_dict(), f, indent=2)   # 再慢慢写 9.8 MB
```

写入期间（约 0.5 秒）文件在读者眼里就是空的。只要另一个进程在这个窗口里
重写同一份 IR，正在读的一方就会拿到 0 字节。本次就是这么撞上的。

## 修复

改 GeneSim 仓的 `src/ir/model_ir.py`（本仓不改代码）：

```python
path = Path(file_path)
tmp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
with open(tmp_path, "w") as f:
    json.dump(self.to_dict(), f, indent=2)
os.replace(tmp_path, path)          # 同目录改名，原子替换
```

先写同目录的临时文件再改名，读者拿到的要么是旧文件、要么是新文件，永远不是
半个文件。临时文件名带 pid，两个 model_parser 同时跑也不会互相踩。

## 测试

| 测试 | 位置 | 结果 |
| --- | --- | --- |
| `test_save_replaces_the_file_instead_of_rewriting_it`（新增） | GeneSim `tests/sim/test_model_ir.py` | 通过 |
| `tests/sim/test_model_ir.py` 全量 | 同上 | 50 passed |
| 全流程重跑 `run_full_pipeline.py --num-stages 4 --skip-simulation` | 本仓 | 通过 |

[E] 步仿真这次没跟着跑（上一次跑它是手工 Ctrl-C 中断的，与本次修复无关）；
它读的是已经写好的 sidecar 和 pimir IR，不受本次改动影响。

新增的测试断言两次 `save()` 之间 inode 发生变化、目录里没有残留临时文件、
文件仍能 load 回来。截断重写不会换 inode，所以这条测试在旧代码上**会失败**
（已手工验证：原地重写 inode 不变）。

## 余留问题

1. 本仓自己的产物（`llama2_7b_*_placed.ir`、placement sidecar、精化后的 IR）
   仍是 `Path.write_text` 式的非原子写。仿真进程正在读这些文件时重跑导出/
   精化，同样会读到空文件。本次没撞上，暂未改。
2. GeneSim 是独立仓（origin `pimtools/genesim`），这个 patch 在对方合并前
   只存在于本机工作区，pull 时要留意是否被覆盖。
3. 流水线 [A] 步会无条件重写共享的图骨架 IR。原子写只保证读者拿到的是完整
   文件，不保证拿到的是本次 run 生成的那一份；想彻底避免，得让并发跑的人
   各用各的 IR 路径。
