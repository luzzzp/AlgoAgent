# AlgoAgent 项目日志

## 记录原则

这份日志用于记录项目推进过程中的关键问题、定位过程、解决方案和结果。后续每完成一个阶段都应更新一次，而不是等项目结束后再回忆。

## 2026-06-04 至 2026-06-08：项目骨架与无泄漏 Agent

目标：

- 搭建算法题 Agent 的基础仓库。
- 定义无泄漏评测 schema。
- 实现 C++ 编译运行器、复杂度检查器和 Agent 状态机。

完成内容：

- 实现 `ProblemSpec`、`TestCase`、`TestSuite`、`OracleMetadata`、`AgentResult` 等核心 schema。
- 将运行时可见信息限制为题面、输入输出、约束、语言和资源限制。
- 将难度、标签、参考解、标准复杂度放入离线 oracle，避免进入 Agent prompt。
- 实现 `CppExecutor`，支持 C++17 编译、运行、超时和 stdout 对比。
- 实现 `AlgoAgent`，支持生成、复杂度检查、repair tests、held-out eval tests 和最多 3 轮修复。

关键决策：

- 不直接使用 LangChain，而是自定义状态机，便于控制输入输出、记录日志和做消融。
- 删除 samples / hidden_tests 的概念，改为 repair_tests 和 eval_tests。
- eval_tests 失败后不把具体失败样例反馈给 Agent，避免评测泄漏。

## 2026-06-08：TACO-verified 数据转换

问题：

- 初始 `make_datasets.py` 读取 `_manifest.json` 时崩溃，报 `KeyError: 'problem'`。

原因：

- `load_problems()` 把目录下所有 `.json` 都当作题目文件读取，包括 manifest 文件。

解决：

- 修改 loader，跳过文件名以 `_` 开头的 JSON。
- 对非 AlgoAgent 题目 JSON 给出更友好的错误。

结果：

- TACO 转换流程可以继续运行。
- 发现原始转换结果没有 verified C++ 解法，因此 SFT/DPO 数据为空，只能生成 GRPO prompts。

## 2026-06-08：Python oracle 验证

目标：

- 先验证 TACO 中的 Python 解法是否能通过测试，再作为翻译源。

完成内容：

- 新增 `PythonExecutor`。
- 新增 `scripts/verify_python_oracles.py`。
- 每题最多尝试前 3 个 Python 解法，通过全部 repair/eval tests 后标记为 `verified=true`。

结果：

- 200 道 TACO-verified 中，195 道找到 verified Python oracle。
- 5 道 Python 解法未通过测试。

结论：

- Python oracle 质量足够作为 C++ 数据构造源。

## 2026-06-09：Python-to-C++ 翻译与验证

目标：

- 使用 Qwen2.5-Coder-7B-Instruct 将 verified Python oracle 翻译为 C++17。
- 翻译结果必须重新编译并通过全部测试。

问题 1：Windows 本地 PyTorch 无法导入

- 报错：`WinError 182`，torch `shm.dll` 加载失败。
- 解决：本地只做 mock smoke test，真实 HF 翻译放到 Linux 4090 服务器上运行。

问题 2：mock 翻译真实题目成功率为 0

- 原因：mock translator 只用于链路测试，不具备真实翻译能力。
- 解决：在 4090 服务器用 HF backend 跑真实模型。

结果：

- 10 道、每题 2 个候选：1 道 verified C++。
- 10 道、每题 6 个候选：2 道 verified C++。
- 50 道、每题 6 个候选：22 道 verified C++。
- 200 道、每题 6 个候选：97 道 verified C++。

阶段结论：

- Python-to-C++ 严格验证转化率约 49.7%。
- 该数据构造方法可行，但速度较慢，适合高质量种子数据，不适合作为唯一扩量方式。

## 2026-06-09：训练脚本兼容问题

问题：

- 运行 `training/sft_train.py` 时报错：
  `SFTConfig.__init__() got an unexpected keyword argument 'max_seq_length'`

原因：

- 服务器安装的 TRL 版本中，`SFTConfig` API 已变化，新版使用 `max_length` 或其他字段。

解决：

- 使用 `inspect.signature()` 动态判断 `SFTConfig` 支持 `max_seq_length` 还是 `max_length`。
- 同时兼容新版 `SFTTrainer(processing_class=...)` 和旧版 `SFTTrainer(tokenizer=...)`。

结果：

- QLoRA SFT 成功跑通。
- 97 条 verified C++ 数据，2 epochs，训练耗时 127.5s。
- train loss 为 0.7539，mean token accuracy 为 0.8355。

## 2026-06-09：SFT v1 评测无明显提升

现象：

- Base eval10 与 SFT eval10 指标几乎相同。
- Verified Success Rate 都为 0。
- Complexity Unknown Rate 接近 0.9。

原因：

- 训练数据输出格式与评测 prompt 不一致。
- 旧训练输出包含 `Algorithm tags` 和 `Time and space complexity`，但评测要求 `Time Complexity`、`Space Complexity` 和 cpp code block。

解决：

- 修改 `make_datasets.py`，让 SFT/DPO 输出格式与运行时 prompt 对齐。
- 删除训练输出中的算法标签，避免不必要的离线 oracle 信息进入答案格式。
- 新增 `convert_taco_verified.py --offset`，用于构造 held-out 评测集。

结果：

- 重新生成 SFT v2 数据并训练。
- SFT v2 在 eval10 上开始出现正向信号。

## 2026-06-10：评测脚本与 Agent 修复

目标：

- 增加 base vs SFT vs SFT+Agent 的统一评测脚本。

完成内容：

- 新增 `scripts/evaluate_hf_model.py`。
- 支持 base model、LoRA adapter、4bit 加载、跳过 explanation、设置 repair turns。
- 每评完一题都会写入报告，避免长时间评测中断后结果丢失。

问题 1：候选程序超时后 stdout/stderr 类型异常

- 报错：`TypeError: a bytes-like object is required, not 'str'`
- 原因：`subprocess.TimeoutExpired` 中的 stdout/stderr 有时是 bytes。
- 解决：`normalize_output` 支持 bytes，并统一 decode。

问题 2：候选程序输出非法 UTF-8 字节导致评测崩溃

- 报错：`UnicodeDecodeError: 'utf-8' codec can't decode byte`
- 原因：候选 C++ 程序输出了非法字节，`subprocess.run(text=True)` 解码失败。
- 解决：在 `CppExecutor` 中设置 `encoding="utf-8", errors="replace"`，将非法输出判为 WA 而不是中断评测。

问题 3：长评测中断后需要从头跑

- 解决：`evaluate_hf_model.py` 增加 `--resume`，读取已有 report，跳过已完成 problem_id，继续写回同一个 JSON。

## 2026-06-10：当前评测结果

Smoke test 50：

| 方法 | Compile Rate | Repair Test Pass | Held-out Pass | Verified Success |
|---|---:|---:|---:|---:|
| Base | 86% | 25.0% | 12% | 12% |
| SFT v2 | 88% | 27.7% | 16% | 16% |
| SFT v2 + Agent | 94% | 29.3% | 18% | 18% |

Held-out 50：

| 方法 | Compile Rate | Repair Test Pass | Held-out Pass | Verified Success |
|---|---:|---:|---:|---:|
| Base | 86% | 21.7% | 8% | 8% |
| SFT v2 | 94% | 29.1% | 12% | 12% |
| SFT v2 + Agent | 94% | 29.1% | 16% | 16% |

阶段结论：

- SFT v2 在 held-out 50 上将编译率从 86% 提升到 94%。
- Verified Success Rate 从 8% 提升到 12%。
- 加入 3 轮 Agent 修复后，Verified Success Rate 提升到 16%。
- 当前失败主要集中在 `REPAIR_TEST_FAILED`，说明主要瓶颈是算法正确性和边界条件。
- `complexity_unknown_rate` 仍为 1.0，复杂度结构化输出尚未解决。

## 下一步计划

- 扩展到 1000 道 TACO-verified，目标获得 400 到 500 条 verified C++ 数据。
- 增加失败案例分析脚本，统计编译错误、超时、WA、边界错误的比例。
- 为 verified C++ 数据补充更可靠的复杂度标注。
- 在 SFT 数据扩大后，再进行 DPO 或 GRPO 实验。

## 2026-06-11：TACO-1000 数据扩展与 SFT-1000 评测

目标：

- 将 verified C++ 训练数据从 97 条扩展到数百条。
- 验证扩大 SFT 数据后，模型在 held-out 100 上是否有更明显提升。

数据构造结果：

- TACO-verified 题目数：1000。
- verified Python oracle：973 / 1000。
- failed Python solution：27 / 1000。
- verified C++ 解法：518 / 973。
- Python-to-C++ 严格验证转化率：53.2%。

观察：

- 虽然使用的是 TACO-verified，仍有 27 道题未找到可通过本地测试的 Python oracle。
- 这说明不能盲目信任数据集标注，重新执行验证是必要的数据清洗步骤。
- Python-to-C++ 转化率从 200 道阶段的 49.7% 提升到 1000 道阶段的 53.2%，说明当前数据生产流水线具有一定稳定性。

Held-out 100 评测结果：

| 方法 | Compile Rate | Repair Test Pass | Held-out Pass | Verified Success |
|---|---:|---:|---:|---:|
| Base | 87% | 24.3% | 11% | 11% |
| SFT-1000 | 90% | 28.3% | 11.9% | 11% |
| SFT-1000 + Agent | 97% | 31.6% | 12.9% | 12% |

结论：

- SFT-1000 提升了编译率和 repair test pass rate，但 Verified Success Rate 没有明显提升。
- Agent 修复将 final Compile Rate 从 90% 提升到 97%，说明反馈闭环对编译错误修复有效。
- Verified Success Rate 仅从 11% 提升到 12%，说明当前主要瓶颈已经不是可编译性，而是算法正确性、边界条件和题意理解。
- 失败类型仍以 `REPAIR_TEST_FAILED` 为主，下一阶段应优先做失败样本分析，而不是继续盲目扩 SFT。

下一步调整：

- 新增失败案例分析脚本，从评测 JSON 中抽取 REPAIR_TEST_FAILED、COMPILE_FAILED、TIMEOUT、HELD_OUT_TEST_FAILED 的代表样例。
- 分析模型错误是题意理解、输入输出格式、边界条件、复杂度还是算法选择问题。
- 在此基础上构造更有针对性的 DPO 数据或 GRPO reward，而不是直接进行泛化的偏好训练。
