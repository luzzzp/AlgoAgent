# AlgoAgent 简历项目描述

## 一句话版本

AlgoAgent 是一个面向算法题的 C++17 解题智能体项目，围绕 TACO-verified 数据构建了 Python oracle 验证、Python-to-C++ 翻译、QLoRA SFT、编译运行反馈和 Agent 自修复评测闭环。

## 简历项目描述

**AlgoAgent：面向算法题的后训练与自修复代码智能体**

- 基于 Qwen2.5-Coder-7B-Instruct 构建算法题解题 Agent，输入题面、约束、时间/空间限制，输出 C++17 代码，并通过编译器和测试用例自动验证。
- 设计无泄漏评测 schema，将难度、标签、参考答案与标准复杂度隔离为离线 oracle，运行时 Agent 仅可见题面、输入输出格式、约束和资源限制。
- 构建 TACO-verified 数据生产流水线：验证 Python oracle，通过大模型翻译为 C++17，再用 C++ 编译器和测试集严格筛选 verified C++ 解法。
- 使用 97 条 verified C++ 数据进行 QLoRA SFT，并在 held-out 50 道题上对比 Base、SFT 和 SFT+Agent。
- 实现编译、运行、超时、Wrong Answer 反馈驱动的 Agent 修复闭环，并记录 Compile Rate、Repair Test Pass Rate、Held-out Pass Rate、Verified Success Rate 等指标。

## 当前可用指标

数据构造：

- TACO-verified 小批量题目数：200
- verified Python oracle：195 / 200
- verified C++ 解法：97 / 195
- Python-to-C++ 严格验证转化率：49.7%
- SFT 训练数据：97 条

SFT 训练：

- 模型：Qwen2.5-Coder-7B-Instruct
- 方法：QLoRA SFT
- GPU：RTX 4090 24GB
- 训练轮数：2 epochs
- 训练耗时：127.5s
- train loss：0.7539
- mean token accuracy：0.8355

Held-out 50 评测：

| 方法 | Compile Rate | Repair Test Pass | Held-out Pass | Verified Success |
|---|---:|---:|---:|---:|
| Base | 86% | 21.7% | 8% | 8% |
| SFT v2 | 94% | 29.1% | 12% | 12% |
| SFT v2 + Agent | 94% | 29.1% | 16% | 16% |

阶段性结论：

- SFT 后编译率从 86% 提升到 94%。
- SFT 后 Verified Success Rate 从 8% 提升到 12%。
- 加入 3 轮 Agent 修复后，Verified Success Rate 进一步提升到 16%。
- 当前主要瓶颈是算法正确性和边界处理，失败类型以 REPAIR_TEST_FAILED 为主。

## 面试讲法

我做这个项目主要是因为通用代码大模型在算法题场景中经常能生成可编译代码，但难以稳定通过隐藏测试，尤其容易在边界条件、复杂度选择和输入输出细节上出错。因此我没有只做普通微调，而是构建了一个可验证的后训练与 Agent 闭环：先用 TACO-verified 的 Python oracle 构造 verified C++ 数据，再进行 QLoRA SFT，最后接入编译器和测试器，让模型根据错误反馈进行修复。

在数据构造上，我没有直接信任数据集里的解法，而是先运行 Python oracle，通过全部测试后再翻译成 C++，翻译后的 C++ 也必须重新编译并通过测试才能进入训练集。这样可以保证 SFT 数据是可执行、可验证的。

在评测上，我使用 Compile Rate、Repair Test Pass Rate、Held-out Pass Rate 和 Verified Success Rate，而不是只看训练 loss。当前 97 条 verified C++ 数据训练后，在 held-out 50 道题上，编译率从 86% 提升到 94%，Verified Success Rate 从 8% 提升到 12%；加入 Agent 修复后提升到 16%。

## 后续优化方向

- 将 TACO 数据扩展到 1000 道，目标获得 400 到 500 条 verified C++ 解法。
- 增加失败日志分析，区分编译错误、超时、边界错误和算法错误。
- 为复杂度字段构造更可靠的标注，降低 Complexity Unknown Rate。
- 在 SFT 数据扩大后进行 DPO 或 GRPO，使用测试通过率作为可验证奖励。

## 最新实验补充：TACO-1000

数据构造：

- TACO-verified 题目数：1000
- verified Python oracle：973 / 1000
- verified C++ 解法：518 / 973
- Python-to-C++ 严格验证转化率：53.2%
- SFT 训练数据：518 条

Held-out 100 评测：

| 方法 | Compile Rate | Repair Test Pass | Held-out Pass | Verified Success |
|---|---:|---:|---:|---:|
| Base | 87% | 24.3% | 11% | 11% |
| SFT-1000 | 90% | 28.3% | 11.9% | 11% |
| SFT-1000 + Agent | 97% | 31.6% | 12.9% | 12% |

实验结论：

- 扩展到 518 条 verified C++ 后，SFT 主要提升了编译率和测试通过比例。
- SFT-1000 的 Compile Rate 从 87% 提升到 90%，Repair Test Pass Rate 从 24.3% 提升到 28.3%。
- 加入 3 轮 Agent 修复后，final Compile Rate 提升到 97%，Verified Success Rate 从 11% 小幅提升到 12%。
- 该实验说明当前 SFT 更擅长改善输出格式和可编译性，但对算法正确性的提升有限，下一阶段需要失败案例分析、DPO/GRPO 或更高质量的题解推理数据。
