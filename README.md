# AlgoAgent：无泄漏算法题自修复智能体

AlgoAgent 是一个面向大模型算法实习的可复现项目。系统要求模型根据真实可见的题面和资源限制生成 C++17 代码，并通过复杂度检查、编译器、修复测试和留出测试验证代码。

核心目标：

- 运行时 Agent 无法访问难度、算法标签、标准复杂度和参考答案。
- 修复测试可以向模型反馈反例；留出测试永远不向模型泄露。
- 代码验证成功后才生成最终解题思路并返回代码。
- 无法验证成功时明确返回 `FAILED`，不返回未验证代码。
- 同时评估理论复杂度可行性和实际测试运行结果。

## 当前实现

- `ProblemSpec`：Agent 可见的题面、数据范围、语言、时间与空间限制。
- `TestSuite`：可反馈的 `repair_tests` 与只用于最终评测的 `eval_tests`。
- `OracleMetadata`：仅用于训练数据构造和离线分析的标注。
- `ComplexityFeasibilityChecker`：估算理论操作量和内存占用。
- `CppExecutor`：使用 `g++ -std=c++17 -O2` 编译并运行测试。
- `AlgoAgent`：复杂度检查、生成、执行、反馈、修复、留出评测和成功后讲解。
- Hugging Face 后端：可接入 Qwen2.5-Coder 或微调后的本地模型。
- 规则模型：仅用于离线验证工程链路，不代表真实大模型效果。

## 环境要求

- Python 3.10+
- `g++` 已加入 `PATH`
- 默认语言：`C++17`
- 默认时间限制：`2s`
- 默认空间限制：`256MB`

```powershell
python --version
g++ --version
```

## 跑通本地 MVP

运行全部测试：

```powershell
python -m unittest discover -s tests -v
```

评测三道演示题：

```powershell
python -m algoagent.cli evaluate `
  --problems data/problems/sample `
  --max-repair-turns 3 `
  --out reports/sample_eval.json
```

单独解一道题：

```powershell
python -m algoagent.cli solve `
  --problem data/problems/sample/two_sum.json `
  --max-repair-turns 3
```

成功输出包含：

```text
SOLVED
解题思路
结构化时间与空间复杂度
资源检查结果
通过修复测试和留出测试的 C++17 代码
```

失败输出包含：

```text
FAILED
失败原因
诊断摘要
```

失败结果不会返回未验证代码。

## 题目数据格式

每道题由三个隔离区域组成：

```json
{
  "problem": {
    "id": "unique_id",
    "title": "题目名称",
    "statement": "完整题面",
    "input_format": "输入格式",
    "output_format": "输出格式",
    "constraints": ["1 <= n <= 100000"],
    "language": "cpp17",
    "time_limit_sec": 2,
    "memory_limit_mb": 256
  },
  "tests": {
    "repair_tests": [
      {
        "stdin": "测试输入\n",
        "expected_stdout": "期望输出\n"
      }
    ],
    "eval_tests": [
      {
        "stdin": "留出测试输入\n",
        "expected_stdout": "留出测试输出\n"
      }
    ]
  },
  "oracle": {
    "difficulty": "medium",
    "tags": ["dynamic-programming"],
    "expected_complexity": "O(n log n) time, O(n) space",
    "reference_solution": "C++17 标准答案",
    "solutions": [
      {
        "language": "python3",
        "code": "Python 标准答案",
        "verified": true
      }
    ],
    "source": "数据来源",
    "url": "原题链接"
  }
}
```

测试名称不是必填字段。加载器会自动生成：

```text
repair-0001
repair-0002
eval-0001
```

`problem` 会进入模型 prompt；`tests` 由 Agent 执行；`oracle` 只允许训练数据构造器和离线分析使用。

## Agent 工作流

```text
读取 ProblemSpec
    ↓
模型生成候选代码和结构化复杂度
    ↓
理论复杂度检查
    ├── 超限：反馈理论 TLE/MLE，重新规划
    └── 通过或未知：继续
    ↓
编译并运行 repair_tests
    ├── 失败：反馈第一个失败反例并修复
    └── 全部通过：继续
    ↓
运行一次 eval_tests
    ├── 失败：返回 FAILED，不泄露反例，不继续修复
    └── 全部通过：继续
    ↓
根据最终代码生成解题思路
    ↓
返回 SOLVED
```

## 复杂度与资源检查

默认 C++ 理论操作预算：

```text
operation_budget = time_limit_sec × 1e8
```

例如：

```text
n = 10000
时间复杂度 = O(n²)
时间限制 = 1s
估算操作量 = 1e8
结果 = 临界通过
```

支持估算常见复杂度：

- `O(1)`
- `O(log n)`
- `O(n)`
- `O(n log n)`
- `O(n²)` / `O(n^2)`
- `O(n*m)` / `O(n+m)`
- `O(2^n)` 和小规模 `O(n!)`

空间检查会结合：

- 模型声明的空间复杂度。
- C++ 静态数组。
- 常见 `vector` 和二维 `vector`。
- C++ 基础类型大小。

复杂度无法可靠解析时返回 `UNKNOWN`，不会伪装成通过。复杂度检查是启发式分析，不能替代真实执行，因此系统仍会按题目时间限制运行每个测试。

## 当前指标

- `initial_compile_rate`：第一次实际执行候选的编译率。
- `final_compile_rate`：最终候选的编译率。
- `repair_test_pass_rate`：最终候选在修复测试上的宏平均通过率。
- `held_out_test_pass_rate`：最终候选在留出测试上的宏平均通过率。
- `verified_success_rate`：资源检查未失败且全部测试通过的题目比例。
- `avg_repair_turns`：平均修复轮数。
- `theoretical_tle_rate`：最终候选被判断理论超时的比例。
- `theoretical_mle_rate`：最终候选被判断理论超内存的比例。
- `complexity_unknown_rate`：最终复杂度检查含 `UNKNOWN` 的比例。
- `failure_breakdown`：失败原因分布。
- `explanation_success_rate`：验证成功题目中成功生成讲解的比例。

## 构造训练数据

```powershell
python scripts/make_datasets.py `
  --problems data/problems/sample `
  --out-dir data/processed
```

生成：

- `sft.jsonl`
- `dpo.jsonl`
- `grpo_prompts.jsonl`

训练 prompt 只使用 `ProblemSpec`，不会包含测试和 oracle。标准答案只用于构造监督输出。

## 使用 TACO-verified

TACO-verified 保留了 TACO 的题面、测试、解法、难度、标签、来源、时间限制和空间限制等字段，并移除了没有正确解的问题。当前转换器默认只导入 stdin/stdout 类型题目，跳过带 `fn_name` 的函数式题，因为本项目现阶段执行器面向标准输入输出。

先在服务器或可联网环境安装依赖：

```bash
pip install datasets
```

小批量转换 200 道可用题：

```bash
python scripts/convert_taco_verified.py \
  --dataset likaixin/TACO-verified \
  --split train \
  --out-dir data/problems/taco_verified_200 \
  --limit 200 \
  --max-repair-tests 20 \
  --max-eval-tests 10
```

转换结果：

```text
data/problems/taco_verified_200/*.json
data/problems/taco_verified_200/_manifest.json
```

字段处理策略：

- `question` → `problem.statement`
- `input_output.inputs/outputs` → `repair_tests` 和 `eval_tests`
- `time_limit` → `problem.time_limit_sec`
- `memory_limit` → `problem.memory_limit_mb`
- `solutions` → `oracle.solutions`
- Python 解法会保存为 `language=python3`
- C++ 解法若存在，会保存为 `language=cpp17` 并可用于 C++ SFT

验证 Python oracle：

```bash
python scripts/verify_python_oracles.py \
  --problems data/problems/taco_verified_200 \
  --out-dir data/problems/taco_verified_200_py_verified \
  --max-solutions-per-problem 3
```

这一步会用 `PythonExecutor` 跑完整 `repair_tests + eval_tests`。只有本地执行通过的 Python 解法才会被标记为 `verified=true`。

将 verified Python 解法翻译为 C++ 并验证：

```bash
python scripts/translate_python_to_cpp.py \
  --problems data/problems/taco_verified_200_py_verified \
  --out-dir data/problems/taco_verified_200_cpp_verified \
  --backend hf \
  --model Qwen/Qwen2.5-Coder-7B-Instruct \
  --candidates-per-problem 2
```

通过全部测试的 C++ 候选会写入 `oracle.solutions`：

```json
{
  "language": "cpp17",
  "code": "...",
  "verified": true
}
```

同时会填充兼容字段 `oracle.reference_solution`。

生成训练数据：

```bash
python scripts/make_datasets.py \
  --problems data/problems/taco_verified_200_cpp_verified \
  --out-dir data/processed/taco_verified_200_cpp_verified
```

检查生成行数：

```bash
wc -l data/processed/taco_verified_200_cpp_verified/sft.jsonl
wc -l data/processed/taco_verified_200_cpp_verified/dpo.jsonl
wc -l data/processed/taco_verified_200_cpp_verified/grpo_prompts.jsonl
```

注意：如果 TACO-verified 中某题只有 Python 解法，它会先进入 `oracle.solutions`，但不会直接进入 C++ SFT 的 `chosen`。这类数据当前用于：

- 测试集和题面导入。
- Python oracle 保留。
- 后续 Python-to-C++ 翻译验证。
- GRPO prompt 构造。

只有本地执行验证通过的 C++ 解法才会进入当前 C++ SFT/DPO 监督数据。

不要复用已经中断过的输出目录。比如某次 `verify_python_oracles.py` 超时后留下了半成品目录，建议换一个新目录名重新跑。

## 使用真实模型

在 GPU 服务器安装依赖：

```bash
pip install -r requirements-train.txt
```

运行 Qwen2.5-Coder baseline：

```bash
python -m algoagent.cli evaluate \
  --problems data/problems/sample \
  --backend hf \
  --model Qwen/Qwen2.5-Coder-7B-Instruct \
  --max-repair-turns 3 \
  --out reports/qwen_base_eval.json
```

训练入口：

```bash
python training/sft_train.py
python training/dpo_train.py
python training/grpo_train.py
```

当前 GRPO 奖励函数仍是轻量占位实现，后续需要接入独立沙箱执行器才能用于正式训练。

## 数据收集原则

1. 从公开且许可清晰的数据集收集题面、测试和标准答案。
2. 将公开样例与其他测试统一处理，再划分为修复测试和留出测试。
3. 每道题必须至少包含一个修复测试和一个留出测试。
4. 留出测试不能用于模型修复、prompt 或训练偏好构造。
5. 使用标准答案验证全部测试，无法稳定通过的题目不得进入正式数据集。
6. 按题目而不是按代码划分训练集和评测集，避免同题泄漏。
7. 记录来源、链接和许可证。

## 项目结构

### 根目录

| 文件 | 作用 |
| --- | --- |
| `README.md` | 项目说明、运行步骤、数据格式、TACO-verified 使用方式和文件说明。 |
| `pyproject.toml` | Python 项目元信息和测试配置，目前主要声明包名、Python 版本和测试目录。 |
| `requirements-train.txt` | GPU 服务器训练和 Hugging Face 推理所需依赖，如 `transformers`、`datasets`、`trl`、`peft`。 |

### `algoagent/`

| 文件 | 作用 |
| --- | --- |
| `algoagent/schema.py` | 定义核心数据结构，包括 `ProblemSpec`、`TestSuite`、`OracleMetadata`、`AgentResult`、复杂度估计和资源检查结果。 |
| `algoagent/agent.py` | 自修复 Agent 主流程：生成代码、复杂度检查、运行修复测试、运行留出测试、成功后生成讲解。 |
| `algoagent/complexity.py` | 理论复杂度与资源限制检查器，根据题目约束和模型声明的复杂度估算操作量与内存占用。 |
| `algoagent/executor.py` | C++17 执行器，负责写入临时代码文件、调用 `g++` 编译、运行 stdin/stdout 测试并收集结果。 |
| `algoagent/python_executor.py` | Python 执行器，用于验证 TACO/APPS 中的 Python oracle，暂不作为主 Agent 输出语言。 |
| `algoagent/model_client.py` | 模型接口定义和本地规则模型。规则模型只用于验证工程链路，不代表真实大模型效果。 |
| `algoagent/hf_model.py` | Hugging Face 模型后端，可加载 Qwen2.5-Coder 或微调后的模型进行真实生成。 |
| `algoagent/evaluation.py` | 统一评测逻辑，计算编译率、修复测试通过率、留出测试通过率、理论 TLE/MLE 等指标，并生成报告。 |
| `algoagent/cli.py` | 命令行入口，提供 `solve` 和 `evaluate` 两个子命令。 |
| `algoagent/__init__.py` | 包初始化文件，声明项目版本。 |

### `scripts/`

| 文件 | 作用 |
| --- | --- |
| `scripts/convert_taco_verified.py` | 将 TACO-verified 数据集转换为本项目的标准题目 JSON；默认跳过函数式题，只导入 stdin/stdout 题。 |
| `scripts/verify_python_oracles.py` | 验证 TACO/APPS 中的 Python oracle 是否能通过本项目测试，通过后标记为 `verified=true`。 |
| `scripts/translate_python_to_cpp.py` | 将 verified Python 解法翻译成 C++17，并用 C++ 执行器验证；通过后写回 verified C++ oracle。 |
| `scripts/make_datasets.py` | 将标准题目 JSON 转换为 `sft.jsonl`、`dpo.jsonl`、`grpo_prompts.jsonl`。 |

### `training/`

| 文件 | 作用 |
| --- | --- |
| `training/sft_train.py` | QLoRA SFT 训练入口，默认基座为 `Qwen/Qwen2.5-Coder-7B-Instruct`。 |
| `training/dpo_train.py` | DPO 偏好优化训练入口，默认从 SFT 输出模型继续训练。 |
| `training/grpo_train.py` | GRPO 实验入口，目前奖励函数仍是轻量占位，后续需要接入真实执行奖励。 |
| `training/configs/sft_qwen25_coder_7b.yaml` | SFT 训练配置示例，记录模型、数据、LoRA 和训练超参。 |

### `data/`

| 文件或目录 | 作用 |
| --- | --- |
| `data/problems/sample/two_sum.json` | Two Sum 演示题，包含 `problem`、`repair_tests/eval_tests` 和 `oracle`。 |
| `data/problems/sample/balanced_parentheses.json` | 括号匹配演示题，用于测试 WA 修复能力。 |
| `data/problems/sample/longest_increasing_subsequence.json` | LIS 演示题，用于测试 `O(n log n)` 复杂度估算。 |
| `data/problems/taco_verified/` | TACO-verified 转换后的题目目录，运行转换脚本后生成。 |
| `data/processed/sft.jsonl` | 从样例题生成的 SFT 数据。 |
| `data/processed/dpo.jsonl` | 从样例题生成的 DPO 偏好数据；当前 rejected 仍是占位样本。 |
| `data/processed/grpo_prompts.jsonl` | 从样例题生成的 GRPO prompt 数据。 |

### `reports/`

| 文件 | 作用 |
| --- | --- |
| `reports/sample_eval.json` | 本地样例题评测报告，记录每题状态、尝试轮数、资源检查结果和指标汇总。 |
| `reports/interview_notes.md` | 面试讲解笔记，整理项目动机、无泄漏评测、资源约束和指标表达。 |

### `tests/`

| 文件 | 作用 |
| --- | --- |
| `tests/test_executor.py` | 测试 C++ 执行器的编译、运行和编译错误处理。 |
| `tests/test_complexity.py` | 测试复杂度检查器，包括 `O(n²)` 边界、理论 TLE、理论 MLE 和 `UNKNOWN`。 |
| `tests/test_agent_reliability.py` | 测试 Agent 的可靠性约束，如 oracle 不进 prompt、留出测试不反馈、失败不返回代码。 |
| `tests/test_evaluation.py` | 测试样例题能通过统一评测并生成新指标。 |
| `tests/test_taco_converter.py` | 测试 TACO-verified 转换器，包括 stdin/stdout 题转换和函数式题跳过。 |
