# 两周 EMNLP 冲刺时间线与 Todo List

## 目标

在两周内完成一个可写成 EMNLP submission 原型的 dLLM remasking 项目，核心目标是：

> 证明 verifier-guided span remasking 能在 training-free 设置下，比 random / heuristic / confidence-style remasking 更有效地修复 dLLM 在数学和代码推理中的错误。

两周结束时应产出：

1. 可运行代码框架
2. GSM8K 初步主结果
3. 至少一个代码任务初步结果，优先 HumanEval 或 MBPP
4. 至少 2–3 个强 case study
5. 一版 paper skeleton
6. 明确的 gap、方法、实验和分析图表

---

## 总体策略

两周内不要追求“所有 baseline 都完整复现”，而是优先完成：

```text
MVP result > clean logging > interpretable analysis > strong narrative > extra baselines
```

核心顺序：

1. 先跑通 LLaDA + GSM8K vanilla
2. 再实现 span remasking
3. 再做 verifier-guided remasking
4. 再补 random / heuristic baselines
5. 再上 HumanEval / MBPP
6. 最后做分析图和 paper skeleton

---

# Week 1：跑通 MVP + 拿到 GSM8K 初步结果

## Day 1：环境、repo、vanilla generation

### 目标

跑通 LLaDA-8B-Instruct 单条 prompt 的 vanilla generation。

### Todo

- [ ] 创建 repo：`verem-dllm`
- [ ] 建立基础目录结构
- [ ] 配置 Python 环境
- [ ] 安装依赖：
  - [ ] torch
  - [ ] transformers
  - [ ] accelerate
  - [ ] datasets
  - [ ] tqdm
  - [ ] numpy
  - [ ] pandas
- [ ] 下载 / 缓存 LLaDA-8B-Instruct
- [ ] 跑通官方 demo prompt
- [ ] 实现 `LLaDAWrapper.generate()`
- [ ] 实现基础 JSONL logger
- [ ] 保存单条 generation 输出

### 当天产出

- [ ] `src/models/llada_wrapper.py`
- [ ] `src/utils/io.py`
- [ ] `outputs/generations/smoke_test.jsonl`

### 风险

- 显存 OOM
- transformers 版本不兼容
- mask token / generation API 不清楚

### 解决原则

- 先使用官方 repo 的 inference 代码，不要自己重写采样逻辑
- batch size 固定为 1
- max_new_tokens 先设为 256
- steps 先设为 32

---

## Day 2：GSM8K loader + vanilla baseline

### 目标

跑通 GSM8K 20 条和 200 条 vanilla baseline。

### Todo

- [ ] 实现 `gsm8k_loader.py`
- [ ] 实现 `gsm8k_verifier.py`
- [ ] 实现 `extract_answer()`
- [ ] 实现 `is_correct()`
- [ ] 实现 `run_gsm8k.py`
- [ ] 跑 20 条 smoke test
- [ ] 检查输出格式
- [ ] 跑 200 条 vanilla
- [ ] 聚合 accuracy / latency / answer extraction failure rate

### 当天产出

- [ ] `outputs/generations/gsm8k_vanilla_20.jsonl`
- [ ] `outputs/generations/gsm8k_vanilla_200.jsonl`
- [ ] `outputs/metrics/gsm8k_vanilla_200.json`

### 指标

- [ ] accuracy
- [ ] avg latency
- [ ] p50 latency
- [ ] p95 latency
- [ ] answer extraction failure rate

### 通过标准

- 200 条可以稳定跑完
- 每条样本包含 prompt、gold、prediction、correct、latency

---

## Day 3：span splitter + random remasking

### 目标

实现 reasoning span 切分和 random span remasking。

### Todo

- [ ] 实现 `Span` dataclass
- [ ] 实现 `sentence_splitter.py`
- [ ] 实现 `math_span.py`
- [ ] 实现 `mask_span(output, span)`
- [ ] 实现 `LLaDAWrapper.infill()`
- [ ] 实现 random span remasking decoder
- [ ] 对 10 条样本打印 revision trace
- [ ] 跑 GSM8K 20 条 random remask
- [ ] 检查 remask 后输出是否合理

### 当天产出

- [ ] `src/spans/base.py`
- [ ] `src/spans/sentence_splitter.py`
- [ ] `src/decoding/random_remask.py`
- [ ] `outputs/generations/gsm8k_random_20.jsonl`

### 风险

- LLaDA infill 接口不直接支持任意 span mask
- mask 后格式破坏
- 重填内容无法对齐原文本

### 解决原则

- 第一版可以把 span 替换为固定数量 mask token
- 若字符级 span 难对齐，改用 token-level span
- 先追求可运行，不追求完美 span 对齐

---

## Day 4：heuristic remasking + verifier-guided remasking

### 目标

实现两个关键 baseline / method：

1. heuristic span remasking
2. VeReM verifier-guided span remasking

### Todo

- [ ] 实现 `is_candidate_span()`
- [ ] 实现 number/operator span priority
- [ ] 实现 `heuristic_remask.py`
- [ ] 实现 `verifier_remask.py`
- [ ] 支持参数：
  - [ ] `max_revision_rounds`
  - [ ] `max_spans_per_example`
  - [ ] `samples_per_span`
  - [ ] `infill_steps`
- [ ] 对 vanilla 错误样本做 revision
- [ ] 记录 revision trace
- [ ] 跑 20 条 sanity check

### 当天产出

- [ ] `src/decoding/heuristic_remask.py`
- [ ] `src/decoding/verifier_remask.py`
- [ ] `outputs/generations/gsm8k_heuristic_20.jsonl`
- [ ] `outputs/generations/gsm8k_verem_20.jsonl`

### 通过标准

- VeReM 至少能修复少量 vanilla 错误样本
- 每个修复样本有可读 revision trace

---

## Day 5：GSM8K 200 条四方法对比

### 目标

跑出第一张核心表。

### Todo

- [ ] 跑 vanilla 200
- [ ] 跑 random remask 200
- [ ] 跑 heuristic remask 200
- [ ] 跑 VeReM 200
- [ ] 聚合 metrics
- [ ] 计算 fix rate
- [ ] 计算 avg extra forwards
- [ ] 计算 avg verifier calls
- [ ] 计算 avg latency
- [ ] 初步检查 case studies

### 当天产出

- [ ] `gsm8k_vanilla_200.jsonl`
- [ ] `gsm8k_random_200.jsonl`
- [ ] `gsm8k_heuristic_200.jsonl`
- [ ] `gsm8k_verem_200.jsonl`
- [ ] `gsm8k_main_table_200.csv`

### 主表模板

| Method | Acc | Fix Rate | Avg Extra Forwards | Avg Verifier Calls | Avg Latency |
|---|---:|---:|---:|---:|---:|
| Vanilla |  | - | 0 | 0 |  |
| Random |  |  |  |  |  |
| Heuristic |  |  |  |  |  |
| VeReM |  |  |  |  |  |

### 通过标准

- VeReM fix rate 高于 random
- VeReM final acc 高于 vanilla
- 有至少 3 个可解释修复案例

---

## Day 6：分析脚本 + case study

### 目标

把结果变成论文可以用的分析。

### Todo

- [ ] 实现 `aggregate_results.py`
- [ ] 实现 `fix_rate.py`
- [ ] 实现 `latency.py`
- [ ] 实现 `case_study.py`
- [ ] 自动导出修复案例
- [ ] 自动导出失败案例
- [ ] 分析哪些 span 被 remask
- [ ] 分析包含数字 / 运算符 span 的修复比例
- [ ] 画 Accuracy vs Extra Forward Calls
- [ ] 画 Fix Rate by Method

### 当天产出

- [ ] `outputs/metrics/gsm8k_main_table_200.csv`
- [ ] `outputs/case_studies/fixed_cases.md`
- [ ] `outputs/case_studies/failed_cases.md`
- [ ] `outputs/figures/accuracy_vs_budget.png`
- [ ] `outputs/figures/fix_rate.png`

### 通过标准

- 至少 2 张可放进 paper draft 的图
- 至少 3 个清楚 case study

---

## Day 7：Week 1 复盘 + 决策

### 目标

判断项目是否继续扩展、如何扩展。

### Todo

- [ ] 汇总 Week 1 结果
- [ ] 检查 VeReM 是否明显优于 random / heuristic
- [ ] 检查成本是否可接受
- [ ] 记录主要 failure modes
- [ ] 确定 Week 2 重点：
  - [ ] GSM8K full
  - [ ] HumanEval / MBPP
  - [ ] confidence baseline
  - [ ] CoRe-style baseline approximation
  - [ ] partial self-consistency

### Go / No-Go 标准

Go 条件：

- [ ] VeReM 在 200 条 GSM8K 上有正向提升
- [ ] revision trace 可解释
- [ ] 运行成本可接受
- [ ] 方法和 baseline 差异清楚

如果不满足：

- [ ] 改 span proposal
- [ ] 增加 samples_per_span
- [ ] 降低 vanilla steps，让 remasking 更有机会修复
- [ ] 转向 HumanEval，因为 unit test verifier 更强

---

# Week 2：扩展实验 + 论文雏形

## Day 8：GSM8K 扩展到更大样本

### 目标

将 GSM8K 从 200 条扩展到 500–1000 条，确认趋势稳定。

### Todo

- [ ] 跑 vanilla 500/1000
- [ ] 跑 random 500/1000
- [ ] 跑 heuristic 500/1000
- [ ] 跑 VeReM 500/1000
- [ ] 对比 200 条和大样本趋势
- [ ] 更新主表
- [ ] 记录运行时间和成本

### 当天产出

- [ ] `gsm8k_main_table_1000.csv`
- [ ] updated `accuracy_vs_budget.png`
- [ ] updated `fix_rate.png`

### 通过标准

- VeReM 提升趋势在更大样本上仍稳定
- 没有严重 latency 爆炸

---

## Day 9：HumanEval / MBPP 最小实现

### 目标

跑通代码任务的 verifier-guided remasking。

### Todo

- [ ] 实现 HumanEval loader
- [ ] 实现代码抽取函数
- [ ] 实现安全执行 wrapper
- [ ] 实现 unit test verifier
- [ ] 跑 HumanEval 10 条 smoke test
- [ ] 跑 HumanEval vanilla full 或 subset
- [ ] 实现 code line span splitter
- [ ] 跑 VeReM code subset

### 当天产出

- [ ] `src/datasets/humaneval_loader.py`
- [ ] `src/verifiers/code_verifier.py`
- [ ] `src/spans/code_span.py`
- [ ] `humaneval_vanilla_subset.jsonl`
- [ ] `humaneval_verem_subset.jsonl`

### 风险

- 代码执行环境不安全
- 模型输出格式混乱
- dLLM infill 破坏代码缩进

### 解决原则

- 先用 subset
- 先按行 remask
- 严格设置 timeout
- 每个测试独立进程执行

---

## Day 10：代码任务结果与 case study

### 目标

拿到 HumanEval 或 MBPP 初步结果。

### Todo

- [ ] 跑 HumanEval / MBPP vanilla
- [ ] 跑 random code line remask
- [ ] 跑 heuristic code remask
- [ ] 跑 VeReM code remask
- [ ] 聚合 pass@1
- [ ] 记录 fix rate
- [ ] 导出代码修复案例
- [ ] 分析 unit test verifier 修复了哪些 bug

### 当天产出

- [ ] `code_main_table.csv`
- [ ] `outputs/case_studies/code_fixed_cases.md`
- [ ] 初步 pass@1 对比

### 通过标准

- 至少有若干 unit-test-guided remasking 成功案例
- 即使总体提升不大，也能支撑“verifier signal works better in code”

---

## Day 11：补强 baseline

### 目标

增加至少两个更有说服力的 baseline。

优先顺序：

1. confidence-style remasking
2. partial self-consistency
3. CoRe-style perturbation approximation

### Todo

- [ ] 如果可获得 token confidence，实现 confidence span score
- [ ] 如果不可获得，使用 logit margin / entropy 近似
- [ ] 实现 full self-consistency baseline：
  - [ ] 生成 N 条完整答案
  - [ ] majority vote / verifier select
- [ ] 实现 partial self-consistency：
  - [ ] 只对 top-k spans 重采样
  - [ ] 和 full self-consistency 比较 extra forward
- [ ] 尝试实现 CoRe-style context perturbation approximation

### 当天产出

- [ ] `confidence_remask.py`
- [ ] `self_consistency.py`
- [ ] `partial_self_consistency.py`
- [ ] baseline comparison table

### 通过标准

- 至少有一个强 baseline 可用于论文主表
- partial self-consistency 的 cost-performance 有可解释趋势

---

## Day 12：机制分析

### 目标

提炼论文核心发现：stable/confident 不等于 correct，verifier-sensitive spans 更值得修。

### Todo

- [ ] 计算 high-confidence error rate，如果 confidence 可用
- [ ] 如果 confidence 不可用，计算 heuristic-stable but wrong cases
- [ ] 计算 verifier influence proxy：
  - [ ] remask 某 span 后 final answer 是否变化
  - [ ] remask 某 span 后 verifier pass rate
- [ ] 分析被修复样本中的 span 类型
- [ ] 分析失败样本类型
- [ ] 画 verifier influence vs fix probability
- [ ] 画 partial self-consistency vs full self-consistency cost curve

### 当天产出

- [ ] `outputs/figures/verifier_influence.png`
- [ ] `outputs/figures/partial_vs_full_sc.png`
- [ ] `outputs/analysis/error_taxonomy.md`

### 通过标准

- 至少有一个分析结果能支撑 paper 的 main claim
- 至少有一个图可以说明“不是普通 heuristic”

---

## Day 13：Paper skeleton

### 目标

写出 EMNLP paper skeleton。

### Todo

- [ ] 写 Abstract draft
- [ ] 写 Introduction outline
- [ ] 写 Related Work taxonomy
- [ ] 写 Problem Formulation
- [ ] 写 Method section
- [ ] 写 Experimental Setup
- [ ] 填入初步主表
- [ ] 填入初步图
- [ ] 选择 2–3 个 case study
- [ ] 写 Limitations

### 当天产出

- [ ] `paper/outline.md`
- [ ] `paper/abstract.md`
- [ ] `paper/method.md`
- [ ] `paper/experiments.md`

### 推荐 paper claim

```text
Existing dLLM remasking methods revise tokens based on confidence,
stability, or context sensitivity. We show that reasoning failures
often arise from spans that are locally stable but task-level incorrect.
We formulate verifier-guided span remasking as a budgeted revision
problem and propose a training-free decoder that selectively revises
verifier-sensitive spans.
```

---

## Day 14：整合、补洞、决定 submission strategy

### 目标

完成两周冲刺总结，明确下一步是否继续冲主会。

### Todo

- [ ] 整理所有实验结果
- [ ] 检查主表是否足够强
- [ ] 检查 baseline 是否足够公平
- [ ] 检查方法差异是否清楚
- [ ] 补跑必要小实验
- [ ] 整理失败案例
- [ ] 更新 paper skeleton
- [ ] 写两周总结文档
- [ ] 制定下一阶段 2–4 周计划

### 当天产出

- [ ] `reports/week2_summary.md`
- [ ] updated paper skeleton
- [ ] final preliminary result tables
- [ ] final case studies

### 决策标准

继续冲 EMNLP 主会，如果满足：

- [ ] GSM8K 有稳定提升
- [ ] HumanEval / MBPP 至少有初步正向结果或强 case study
- [ ] VeReM 明显优于 random / heuristic
- [ ] 有清楚机制分析
- [ ] 方法无需训练，复现成本低
- [ ] 和 CoRe / STDD / Saber / DCD 的差异清楚

如果暂时不满足：

- [ ] 改投 workshop
- [ ] 转向 code verifier 场景
- [ ] 强化 partial self-consistency 故事
- [ ] 改成 analysis paper：stable-but-wrong spans in dLLM decoding

---

# 每日固定检查清单

每天结束前检查：

- [ ] 今天是否有可复现 commit
- [ ] 是否记录了运行命令
- [ ] 是否保存了 JSONL 输出
- [ ] 是否保存了 metrics
- [ ] 是否有失败日志
- [ ] 是否有 1–2 个观察记录
- [ ] 是否更新实验表格
- [ ] 是否更新 todo

---

# 实验命名规范

建议所有实验输出遵循：

```text
{dataset}_{model}_{method}_{num_samples}_s{steps}_is{infill_steps}_r{rounds}_k{spans}_n{samples}_seed{seed}.jsonl
```

示例：

```text
gsm8k_llada8b_verem_200_s64_is32_r1_k3_n2_seed42.jsonl
```

---

# 最重要的四个结果文件

两周内务必产出：

```text
outputs/metrics/gsm8k_main_table.csv
outputs/metrics/code_main_table.csv
outputs/figures/accuracy_vs_budget.png
outputs/case_studies/fixed_cases.md
```

这四个文件基本决定项目能否继续推进。

---

# 优先级排序

## P0：必须完成

- [ ] LLaDA vanilla generation
- [ ] GSM8K verifier
- [ ] GSM8K vanilla 200
- [ ] span splitter
- [ ] random remask
- [ ] VeReM remask
- [ ] GSM8K 200 主表
- [ ] revision trace logging
- [ ] case studies

## P1：强烈建议完成

- [ ] GSM8K 500/1000
- [ ] heuristic remask
- [ ] HumanEval / MBPP subset
- [ ] unit test verifier
- [ ] partial self-consistency
- [ ] latency / budget analysis
- [ ] paper skeleton

## P2：有时间再做

- [ ] Dream-7B
- [ ] CoRe-style approximation
- [ ] STDD-style approximation
- [ ] MATH500
- [ ] JSON / SQL constrained generation
- [ ] confidence entropy analysis

## 不要在两周内做

- [ ] LoRA training
- [ ] RL / DPO / SimPO
- [ ] learned confidence head
- [ ] multi-GPU training
- [ ] 完整 KV-cache 改造
- [ ] 从零实现 dLLM sampling

---

# 两周后的理想状态

如果执行顺利，两周后你应该能说：

1. 我们实现了一个 training-free verifier-guided span remasking decoder。
2. 在 GSM8K 上，VeReM 相比 vanilla 和 random / heuristic remasking 有稳定提升。
3. 在代码任务上，unit-test-guided local remasking 能修复部分语义错误。
4. 现有 confidence / stability remasking 无法充分处理 stable-but-wrong reasoning spans。
5. dLLM 的局部 infilling 能把 full self-consistency 转化为更便宜的 partial self-consistency。
6. 该方向有明确 EMNLP paper story，可以继续扩展 baseline 和分析。
