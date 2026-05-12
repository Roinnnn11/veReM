# VeReM 初步模型方法与代码架构设计

## 1. 项目目标

本项目目标是实现一个 **training-free 的 dLLM remasking 解码框架**，用于快速验证以下研究假设：

> 在 dLLM 推理任务中，错误往往不是来自低置信 token，而是来自“高置信但语义错误”的 reasoning span。相比 token-level confidence remasking，使用 task verifier 指导 span-level remasking，可以在有限推理预算下更有效地修复数学和代码推理错误。

初步方法命名为：

**VeReM: Verifier-Guided Span Remasking for Diffusion Language Model Reasoning**

第一阶段聚焦：

- 模型：LLaDA-8B-Instruct
- 任务：GSM8K
- 方法：vanilla decoding、random span remasking、heuristic span remasking、verifier-guided span remasking
- 目标：在 200–500 条 GSM8K 上跑出第一组 accuracy / latency / fix-rate 结果

---

## 2. 方法概览

### 2.1 基本流程

给定输入问题 `x`：

1. 使用 dLLM 生成完整初始答案 `y0`
2. 从 `y0` 中切分 reasoning spans
3. 对候选 span 进行 remasking
4. 使用 dLLM infill 生成修正版 `y'`
5. 使用 verifier 评估 `y'`
6. 如果 verifier 通过，接受修正版；否则继续尝试下一个 span 或保留原答案

流程图：

```text
Prompt x
  |
  v
Initial dLLM Generation
  |
  v
Output y0 = reasoning + answer
  |
  v
Span Proposal
  |
  v
Candidate Span Remasking
  |
  v
dLLM Infill / Revision
  |
  v
Task Verifier
  |
  +-- pass --> accept revised output
  |
  +-- fail --> try next span / keep original
```

---

## 3. Problem Formulation

给定：

- prompt: `x`
- 初始输出: `y`
- candidate spans: `S = {s1, s2, ..., sk}`
- task verifier: `V(x, y) -> {0, 1}` 或连续分数
- revision budget: `B`

目标是在有限预算内选择最值得 remask 的 spans，使最终输出 `y*` 的 verifier score 最大：

```text
maximize    V(x, y*)
subject to  model_forward_calls + verifier_calls <= B
```

其中 `y*` 由 dLLM 对部分 masked spans 重新 infill 得到。

---

## 4. 方法模块

### 4.1 Initial Generation

使用标准 dLLM decoding 生成完整答案。

输入：

```python
prompt: str
max_new_tokens: int
steps: int
temperature: float
```

输出：

```python
{
    "text": str,
    "latency": float,
    "num_steps": int,
    "num_forwards": int,
    "token_confidences": Optional[list[float]]
}
```

第一阶段如果模型 wrapper 不方便返回 token confidence，可以先忽略 confidence，优先跑通 vanilla 和 span remasking。

---

### 4.2 Span Proposal

第一版使用规则切分。

GSM8K 推荐顺序：

1. 按换行切分
2. 若无换行，则按句号、问号、感叹号切分
3. 优先保留包含数字、等号、运算符的 span
4. 可选：不 remask final answer 行

候选 span 格式：

```python
@dataclass
class Span:
    span_id: int
    start: int
    end: int
    text: str
    span_type: str  # "sentence", "equation", "number", "answer", etc.
```

候选规则：

```python
def is_candidate_span(span_text: str) -> bool:
    has_number = bool(re.search(r"\d", span_text))
    has_operator = any(op in span_text for op in ["=", "+", "-", "*", "/", "%"])
    is_too_short = len(span_text.strip()) < 5
    is_final_answer = "####" in span_text

    return (has_number or has_operator) and not is_too_short and not is_final_answer
```

---

### 4.3 Remasking Strategy

第一阶段实现三类策略。

#### A. Random Span Remasking

随机选择一个 candidate span remask。

用途：

- sanity check
- 证明不是“随便改改就能提升”

#### B. Heuristic Span Remasking

优先选择包含数字、等号、计算符号的 span。

用途：

- 简单强 baseline
- 检查数学 span 是否确实更关键

#### C. Verifier-Guided Span Remasking

对多个 span 逐个 remask + infill，使用 verifier 判断是否接受。

MVP 版本：

```python
for span in candidate_spans[:max_spans_per_example]:
    masked_output = mask_span(output, span)
    for _ in range(samples_per_span):
        revised_output = model.infill(prompt, masked_output, steps=infill_steps)
        if verifier(revised_output, gold_answer):
            return revised_output
return original_output
```

后续版本可以加入：

- span risk score
- verifier influence score
- 多轮 revision
- partial self-consistency
- acceptance ranking

---

### 4.4 Verifier

#### GSM8K Verifier

第一阶段使用 final numeric answer exact match。

核心函数：

```python
def extract_answer(text: str) -> str | None:
    # 优先抽取 "#### 42"
    # 否则抽取最后一个数字
    ...

def is_correct(pred_text: str, gold_text: str) -> bool:
    return extract_answer(pred_text) == extract_answer(gold_text)
```

注意：

- 需要去除逗号
- 需要处理整数、小数、负数
- 需要记录无法抽取答案的比例

#### HumanEval / MBPP Verifier

第二阶段实现：

- 提取代码块
- 拼接测试用例
- sandbox 执行
- pass/fail 作为 verifier

第一阶段暂不实现，避免拉长工程周期。

---

## 5. 推荐代码架构

```text
verem-dllm/
  README.md
  requirements.txt
  pyproject.toml

  configs/
    gsm8k_llada_vanilla.yaml
    gsm8k_llada_verem.yaml
    humaneval_llada_verem.yaml

  src/
    __init__.py

    models/
      __init__.py
      base.py
      llada_wrapper.py
      dream_wrapper.py

    decoding/
      __init__.py
      vanilla.py
      random_remask.py
      heuristic_remask.py
      verifier_remask.py
      span_remask.py

    spans/
      __init__.py
      base.py
      sentence_splitter.py
      math_span.py
      code_span.py

    verifiers/
      __init__.py
      gsm8k_verifier.py
      code_verifier.py

    datasets/
      __init__.py
      gsm8k_loader.py
      humaneval_loader.py
      mbpp_loader.py

    eval/
      __init__.py
      run_gsm8k.py
      run_humaneval.py
      aggregate_results.py

    analysis/
      __init__.py
      fix_rate.py
      latency.py
      stable_wrong.py
      case_study.py

    utils/
      __init__.py
      io.py
      logging.py
      parsing.py
      seeds.py
      timing.py

  scripts/
    run_gsm8k_vanilla.sh
    run_gsm8k_random.sh
    run_gsm8k_heuristic.sh
    run_gsm8k_verem.sh
    aggregate_gsm8k.sh

  outputs/
    generations/
    metrics/
    logs/
    case_studies/
```

---

## 6. 关键类与接口设计

### 6.1 Model Wrapper

```python
class BaseDLLM:
    def generate(
        self,
        prompt: str,
        max_new_tokens: int,
        steps: int,
        temperature: float = 0.0,
        **kwargs,
    ) -> dict:
        raise NotImplementedError

    def infill(
        self,
        prompt: str,
        text_with_masks: str,
        steps: int,
        temperature: float = 0.0,
        **kwargs,
    ) -> dict:
        raise NotImplementedError
```

`LLaDAWrapper` 继承 `BaseDLLM`。

需要统一输出：

```python
{
    "text": str,
    "latency": float,
    "num_steps": int,
    "num_forwards": int,
    "metadata": dict
}
```

---

### 6.2 Decoder Interface

```python
class BaseDecoder:
    def decode(self, example: dict) -> dict:
        raise NotImplementedError
```

输出统一为：

```python
{
    "id": str,
    "prompt": str,
    "gold": str,
    "initial_output": str,
    "final_output": str,
    "initial_correct": bool,
    "final_correct": bool,
    "method": str,
    "latency": float,
    "num_model_forwards": int,
    "num_verifier_calls": int,
    "num_remasked_spans": int,
    "revision_trace": list[dict],
}
```

---

### 6.3 Revision Trace

每次 remask 都要记录：

```python
{
    "round": int,
    "span_id": int,
    "span_text": str,
    "masked_output": str,
    "revised_output": str,
    "verifier_pass": bool,
    "latency": float,
}
```

这个 trace 非常重要，用于：

- case study
- error analysis
- 计算 fix rate
- 计算 verifier budget
- 论文中的 qualitative example

---

## 7. 第一阶段命令设计

### Vanilla

```bash
python -m src.eval.run_gsm8k \
  --model llada-8b-instruct \
  --method vanilla \
  --num_samples 200 \
  --steps 64 \
  --max_new_tokens 512 \
  --output outputs/generations/gsm8k_vanilla_200.jsonl
```

### Random Span Remasking

```bash
python -m src.eval.run_gsm8k \
  --model llada-8b-instruct \
  --method random_remask \
  --num_samples 200 \
  --steps 64 \
  --infill_steps 32 \
  --max_revision_rounds 1 \
  --max_spans_per_example 3 \
  --samples_per_span 1 \
  --output outputs/generations/gsm8k_random_200.jsonl
```

### Heuristic Span Remasking

```bash
python -m src.eval.run_gsm8k \
  --model llada-8b-instruct \
  --method heuristic_remask \
  --num_samples 200 \
  --steps 64 \
  --infill_steps 32 \
  --max_revision_rounds 1 \
  --max_spans_per_example 3 \
  --samples_per_span 1 \
  --output outputs/generations/gsm8k_heuristic_200.jsonl
```

### VeReM

```bash
python -m src.eval.run_gsm8k \
  --model llada-8b-instruct \
  --method verem \
  --num_samples 200 \
  --steps 64 \
  --infill_steps 32 \
  --max_revision_rounds 1 \
  --max_spans_per_example 3 \
  --samples_per_span 2 \
  --output outputs/generations/gsm8k_verem_200.jsonl
```

---

## 8. 第一阶段主要指标

### Accuracy

```text
accuracy = final_correct / total_examples
```

### Initial Accuracy

```text
initial_accuracy = initial_correct / total_examples
```

### Fix Rate

只看 vanilla 初始错误样本：

```text
fix_rate = fixed_wrong_examples / initially_wrong_examples
```

### Harm Rate

如果对所有样本都做 revision，需要记录：

```text
harm_rate = originally_correct_but_revised_wrong / initially_correct_examples
```

第一版可以只修正 initially wrong examples，使 harm rate 暂时为 0。

### Average Extra Forward Calls

```text
avg_extra_forwards = extra_model_forwards / total_examples
```

### Average Verifier Calls

```text
avg_verifier_calls = verifier_calls / total_examples
```

### Latency

```text
avg_latency
p50_latency
p95_latency
```

---

## 9. 最小主结果表

| Method | Initial Acc | Final Acc | Fix Rate | Avg Extra Forwards | Avg Verifier Calls | Avg Latency |
|---|---:|---:|---:|---:|---:|---:|
| Vanilla | - |  | - | 0 | 0 |  |
| Random Span Remask |  |  |  |  |  |  |
| Heuristic Span Remask |  |  |  |  |  |  |
| VeReM |  |  |  |  |  |  |

---

## 10. 第二阶段扩展

### 10.1 更强 Span Proposal

- 数学表达式 span
- 方程行 span
- final answer dependency span
- code AST block span

### 10.2 更强 Risk Scoring

```text
risk(span) =
  lambda_1 * uncertainty(span)
+ lambda_2 * instability(span)
+ lambda_3 * verifier_influence(span)
```

### 10.3 HumanEval / MBPP

增加代码任务，用 unit tests 作为 verifier。

### 10.4 CoRe-style Baseline

实现 context perturbation sensitivity：

1. 对上下文局部 mask
2. 多次 infill
3. 计算目标 span 变化率
4. 高变化率 span 优先 remask

### 10.5 Partial Self-Consistency

对比：

- full self-consistency: 生成 N 条完整答案
- partial self-consistency: 只重采样 top-k risky spans

核心指标：

```text
accuracy gain per extra forward
```

---

## 11. 工程注意事项

1. **先 batch size = 1**  
   优先跑通逻辑，不要先优化吞吐。

2. **所有输出存 JSONL**  
   不要只存最终 accuracy。revision trace 是后续论文分析的核心资产。

3. **先做 oracle-style verifier**  
   GSM8K 用 gold answer，HumanEval 用 unit tests。先证明方法有效，再考虑 deployment setting。

4. **控制变量**  
   比较不同方法时保持：
   - same prompt
   - same max_new_tokens
   - same initial decoding steps
   - same infill steps
   - same revision budget

5. **固定随机种子**  
   每次实验记录 seed，方便复现。

6. **先做小样本 smoke test**  
   每个方法先跑 20 条，再跑 200 条，最后跑 full set。

---

## 12. 近期优先级

### 必做

- LLaDA vanilla generation
- GSM8K loader
- answer extraction verifier
- span splitter
- random remask
- verifier-guided remask
- JSONL logging
- aggregate metrics

### 暂缓

- LoRA / RL / DPO
- Dream-7B
- HumanEval
- CoRe 完整复现
- STDD 完整复现
- KV-cache / block decoding
- 多 GPU 并行

---

## 13. 第一版成功标准

第一版不追求完整论文，只追求验证核心假设。

成功标准：

1. LLaDA-8B-Instruct 能稳定跑 GSM8K 200 条
2. Vanilla baseline 有可复现 accuracy
3. VeReM 能修复一部分 vanilla 错误样本
4. VeReM 的 fix rate 明显高于 random span remasking
5. 每个修复样本有清晰 revision trace，可用于 case study

一旦满足以上条件，就可以进入论文级实验扩展。
