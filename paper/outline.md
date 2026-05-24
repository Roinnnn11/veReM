# VeReM: Verifier-Guided Reranking for Diffusion LLM Reasoning

> Working title (备选)：
> - "Verifier-Guided Reranking is a Strong Decoder for Diffusion Language Models on Reasoning"
> - "Stable-but-Wrong: Why Span Remasking Fails in dLLMs and What to Do Instead"
> - "Cheap Perturbation, Strong Verifier: Test-Time Scaling for Diffusion LLM Reasoning"

EMNLP 2026 short paper（8p）or long（10p）。建议先按 short 写，结果够强再扩。

---

## 0. One-line claim

> Training-free **verifier-guided reranking** turns the cheap multi-sample property of dLLMs into a **+28.5pp absolute** GSM8K boost on LLaDA-8B-Instruct, and **+17pp** over self-consistency at the same forward budget.

---

## 1. Abstract draft (≤200 words)

```
Diffusion language models (dLLMs) such as LLaDA generate text by iteratively
denoising masked positions, a paradigm that admits cheap *local* perturbation
of the output via partial remasking. Prior work has explored confidence-based
or context-perturbation remasking strategies (CoRe, STDD, Saber, DCD) at the
*span* level, motivated by the intuition that locally unstable spans are more
likely to be wrong. We first present a negative result: on GSM8K, span-level
remasking achieves only a 1.4% fix rate over greedy decoding, because the
infilling step tends to *restore the original incorrect answer* whenever the
surrounding reasoning chain is itself wrong. We then show that the right
granularity is *whole-sample*, not span: by sampling N candidate generations
at low temperature and selecting the answer that passes a task-specific
verifier (the canonical GSM8K answer extractor, no LM-as-judge), we obtain
**92.5% accuracy on GSM8K with N=5** (vs 64.0% greedy, +28.5pp). Against
self-consistency at the same forward budget, our method is **+17pp** at
N=5 and **+13pp** at N=3. The approach is training-free, requires no
auxiliary model, and exploits the unique property that dLLMs sample without
KV-cache and therefore pay no extra latency penalty for diverse generation.
```

---

## 2. Section outline

### §1 Introduction (≈1 page)

- Hook: AR-LMs use Self-Consistency (Wang et al. 2023) at N=40 to gain ~10pp on math reasoning. dLLMs have a **structural advantage** for cheap multi-sample but it has not been exploited.
- Setup: LLaDA-8B-Instruct on GSM8K. Vanilla 64.0%.
- Negative result first (counterintuitive hook): span-level remasking — the natural extension of CoRe/STDD/DCD — only fixes 1.4% of errors. Why? **Stable-but-wrong** reasoning chains.
- Positive result: verifier-guided reranking @ N=5 → 92.5% (+28.5pp). Same forward budget as N=5 SC but +17pp.
- Three contributions:
  1. First systematic comparison of span-level vs sample-level perturbation in dLLMs.
  2. Negative result on span remasking with mechanistic explanation.
  3. Strong positive baseline (VRerank) that should be the default for any future dLLM reasoning work.

### §2 Related Work (≈0.5 page)

四类，下面 §6 给完整文献清单：

1. **dLLM remasking / sampling**：CoRe, STDD, DCD, Saber, LLaDA's own confidence remasking.
2. **Test-time scaling for reasoning**：Self-Consistency (Wang 2023), Best-of-N, MCTS, ToT.
3. **Verifier-guided decoding for AR-LMs**：Cobbe 2021 (GSM8K verifier), Lightman 2023 (PRM800K), Hosseini 2024 (V-STaR).
4. **Diffusion LMs**：Austin 2021 (D3PM), Lou 2024 (SEDD), Nie 2025 (LLaDA), DiffuSeq, Dream-7B.

### §3 Background

- §3.1 LLaDA briefly: masked diffusion, block-wise semi-autoregressive decoding, no KV-cache.
- §3.2 Why dLLM sampling is structurally cheap:
  > AR-LM: N samples ≈ N×T forwards, dominated by KV-cache rebuild on rejection.
  > dLLM: N samples = N×S forwards (S=sampling steps). **Each forward is identical to greedy** — diversity is free.
- §3.3 Verifier model: define `V: Output → {0,1}` (GSM8K canonical answer extractor; pass = `extract_answer(o) is not None`).

### §4 Method

#### §4.1 Sample-Level Verifier-Guided Reranking (VRerank)

```
def vrerank(prompt, N, T, verifier):
    cands = [model.generate(prompt, temperature=T) for _ in range(N)]
    scored = [(c, verifier_score(c)) for c in cands]
    return argmax(scored, key=lambda x: x[1])
```

- `verifier_score(c)` = 1 if `extract_answer(c) is not None and answer is well-formed` else 0; tie-break by majority vote on extracted answers.
- Total cost = N × greedy cost. No verifier-time training, no LM-as-judge.

#### §4.2 Span-Level Remasking (negative result baseline)

为对比，复述 CoRe/STDD 的 span remasking 框架，给出形式化定义与我们的实现 (heuristic / random / verem-span 三种)。强调：所有变体在 GSM8K 上 fix rate ≤ 1.5%。

#### §4.3 Why span remasking fails: stable-but-wrong

- Mechanism analysis: given a wrong reasoning chain, the LM's `p(answer | masked answer position, full reasoning)` is sharply peaked at the *original* wrong answer because the reasoning constrains it. Infilling adds **no new information**.
- Sample-level perturbation, in contrast, perturbs the **reasoning chain** itself, opening the door to a different (correct) trajectory.

### §5 Experiments

#### §5.1 Setup

- Model: LLaDA-8B-Instruct, bf16, 1×A6000 (48GB).
- Dataset: GSM8K test, N=200 (random, seed=42). Seeds {0, 1, 42} for variance.
- Decoding: max_new_tokens=256, steps=256, block_length=32. Temperature 0.0 (greedy) or 0.1 (rerank).
- Verifier: canonical GSM8K extractor (regex on `#### <number>`).

#### §5.2 Main Results

**Table 1**：上面 outline 已经写好的主表。

#### §5.3 Ablations

- **N**: 1, 3, 5（已有）
- **Temperature**: 0.1, 0.3, (0.5 if time)（在跑）
- **Seed variance**: {0, 1, 42}（在跑）
- **Span vs Sample**：直接对比 fix rate 和 final acc。

#### §5.4 Cost analysis

- Forward count vs accuracy curve.
- Latency vs accuracy curve（强调 dLLM 的 sampling 不付额外 KV-cache 代价）。

#### §5.5 Case studies (3 个)

- 一个 stable-but-wrong: greedy 错，N=5 中第 3 个 sample 推理链不同，verifier 选中。
- 一个 verifier ties: 所有 N 个候选都 pass verifier，majority vote 兜底。
- 一个 hard failure: 所有 N 个候选都 fail verifier（reasoning skeleton 错）→ limitation。

### §6 Discussion / Limitations

- 依赖 verifier 信号 → 适用于数学、代码、constrained generation；不适用于 open-ended QA。
- N 增大遵循 sublinear scaling（待补图）。
- 不能修复 reasoning skeleton 错误（hard failure 类）。
- dLLM single-step cost 高于 AR-LM（领域共性问题，非本方法独家）。

### §7 Conclusion

四句话：dLLM has structurally cheap sampling → span remasking surprisingly fails → sample-level + verifier wins → strong default for future dLLM reasoning work.

---

## 3. Required figures / tables

| # | Title | Status |
|---|---|---|
| Tab 1 | Main GSM8K table (7 rows) | ✅ have data |
| Tab 2 | Ablation: N ∈ {1,3,5}, T ∈ {0.1, 0.3} | 🔄 partial |
| Tab 3 | Seed variance (mean ± std) | 🔄 跑中 |
| Fig 1 | Accuracy vs Forwards (Pareto) | ✏️ to plot |
| Fig 2 | Span vs Sample fix rate bar | ✏️ to plot |
| Box 1 | Algorithm: VRerank pseudocode | ✏️ to write |
| Box 2 | Case study (stable-but-wrong) | ✏️ to extract |

---

## 4. 写作顺序建议（最高 ROI 优先）

1. **Method §4** + **Algorithm Box**（最容易写，30 分钟）
2. **Main Table §5.2**（数据已齐）
3. **Negative result §4.3** + **§5.3 Span vs Sample** ablation（这是论文的"洞察"，要打磨）
4. **Introduction**（最后写，前面有了再回来）
5. **Related Work**（参考下面 §6 文献）
6. **Abstract**（最最后）

---

## 5. 风险 & B 计划

| 风险 | 概率 | B 计划 |
|---|---|---|
| Reviewer 说"verifier-guided 不新" | 高 | 强调 dLLM 特殊性 + 负面结果价值；short paper 角度更稳 |
| Reviewer 说"GSM8K 不够" | 中 | 加 MATH-500 subset 或 HumanEval（Day 9-10 计划已有） |
| Reviewer 说"baseline 不全" | 中 | 至少加 PRM-style verifier 对比；CoRe-style perturbation 复现 |
| 多 seed 结果方差很大 | 低 | 公开报告，加 N=10 reduce variance |

---

## 6. 推荐阅读清单（按重要性）

### 6.1 必读 — dLLM 自身（理解 baseline 和 backbone）

| Tier | Paper | 一句话 | Link |
|---|---|---|---|
| ⭐⭐⭐ | **LLaDA**: Large Language Diffusion Models (Nie et al. 2025) | 本文使用的 backbone，一定要读 generate.py | arXiv:2502.09992 |
| ⭐⭐⭐ | **D3PM**: Structured Denoising Diffusion Models in Discrete State-Spaces (Austin et al. 2021) | 离散 diffusion 的奠基论文 | arXiv:2107.03006 |
| ⭐⭐⭐ | **SEDD**: Discrete Diffusion Modeling by Estimating the Ratios of the Data Distribution (Lou et al. 2024) | 当前 SOTA 离散 diffusion，理论 + 实证 | arXiv:2310.16834 |
| ⭐⭐ | **MDLM**: Simple and Effective Masked Diffusion Language Models (Sahoo et al. 2024) | 极简的 masked diffusion 公式化 | arXiv:2406.07524 |
| ⭐⭐ | **DiffuSeq** (Gong et al. 2022) | Seq2seq diffusion | arXiv:2210.08933 |
| ⭐ | **Dream-7B** (HKUNLP 2024) | 另一个公开 dLLM，可作为补充 backbone | tech report |

### 6.2 必读 — dLLM remasking / 推理时改进（你的直接对手）

| Tier | Paper | 一句话 | 你要看的关键点 |
|---|---|---|---|
| ⭐⭐⭐ | **CoRe**: Context-aware Re-masking (NeurIPS 2024 candidate, 写 related work 必引) | context perturbation 给 span 打分再 remask | 它的 fix rate / acc 数字 vs 我们 |
| ⭐⭐⭐ | **STDD**: Self-Distillation Through Time (ICLR 2024) | 用更长 sampling chain 的自蒸馏 | sampling cost 模型 |
| ⭐⭐ | **DCD**: Diffusion Confidence Decoding | confidence-based remasking | confidence 信号有效性的边界 |
| ⭐⭐ | **Saber**: Self-Aware Bidirectional Reasoning (2024) | dLLM reasoning 改进 | 它的 reasoning task setup |
| ⭐⭐ | **LLaDA paper §4 Generation**：generate.py 里的 low-confidence remasking 和 random remasking | 我们的 random/heuristic baseline 来源 | 必须复述并对比 |

### 6.3 必读 — verifier-guided 推理（你的方法学血统）

| Tier | Paper | 一句话 |
|---|---|---|
| ⭐⭐⭐ | **GSM8K paper** (Cobbe et al. 2021) | verifier-guided sampling 的鼻祖；本文的 verifier 直接来自这里 |
| ⭐⭐⭐ | **Self-Consistency** (Wang et al. 2023) | 你的主要 baseline；要把 cost 对比讲清楚 |
| ⭐⭐⭐ | **Let's Verify Step by Step** (Lightman et al. 2023) | PRM 的代表作；说明为什么我们用 outcome verifier 而非 process |
| ⭐⭐ | **V-STaR** (Hosseini et al. 2024) | verifier + STaR 训练，证明 verifier 价值 |
| ⭐⭐ | **Best-of-N is Robust** (Stiennon 2020 / Cobbe 2021 后续) | BoN 经典分析 |
| ⭐ | **Math-Shepherd** (Wang et al. 2024) | 自动化 PRM，未来扩展可参考 |

### 6.4 必读 — 测试时扩展（讲 narrative 用）

| Tier | Paper | 一句话 |
|---|---|---|
| ⭐⭐⭐ | **Test-Time Compute Scaling Laws** (Snell et al. 2024) | OpenAI o1 之前的奠基；BoN/PRM/Lookahead 的统一框架 |
| ⭐⭐ | **Tree of Thoughts** (Yao et al. 2023) | 用来对比 ToT 的 cost |
| ⭐⭐ | **Self-Refine** (Madaan et al. 2023) | LM-as-judge 的代表，对比说明我们用 task verifier 的优势 |
| ⭐ | **MCTS for LLM reasoning** (e.g. ReST-MCTS, Zhang 2024) | 高 cost baseline 参考 |

### 6.5 强烈建议泛读 — 推理失败模式分析

| Tier | Paper | 一句话 |
|---|---|---|
| ⭐⭐ | **Faithfulness vs. Plausibility** (Turpin et al. 2023) | reasoning chain 看起来对但答案错的现象 |
| ⭐⭐ | **CoT Faithfulness** (Lanham et al. 2023) | 你 stable-but-wrong claim 的相邻文献 |
| ⭐ | **Sycophancy in LMs** (Sharma et al. 2024) | 模型坚持错误答案的倾向 |

---

## 7. 立即能做的下一步（你写论文时我并行做）

- [ ] 把 N=5 vs SC=5 的 case study 抽 3 个出来（自动）
- [ ] 画 Accuracy vs Forwards Pareto 图
- [ ] 等 seed=0/1 跑完，算 mean ± std
- [ ] 等 t=0.3 跑完，做 temperature ablation 图
- [ ] （optional）跑 1 条 LLaDA-Base 验证不依赖 instruction tuning
- [ ] （optional）写 §4 method 的 LaTeX algorithm box
