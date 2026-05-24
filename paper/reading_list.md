# 必读论文清单（按阅读优先级排序）

孩子读论文的顺序就按 P0 → P1 → P2，P0 三天读完，P1 一周读完。

---

## P0 — 必须精读（论文写作直接引用）

### 1. dLLM 主线：LLaDA & 同期 dLLM

| # | 标题 | venue / year | 为什么必读 |
|---|---|---|---|
| 1 | **LLaDA: Large Language Diffusion with mAsking** | arXiv 2025 (Nie et al., GSAI) | 我们的 base model；要复述其 block-wise decoding 和 confidence remasking |
| 2 | **Dream 7B** | technical report 2025 | 同期最强 dLLM，要在 related work 引用 |
| 3 | **SEDD: Score Entropy Discrete Diffusion** (Lou et al.) | ICML 2024 | 现代 dLLM 的训练目标 |
| 4 | **D3PM: Structured Denoising Diffusion Models in Discrete State-Spaces** (Austin et al.) | NeurIPS 2021 | 离散扩散的奠基论文 |

### 2. dLLM remasking / decoding 改进（直接竞品）

| # | 标题 | venue / year | 关系 |
|---|---|---|---|
| 5 | **CoRe: Context-Perturbed Remasking for Diffusion Language Models** | NeurIPS 2024 | 我们的 main competitor，必须仔细比较 |
| 6 | **STDD: Self-Distillation Through Time for dLLM** | 2024 | 训练态 remasking baseline |
| 7 | **DCD: Diffusion Confidence Decoding** | 2024 | confidence-based remasking |
| 8 | **Saber: Stability-Aware Beam-search-like Remasking** | 2024 | stability-based remasking |

> 注：CoRe / STDD / DCD / Saber 这几个的 venue 我没有 100% 确认，找的时候直接 Google Scholar + "diffusion language model remasking" 一键搜到。

### 3. AR-LM 的 verifier / Best-of-N / Self-Consistency

| # | 标题 | venue / year | 关系 |
|---|---|---|---|
| 9 | **Self-Consistency Improves Chain-of-Thought Reasoning** (Wang et al.) | ICLR 2023 | 我们的 main baseline |
| 10 | **Training Verifiers to Solve Math Word Problems** (Cobbe et al.) | arXiv 2021 | GSM8K 数据集 + 第一篇 verifier 论文 |
| 11 | **Let's Verify Step by Step** (Lightman et al., OpenAI PRM800K) | ICLR 2024 | Process reward model，verifier 概念升级版 |
| 12 | **V-STaR: Training Verifiers for Self-Taught Reasoners** (Hosseini et al.) | COLM 2024 | verifier 训练方法 |

---

## P1 — 强烈推荐（写 Related Work 用）

### 4. Test-time scaling / inference-time compute

| # | 标题 | venue / year |
|---|---|---|
| 13 | **Scaling LLM Test-Time Compute Optimally Can be More Effective Than Scaling Model Parameters** (Snell et al.) | 2024 |
| 14 | **Tree of Thoughts** (Yao et al.) | NeurIPS 2023 |
| 15 | **Best-of-N: A Case Study on Test-Time Compute** | 2024 |
| 16 | **Chain-of-Thought Prompting** (Wei et al.) | NeurIPS 2022 |

### 5. 数学推理 benchmark / verifier 设计

| # | 标题 | venue / year |
|---|---|---|
| 17 | **Measuring Mathematical Problem Solving with the MATH Dataset** (Hendrycks et al.) | NeurIPS 2021 |
| 18 | **Math-Shepherd** (Wang et al.) | ACL 2024 |
| 19 | **OpenMathInstruct-1/2** | NeurIPS 2024 |

### 6. 代码任务（如果你做 HumanEval）

| # | 标题 | venue / year |
|---|---|---|
| 20 | **HumanEval / Codex** (Chen et al.) | arXiv 2021 |
| 21 | **MBPP** (Austin et al.) | 2021 |
| 22 | **CodeT: Code Generation with Generated Tests** (Chen et al.) | ICLR 2023 |
| 23 | **AlphaCode** (Li et al.) | Science 2022 |

---

## P2 — 加分项（有时间再看）

### 7. 其他 dLLM / non-AR

| # | 标题 |
|---|---|
| 24 | DiffuSeq (Gong et al., 2022) |
| 25 | Plaid (Gulrajani & Hashimoto, 2024) |
| 26 | MD4: Masked Diffusion Models (Shi et al., 2024) |
| 27 | MDLM (Sahoo et al., NeurIPS 2024) |
| 28 | Diffusion-LM (Li et al., NeurIPS 2022) |

### 8. Reranking / refining 一般化

| # | 标题 |
|---|---|
| 29 | **Self-Refine** (Madaan et al., NeurIPS 2023) |
| 30 | **Reflexion** (Shinn et al., NeurIPS 2023) |
| 31 | **CRITIC** (Gou et al., ICLR 2024) |

---

## 阅读策略（孩子省时间用）

**P0 第一遍**（每篇 30–60 min）：
1. 跳读 abstract + intro + experiments table。
2. 抄下 main claim、main numbers、main figure 标题。
3. 标记我们 paper 中要引用的位置（intro hook / related work / method 对比 / experiment baseline）。

**P0 第二遍**（每篇 30 min，仅 dLLM 主线 #1, #5）：
- 看 method section。CoRe 怎么做 context perturbation？LLaDA 的 confidence remasking 公式怎么写？这两个我们 method section 里要严格对比。

**P1**：只看 abstract + intro + 一张主表，能在 related work 写一句话即可。

**P2**：扫 abstract，知道存在即可。

---

## 一些找论文的实用 tips

1. **CoRe / STDD / DCD / Saber 这种缩写**：直接 Google Scholar 搜全名 "context-perturbed remasking diffusion"，第一个就是。
2. **dLLM 综述**：搜 "survey diffusion language models 2025"，找一篇近期综述能一次扫到所有同期工作。
3. **OpenReview 直接刷**：ICLR 2025 / NeurIPS 2024 投稿区搜 "diffusion language" + "discrete"，能看到刚拒未发的工作。
4. **ar5iv / arXiv 阅读器**：长文用 ar5iv.org 看，比 PDF 快。

---

## 我们论文的 "差异化" 一句话总结（写 related work 必备）

> 与 CoRe / STDD / DCD / Saber 这些 **span-level** remasking 工作不同，本文主张 **sample-level** verifier-guided reranking 是更适合 dLLM reasoning 的 inference-time scaling 范式：(i) span-level remasking 在 reasoning 任务上 fail（我们的 negative result），(ii) sample-level 利用 dLLM 的 KV-cache-free 性质做廉价采样，(iii) 在同 forward budget 下显著优于 self-consistency。

这句话直接就是 related work 的小结段。
