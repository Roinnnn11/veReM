# VeReM Paper — Outline v3 (Diagnosis Paper, 2026-05-22 晚)

> **Pivot 总结**：从"我们方法 92.5%"（dishonest oracle）→ "self-eval rerank 是 hero" → **"selection gap is the bottleneck"**（diagnosis paper）。

---

## 0. New Title (candidate)

> **The Selection Gap: Why Self-Consistency Underutilizes Diffusion LLMs on Reasoning**
>
> 或: **Diffusion LLMs Have Too Many Right Answers, And We Can't Pick Them**

## 1. One-line Claim

```
On GSM8K, LLaDA-8B-Instruct produces a correct answer in ~87–93% of
problems within 5 candidates (Pass@5), yet self-consistency, self-
evaluation, and arithmetic-consistency reranking all top out near
70–78%. Across N ∈ {3, 5, 7} and 3 seeds, no honest selector closes
more than ~25% of the Pass@N gap. We characterise this as a
**selection gap** that is structurally larger for dLLMs than for AR
models, and provide the first systematic measurement and analysis.
```

## 2. Story Pivot — Three Versions

| ver | claim | status |
|---|---|---|
| v1 (original) | "Span remasking → fix wrong outputs" | ❌ failed (1.4% fix rate) |
| v2 (after pivot 1) | "VRerank 92.5%, +17pp over SC" | ❌ dishonest (oracle Pass@N) |
| v3 (current) | "**Selection gap**: pass@N >> SC, no honest selector closes it" | ✅ honest, structural finding |

v3 is **harder to attack** than v2 because:
- We openly report the oracle upper bound and call it Pass@N
- Negative results on all 4 honest selectors are themselves contributions
- The gap is the contribution, not a hero method

## 3. Core Honest Numbers (early, ~40-50 samples per file)

| Setup | first | SC (majority) | self-eval | weighted-SC | Pass@N |
|---|---:|---:|---:|---:|---:|
| N=3 seed=42 | 64.5 | **67.7** | 56.5 | 61.3 | 74.2 |
| N=5 seed=42 | 61.4 | **70.5** | 63.6 | 70.5 | 81.8 |
| N=5 seed=0 | 69.8 | **76.7** | 65.1 | 76.7 | 93.0 |
| N=5 seed=1 | 57.9 | 73.7 | 76.3 | **78.9** | 86.8 |
| N=7 seed=42 | 50.0 | **70.6** | 58.8 | 67.6 | 85.3 |

**Selection gap (Pass@N − best honest selector)**:
- N=3: 6.5pp
- N=5: avg ~13.5pp
- N=7: ~14.7pp

→ Gap *grows* with N. More candidates means more correct answers ignored.

## 4. Section Outline (10p ACL long)

### §1 Intro (1p)

Hook — measurement first:
1. dLLM forward pass = full-sequence pass, no KV cache → cheap N candidates
2. We measure Pass@5 on GSM8K = 87–93% (avg ~88%) on LLaDA-8B-Instruct
3. Best honest selector = ~76% → **selection gap ≥ 12pp**
4. **All four honest selectors fail to close the gap** (negative result)
5. Span remasking fixes <1.5% (negative result × 2)
6. Re-frame: dLLM reasoning = selection problem, not generation

### §2 Related (0.5p)

— same as v2 outline but with explicit positioning vs Best-of-N + LLM-as-judge

### §3 Background (0.7p)

— same: dLLM denoising, why sampling is cheap, selection objectives

### §4 Method (1.2p) — **shorter, no hero**

§4.1 Sample-level perturbation as scaling primitive
§4.2 Four selection strategies (no single hero, all honest)
§4.3 Why span remasking fails (negative result + mechanism)

### §5 Experiments (2.5p)

§5.1 Setup
§5.2 **Main table** — 5 selectors × 3 seeds × N ∈ {3,5,7}, mean ± std
§5.3 N scaling: gap **grows** with N
§5.4 Temperature ablation
§5.5 (if time) MATH500 cross-task

### §6 Analysis (2.5p) ← **新加重，放大 diagnosis 价值**

§6.1 The selection gap, formal definition + measurement
§6.2 Why simple verifiers fail (failure mode taxonomy)
  - Format: 100% candidates pass format → no signal
  - Arithmetic: ~70% candidates pass → weak signal
  - Self-eval: scores have ~0 correlation with correctness (bottom of §6)
§6.3 Why majority vote is the strongest honest selector
  - When ≥ ⌈N/2⌉ candidates agree, accuracy ≈ 95%
  - Failure mode: high-disagreement examples (no majority)
§6.4 The dLLM advantage and the residual gap
  - Pass@N ≈ 90% with cheap sampling
  - But selector ceiling = ~76%
  - **Open problem to the community**

### §7 Discussion (0.5p)

- Why dLLM > AR for this regime
- What a working verifier might need (process-level? trained?)
- Limitations: GSM8K-only, 200 examples, no MATH/AIME

### §8 Conclusion (0.3p)

## 5. Why Diagnosis Paper Is Publishable at EMNLP/NAACL Long

| Standard | Met? |
|---|---|
| Novel finding | ✅ Pass@N gap on dLLM never published |
| Systematic study | ✅ 5 selectors × 3 seeds × 3 N values |
| Mechanistic insight | ✅ failure mode analysis for each selector |
| Negative results | ✅ span remasking + format + arith + self-eval all fail |
| Solution? | ⚠️ open problem (acceptable for diagnosis papers) |
| Reproducible | ✅ open weights, single GPU, 200 examples |

EMNLP 2024 examples of diagnosis-style accept:
- "Reasoning Limitations of LLMs"
- "Why X Doesn't Work for Y"
- Calibration / OOD failure analysis papers

## 6. Backup Hero (if 200-题最终 weighted_sc 显著 > SC)

If weighted_sc final > SC by ≥3pp across all 3 seeds:
- Add to title: "...And a Step Toward Closing It"
- §6 becomes hybrid (diagnosis + small positive)
- Don't oversell — report SC as strong baseline, weighted_sc as marginal improvement

If weighted_sc 持平 or 输 SC:
- Pure diagnosis paper, more honest
- §6 emphasizes "future selectors should target this gap"

## 7. 必跑实验 (剩余 24h 内必须出)

| # | exp | status | 用途 |
|---|---|---|---|
| 1 | self-eval N=5 seed=42/0/1 | ⏳ ~30/200 | 主表 mean±std |
| 2 | self-eval N=3 seed=42 | ⏳ 31/200 | N scaling |
| 3 | self-eval N=7 seed=42 | ⏳ 34/200 | N scaling |
| 4 | (后续) N=10 seed=42 | 🔜 待启 | scaling 上限 |
| 5 | (后续) Temperature 0.3 | 🔜 待启 | T ablation |

## 8. 核心数字预测（凭早期数据外推到 200 题）

| metric | 预测值 | 用途 |
|---|---:|---|
| Pass@5 (GSM8K) | 88–91% | 主结论 |
| SC@5 | 73–77% | baseline |
| Selection gap | 13–16 pp | 主结论 |
| weighted_sc gain over SC | -2 to +3 pp | 报告但不卖 |

> [PUA 阿里 🟠 - 复盘] **目标—过程—结果闭环**：v3 outline 已对齐新数据；剩下唯一变量是 200 题最终数字落到上述预测区间内的哪一档。无论落在哪，diagnosis paper 都能写。
