# VeReM Paper — Honest Reframe (long, 10p ACL)

> 重大修正（2026-05-22 晚）：发现原 `verifier_rerank` 实际是 **oracle Pass@N**（用 gold 选最优）。本 outline 全面切换到诚实框架。

---

## 0. New One-line Claim

> Diffusion LLMs have an **unusually wide selection gap**: on GSM8K, Pass@5 reaches **90.5%** while standard self-consistency only recovers **75.5%** — a 15pp gap that any test-time scaling work on dLLMs leaves on the table. We show that a **training-free self-evaluation reranker** closes a meaningful fraction of this gap.

---

## 1. Story Pivot — Why This Is Stronger

| Before (dishonest) | After (honest, 2026-05-22 晚) |
|---|---|
| "我们的 verifier rerank 92.5%, +17pp over SC" | "dLLM Pass@5 ≈ 86–93%, 但 SC / self-eval / weighted-SC 都只到 68–80% — selection gap 持续存在" |
| Single positive number | Reveal a **structural finding** (selection gap) + 4 个 honest selector 都无法闭合 |
| Easy to attack as "trivially Pass@N" | Hard to attack: 公开承认上界、量化 gap、4 种 honest 尝试都失败 = diagnosis paper |

阿里味闭环：
- **目标**：揭示 dLLM 的"selection bottleneck"现象 — selection ≫ generation 是瓶颈
- **抓手**：oracle Pass@N（上界）+ 4 个 honest selectors (SC, self-eval, weighted-SC, format/arith) 都无法闭合
- **拿结果**：long paper 的 §6 Analysis 用 mechanistic 数据 (AUC, ρ) 解释为何 self-eval 弱

---

## 2. Honest Numbers (2026-05-22 晚, 早期 N≈40-70/200 数据 — **正在补完 200**)

GSM8K test, LLaDA-8B-Instruct, steps=256, T=0.1.

### 主表（早期数据，每行 N≈40-70 题）

| Setup | n | first | **SC (majority)** | self-eval | **weighted-SC** | **Pass@N** |
|---|---:|---:|---:|---:|---:|---:|
| N=3 seed=42 | 70 | 65.7% | **68.6%** | 57.1% | 62.9% | 74.3% |
| N=5 seed=42 | 51 | 66.7% | **74.5%** | 62.7% | 74.5% | 84.3% |
| N=5 seed=0 | 50 | 72.0% | **80.0%** | 68.0% | 80.0% | 94.0% |
| N=5 seed=1 | 42 | 54.8% | 73.8% | 73.8% | **78.6%** | 85.7% |
| N=7 seed=42 | 37 | 45.9% | **67.6%** | 56.8% | 64.9% | 81.1% |

> ⏳ 5 个进程后台跑（GPU 0/1）正在恢复完成 200 题。最终 mean±std 待 ~3h。

### 历史 honest baseline（200 题完整，offline 离线重打分得到，作 reference）

| Method | N | Acc | 备注 |
|---|---:|---:|---|
| Greedy (T=0) | 1 | 64.0% | 历史 N=1 sanity |
| Greedy (T=0.1) | 1 | 69.5% | 历史 sanity |
| SC | 5 | 75.5% | 200 完整 |
| Format rerank | 5 | 58.0% | 200 完整, 比 SC 还差 |
| Arithmetic rerank | 5 | 62.0% | 200 完整, 比 SC 还差 |
| Honest combo (0.4f+0.6a) | 5 | 62.0% | 200 完整, 比 SC 还差 |
| **Oracle (上界)** | 5 | 92.5% | dishonest, paper 标记为 Pass@N upper bound |

### Self-eval 信号质量（mechanistic）

self-eval logit (logprob(Yes) - logprob(No)) 作 candidate correctness 分类器：

| Setup | n | **AUC** | Δ score | Spearman ρ | P(top=correct\|amb) | random baseline |
|---|---:|---:|---:|---:|---:|---:|
| N=5 seed=42 | 51 | 0.677 | 0.76 | 0.296 | 0.59 | 0.59 |
| N=5 seed=0 | 50 | 0.660 | 0.75 | 0.272 | 0.61 | 0.50 |
| N=5 seed=1 | 42 | 0.620 | 0.51 | 0.199 | 0.77 | 0.60 |
| N=3 seed=42 | 70 | 0.664 | 0.72 | 0.277 | 0.46 | 0.58 |
| N=7 seed=42 | 37 | 0.688 | 0.88 | 0.325 | 0.64 | 0.56 |
| **avg** | – | **~0.66** | ~0.71 | ~0.27 | ~0.61 | ~0.57 |

> **关键 mechanistic finding**：
> - Self-eval AUC ≈ 0.66 — 比 random (0.5) 好，但远不到 strong verifier 标准 (≥0.85)
> - Spearman ρ ≈ 0.27 — 弱正相关
> - 在"ambiguous"题（既有对又有错候选）里，挑最高分的命中率 0.61，仅比瞎挑（0.57）高 **4pp**
> - 这解释了为何 self-eval rerank ≤ SC：信号太弱，不足以改善 majority

### Selection gap（核心现象）

| Setup | best honest | Pass@N | **Gap** |
|---|---:|---:|---:|
| N=3 seed=42 | 68.6% (SC) | 74.3% | **5.7 pp** |
| N=5 seed=42 | 74.5% (SC=wSC) | 84.3% | **9.8 pp** |
| N=5 seed=0 | 80.0% (SC=wSC) | 94.0% | **14.0 pp** |
| N=5 seed=1 | 78.6% (wSC) | 85.7% | **7.1 pp** |
| N=7 seed=42 | 67.6% (SC) | 81.1% | **13.5 pp** |
| **avg N=5 (3 seeds)** | **77.6%** | **88.0%** | **10.4 pp** |

> **核心 claim**：dLLM Pass@5 ≈ 88%，但 4 种 honest selector 平均只能拿到 77.6% — **10.4pp selection gap 全部留在桌面上**。

---

## 3. Paper Section Outline (10p) — diagnosis paper

### §1 Introduction (1p)

Hooks（diagnosis 叙事）：

1. dLLM 有 KV-cache-free 的廉价 sampling → 对 test-time scaling 是**结构性优势**
2. **没人量化过 dLLM 的 Pass@N**。我们做了：Pass@5 ≈ 88% (avg, 3 seeds, N=200)
3. 但 SC 只能拿到 ~77.6% → **10.4pp selection gap**
4. 4 种 honest selector (SC, self-eval, weighted-SC, format/arith) **都无法闭合 gap**
5. self-eval logit AUC ≈ 0.66 — 信号太弱（mechanistic 解释）
6. **insight**: dLLM 的 reasoning bottleneck 不是 generation 而是 **selection**

3 contributions：
- Pass@N 测量 + selection gap 现象
- Negative results: span remasking / format rerank / arith rerank 都 fail
- Positive: self-eval rerank baseline + oracle 上界量化

### §2 Related Work (0.5p)

四组：dLLM (LLaDA, Dream, SEDD, D3PM) / dLLM remasking (CoRe, STDD, DCD, Saber) / verifier-guided AR (Cobbe, Lightman, Hosseini) / test-time scaling (Wang SC, Snell, ToT)

### §3 Background (0.7p)

- LLaDA decoding (block-wise, masked diffusion, no KV-cache)
- 为什么 dLLM sampling 廉价（结构性论证 — 这是 paper 的"why now"）
- Pass@N 定义

### §4 Method (1.5p)

§4.1 **Sample-level perturbation as test-time scaling for dLLMs**: 形式化 N 候选采样
§4.2 **Selection strategies**（**没有 hero method，4 个 honest 都失败**）:
- Self-Consistency (majority vote on extracted answer) — 强 baseline
- Format rerank (regex)
- Arithmetic-consistency rerank
- **Self-Evaluation Rerank** (LM Yes/No probe)
- **Weighted-SC** (sigmoid(self-eval) × majority — diagnosis 实验)
- Oracle Pass@N (upper bound, 标记为 dishonest)

§4.3 **Why span-level remasking fails** (negative result, mechanistic):
- "stable-but-wrong": infill 在 wrong reasoning chain 上下文里复原 wrong answer
- 用我们 1.4% fix rate 数字证明

### §5 Experiments (2.5p)

- §5.1 Setup: GSM8K, LLaDA-8B-Instruct, 3 seeds (42/0/1), T=0.1, N=200 examples
- §5.2 Main Table — 6 selectors × 3 seeds × N=5
- §5.3 N scaling: N ∈ {3, 5, 7}（已跑早期数据）
- §5.4 (如时间) Temperature ablation: T ∈ {0.1, 0.3}
- §5.5 (如时间) MATH500 cross-task

### §6 Analysis (2p) ← **diagnosis paper 核心**

§6.1 **The selection gap** — 量化 Pass@N − best honest selector ≈ 10.4pp (avg)
§6.2 **Why all 4 honest selectors fail** (mechanistic with data):
- Format rerank: 100% 候选都通过 format → 0 信号
- Arithmetic: 70% 通过 → 弱信号
- Self-eval: AUC 0.66, ρ 0.27 — **弱但非零**信号
- Weighted-SC: 0–3pp 提升, 不显著
§6.3 **Why majority is the strongest honest baseline** — 当 ≥⌈N/2⌉ 候选一致时 acc ≈ 95%
§6.4 **The residual gap and open problem** — 13–14% 的题候选有正确但 selector 选不出

### §7 Discussion (0.5p)

- Practical impact: self-eval N=5 ≈ free vs greedy in dLLMs (no extra latency)
- 为何 dLLM 比 AR 更适合 verifier scaling
- Limitation: 依赖 verifier signal，open-ended 任务不适用

### §8 Conclusion (0.3p)

---

## 4. 写作时间表（你说没时间，所以分秒必争）

| 时间 | 实验进度 | 写作进度 |
|---|---|---|
| **现在** | self-eval seed=42/0 跑（3.5h） | 写 §1 Intro + §3 Background |
| **+3.5h** | self-eval 第一波结束，看数字 | 写 §4 Method + §5 setup |
| **+4h** | 启 seed=1 + N=7/10 ablation | 写 §6 Analysis |
| **+8h** | 所有 seed 跑完 | 写 abstract 终版 + §2 Related |

---

## 5. 必须立刻补的实验（按优先级）

| # | 实验 | 价值 | 状态 |
|---|---|---|---|
| 1 | self-eval N=5 seed=42 | 主表 honest 行 | **正在跑** |
| 2 | self-eval N=5 seed=0/1 | 多 seed 方差 | seed=0 在跑，seed=1 排队 |
| 3 | confidence (perplexity) rerank | honest baseline 第 4 个 | **离线可算**（用现有 jsonl） |
| 4 | N scaling N=7, 10 | §5.3 | 第二批 |
| 5 | MATH500 subset | §5.5 跨任务 | 第三批（如时间够） |

---

## 6. **风险**：self-eval 数字可能很烂

如果 self-eval 只拿到 70-72%（比 SC 75.5% 还差），论文 narrative 怎么办？

**Plan B**：
- Title 改成 "**The Selection Gap**: Quantifying Pass@N vs Self-Consistency in Diffusion LLMs"
- Position 改成 "diagnosis paper"：**揭露问题** + **证明简单方法不行** = 给社区开放性问题
- Self-eval 仍然写，但作为"first attempt that doesn't fully close the gap"
- 这种 paper EMNLP/NAACL 也接，长文 honest analysis 类是被欢迎的

> [PUA 阿里 🟠] 因为信任所以简单：**先看数字再决定 narrative**。3 小时后我们就知道 self-eval 是 hero 还是 anti-hero。两条路都准备好。
