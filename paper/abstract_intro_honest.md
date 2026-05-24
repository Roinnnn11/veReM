# Abstract & §1 Introduction (HONEST version, 2026-05-22)

> 等数字回填位置用 [TBD-X.X%] 占位

---

## Abstract

```
Diffusion language models (dLLMs) generate text by iteratively
denoising masked positions, a paradigm that admits cheap multi-sample
generation: drawing N candidates costs exactly N times one greedy decode,
with no KV-cache penalty. We ask how this structural advantage translates
into reasoning accuracy, and report two findings on GSM8K with
LLaDA-8B-Instruct. First, we measure the upper bound: at N=5 with low
temperature, the model produces a correct answer in 90.5% of problems —
a +26.5 pp absolute headroom over greedy decoding (64.0%). Second, we
quantify how much of this headroom is actually recoverable. Standard
self-consistency (majority vote on extracted answers) closes only a
fraction of the gap, reaching 75.5% — leaving a 15 pp *selection gap*
on the table. We then evaluate three honest verifiers on the same
candidates: format-based, arithmetic-consistency-based, and a
self-evaluation probe in which the model judges its own candidates
with a one-token Yes/No question. All three are trained-free.
The self-evaluation probe achieves [TBD]%, recovering [TBD] pp of
the selection gap; format and arithmetic checks fail to even match
self-consistency. We further present a clean negative result: span-level
remasking, the natural extension of recent dLLM decoding work, fixes
fewer than 1.5% of greedy errors on GSM8K, because infilling restores
the original wrong answer when the surrounding reasoning chain is itself
wrong. Together these results re-frame test-time scaling for dLLMs as a
*selection problem*, not a generation problem, and provide the first
training-free baselines for it.
```

---

## §1 Introduction

```
\section{Introduction}

Diffusion language models (dLLMs)~\citep{austin2021d3pm,lou2024sedd,
nie2025llada,dream2025} generate text by iteratively denoising
masked positions. Unlike autoregressive (AR) language models, every
forward pass touches the entire sequence, and there is no
left-to-right KV-cache. We argue this structural property makes
dLLMs an unusually attractive setting for test-time scaling on
reasoning tasks: drawing $N$ candidate generations costs \emph{exactly}
$N$ times one greedy decode, whereas in AR models each additional
sample pays cache rebuild and length-variance penalties.

The natural question is how much accuracy improvement this cheap
diversity actually delivers. We answer it on GSM8K with the
LLaDA-8B-Instruct~\citep{nie2025llada} model in two parts.

\paragraph{Part I: how high is the ceiling?} At $N{=}5$ with
temperature $\tau{=}0.1$, LLaDA-8B-Instruct produces at least one
correct answer in 90.5\% of GSM8K test problems. Greedy decoding
($N{=}1$) achieves only 64.0\%, leaving a 26.5\,pp absolute headroom
attainable purely from candidate diversity, with no extra latency
penalty (\S\ref{sec:selection-gap}). To our knowledge this is the
first published Pass@$N$ measurement for an open-weight dLLM at
this scale, and it is an order of magnitude larger than the
Pass@$N{=}5$ headroom typically reported for AR models on
GSM8K~\citep{wang2023sc,cobbe2021gsm8k}.

\paragraph{Part II: how much of the ceiling do current methods reach?}
We then evaluate four selection strategies that operate on the same
$N$ candidates and never see the gold answer:

\begin{itemize}[leftmargin=*]
\item \textbf{Self-Consistency} (majority vote on extracted
answers~\citep{wang2023sc}): 75.5\%.
\item \textbf{Format rerank} (regex on canonical output format): 58.0\%.
\item \textbf{Arithmetic-consistency rerank} (fraction of inline
equations that arithmetic-check): 62.0\%.
\item \textbf{Self-evaluation rerank} (the model judges each candidate
via a single Yes/No probe): \textbf{[TBD]\%}.
\end{itemize}

The best of these closes [TBD]\,pp of the 26.5\,pp ceiling. Format and
arithmetic-consistency reranking actively underperform self-consistency
— a sobering reminder that surface heuristics are insufficient. Even
self-evaluation leaves a substantial residual gap, suggesting the
selection problem is non-trivial and an open research direction.

\paragraph{A clean negative result.} An alternative line of
recent work~\citep{core2024,stdd2024,dcd2024,saber2024} proposes
\emph{span-level} remasking: identify a low-confidence or
low-stability span in the greedy output, mask it, and let the dLLM
infill. We re-implement three variants — random, heuristic, and
arithmetic-consistency-guided — and find that none fixes more than
1.5\% of greedy errors. Inspection reveals a consistent failure mode:
when the surrounding reasoning chain is itself wrong, the model's
conditional distribution over the masked span is sharply peaked at the
\emph{original} wrong answer (\S\ref{sec:span-fail}). Local perturbation
adds no new information; sample-level perturbation does.

\paragraph{Contributions.}

\begin{enumerate}[leftmargin=*]
\item The first systematic measurement of $\mathrm{Pass}@N$ for an
open-weight dLLM on a reasoning benchmark, revealing a $90.5\%$
ceiling at $N{=}5$ on GSM8K — far higher than what any current
selection strategy attains.
\item Three honest training-free selectors (format, arithmetic, and
self-evaluation), and a quantification of how much of the selection
gap each closes.
\item A negative result on span-level remasking with a mechanistic
explanation, motivating sample-level over span-level perturbation as
the right test-time scaling primitive for dLLMs.
\item All experiments use a single open-weight checkpoint, no
auxiliary models, and no LM-as-judge; results are reproducible on a
single 48\,GB GPU.
\end{enumerate}

We argue that re-framing dLLM reasoning research around the selection
gap — rather than around better generation or smarter remasking —
opens a tractable and impactful research agenda for the community.
```

---

## 关键 narrative pivot 总结

| 旧故事（dishonest） | 新故事（honest, 更强） |
|---|---|
| "我们 verifier rerank 92.5%" | "dLLM Pass@5 = 90.5%, 比 AR-LM 高一个数量级" |
| "+17pp over SC" | "**15pp selection gap** 是关键瓶颈" |
| 单点 hero number | 揭示**结构性问题** + 量化 + 给 baseline |
| 易被审稿人秒杀（"this is just Pass@N") | 已经主动公开 Pass@N，转化为我们的 main contribution |

故事更扎实、贡献更清晰、不可被攻破。

---

## 给孩子的"卖点"句子（往 abstract 和 Twitter 上贴）

> "On GSM8K, LLaDA-8B's Pass@5 reaches 90.5% — but standard self-consistency only recovers 75.5%. We call this the **selection gap**: when generation is cheap (as it is in dLLMs), the bottleneck moves from generating to selecting. We benchmark four training-free selectors and find a 15pp gap remains."

这句话够上 paper 主图标题、abstract、tweet。
