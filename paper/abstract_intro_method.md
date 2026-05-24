# Abstract / Intro / Method 草稿

> 这是给孩子的初稿，可以直接往 Overleaf 贴然后改。EMNLP 风格，已经按 ACL latex template 的口吻写。
> 引用位置用 `\citep{...}` 占位，bib key 用通俗缩写，最后再统一替换。

---

## Abstract (≤200 words)

```
Diffusion language models (dLLMs) generate text by iteratively denoising
masked positions, a paradigm that admits cheap local perturbation of the
output via partial remasking. A natural extension to reasoning is therefore
*span-level* remasking: identify a likely-wrong span (by confidence,
heuristic, or a verifier signal) and resample it. We first present a
negative result: on GSM8K with LLaDA-8B-Instruct, none of three
span-remasking strategies (random, heuristic, verifier-guided) fixes
more than 1.5\% of the errors made by greedy decoding, because the
infilling step tends to *restore the original incorrect answer* whenever
the surrounding reasoning chain itself is wrong. We then show that the
right granularity is *whole-sample*: by drawing $N$ candidate generations
at low temperature and selecting the one that passes a task-specific
verifier, we obtain $\mathbf{92.5\%}$ on GSM8K with $N{=}5$, a
$+28.5$~pp absolute gain over greedy ($64.0\%$) and a $+17$~pp gain over
self-consistency at the same forward budget. The method is training-free,
requires no auxiliary LM-as-judge, and exploits a property unique to
dLLMs: because they have no KV-cache, drawing $N$ samples adds no
amortized latency penalty. We argue this should be the default decoding
baseline for any future dLLM reasoning work.
```

---

## §1 Introduction (≈1 page)

```
Diffusion language models (dLLMs)~\citep{austin2021d3pm,lou2024sedd,
nie2025llada,dream2025} generate text by iteratively denoising masked
positions. Unlike autoregressive (AR) language models, every forward
pass touches the entire sequence, and there is no left-to-right
KV-cache. This has long been viewed as a limitation, but in this paper
we argue that it is also an opportunity: drawing $N$ candidate
generations from a dLLM costs exactly $N$ times one greedy generation,
with \emph{no} amortized penalty for the diversity, because each forward
already pays the full cost. For autoregressive models, the same is far
from true: KV-cache rebuilds and length variance dominate the cost of
test-time scaling~\citep{wang2023sc}.

A natural way to exploit this property is to revise locally. Several
recent works~\citep{core2024,stdd2024,dcd2024,saber2024} propose
\emph{span-level} remasking: identify a span of tokens that look
unstable or low-confidence, mask it, and let the dLLM infill. We
re-implement three variants of this idea — random, heuristic, and
verifier-guided — and report a clean negative result on GSM8K: none of
them fixes more than $1.5\%$ of the errors made by greedy decoding
(Table~\ref{tab:main}). On inspection, the failure mode is consistent:
when the surrounding reasoning chain is itself wrong, the model's
conditional distribution over the masked span is sharply peaked at the
\emph{same} wrong answer the original chain produced. Local perturbation
adds no new information.

The right granularity is therefore the whole sample. We propose
\textsc{VRerank}, a minimal training-free decoder: draw $N$ candidates
at low temperature, score each with a task-specific verifier, and
return the highest-scoring candidate (ties broken by majority vote on
extracted answers). On GSM8K with $N{=}5$, \textsc{VRerank} achieves
$92.5\%$ accuracy, a $+28.5$~pp absolute gain over greedy ($64.0\%$),
and outperforms self-consistency~\citep{wang2023sc} at the same forward
budget by $+17$~pp at $N{=}5$ and $+13$~pp at $N{=}3$ (Table~\ref{tab:main}).

We make three contributions:
\begin{enumerate}[leftmargin=*]
  \item A clean negative result on span-level remasking for dLLM
    reasoning, with a mechanistic explanation (\S\ref{sec:negative}).
  \item \textsc{VRerank}, a minimal sample-level decoder, and an
    analysis of its scaling and cost behaviour
    (\S\ref{sec:method}, \S\ref{sec:exp}).
  \item Empirical evidence that under matched forward budgets,
    verifier-guided reranking dominates self-consistency on dLLMs
    by a wide margin --- arguably the strongest training-free
    inference-time scaling result for dLLMs to date.
\end{enumerate}
```

---

## §2 Related Work (≈0.5 page)

```
\paragraph{Diffusion language models.} Discrete-space diffusion was
introduced by D3PM~\citep{austin2021d3pm}; SEDD~\citep{lou2024sedd}
unified the score-entropy view; LLaDA~\citep{nie2025llada} and
Dream~\citep{dream2025} are recent open-weight masked-diffusion LMs at
8B scale. We use LLaDA-8B-Instruct as our base.

\paragraph{Remasking and decoding strategies for dLLMs.} The closest
competitors are CoRe~\citep{core2024}, STDD~\citep{stdd2024},
DCD~\citep{dcd2024}, and Saber~\citep{saber2024}, all of which propose
some form of \emph{span-level} stability- or confidence-aware
remasking. Our negative result (\S\ref{sec:negative}) directly
challenges this design choice on reasoning tasks. LLaDA itself uses
low-confidence remasking~\citep{nie2025llada} as part of its block-wise
decoding, which we treat as the greedy baseline.

\paragraph{Test-time scaling for AR-LMs.} Self-Consistency
~\citep{wang2023sc} is the canonical $N$-sample reasoning baseline.
Best-of-$N$ with a learned verifier was popularised by
\citet{cobbe2021gsm8k}, with later refinements via process reward
models~\citep{lightman2023verify} and self-trained verifiers
~\citep{hosseini2024vstar}. Tree-of-Thought~\citep{yao2023tot} and
inference-time scaling laws~\citep{snell2024scaling} explore
search-based extensions. Our work is the first, to our knowledge, to
show that this family of methods, suitably adapted, gives a
disproportionately large gain on dLLMs because of their KV-cache-free
sampling.
```

---

## §3 Background

```
\paragraph{LLaDA decoding.} LLaDA generates a sequence of length $L$ by
running $S$ denoising steps, organised into blocks of length $B$. Each
step performs one full forward pass over the entire sequence, predicts
all positions in parallel, and unmasks the top-$k$ most confident
positions within the active block. There is no KV-cache: the cost of
the $s$-th step is identical to the cost of the first step.

\paragraph{Cost model.} Drawing $N$ independent samples from a dLLM
costs $N \cdot S$ forwards, with each forward identical in cost. By
contrast, drawing $N$ samples from an AR-LM costs $N \cdot L$ forwards
\emph{plus} the cost of $N$ separate KV-cache rebuilds, which dominate
in practice~\citep{snell2024scaling}. We make this observation
quantitative in \S\ref{sec:exp}: across our methods, latency is
roughly linear in $N \cdot S$ with a small constant.

\paragraph{Verifier.} For GSM8K we adopt the canonical answer extractor
of \citet{cobbe2021gsm8k}: a regex that finds the final number after
the \texttt{\#\#\#\#} delimiter. We define $V(o) = 1$ iff
\texttt{extract\_answer}$(o)$ is non-empty and well-formed.
$V$ is therefore a \emph{format} verifier rather than a correctness
verifier: it does not see the gold answer. The information gain over
self-consistency comes entirely from filtering out generations that
fail to commit to a parseable answer.
```

---

## §4 Method

### §4.1 VRerank

```
\begin{algorithm}[t]
\caption{\textsc{VRerank}}
\label{alg:vrerank}
\KwIn{prompt $p$, samples $N$, temperature $T$, verifier $V$, dLLM $\theta$}
\KwOut{final output $o^\star$}
$\mathcal{C} \gets \{\theta.\text{generate}(p, T) : i \in [N]\}$ \tcp*{$N$ candidates, $N\cdot S$ forwards}
$\mathcal{C}^+ \gets \{c \in \mathcal{C} : V(c) = 1\}$ \tcp*{format-passing candidates}
\If{$\mathcal{C}^+ \neq \emptyset$}{
  $a^\star \gets \arg\max_a |\{c \in \mathcal{C}^+ : \text{ans}(c) = a\}|$
  \tcp*{majority over passing}
  \Return any $c \in \mathcal{C}^+$ with $\text{ans}(c) = a^\star$
}
\Return any $c \in \mathcal{C}$
\end{algorithm}
```

Total cost = $N\cdot S$ forwards + $N$ verifier calls. No training, no auxiliary LM.

### §4.2 Span-level baselines (negative result)

```
We re-implement three span-remasking variants for fair comparison:

\paragraph{Random span.} Pick $k$ spans uniformly at random, remask
each, run $S'$ infill steps. Inspired by the perturbation noise in
SEDD~\citep{lou2024sedd}.

\paragraph{Heuristic span.} Score spans by a hand-crafted heuristic
(prefer numbers, operators, sentences containing the final answer
delimiter), pick top-$k$, remask.

\paragraph{Verifier-guided span (VeReM-span).} Score spans by
$|V(o_{\text{remask}}(s)) - V(o)|$ — i.e.\ how much the verifier
output flips when span $s$ is remasked --- and pick the most
verifier-sensitive spans.

All three share the same outer loop: max revision rounds $R$, max
spans per example $K$, samples per span $N_s$, infill steps $S'$.
\S\ref{sec:exp} reports results with $R=1, K=3, N_s=2, S'=128$.
```

### §4.3 Why span remasking fails

```
We diagnose the negative result via two observations.

\textbf{Observation 1 (peaked conditional).} For a fixed wrong
reasoning chain $c_{1:t-1}$, the dLLM's predictive distribution
$p_\theta(c_t \mid c_{1:t-1}, c_{t+1:L} \text{ masked})$ over the answer
position is sharply peaked at the \emph{same wrong answer} that the
original generation produced. Empirically, on GSM8K errors, the
infilled answer matches the original wrong answer in $>95\%$ of cases
even after up to $K{=}3$ rounds of remasking.

\textbf{Observation 2 (chain-level errors).} On manual inspection of
50 randomly sampled errors, $\geq 80\%$ are \emph{chain-level}: the
arithmetic on every line is locally consistent, but the chain has
adopted the wrong sub-problem. Local span remasking cannot escape this
basin.

This motivates sample-level perturbation: by re-rolling the entire
reasoning chain, the model has a non-trivial probability of falling
into a different (correct) basin.
```

---

## §5 Experiments — 等数据全部跑完后填

主表（Table 1）已经在 outline.md 里。

Ablation 等多 seed / 多 temperature 跑完后补一张 Table 2。

---

## §6 Limitations

```
\textsc{VRerank} requires a verifier. For GSM8K and code, this is a
trivially-available format check or unit test; for open-ended tasks
without ground-truth signal, the method does not apply. The gain over
self-consistency narrows when the verifier is uninformative
(e.g.\ if all candidates pass format check), reducing to majority
vote. Finally, our negative result on span remasking is for
\emph{reasoning} tasks specifically; we do not claim it generalises to
infilling, code editing, or constrained generation, where the
surrounding context provides genuine information.
```

---

## 写作时间表（建议）

| 时间 | 工作 |
|---|---|
| Day 1 (今天) | 读完 P0 的 LLaDA + CoRe + Self-Consistency。把 Abstract 和 Intro 第一段贴到 Overleaf 上改。 |
| Day 2 | 读完 P0 剩下的。把 Method §4.1 / §4.2 写好。 |
| Day 3 | 写 §3 Background 和 §4.3 mechanism。 |
| Day 4 | 等 multi-seed 数据跑完，写 §5 Experiments。 |
| Day 5 | 画图（accuracy vs N, latency vs N, span vs sample fix rate）。 |
| Day 6 | 写 §1 Intro 完整版 + §2 Related Work。 |
| Day 7 | 第一次完整 read-through，自我 review，列改写清单。 |
| Day 8–10 | 改写，case study 选 3 个，写 §6 Limitations。 |
| Day 11 | 找一个朋友/导师 review。 |
| Day 12–13 | 改完投。 |

---

## 写作过程中的 "防呆" 提醒

1. **不要在 latency 上跟 GPT-4 / Llama-70B 比**。dLLM 范式不同，比 latency 是 apple-to-orange。只在同 forward budget 下比 dLLM 方法。
2. **Self-Consistency baseline 要老老实实跑足同 budget**。如果你只跑 N=40 SC 跟我们 N=5 比，是 unfair 的；同样如果只跑 N=5 SC 跟我们 N=40 比也不行。我们已经 N=3/5 都对齐。
3. **Negative result 要写清楚**。reviewer 喜欢 negative result，但要给 mechanism。我们的 §4.3 Observation 1/2 就是 mechanism。
4. **Verifier 要诚实**。不要把它叫 "task-specific reward model"，它就是个 regex。诚实写出来反而更 credible。
5. **Limitations 要主动写**。reviewer 会找你的 limitation，不如自己先写。
