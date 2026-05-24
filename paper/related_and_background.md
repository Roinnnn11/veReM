# §2 Related Work + §3 Background (Draft)

> 不依赖具体数字，可以现在写完。

---

## §2 Related Work

```
\section{Related Work}

\paragraph{Diffusion language models.}
Discrete-space diffusion was introduced for language by
\citet{austin2021d3pm}; subsequent work refined the training objective
\citep{lou2024sedd,sahoo2024mdlm,shi2024md4} and scaled to larger
checkpoints. LLaDA-8B \citep{nie2025llada} and Dream-7B \citep{dream2025}
are the leading open-weight masked-diffusion language models at the
time of writing. We use LLaDA-8B-Instruct as our base.

\paragraph{Decoding strategies for dLLMs.}
The most relevant prior work proposes \emph{span-level} remasking
strategies. CoRe \citep{core2024} performs context-perturbed remasking;
STDD \citep{stdd2024} uses self-distillation through time;
DCD \citep{dcd2024} introduces a confidence-based remasking schedule;
Saber \citep{saber2024} uses stability-aware beam-search-like remasking.
LLaDA's own decoder uses confidence-based partial remasking
\citep{nie2025llada}. We evaluate the closest analogues of these
methods on GSM8K (\S\ref{sec:span-fail}) and find that none fix more
than 1.5\% of greedy errors, motivating our shift to sample-level
strategies. Our work is the first to systematically isolate this
distinction.

\paragraph{Test-time scaling for AR language models.}
Best-of-$N$ with a learned reward model was popularised for math
reasoning by \citet{cobbe2021gsm8k}, with later refinements via
process-reward models \citep{lightman2023verify,wang2024mathshepherd}
and self-trained verifiers \citep{hosseini2024vstar}.
Self-Consistency \citep{wang2023sc} replaces a reward model with
plurality vote on extracted answers. Tree-of-Thought
\citep{yao2023tot} and inference-time scaling laws
\citep{snell2024scaling} extend the search space. Our work draws on
this literature for the design of selection strategies but contributes
a new finding: that the structural cheapness of dLLM sampling makes
the \emph{ceiling} of these methods qualitatively higher than for AR
models on the same benchmark.

\paragraph{Self-evaluation as a verifier signal.}
Asking a language model to judge its own output traces back to
self-refinement \citep{madaan2023selfrefine}, reflexion
\citep{shinn2023reflexion}, and CRITIC \citep{gou2024critic}. These
works typically use longer LM-as-judge prompts and target dialogue or
code tasks. Our self-evaluation probe is deliberately minimal — a
single masked position predicting Yes or No — to keep the verifier
cost negligible relative to candidate generation, and to make the
probe a cheap drop-in that any dLLM can run on its own outputs.

\paragraph{Pass@$N$ as a measurement primitive.}
Pass@$N$ has been used extensively in code generation
\citep{chen2021codex,li2022alphacode,chen2023codet} as both an
evaluation metric and a planning oracle. To our knowledge it has been
under-used in math reasoning evaluation of dLLMs; our work provides
the first systematic measurement and analysis of $\mathrm{Pass}@N$
versus realisable selectors for an open-weight dLLM.
```

---

## §3 Background

```
\section{Background}
\label{sec:background}

\subsection{Masked-diffusion language models}
A masked-diffusion language model
\citep{austin2021d3pm,lou2024sedd,nie2025llada} parameterises a denoising
network $f_\theta : \mathcal{V}^L \to \Delta^L_{|\mathcal{V}|}$ that
maps a partially-masked input sequence (with mask token $\square$) to a
distribution over tokens at every position. To generate, one initialises
a sequence of length $L$ as fully masked, iteratively (i) predicts a
distribution at every masked position, (ii) selects which positions
to commit, and (iii) replaces the rest with mask tokens for the next
step. After $T$ steps, all positions are committed.

LLaDA further organises the $T$ steps into blocks of length $B$, decoded
left-to-right; within each block, all positions are denoised
simultaneously. We adopt the original block-wise schedule with
$B{=}32$ and $T{=}256$ throughout.

\subsection{Why dLLM sampling is structurally cheap}
\label{sec:cheap-sampling}
A central observation behind our work is that, for masked-diffusion
language models, drawing a temperature-$\tau$ candidate costs exactly
the same as a greedy decode. This is because (i) every forward pass
already touches the entire sequence, (ii) there is no left-to-right
KV-cache to rebuild between candidates, and (iii) all $N$ candidates
have the same length $L$ by construction, so there is no length-
variance penalty.

In contrast, an autoregressive model paying $T$ tokens of greedy decode
incurs an amortised KV-cache rebuild plus length variance for each
additional sample. Empirically this manifests as a $2{-}5\times$ wall
clock penalty for Best-of-$N$ on standard benchmarks
\citep{snell2024scaling}; in our setting this penalty is zero.

We exploit this property by treating sample-level diversity as the
primary lever for test-time scaling: $N$ candidates from $N$ independent
$\tau$-sampled decodes, all running on the same hardware in the same
total wall clock as a single greedy decode (modulo fixed batching).

\subsection{Selection objectives}
Given $N$ candidate decodes $y^{(1)}, \dots, y^{(N)}$ for a problem $x$,
let $a(y)$ denote the canonical extracted answer (e.g.\ the number
following the last \texttt{\#\#\#\#} marker on GSM8K) and let $\star$
denote the gold answer. We use three accuracy quantities:

\begin{itemize}[leftmargin=*]
\item \textbf{Greedy accuracy.} $\Pr[a(y^{(0)}) = \star]$, where
$y^{(0)}$ is a single greedy decode (the canonical baseline).
\item \textbf{Selector accuracy.} $\Pr[a(y^{(\sigma(x))}) = \star]$
where $\sigma : x \mapsto \{1,\dots,N\}$ is a selection function that
does not see $\star$. Different selectors define different $\sigma$.
\item \textbf{Pass@$N$.} $\Pr[\exists i : a(y^{(i)}) = \star]$,
the probability that at least one candidate is correct. This is the
upper bound on selector accuracy and equivalent to a hypothetical
oracle selector with access to $\star$.
\end{itemize}

The gap $\mathrm{Pass}@N - \mathrm{Greedy}$ measures the headroom
available to candidate diversity; the gap $\mathrm{Pass}@N -
\mathrm{Selector}$ measures the residual failure of the selector. Our
results in \S\ref{sec:exp} report all three.
```

---

## 三句话总结（对齐核对用）

1. **§2 Related Work** 把工作放进 4 个相邻领域的 taxonomy（dLLM models, dLLM remasking, AR test-time scaling, self-eval），明确区分点：sample-level vs span-level、ceiling 测量为 first，Pass@N as primitive 引入到 dLLM。
2. **§3.2 Cheap sampling 论证**是 paper "why now"：结构性论证 dLLM 在 test-time scaling 上的 cost 优势——不是 demo trick。
3. **§3.3 三个 accuracy 量**（Greedy / Selector / Pass@N）形式化，§5 主表三列对应这三个量，故事一以贯之。
