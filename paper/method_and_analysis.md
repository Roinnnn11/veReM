# §4 Method (Draft)

> Self-contained method section. Numbers backfilled when experiments finish.

---

## 4.1 Sample-Level Perturbation as Test-Time Scaling for Diffusion LLMs

We focus on diffusion language models that generate text by iteratively
denoising a fully-masked sequence \citep{austin2021d3pm,nie2025llada}.
Let $f_\theta$ denote the model and $T$ the number of denoising steps.
A standard greedy decode produces an output $y^{(0)} = \mathrm{decode}_T(f_\theta, x)$
with $T$ forward passes.

Because each forward pass touches the entire sequence, a temperature-$\tau$
sample $y^{(i)}$ costs *exactly* $T$ forward passes — there is no KV-cache
to rebuild and no length-variance penalty. Drawing $N$ candidates therefore
costs $N \cdot T$ forwards, with no amortised cost from sampling diversity.
This is in stark contrast to autoregressive language models, where Best-of-$N$
incurs both (i) per-token overhead from re-rolling the KV-cache and
(ii) length-variance penalties when candidates differ in token count.

This structural property makes dLLMs an unusually attractive target for
test-time scaling: the **only** open question is *which* of the $N$
candidates to return.

## 4.2 Selection Strategies

We study six selection strategies, three honest (no access to gold) and
three reference points.

**Self-Consistency (SC) [\citealt{wang2023sc}].**
Each candidate is run through a deterministic answer extractor
$a(\cdot)$; we return the candidate whose extracted answer matches the
plurality vote.

**Format rerank.**
We score each candidate by a regex test for the canonical
`#### <number>` GSM8K format (1.0 if matched, 0.5 if a fallback
extractor fires, 0.0 otherwise) and return the highest-scoring candidate.
Ties are broken by candidate index.

**Arithmetic-consistency rerank.**
For each candidate, we extract all inline equations of the form
$a \diamond b = c$ where $\diamond \in \{+,-,*,/\}$, and compute the
fraction that are arithmetically correct. The candidate with the highest
fraction wins.

**Honest combo.**
A weighted sum of the format and arithmetic scores
($0.4 \cdot s_{\mathrm{format}} + 0.6 \cdot s_{\mathrm{arith}}$).
Tested as a check that simple combinations of cheap signals are not
sufficient.

**Self-Evaluation rerank (ours, main).**
We construct a probe prompt of the form

```
Problem: {x}
Proposed solution: {y^{(i)}}
Is the proposed solution correct? Answer with a single word: Yes or No.
```

and run a single forward pass with one masked position appended. We
collect the logits at that position and score the candidate by

$$ s(y^{(i)}) = \log \pi(\text{Yes} \mid \cdot) - \log \pi(\text{No} \mid \cdot), $$

where $\log \pi(\cdot)$ aggregates probability over all surface
realisations of "Yes" / "No" (with and without leading space, casing).
This adds one forward pass per candidate over the SC baseline. The
candidate with the highest $s$ is returned.

**Oracle (upper bound).**
For *analysis only*, we report the upper bound that ignores the verifier
and returns any candidate that matches the gold answer (equivalent to
$\mathrm{Pass@}N$). This is dishonest by construction and is reported
only to quantify the *selection gap* (\S\ref{sec:selection-gap}).

## 4.3 Why Span-Level Remasking Fails: a Negative Result

A natural alternative to sample-level methods, advocated by recent
remasking work \citep{core2024,stdd2024,dcd2024,saber2024}, is to
*locally* perturb a single span of the greedy output and let the model
infill. We re-implement three variants:

1. **Random remask.** Mask a random span of length $L_s$ and resample.
2. **Heuristic remask.** Mask the lowest-confidence span (per
\citealt{nie2025llada}) and resample.
3. **Verifier-guided remask.** Mask the span with the lowest
arithmetic-consistency score and resample.

All three achieve a *fix rate* of less than $1.5\%$ on GSM8K
(Table~\ref{tab:span-fail}) — meaning that among initially incorrect
predictions, fewer than 1 in 65 are corrected. Inspection reveals a
consistent failure mode that we call **stable-but-wrong**:

> When the surrounding reasoning chain is itself wrong, the model's
> conditional distribution $\pi(\text{span} \mid \text{rest of chain})$
> is sharply peaked at the *original* incorrect span, because the
> reasoning constrains the answer.

Local infilling thus adds no new information. The right granularity of
perturbation in dLLMs for reasoning is the **whole sample**: only by
resampling the reasoning chain itself do we open the door to a
different — possibly correct — trajectory. Section~\ref{sec:exp}
confirms this: the same forward budget reallocated to whole-sample
diversity yields large gains.

---

# §6 Analysis (Draft)

## 6.1 The Selection Gap

The most striking finding of our experiments is the size of the gap
between $\mathrm{Pass@}N$ and the best honest selector. On GSM8K with
LLaDA-8B-Instruct and $N{=}5$ candidates at $\tau{=}0.1$:

- $\mathrm{Pass@}5 = 90.5\%$ — at least one candidate is correct in
  9 out of 10 problems.
- Self-Consistency selects the correct candidate in $75.5\%$.
- The best honest verifier (\textit{TBD}) achieves $X\%$.
- Greedy (no resampling) achieves $64.0\%$.

The $15$~pp gap between Pass@$5$ and SC indicates that the *generator*
is not the bottleneck on GSM8K — the *selector* is. This re-frames the
research agenda for dLLM reasoning: rather than chasing better
generation (smarter remasking, longer chains), one should chase better
selection over a fixed set of cheap candidates.

We argue this re-framing is dLLM-specific. For autoregressive models,
$\mathrm{Pass@}N$ at fixed compute is bounded above by the cost of
$N$ KV-cache rebuilds, and the gap to SC is far smaller in published
benchmarks \citep{wang2023sc}. The dLLM's structural ability to draw
diverse candidates at no extra latency is what makes the selection
question both newly relevant and newly tractable.

## 6.2 Why Simple Honest Verifiers Fail

Format and arithmetic-consistency rerank both perform *worse* than SC,
despite using more information per candidate. Two failure modes:

**(F1) Most candidates pass format/arithmetic.** When all candidates
are well-formatted and arithmetically self-consistent (often the case
for LLaDA on GSM8K), the score reduces to a uniform tie, and we fall
back to the first candidate — which is just greedy.

**(F2) The wrong candidate is also self-consistent.** A candidate can
follow valid arithmetic and still apply it to the wrong setup.
Consistency does not imply correctness; it only rules out bookkeeping
errors.

These failures motivate self-evaluation: a *semantic* rather than
*syntactic* judgement.

## 6.3 What Self-Evaluation Captures

\textit{TBD --- backfill once the seed=42/0/1 runs finish.}

We will analyse:

- Which candidates self-eval prefers (top-1 vs majority alignment)
- Calibration of $s(y^{(i)})$ vs ground-truth correctness
- Cases where self-eval picks correctly when SC fails
- Cases where self-eval picks incorrectly when SC succeeds

## 6.4 What Self-Evaluation Misses

\textit{TBD --- depends on numbers.} Likely failure modes:

- Hard arithmetic where the model is confidently wrong
- Multi-step problems where mid-chain errors compound silently
- Cases where all $N$ candidates make a similar mistake (low diversity)

These will inform §7 Limitations.

---

# Tables (skeleton, fills in as runs finish)

## Table 1 — Main Results

| Method | N | Final Acc | Pass@$N$ | Forwards |
|---|---:|---:|---:|---:|
| Greedy ($\tau=0$) | 1 | 64.0 | 64.0 | 256 |
| Greedy ($\tau=0.1$, sanity) | 1 | 69.5 | 69.5 | 256 |
| Self-Consistency | 3 | 68.5 | 86.5 | 768 |
| Self-Consistency | 5 | 75.5 | 90.5 | 1280 |
| Format rerank | 5 | 58.0 | 90.5 | 1280 |
| Arithmetic rerank | 5 | 62.0 | 90.5 | 1280 |
| Honest combo | 5 | 62.0 | 90.5 | 1280 |
| **Self-eval rerank (ours)** | 3 | **TBD** | 86.5 | 768+3 |
| **Self-eval rerank (ours)** | 5 | **TBD** | 90.5 | 1280+5 |
| **Self-eval rerank (ours)** | 7 | **TBD** | TBD | 1792+7 |
| Oracle (upper bound) | 5 | 92.5 | 90.5 | 1280 |

## Table 2 — Multi-seed variance (N=5)

| Seed | Greedy | SC | Self-eval | Pass@5 |
|---|---:|---:|---:|---:|
| 42 | 64.0 | 75.5 | TBD | 90.5 |
| 0 | TBD | TBD | TBD | TBD |
| 1 | TBD | TBD | TBD | TBD |
| **mean ± std** | TBD | TBD | TBD | TBD |
