# Three-seed, two-size CPU comparison

Training seeds: [17, 29, 43]; shared data seed: 2026. Widths: [64, 96]; two layers, four heads, feed-forward width 4×model width, context 64, vocabulary 22. FP32 CPU, deterministic algorithms, cached decoding for every timed method.

Each target trains for 800 AdamW steps (batch 32, learning rate 0.003) on 256 sequences per task. 12 held-out cases per task; 1 intra-op and one inter-op thread; 2 warmups and 9 repetitions. Draft lengths: [1, 2, 4, 8]; output limit 25.

All six configurations use byte-identical training and evaluation data; all evaluation prefixes are excluded from training. Seeds vary initialization and minibatch sampling, not the dataset. One random method permutation per case is cyclically rotated across repetitions, balancing each method at every position within each case. Case order is shuffled. This balances position, not all possible carryover effects. Configurations run sequentially with sizes interleaved.

All 17,496 measured outputs matched their target's uncached greedy reference. Different targets need not produce the same outputs. Warmups and correctness audits also require agreement.

## Variation across training seeds

Each cell is a within-case paired latency speedup, summarized separately per trained target, then described across three seeds. Values below 1 are slowdowns. No samples or slow settings are omitted. The range is descriptive, not a confidence interval; timing repeats are not independent model trials.

| Width | Task | Method | Seed 17 | Seed 29 | Seed 43 | Median | Min–max | Slowdown seeds |
|---:|---|---|---:|---:|---:|---:|---|---|
| 64 | repetitive | greedy | 1.000× | 1.000× | 1.000× | 1.000× | 1.000–1.000 | — |
| 64 | repetitive | prompt-k1 | 1.552× | 1.584× | 1.638× | 1.584× | 1.552–1.638 | — |
| 64 | repetitive | prompt-k2 | 2.084× | 2.173× | 2.208× | 2.173× | 2.084–2.208 | — |
| 64 | repetitive | prompt-k4 | 3.338× | 3.514× | 3.628× | 3.514× | 3.338–3.628 | — |
| 64 | repetitive | prompt-k8 | 4.815× | 5.030× | 5.196× | 5.030× | 4.815–5.196 | — |
| 64 | repetitive | ngram-k1 | 0.890× | 0.871× | 0.894× | 0.890× | 0.871–0.894 | 17, 29, 43 |
| 64 | repetitive | ngram-k2 | 0.865× | 0.867× | 0.871× | 0.867× | 0.865–0.871 | 17, 29, 43 |
| 64 | repetitive | ngram-k4 | 0.862× | 0.836× | 0.852× | 0.852× | 0.836–0.862 | 17, 29, 43 |
| 64 | repetitive | ngram-k8 | 0.834× | 0.813× | 0.831× | 0.831× | 0.813–0.834 | 17, 29, 43 |
| 64 | structured | greedy | 1.000× | 1.000× | 1.000× | 1.000× | 1.000–1.000 | — |
| 64 | structured | prompt-k1 | 1.238× | 1.256× | 1.313× | 1.256× | 1.238–1.313 | — |
| 64 | structured | prompt-k2 | 1.786× | 1.760× | 1.881× | 1.786× | 1.760–1.881 | — |
| 64 | structured | prompt-k4 | 1.807× | 1.826× | 1.989× | 1.826× | 1.807–1.989 | — |
| 64 | structured | prompt-k8 | 1.981× | 1.881× | 2.037× | 1.981× | 1.881–2.037 | — |
| 64 | structured | ngram-k1 | 1.292× | 1.324× | 1.342× | 1.324× | 1.292–1.342 | — |
| 64 | structured | ngram-k2 | 2.215× | 2.181× | 2.202× | 2.202× | 2.181–2.215 | — |
| 64 | structured | ngram-k4 | 2.200× | 2.158× | 2.172× | 2.172× | 2.158–2.200 | — |
| 64 | structured | ngram-k8 | 2.345× | 2.291× | 2.277× | 2.291× | 2.277–2.345 | — |
| 64 | unpredictable | greedy | 1.000× | 1.000× | 1.000× | 1.000× | 1.000–1.000 | — |
| 64 | unpredictable | prompt-k1 | 0.920× | 0.885× | 0.940× | 0.920× | 0.885–0.940 | 17, 29, 43 |
| 64 | unpredictable | prompt-k2 | 0.921× | 0.879× | 0.961× | 0.921× | 0.879–0.961 | 17, 29, 43 |
| 64 | unpredictable | prompt-k4 | 0.901× | 0.871× | 0.928× | 0.901× | 0.871–0.928 | 17, 29, 43 |
| 64 | unpredictable | prompt-k8 | 0.879× | 0.841× | 0.912× | 0.879× | 0.841–0.912 | 17, 29, 43 |
| 64 | unpredictable | ngram-k1 | 0.937× | 0.883× | 0.944× | 0.937× | 0.883–0.944 | 17, 29, 43 |
| 64 | unpredictable | ngram-k2 | 0.905× | 0.871× | 0.921× | 0.905× | 0.871–0.921 | 17, 29, 43 |
| 64 | unpredictable | ngram-k4 | 0.891× | 0.863× | 0.915× | 0.891× | 0.863–0.915 | 17, 29, 43 |
| 64 | unpredictable | ngram-k8 | 0.867× | 0.815× | 0.900× | 0.867× | 0.815–0.900 | 17, 29, 43 |
| 96 | repetitive | greedy | 1.000× | 1.000× | 1.000× | 1.000× | 1.000–1.000 | — |
| 96 | repetitive | prompt-k1 | 1.590× | 1.606× | 1.576× | 1.590× | 1.576–1.606 | — |
| 96 | repetitive | prompt-k2 | 2.181× | 2.199× | 2.184× | 2.184× | 2.181–2.199 | — |
| 96 | repetitive | prompt-k4 | 3.451× | 3.502× | 3.456× | 3.456× | 3.451–3.502 | — |
| 96 | repetitive | prompt-k8 | 4.961× | 4.979× | 4.945× | 4.961× | 4.945–4.979 | — |
| 96 | repetitive | ngram-k1 | 0.870× | 0.873× | 0.860× | 0.870× | 0.860–0.873 | 17, 29, 43 |
| 96 | repetitive | ngram-k2 | 0.862× | 0.874× | 0.855× | 0.862× | 0.855–0.874 | 17, 29, 43 |
| 96 | repetitive | ngram-k4 | 0.837× | 0.841× | 0.841× | 0.841× | 0.837–0.841 | 17, 29, 43 |
| 96 | repetitive | ngram-k8 | 0.807× | 0.805× | 0.808× | 0.807× | 0.805–0.808 | 17, 29, 43 |
| 96 | structured | greedy | 1.000× | 1.000× | 1.000× | 1.000× | 1.000–1.000 | — |
| 96 | structured | prompt-k1 | 1.215× | 1.207× | 1.188× | 1.207× | 1.188–1.215 | — |
| 96 | structured | prompt-k2 | 1.707× | 1.774× | 1.672× | 1.707× | 1.672–1.774 | — |
| 96 | structured | prompt-k4 | 1.745× | 1.742× | 1.630× | 1.742× | 1.630–1.745 | — |
| 96 | structured | prompt-k8 | 1.789× | 1.806× | 1.714× | 1.789× | 1.714–1.806 | — |
| 96 | structured | ngram-k1 | 1.287× | 1.343× | 1.299× | 1.299× | 1.287–1.343 | — |
| 96 | structured | ngram-k2 | 2.234× | 2.146× | 2.161× | 2.161× | 2.146–2.234 | — |
| 96 | structured | ngram-k4 | 2.145× | 2.247× | 2.139× | 2.145× | 2.139–2.247 | — |
| 96 | structured | ngram-k8 | 2.168× | 2.479× | 2.180× | 2.180× | 2.168–2.479 | — |
| 96 | unpredictable | greedy | 1.000× | 1.000× | 1.000× | 1.000× | 1.000–1.000 | — |
| 96 | unpredictable | prompt-k1 | 0.919× | 0.906× | 0.901× | 0.906× | 0.901–0.919 | 17, 29, 43 |
| 96 | unpredictable | prompt-k2 | 0.909× | 0.909× | 0.914× | 0.909× | 0.909–0.914 | 17, 29, 43 |
| 96 | unpredictable | prompt-k4 | 0.888× | 0.889× | 0.886× | 0.888× | 0.886–0.889 | 17, 29, 43 |
| 96 | unpredictable | prompt-k8 | 0.866× | 0.867× | 0.862× | 0.866× | 0.862–0.867 | 17, 29, 43 |
| 96 | unpredictable | ngram-k1 | 0.910× | 0.931× | 0.897× | 0.910× | 0.897–0.931 | 17, 29, 43 |
| 96 | unpredictable | ngram-k2 | 0.908× | 0.909× | 0.893× | 0.908× | 0.893–0.909 | 17, 29, 43 |
| 96 | unpredictable | ngram-k4 | 0.887× | 0.893× | 0.871× | 0.887× | 0.871–0.893 | 17, 29, 43 |
| 96 | unpredictable | ngram-k8 | 0.836× | 0.843× | 0.830× | 0.836× | 0.830–0.843 | 17, 29, 43 |

## Target quality and individual measurements

Accuracy is against the same held-out truths, independently of decoding agreement. Token accuracy includes EOS except the payload column. Missing generated tokens count as wrong. Unpredictable continuations cannot be inferred from their prompts; repetitive model output can be fast to draft despite low task accuracy.

| Configuration | Parameters | Task | Teacher-forced | Payload only | Free-running | Exact continuation |
|---|---:|---|---:|---:|---:|---:|
| [width64-seed17](width64-seed17/RESULTS.md) | 107,008 | repetitive | 97.3% | 97.2% | 95.3% | 66.7% |
| [width64-seed17](width64-seed17/RESULTS.md) | 107,008 | structured | 72.7% | 71.5% | 44.0% | 0.0% |
| [width64-seed17](width64-seed17/RESULTS.md) | 107,008 | unpredictable | 10.3% | 6.6% | 9.0% | 0.0% |
| [width96-seed17](width96-seed17/RESULTS.md) | 234,240 | repetitive | 99.7% | 99.7% | 99.7% | 91.7% |
| [width96-seed17](width96-seed17/RESULTS.md) | 234,240 | structured | 74.3% | 73.3% | 47.3% | 0.0% |
| [width96-seed17](width96-seed17/RESULTS.md) | 234,240 | unpredictable | 11.3% | 7.6% | 10.0% | 0.0% |
| [width64-seed29](width64-seed29/RESULTS.md) | 107,008 | repetitive | 99.7% | 99.7% | 99.7% | 91.7% |
| [width64-seed29](width64-seed29/RESULTS.md) | 107,008 | structured | 72.3% | 71.2% | 40.7% | 0.0% |
| [width64-seed29](width64-seed29/RESULTS.md) | 107,008 | unpredictable | 11.0% | 7.3% | 9.7% | 0.0% |
| [width96-seed29](width96-seed29/RESULTS.md) | 234,240 | repetitive | 99.3% | 99.3% | 99.3% | 83.3% |
| [width96-seed29](width96-seed29/RESULTS.md) | 234,240 | structured | 72.7% | 71.5% | 45.3% | 0.0% |
| [width96-seed29](width96-seed29/RESULTS.md) | 234,240 | unpredictable | 9.3% | 5.6% | 8.7% | 0.0% |
| [width64-seed43](width64-seed43/RESULTS.md) | 107,008 | repetitive | 99.7% | 99.7% | 99.7% | 91.7% |
| [width64-seed43](width64-seed43/RESULTS.md) | 107,008 | structured | 72.7% | 71.5% | 45.3% | 0.0% |
| [width64-seed43](width64-seed43/RESULTS.md) | 107,008 | unpredictable | 9.0% | 5.2% | 9.7% | 0.0% |
| [width96-seed43](width96-seed43/RESULTS.md) | 234,240 | repetitive | 99.7% | 99.7% | 99.3% | 91.7% |
| [width96-seed43](width96-seed43/RESULTS.md) | 234,240 | structured | 70.3% | 69.1% | 41.3% | 0.0% |
| [width96-seed43](width96-seed43/RESULTS.md) | 234,240 | unpredictable | 10.7% | 6.9% | 7.0% | 0.0% |

Each linked report contains latency, acceptance, target calls and drafting overhead for every task/method. Adjacent files preserve all raw timings, inputs, correctness records, weights, hashes, training traces, dependency versions and hardware details. suite-report.json preserves unrounded per-configuration results and across-seed ranges.

## Limits

Only three initializations, two tiny widths, one shared synthetic dataset, one CPU environment and greedy decoding were tested. Width changes also change parameter count and training compute; the step budget, not training FLOPs or achieved quality, is held fixed. This does not isolate model size from output quality or draftability. Small held-out sets and shared-host scheduling limit precision. No extrapolation to large models, natural language, GPUs or sampling is justified. There is no novelty claim: this extends the existing greedy speculative verification, prompt lookup and n-gram baseline experiment with replication.
