# Dataset replication on CPU

Phase: **development**. 1 targets; 24 held-out case/target audits; 432 raw timing samples. All audited, warmup and measured outputs matched uncached greedy.

Data seeds [11003], paired training seeds [7]. Width 64, two layers, four heads, context 64, vocabulary 22, FP32, 1 CPU thread. 800 training steps, batch 32, learning rate 0.003; 256 training and 8 held-out examples per task. First 4 held-out IDs per task timed; 2 warmups, 6 repetitions. Draft lengths [2, 8]; 25 output-token limit.

The protocol was frozen before final training/evaluation. Each length has its own three-method block and paired greedy baseline; positions are balanced within each case. Configuration order rotates datasets across training seeds; length order alternates. No result-dependent stopping or selection.

## Dataset variation

Each dataset column is the geometric mean of its three training-seed estimates. The range across datasets is descriptive, not a confidence interval. Below 1 means slowdown.

| Task | Length | Method | 11003 | Dataset min–max |
|---|---:|---|---:|---|
| repetitive | 2 | greedy | 1.000× | 1.000–1.000 |
| repetitive | 2 | prompt-k2 | 2.245× | 2.245–2.245 |
| repetitive | 2 | ngram-k2 | 0.891× | 0.891–0.891 |
| repetitive | 8 | greedy | 1.000× | 1.000–1.000 |
| repetitive | 8 | prompt-k8 | 4.995× | 4.995–4.995 |
| repetitive | 8 | ngram-k8 | 0.817× | 0.817–0.817 |
| structured | 2 | greedy | 1.000× | 1.000–1.000 |
| structured | 2 | prompt-k2 | 1.467× | 1.467–1.467 |
| structured | 2 | ngram-k2 | 2.243× | 2.243–2.243 |
| structured | 8 | greedy | 1.000× | 1.000–1.000 |
| structured | 8 | prompt-k8 | 1.557× | 1.557–1.557 |
| structured | 8 | ngram-k8 | 2.641× | 2.641–2.641 |
| unpredictable | 2 | greedy | 1.000× | 1.000–1.000 |
| unpredictable | 2 | prompt-k2 | 0.925× | 0.925–0.925 |
| unpredictable | 2 | ngram-k2 | 0.898× | 0.898–0.898 |
| unpredictable | 8 | greedy | 1.000× | 1.000–1.000 |
| unpredictable | 8 | prompt-k8 | 0.921× | 0.921–0.921 |
| unpredictable | 8 | ngram-k8 | 0.861× | 0.861–0.861 |

## Every fitted target

Estimate: median across cases of each case's median paired latency ratio. Intervals: 2000 whole-case bootstrap draws, seed 701, percentile 95%. These intervals are conditional on this model/dataset and do not capture training, dataset or host variation. Six repetitions stay together; they are not six independent experiments.

| Dataset | Train seed | Task | Length | Method | Speedup | Conditional 95% interval | Slowdown |
|---:|---:|---|---:|---|---:|---|---|
| 11003 | 7 | repetitive | 2 | ngram-k2 | 0.891× | 0.847–1.252 | yes |
| 11003 | 7 | repetitive | 2 | prompt-k2 | 2.245× | 2.117–2.449 | no |
| 11003 | 7 | structured | 2 | ngram-k2 | 2.243× | 2.203–2.334 | no |
| 11003 | 7 | structured | 2 | prompt-k2 | 1.467× | 1.419–1.638 | no |
| 11003 | 7 | unpredictable | 2 | ngram-k2 | 0.898× | 0.890–0.978 | yes |
| 11003 | 7 | unpredictable | 2 | prompt-k2 | 0.925× | 0.861–1.015 | yes |
| 11003 | 7 | repetitive | 8 | ngram-k8 | 0.817× | 0.810–1.026 | yes |
| 11003 | 7 | repetitive | 8 | prompt-k8 | 4.995× | 4.967–5.556 | no |
| 11003 | 7 | structured | 8 | ngram-k8 | 2.641× | 2.224–2.834 | no |
| 11003 | 7 | structured | 8 | prompt-k8 | 1.557× | 1.397–1.854 | no |
| 11003 | 7 | unpredictable | 8 | ngram-k8 | 0.861× | 0.820–0.917 | yes |
| 11003 | 7 | unpredictable | 8 | prompt-k8 | 0.921× | 0.859–0.964 | yes |

## Target accuracy

Accuracy uses all held-out inputs, separately from decoding equivalence. Exact continuation includes EOS; missing output tokens count as wrong. Unpredictable continuations cannot be inferred from the prompt; fast repetitive outputs can still be inaccurate.

| Dataset | Train seed | Task | Teacher-forced | Payload only | Free-running | Exact continuation |
|---:|---:|---|---:|---:|---:|---:|
| 11003 | 7 | repetitive | 100.00% | 100.00% | 100.00% | 100.00% |
| 11003 | 7 | structured | 70.00% | 68.75% | 42.00% | 0.00% |
| 11003 | 7 | unpredictable | 11.00% | 7.29% | 12.00% | 0.00% |

## Artifacts and limits

protocol.json freezes settings, dataset hashes, timing IDs, analysis and source hashes. Each dataset shard stores exact train/evaluation inputs. Each target directory stores its checkpoint, config, training trace, hardware/dependencies, task correctness shards, raw timing shards, and report.json with acceptance rates, target calls, draft overhead and all unrounded estimates. Gzip JSONL shards use deterministic compression and can be read with Python's gzip module.

Three synthetic datasets and three paired training seeds, one tiny FP32 CPU model size, shared host. Conditional case bootstrap excludes training/dataset and host uncertainty. No natural-language, large-model, GPU or sampling generalization.

See the project README and PROTOCOL.md for reproduction and existing work. This is replication of established greedy speculative verification and lookup approaches, not a novel algorithm.
