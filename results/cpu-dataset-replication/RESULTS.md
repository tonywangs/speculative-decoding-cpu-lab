# Dataset replication on CPU

Phase: **final**. 9 targets; 3,456 held-out case/target audits; 15,552 raw timing samples. All audited, warmup and measured outputs matched uncached greedy.

Data seeds [3101, 6203, 9307], paired training seeds [17, 29, 43]. Width 64, two layers, four heads, context 64, vocabulary 22, FP32, 1 CPU thread. 800 training steps, batch 32, learning rate 0.003; 256 training and 128 held-out examples per task. First 16 held-out IDs per task timed; 2 warmups, 6 repetitions. Draft lengths [2, 8]; 25 output-token limit.

The protocol was frozen before final training/evaluation. Each length has its own three-method block and paired greedy baseline; positions are balanced within each case. Configuration order rotates datasets across training seeds; length order alternates. No result-dependent stopping or selection.

## Dataset variation

Each dataset column is the geometric mean of its three training-seed estimates. The range across datasets is descriptive, not a confidence interval. Below 1 means slowdown.

| Task | Length | Method | 3101 | 6203 | 9307 | Dataset min–max |
|---|---:|---|---:|---:|---:|---|
| repetitive | 2 | greedy | 1.000× | 1.000× | 1.000× | 1.000–1.000 |
| repetitive | 2 | prompt-k2 | 2.251× | 2.245× | 2.292× | 2.245–2.292 |
| repetitive | 2 | ngram-k2 | 0.879× | 0.949× | 0.884× | 0.879–0.949 |
| repetitive | 8 | greedy | 1.000× | 1.000× | 1.000× | 1.000–1.000 |
| repetitive | 8 | prompt-k8 | 5.311× | 5.250× | 5.203× | 5.203–5.311 |
| repetitive | 8 | ngram-k8 | 0.834× | 0.898× | 0.838× | 0.834–0.898 |
| structured | 2 | greedy | 1.000× | 1.000× | 1.000× | 1.000–1.000 |
| structured | 2 | prompt-k2 | 1.836× | 1.768× | 1.785× | 1.768–1.836 |
| structured | 2 | ngram-k2 | 2.235× | 2.265× | 2.251× | 2.235–2.265 |
| structured | 8 | greedy | 1.000× | 1.000× | 1.000× | 1.000–1.000 |
| structured | 8 | prompt-k8 | 1.928× | 1.908× | 1.951× | 1.908–1.951 |
| structured | 8 | ngram-k8 | 2.465× | 2.344× | 2.505× | 2.344–2.505 |
| unpredictable | 2 | greedy | 1.000× | 1.000× | 1.000× | 1.000–1.000 |
| unpredictable | 2 | prompt-k2 | 0.926× | 0.940× | 0.929× | 0.926–0.940 |
| unpredictable | 2 | ngram-k2 | 0.911× | 0.921× | 0.918× | 0.911–0.921 |
| unpredictable | 8 | greedy | 1.000× | 1.000× | 1.000× | 1.000–1.000 |
| unpredictable | 8 | prompt-k8 | 0.902× | 0.916× | 0.872× | 0.872–0.916 |
| unpredictable | 8 | ngram-k8 | 0.873× | 0.891× | 0.892× | 0.873–0.892 |

## Every fitted target

Estimate: median across cases of each case's median paired latency ratio. Intervals: 2000 whole-case bootstrap draws, seed 701, percentile 95%. These intervals are conditional on this model/dataset and do not capture training, dataset or host variation. Six repetitions stay together; they are not six independent experiments.

| Dataset | Train seed | Task | Length | Method | Speedup | Conditional 95% interval | Slowdown |
|---:|---:|---|---:|---|---:|---|---|
| 3101 | 17 | repetitive | 2 | ngram-k2 | 0.868× | 0.854–0.877 | yes |
| 3101 | 17 | repetitive | 2 | prompt-k2 | 2.238× | 2.213–2.293 | no |
| 3101 | 17 | structured | 2 | ngram-k2 | 2.233× | 2.183–2.343 | no |
| 3101 | 17 | structured | 2 | prompt-k2 | 1.864× | 1.638–1.957 | no |
| 3101 | 17 | unpredictable | 2 | ngram-k2 | 0.908× | 0.869–0.955 | yes |
| 3101 | 17 | unpredictable | 2 | prompt-k2 | 0.918× | 0.880–0.958 | yes |
| 3101 | 17 | repetitive | 8 | ngram-k8 | 0.834× | 0.819–0.860 | yes |
| 3101 | 17 | repetitive | 8 | prompt-k8 | 5.373× | 5.287–5.502 | no |
| 3101 | 17 | structured | 8 | ngram-k8 | 2.457× | 2.215–2.601 | no |
| 3101 | 17 | structured | 8 | prompt-k8 | 1.876× | 1.681–2.085 | no |
| 3101 | 17 | unpredictable | 8 | ngram-k8 | 0.868× | 0.849–0.918 | yes |
| 3101 | 17 | unpredictable | 8 | prompt-k8 | 0.877× | 0.848–0.906 | yes |
| 6203 | 17 | repetitive | 8 | ngram-k8 | 0.967× | 0.818–1.049 | yes |
| 6203 | 17 | repetitive | 8 | prompt-k8 | 5.327× | 5.056–5.395 | no |
| 6203 | 17 | structured | 8 | ngram-k8 | 2.313× | 2.190–2.376 | no |
| 6203 | 17 | structured | 8 | prompt-k8 | 1.886× | 1.669–2.139 | no |
| 6203 | 17 | unpredictable | 8 | ngram-k8 | 0.903× | 0.836–0.928 | yes |
| 6203 | 17 | unpredictable | 8 | prompt-k8 | 0.926× | 0.832–0.981 | yes |
| 6203 | 17 | repetitive | 2 | ngram-k2 | 0.971× | 0.852–1.129 | yes |
| 6203 | 17 | repetitive | 2 | prompt-k2 | 2.267× | 2.201–2.337 | no |
| 6203 | 17 | structured | 2 | ngram-k2 | 2.240× | 2.191–2.267 | no |
| 6203 | 17 | structured | 2 | prompt-k2 | 1.788× | 1.641–1.900 | no |
| 6203 | 17 | unpredictable | 2 | ngram-k2 | 0.927× | 0.917–0.967 | yes |
| 6203 | 17 | unpredictable | 2 | prompt-k2 | 0.932× | 0.888–0.961 | yes |
| 9307 | 17 | repetitive | 2 | ngram-k2 | 0.869× | 0.852–0.884 | yes |
| 9307 | 17 | repetitive | 2 | prompt-k2 | 2.271× | 2.147–2.341 | no |
| 9307 | 17 | structured | 2 | ngram-k2 | 2.239× | 2.200–2.294 | no |
| 9307 | 17 | structured | 2 | prompt-k2 | 1.704× | 1.651–1.860 | no |
| 9307 | 17 | unpredictable | 2 | ngram-k2 | 0.945× | 0.879–0.999 | yes |
| 9307 | 17 | unpredictable | 2 | prompt-k2 | 0.929× | 0.887–0.949 | yes |
| 9307 | 17 | repetitive | 8 | ngram-k8 | 0.835× | 0.828–0.850 | yes |
| 9307 | 17 | repetitive | 8 | prompt-k8 | 5.191× | 4.425–5.254 | no |
| 9307 | 17 | structured | 8 | ngram-k8 | 2.692× | 2.367–2.977 | no |
| 9307 | 17 | structured | 8 | prompt-k8 | 1.852× | 1.640–2.057 | no |
| 9307 | 17 | unpredictable | 8 | ngram-k8 | 0.900× | 0.851–0.949 | yes |
| 9307 | 17 | unpredictable | 8 | prompt-k8 | 0.863× | 0.821–0.910 | yes |
| 6203 | 29 | repetitive | 8 | ngram-k8 | 0.869× | 0.836–1.054 | yes |
| 6203 | 29 | repetitive | 8 | prompt-k8 | 5.235× | 4.934–5.346 | no |
| 6203 | 29 | structured | 8 | ngram-k8 | 2.491× | 2.375–2.613 | no |
| 6203 | 29 | structured | 8 | prompt-k8 | 2.050× | 1.943–2.208 | no |
| 6203 | 29 | unpredictable | 8 | ngram-k8 | 0.875× | 0.842–0.909 | yes |
| 6203 | 29 | unpredictable | 8 | prompt-k8 | 0.903× | 0.860–0.945 | yes |
| 6203 | 29 | repetitive | 2 | ngram-k2 | 0.953× | 0.856–1.041 | yes |
| 6203 | 29 | repetitive | 2 | prompt-k2 | 2.268× | 2.195–2.380 | no |
| 6203 | 29 | structured | 2 | ngram-k2 | 2.257× | 2.208–2.307 | no |
| 6203 | 29 | structured | 2 | prompt-k2 | 1.857× | 1.821–1.896 | no |
| 6203 | 29 | unpredictable | 2 | ngram-k2 | 0.927× | 0.908–0.960 | yes |
| 6203 | 29 | unpredictable | 2 | prompt-k2 | 0.945× | 0.890–0.991 | yes |
| 9307 | 29 | repetitive | 2 | ngram-k2 | 0.881× | 0.860–0.913 | yes |
| 9307 | 29 | repetitive | 2 | prompt-k2 | 2.232× | 2.195–2.302 | no |
| 9307 | 29 | structured | 2 | ngram-k2 | 2.231× | 2.187–2.364 | no |
| 9307 | 29 | structured | 2 | prompt-k2 | 1.885× | 1.702–2.077 | no |
| 9307 | 29 | unpredictable | 2 | ngram-k2 | 0.875× | 0.855–0.922 | yes |
| 9307 | 29 | unpredictable | 2 | prompt-k2 | 0.910× | 0.880–0.948 | yes |
| 9307 | 29 | repetitive | 8 | ngram-k8 | 0.845× | 0.811–0.863 | yes |
| 9307 | 29 | repetitive | 8 | prompt-k8 | 5.140× | 4.973–5.590 | no |
| 9307 | 29 | structured | 8 | ngram-k8 | 2.331× | 2.150–2.532 | no |
| 9307 | 29 | structured | 8 | prompt-k8 | 2.058× | 1.959–2.263 | no |
| 9307 | 29 | unpredictable | 8 | ngram-k8 | 0.890× | 0.824–0.929 | yes |
| 9307 | 29 | unpredictable | 8 | prompt-k8 | 0.867× | 0.834–0.932 | yes |
| 3101 | 29 | repetitive | 8 | ngram-k8 | 0.830× | 0.810–0.856 | yes |
| 3101 | 29 | repetitive | 8 | prompt-k8 | 5.218× | 3.555–5.293 | no |
| 3101 | 29 | structured | 8 | ngram-k8 | 2.359× | 2.274–2.461 | no |
| 3101 | 29 | structured | 8 | prompt-k8 | 2.032× | 1.635–2.158 | no |
| 3101 | 29 | unpredictable | 8 | ngram-k8 | 0.873× | 0.827–0.935 | yes |
| 3101 | 29 | unpredictable | 8 | prompt-k8 | 0.913× | 0.890–0.973 | yes |
| 3101 | 29 | repetitive | 2 | ngram-k2 | 0.873× | 0.853–0.902 | yes |
| 3101 | 29 | repetitive | 2 | prompt-k2 | 2.240× | 2.149–2.275 | no |
| 3101 | 29 | structured | 2 | ngram-k2 | 2.247× | 2.187–2.288 | no |
| 3101 | 29 | structured | 2 | prompt-k2 | 1.851× | 1.634–1.981 | no |
| 3101 | 29 | unpredictable | 2 | ngram-k2 | 0.891× | 0.870–0.947 | yes |
| 3101 | 29 | unpredictable | 2 | prompt-k2 | 0.946× | 0.868–0.981 | yes |
| 9307 | 43 | repetitive | 2 | ngram-k2 | 0.901× | 0.853–0.926 | yes |
| 9307 | 43 | repetitive | 2 | prompt-k2 | 2.374× | 2.268–2.415 | no |
| 9307 | 43 | structured | 2 | ngram-k2 | 2.284× | 2.188–2.319 | no |
| 9307 | 43 | structured | 2 | prompt-k2 | 1.772× | 1.636–1.985 | no |
| 9307 | 43 | unpredictable | 2 | ngram-k2 | 0.934× | 0.920–0.978 | yes |
| 9307 | 43 | unpredictable | 2 | prompt-k2 | 0.947× | 0.889–0.988 | yes |
| 9307 | 43 | repetitive | 8 | ngram-k8 | 0.833× | 0.809–0.852 | yes |
| 9307 | 43 | repetitive | 8 | prompt-k8 | 5.281× | 5.110–5.462 | no |
| 9307 | 43 | structured | 8 | ngram-k8 | 2.506× | 2.179–2.685 | no |
| 9307 | 43 | structured | 8 | prompt-k8 | 1.947× | 1.706–2.211 | no |
| 9307 | 43 | unpredictable | 8 | ngram-k8 | 0.887× | 0.858–0.917 | yes |
| 9307 | 43 | unpredictable | 8 | prompt-k8 | 0.887× | 0.837–0.931 | yes |
| 3101 | 43 | repetitive | 8 | ngram-k8 | 0.838× | 0.817–0.860 | yes |
| 3101 | 43 | repetitive | 8 | prompt-k8 | 5.344× | 4.922–5.457 | no |
| 3101 | 43 | structured | 8 | ngram-k8 | 2.584× | 2.462–2.682 | no |
| 3101 | 43 | structured | 8 | prompt-k8 | 1.881× | 1.730–2.125 | no |
| 3101 | 43 | unpredictable | 8 | ngram-k8 | 0.877× | 0.848–0.894 | yes |
| 3101 | 43 | unpredictable | 8 | prompt-k8 | 0.916× | 0.855–0.954 | yes |
| 3101 | 43 | repetitive | 2 | ngram-k2 | 0.895× | 0.878–0.915 | yes |
| 3101 | 43 | repetitive | 2 | prompt-k2 | 2.276× | 2.253–2.319 | no |
| 3101 | 43 | structured | 2 | ngram-k2 | 2.224× | 2.188–2.293 | no |
| 3101 | 43 | structured | 2 | prompt-k2 | 1.793× | 1.648–1.920 | no |
| 3101 | 43 | unpredictable | 2 | ngram-k2 | 0.935× | 0.899–0.956 | yes |
| 3101 | 43 | unpredictable | 2 | prompt-k2 | 0.915× | 0.905–0.940 | yes |
| 6203 | 43 | repetitive | 2 | ngram-k2 | 0.924× | 0.865–1.101 | yes |
| 6203 | 43 | repetitive | 2 | prompt-k2 | 2.201× | 2.110–2.266 | no |
| 6203 | 43 | structured | 2 | ngram-k2 | 2.298× | 2.231–2.344 | no |
| 6203 | 43 | structured | 2 | prompt-k2 | 1.664× | 1.614–1.962 | no |
| 6203 | 43 | unpredictable | 2 | ngram-k2 | 0.909× | 0.870–0.996 | yes |
| 6203 | 43 | unpredictable | 2 | prompt-k2 | 0.942× | 0.895–1.008 | yes |
| 6203 | 43 | repetitive | 8 | ngram-k8 | 0.860× | 0.817–1.038 | yes |
| 6203 | 43 | repetitive | 8 | prompt-k8 | 5.188× | 5.110–5.253 | no |
| 6203 | 43 | structured | 8 | ngram-k8 | 2.236× | 2.052–2.477 | no |
| 6203 | 43 | structured | 8 | prompt-k8 | 1.796× | 1.659–1.959 | no |
| 6203 | 43 | unpredictable | 8 | ngram-k8 | 0.896× | 0.860–0.959 | yes |
| 6203 | 43 | unpredictable | 8 | prompt-k8 | 0.920× | 0.894–0.960 | yes |

## Target accuracy

Accuracy uses all held-out inputs, separately from decoding equivalence. Exact continuation includes EOS; missing output tokens count as wrong. Unpredictable continuations cannot be inferred from the prompt; fast repetitive outputs can still be inaccurate.

| Dataset | Train seed | Task | Teacher-forced | Payload only | Free-running | Exact continuation |
|---:|---:|---|---:|---:|---:|---:|
| 3101 | 17 | repetitive | 98.81% | 98.80% | 98.19% | 78.91% |
| 3101 | 17 | structured | 71.62% | 70.44% | 43.19% | 0.00% |
| 3101 | 17 | unpredictable | 9.47% | 5.70% | 10.09% | 0.00% |
| 6203 | 17 | repetitive | 97.91% | 97.82% | 97.31% | 71.88% |
| 6203 | 17 | structured | 71.66% | 70.48% | 42.53% | 0.00% |
| 6203 | 17 | unpredictable | 9.41% | 5.63% | 9.94% | 0.00% |
| 9307 | 17 | repetitive | 98.34% | 98.31% | 97.50% | 70.31% |
| 9307 | 17 | structured | 72.84% | 71.71% | 44.88% | 0.00% |
| 9307 | 17 | unpredictable | 10.72% | 7.00% | 10.56% | 0.00% |
| 6203 | 29 | repetitive | 97.88% | 97.82% | 97.62% | 75.78% |
| 6203 | 29 | structured | 72.56% | 71.42% | 44.94% | 0.00% |
| 6203 | 29 | unpredictable | 9.38% | 5.60% | 10.03% | 0.00% |
| 9307 | 29 | repetitive | 98.62% | 98.57% | 98.28% | 79.69% |
| 9307 | 29 | structured | 72.12% | 70.96% | 44.44% | 0.00% |
| 9307 | 29 | unpredictable | 9.53% | 5.76% | 9.88% | 0.00% |
| 3101 | 29 | repetitive | 98.19% | 98.11% | 98.03% | 74.22% |
| 3101 | 29 | structured | 71.75% | 70.57% | 42.41% | 0.00% |
| 3101 | 29 | unpredictable | 10.16% | 6.41% | 10.38% | 0.00% |
| 9307 | 43 | repetitive | 98.91% | 98.86% | 98.53% | 78.91% |
| 9307 | 43 | structured | 71.38% | 70.18% | 43.31% | 0.00% |
| 9307 | 43 | unpredictable | 10.00% | 6.25% | 10.72% | 0.00% |
| 3101 | 43 | repetitive | 99.09% | 99.06% | 98.81% | 82.81% |
| 3101 | 43 | structured | 72.22% | 71.06% | 43.44% | 0.00% |
| 3101 | 43 | unpredictable | 9.72% | 5.96% | 10.22% | 0.00% |
| 6203 | 43 | repetitive | 99.41% | 99.38% | 98.97% | 88.28% |
| 6203 | 43 | structured | 71.78% | 70.61% | 44.38% | 0.00% |
| 6203 | 43 | unpredictable | 9.75% | 5.99% | 10.53% | 0.00% |

## Artifacts and limits

protocol.json freezes settings, dataset hashes, timing IDs, analysis and source hashes. Each dataset shard stores exact train/evaluation inputs. Each target directory stores its checkpoint, config, training trace, hardware/dependencies, task correctness shards, raw timing shards, and report.json with acceptance rates, target calls, draft overhead and all unrounded estimates. Gzip JSONL shards use deterministic compression and can be read with Python's gzip module.

Three synthetic datasets and three paired training seeds, one tiny FP32 CPU model size, shared host. Conditional case bootstrap excludes training/dataset and host uncertainty. No natural-language, large-model, GPU or sampling generalization.

See the project README and PROTOCOL.md for reproduction and existing work. This is replication of established greedy speculative verification and lookup approaches, not a novel algorithm.
