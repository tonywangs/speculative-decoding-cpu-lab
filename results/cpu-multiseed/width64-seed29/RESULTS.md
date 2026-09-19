# Measured CPU results

Recorded at 2026-09-18T18:59:44.137338+00:00.

CPU: DO-Regular; PyTorch 2.8.0+cpu; 1 intra-op thread(s), FP32.

Speedup is the median of greedy/method latency ratios paired by case and repetition. Values below 1 are slowdowns. Timings include prefill, drafting, verification and Python overhead; training, n-gram fitting, correctness audits and warmups are excluded.

| Task | Method | Median ms | Paired speedup | Acceptance | Mean calls | Draft ms |
|---|---|---:|---:|---:|---:|---:|
| repetitive | greedy | 37.842 | 1.000× | — | 26.00 | 0.000 |
| repetitive | prompt-k1 | 22.973 | 1.584× | 100.0% | 14.00 | 0.158 |
| repetitive | prompt-k2 | 16.550 | 2.173× | 100.0% | 10.00 | 0.110 |
| repetitive | prompt-k4 | 10.498 | 3.514× | 99.2% | 6.17 | 0.070 |
| repetitive | prompt-k8 | 7.415 | 5.030× | 100.0% | 4.08 | 0.044 |
| repetitive | ngram-k1 | 41.824 | 0.871× | 2.5% | 25.42 | 0.398 |
| repetitive | ngram-k2 | 42.011 | 0.867× | 1.3% | 25.42 | 0.476 |
| repetitive | ngram-k4 | 43.856 | 0.836× | 0.7% | 25.42 | 0.581 |
| repetitive | ngram-k8 | 44.641 | 0.813× | 0.4% | 25.42 | 0.764 |
| structured | greedy | 37.513 | 1.000× | — | 26.00 | 0.000 |
| structured | prompt-k1 | 30.428 | 1.256× | 58.3% | 18.08 | 0.486 |
| structured | prompt-k2 | 20.886 | 1.760× | 75.5% | 12.67 | 0.398 |
| structured | prompt-k4 | 20.979 | 1.826× | 43.1% | 12.42 | 0.400 |
| structured | prompt-k8 | 19.512 | 1.881× | 28.1% | 11.67 | 0.388 |
| structured | ngram-k1 | 27.508 | 1.324× | 58.2% | 17.17 | 0.231 |
| structured | ngram-k2 | 17.590 | 2.181× | 88.9% | 10.00 | 0.180 |
| structured | ngram-k4 | 17.334 | 2.158× | 49.5% | 9.75 | 0.204 |
| structured | ngram-k8 | 16.362 | 2.291× | 30.5% | 9.00 | 0.236 |
| unpredictable | greedy | 36.180 | 1.000× | — | 26.00 | 0.000 |
| unpredictable | prompt-k1 | 42.479 | 0.885× | 9.2% | 24.17 | 1.116 |
| unpredictable | prompt-k2 | 42.187 | 0.879× | 5.0% | 24.08 | 1.121 |
| unpredictable | prompt-k4 | 42.327 | 0.871× | 2.6% | 24.08 | 1.131 |
| unpredictable | prompt-k8 | 42.631 | 0.841× | 1.5% | 24.08 | 1.127 |
| unpredictable | ngram-k1 | 42.710 | 0.883× | 4.0% | 25.08 | 0.390 |
| unpredictable | ngram-k2 | 43.325 | 0.871× | 2.1% | 25.08 | 0.483 |
| unpredictable | ngram-k4 | 43.151 | 0.863× | 1.1% | 25.08 | 0.577 |
| unpredictable | ngram-k8 | 45.069 | 0.815× | 0.6% | 25.08 | 0.755 |

| Task | Teacher-forced accuracy | Free-running accuracy | Exact continuation |
|---|---:|---:|---:|
| repetitive | 99.7% | 99.7% | 91.7% |
| structured | 72.3% | 40.7% | 0.0% |
| unpredictable | 11.0% | 9.7% | 0.0% |

Accuracy is against held-out synthetic continuations, including EOS. See report.json for payload-only teacher-forced accuracy. Missing output tokens count as incorrect.

Every measured output matched the uncached greedy reference. This is empirical token agreement, not bitwise equality of floating-point logits or a guarantee on other hardware.

These are small synthetic workloads and one trained target. Random task continuations are not predictable from the prompt. Target repetition can make them easy to draft despite low task accuracy. Repeated timing samples share prompts and weights and are not independent model trials. Shared-host load can affect latency. No claims about natural language or large-model speedups.
