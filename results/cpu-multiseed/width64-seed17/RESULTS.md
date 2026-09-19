# Measured CPU results

Recorded at 2026-09-18T18:52:23.715846+00:00.

CPU: DO-Regular; PyTorch 2.8.0+cpu; 1 intra-op thread(s), FP32.

Speedup is the median of greedy/method latency ratios paired by case and repetition. Values below 1 are slowdowns. Timings include prefill, drafting, verification and Python overhead; training, n-gram fitting, correctness audits and warmups are excluded.

| Task | Method | Median ms | Paired speedup | Acceptance | Mean calls | Draft ms |
|---|---|---:|---:|---:|---:|---:|
| repetitive | greedy | 42.311 | 1.000× | — | 26.00 | 0.000 |
| repetitive | prompt-k1 | 26.665 | 1.552× | 94.6% | 14.42 | 0.190 |
| repetitive | prompt-k2 | 19.873 | 2.084× | 90.1% | 10.75 | 0.131 |
| repetitive | prompt-k4 | 12.683 | 3.338× | 89.4% | 7.08 | 0.082 |
| repetitive | prompt-k8 | 8.899 | 4.815× | 79.0% | 5.33 | 0.051 |
| repetitive | ngram-k1 | 47.966 | 0.890× | 2.5% | 25.42 | 0.430 |
| repetitive | ngram-k2 | 48.426 | 0.865× | 1.5% | 25.33 | 0.524 |
| repetitive | ngram-k4 | 49.460 | 0.862× | 0.8% | 25.33 | 0.616 |
| repetitive | ngram-k8 | 50.759 | 0.834× | 0.5% | 25.25 | 0.813 |
| structured | greedy | 41.307 | 1.000× | — | 26.00 | 0.000 |
| structured | prompt-k1 | 33.788 | 1.238× | 60.5% | 17.83 | 0.510 |
| structured | prompt-k2 | 22.705 | 1.786× | 76.4% | 12.50 | 0.434 |
| structured | prompt-k4 | 22.922 | 1.807× | 45.0% | 12.17 | 0.438 |
| structured | prompt-k8 | 20.635 | 1.981× | 30.1% | 11.25 | 0.415 |
| structured | ngram-k1 | 32.040 | 1.292× | 59.1% | 17.08 | 0.258 |
| structured | ngram-k2 | 19.478 | 2.215× | 88.9% | 10.00 | 0.189 |
| structured | ngram-k4 | 18.952 | 2.200× | 50.0% | 9.67 | 0.218 |
| structured | ngram-k8 | 18.059 | 2.345× | 30.9% | 9.00 | 0.251 |
| unpredictable | greedy | 42.521 | 1.000× | — | 26.00 | 0.000 |
| unpredictable | prompt-k1 | 45.099 | 0.920× | 8.5% | 24.33 | 1.191 |
| unpredictable | prompt-k2 | 47.165 | 0.921× | 4.8% | 24.17 | 1.237 |
| unpredictable | prompt-k4 | 46.007 | 0.901× | 2.6% | 24.17 | 1.183 |
| unpredictable | prompt-k8 | 46.959 | 0.879× | 1.5% | 24.17 | 1.225 |
| unpredictable | ngram-k1 | 44.716 | 0.937× | 6.6% | 24.50 | 0.406 |
| unpredictable | ngram-k2 | 44.990 | 0.905× | 3.6% | 24.42 | 0.490 |
| unpredictable | ngram-k4 | 47.606 | 0.891× | 1.9% | 24.42 | 0.598 |
| unpredictable | ngram-k8 | 48.718 | 0.867× | 1.1% | 24.42 | 0.767 |

| Task | Teacher-forced accuracy | Free-running accuracy | Exact continuation |
|---|---:|---:|---:|
| repetitive | 97.3% | 95.3% | 66.7% |
| structured | 72.7% | 44.0% | 0.0% |
| unpredictable | 10.3% | 9.0% | 0.0% |

Accuracy is against held-out synthetic continuations, including EOS. See report.json for payload-only teacher-forced accuracy. Missing output tokens count as incorrect.

Every measured output matched the uncached greedy reference. This is empirical token agreement, not bitwise equality of floating-point logits or a guarantee on other hardware.

These are small synthetic workloads and one trained target. Random task continuations are not predictable from the prompt. Target repetition can make them easy to draft despite low task accuracy. Repeated timing samples share prompts and weights and are not independent model trials. Shared-host load can affect latency. No claims about natural language or large-model speedups.
