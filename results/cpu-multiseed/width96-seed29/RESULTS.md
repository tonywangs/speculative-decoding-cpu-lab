# Measured CPU results

Recorded at 2026-09-18T19:04:01.473401+00:00.

CPU: DO-Regular; PyTorch 2.8.0+cpu; 1 intra-op thread(s), FP32.

Speedup is the median of greedy/method latency ratios paired by case and repetition. Values below 1 are slowdowns. Timings include prefill, drafting, verification and Python overhead; training, n-gram fitting, correctness audits and warmups are excluded.

| Task | Method | Median ms | Paired speedup | Acceptance | Mean calls | Draft ms |
|---|---|---:|---:|---:|---:|---:|
| repetitive | greedy | 44.700 | 1.000× | — | 26.00 | 0.000 |
| repetitive | prompt-k1 | 27.449 | 1.606× | 99.3% | 14.00 | 0.175 |
| repetitive | prompt-k2 | 20.200 | 2.199× | 97.4% | 10.17 | 0.123 |
| repetitive | prompt-k4 | 12.491 | 3.502× | 97.1% | 6.25 | 0.075 |
| repetitive | prompt-k8 | 8.824 | 4.979× | 94.9% | 4.25 | 0.046 |
| repetitive | ngram-k1 | 50.878 | 0.873× | 2.1% | 25.50 | 0.432 |
| repetitive | ngram-k2 | 50.467 | 0.874× | 1.1% | 25.50 | 0.526 |
| repetitive | ngram-k4 | 51.836 | 0.841× | 0.6% | 25.50 | 0.628 |
| repetitive | ngram-k8 | 54.244 | 0.805× | 0.3% | 25.50 | 0.810 |
| structured | greedy | 43.770 | 1.000× | — | 26.00 | 0.000 |
| structured | prompt-k1 | 35.829 | 1.207× | 55.5% | 18.42 | 0.584 |
| structured | prompt-k2 | 24.431 | 1.774× | 75.2% | 12.83 | 0.464 |
| structured | prompt-k4 | 24.922 | 1.742× | 44.4% | 12.42 | 0.463 |
| structured | prompt-k8 | 23.867 | 1.806× | 27.9% | 12.00 | 0.448 |
| structured | ngram-k1 | 31.586 | 1.343× | 63.6% | 16.67 | 0.246 |
| structured | ngram-k2 | 19.930 | 2.146× | 88.9% | 10.00 | 0.183 |
| structured | ngram-k4 | 19.171 | 2.247× | 52.5% | 9.42 | 0.209 |
| structured | ngram-k8 | 17.528 | 2.479× | 33.9% | 8.42 | 0.231 |
| unpredictable | greedy | 45.274 | 1.000× | — | 26.00 | 0.000 |
| unpredictable | prompt-k1 | 49.136 | 0.906× | 8.0% | 24.42 | 1.185 |
| unpredictable | prompt-k2 | 49.283 | 0.909× | 4.1% | 24.42 | 1.214 |
| unpredictable | prompt-k4 | 49.935 | 0.889× | 2.2% | 24.42 | 1.243 |
| unpredictable | prompt-k8 | 50.868 | 0.867× | 1.2% | 24.42 | 1.232 |
| unpredictable | ngram-k1 | 48.072 | 0.931× | 6.7% | 24.50 | 0.410 |
| unpredictable | ngram-k2 | 49.687 | 0.909× | 3.9% | 24.33 | 0.502 |
| unpredictable | ngram-k4 | 48.069 | 0.893× | 2.1% | 24.33 | 0.587 |
| unpredictable | ngram-k8 | 51.629 | 0.843× | 1.2% | 24.33 | 0.767 |

| Task | Teacher-forced accuracy | Free-running accuracy | Exact continuation |
|---|---:|---:|---:|
| repetitive | 99.3% | 99.3% | 83.3% |
| structured | 72.7% | 45.3% | 0.0% |
| unpredictable | 9.3% | 8.7% | 0.0% |

Accuracy is against held-out synthetic continuations, including EOS. See report.json for payload-only teacher-forced accuracy. Missing output tokens count as incorrect.

Every measured output matched the uncached greedy reference. This is empirical token agreement, not bitwise equality of floating-point logits or a guarantee on other hardware.

These are small synthetic workloads and one trained target. Random task continuations are not predictable from the prompt. Target repetition can make them easy to draft despite low task accuracy. Repeated timing samples share prompts and weights and are not independent model trials. Shared-host load can affect latency. No claims about natural language or large-model speedups.
