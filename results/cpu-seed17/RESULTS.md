# Measured CPU results

Recorded at 2026-09-18T06:58:45.144366+00:00.

CPU: DO-Regular; PyTorch 2.8.0+cpu; 1 intra-op thread(s), FP32.

Speedup is the median of greedy/method latency ratios paired by case and repetition. Values below 1 are slowdowns. Timings include prefill, drafting, verification and Python overhead; training, n-gram fitting, correctness audits and warmups are excluded.

| Task | Method | Median ms | Paired speedup | Acceptance | Mean calls | Draft ms |
|---|---|---:|---:|---:|---:|---:|
| repetitive | greedy | 34.789 | 1.000× | — | 26.00 | 0.000 |
| repetitive | prompt-k1 | 22.405 | 1.647× | 97.3% | 14.08 | 0.155 |
| repetitive | prompt-k2 | 15.857 | 2.204× | 95.5% | 10.25 | 0.113 |
| repetitive | prompt-k4 | 10.057 | 3.496× | 92.9% | 6.42 | 0.067 |
| repetitive | prompt-k8 | 7.113 | 5.149× | 90.6% | 4.42 | 0.042 |
| repetitive | ngram-k1 | 35.073 | 1.012× | 10.7% | 23.67 | 0.317 |
| repetitive | ngram-k2 | 36.092 | 0.931× | 7.7% | 22.92 | 0.405 |
| repetitive | ngram-k4 | 39.502 | 0.888× | 4.5% | 22.67 | 0.518 |
| repetitive | ngram-k8 | 41.569 | 0.868× | 2.8% | 22.50 | 0.662 |
| structured | greedy | 36.406 | 1.000× | — | 26.00 | 0.000 |
| structured | prompt-k1 | 28.152 | 1.282× | 52.7% | 18.67 | 0.494 |
| structured | prompt-k2 | 20.807 | 1.804× | 72.0% | 13.17 | 0.425 |
| structured | prompt-k4 | 21.367 | 1.748× | 39.8% | 13.08 | 0.424 |
| structured | prompt-k8 | 19.597 | 1.895× | 24.1% | 12.50 | 0.408 |
| structured | ngram-k1 | 26.156 | 1.366× | 59.9% | 16.92 | 0.206 |
| structured | ngram-k2 | 15.893 | 2.300× | 88.9% | 10.00 | 0.152 |
| structured | ngram-k4 | 15.844 | 2.327× | 50.3% | 9.75 | 0.179 |
| structured | ngram-k8 | 14.672 | 2.425× | 31.4% | 8.75 | 0.217 |
| unpredictable | greedy | 32.635 | 1.000× | — | 26.00 | 0.000 |
| unpredictable | prompt-k1 | 37.489 | 0.905× | 9.1% | 24.17 | 1.031 |
| unpredictable | prompt-k2 | 35.774 | 0.937× | 5.4% | 23.92 | 0.927 |
| unpredictable | prompt-k4 | 38.322 | 0.898× | 2.9% | 23.92 | 1.053 |
| unpredictable | prompt-k8 | 39.416 | 0.882× | 1.6% | 23.92 | 1.014 |
| unpredictable | ngram-k1 | 37.008 | 0.916× | 7.0% | 24.42 | 0.330 |
| unpredictable | ngram-k2 | 37.275 | 0.924× | 3.8% | 24.33 | 0.414 |
| unpredictable | ngram-k4 | 38.427 | 0.896× | 2.1% | 24.33 | 0.505 |
| unpredictable | ngram-k8 | 38.813 | 0.861× | 1.2% | 24.33 | 0.655 |

| Task | Teacher-forced accuracy | Free-running accuracy | Exact continuation |
|---|---:|---:|---:|
| repetitive | 98.7% | 98.7% | 75.0% |
| structured | 70.7% | 42.7% | 0.0% |
| unpredictable | 11.0% | 11.7% | 0.0% |

Accuracy is against held-out synthetic continuations, including EOS. See report.json for payload-only teacher-forced accuracy. Missing output tokens count as incorrect.

Every measured output matched the uncached greedy reference. This is empirical token agreement, not bitwise equality of floating-point logits or a guarantee on other hardware.

These are small synthetic workloads and one trained target. Random task continuations are not predictable from the prompt. Target repetition can make them easy to draft despite low task accuracy. Repeated timing samples share prompts and weights and are not independent model trials. Shared-host load can affect latency. No claims about natural language or large-model speedups.
