# Measured CPU results

Recorded at 2026-09-18T18:56:29.045208+00:00.

CPU: DO-Regular; PyTorch 2.8.0+cpu; 1 intra-op thread(s), FP32.

Speedup is the median of greedy/method latency ratios paired by case and repetition. Values below 1 are slowdowns. Timings include prefill, drafting, verification and Python overhead; training, n-gram fitting, correctness audits and warmups are excluded.

| Task | Method | Median ms | Paired speedup | Acceptance | Mean calls | Draft ms |
|---|---|---:|---:|---:|---:|---:|
| repetitive | greedy | 43.089 | 1.000× | — | 26.00 | 0.000 |
| repetitive | prompt-k1 | 26.996 | 1.590× | 100.0% | 14.00 | 0.171 |
| repetitive | prompt-k2 | 19.586 | 2.181× | 99.0% | 10.08 | 0.117 |
| repetitive | prompt-k4 | 12.418 | 3.451× | 98.8% | 6.17 | 0.072 |
| repetitive | prompt-k8 | 8.733 | 4.961× | 98.9% | 4.17 | 0.046 |
| repetitive | ngram-k1 | 48.309 | 0.870× | 2.1% | 25.50 | 0.416 |
| repetitive | ngram-k2 | 50.049 | 0.862× | 1.1% | 25.50 | 0.505 |
| repetitive | ngram-k4 | 51.163 | 0.837× | 0.6% | 25.50 | 0.608 |
| repetitive | ngram-k8 | 54.332 | 0.807× | 0.3% | 25.50 | 0.805 |
| structured | greedy | 43.087 | 1.000× | — | 26.00 | 0.000 |
| structured | prompt-k1 | 35.284 | 1.215× | 57.9% | 18.33 | 0.622 |
| structured | prompt-k2 | 24.925 | 1.707× | 73.3% | 13.17 | 0.528 |
| structured | prompt-k4 | 24.868 | 1.745× | 43.7% | 12.67 | 0.508 |
| structured | prompt-k8 | 23.756 | 1.789× | 28.9% | 12.00 | 0.519 |
| structured | ngram-k1 | 33.291 | 1.287× | 53.4% | 17.58 | 0.258 |
| structured | ngram-k2 | 19.569 | 2.234× | 88.9% | 10.00 | 0.184 |
| structured | ngram-k4 | 20.130 | 2.145× | 48.2% | 9.92 | 0.213 |
| structured | ngram-k8 | 20.435 | 2.168× | 27.1% | 9.58 | 0.276 |
| unpredictable | greedy | 44.514 | 1.000× | — | 26.00 | 0.000 |
| unpredictable | prompt-k1 | 46.701 | 0.919× | 7.4% | 24.50 | 1.193 |
| unpredictable | prompt-k2 | 48.249 | 0.909× | 4.1% | 24.42 | 1.224 |
| unpredictable | prompt-k4 | 48.667 | 0.888× | 2.2% | 24.42 | 1.231 |
| unpredictable | prompt-k8 | 50.318 | 0.866× | 1.2% | 24.42 | 1.235 |
| unpredictable | ngram-k1 | 47.713 | 0.910× | 4.7% | 24.92 | 0.416 |
| unpredictable | ngram-k2 | 48.565 | 0.908× | 2.8% | 24.75 | 0.503 |
| unpredictable | ngram-k4 | 49.595 | 0.887× | 1.5% | 24.75 | 0.602 |
| unpredictable | ngram-k8 | 52.278 | 0.836× | 0.9% | 24.75 | 0.773 |

| Task | Teacher-forced accuracy | Free-running accuracy | Exact continuation |
|---|---:|---:|---:|
| repetitive | 99.7% | 99.7% | 91.7% |
| structured | 74.3% | 47.3% | 0.0% |
| unpredictable | 11.3% | 10.0% | 0.0% |

Accuracy is against held-out synthetic continuations, including EOS. See report.json for payload-only teacher-forced accuracy. Missing output tokens count as incorrect.

Every measured output matched the uncached greedy reference. This is empirical token agreement, not bitwise equality of floating-point logits or a guarantee on other hardware.

These are small synthetic workloads and one trained target. Random task continuations are not predictable from the prompt. Target repetition can make them easy to draft despite low task accuracy. Repeated timing samples share prompts and weights and are not independent model trials. Shared-host load can affect latency. No claims about natural language or large-model speedups.
