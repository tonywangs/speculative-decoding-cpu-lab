# Measured CPU results

Recorded at 2026-09-18T19:11:44.926371+00:00.

CPU: DO-Regular; PyTorch 2.8.0+cpu; 1 intra-op thread(s), FP32.

Speedup is the median of greedy/method latency ratios paired by case and repetition. Values below 1 are slowdowns. Timings include prefill, drafting, verification and Python overhead; training, n-gram fitting, correctness audits and warmups are excluded.

| Task | Method | Median ms | Paired speedup | Acceptance | Mean calls | Draft ms |
|---|---|---:|---:|---:|---:|---:|
| repetitive | greedy | 45.154 | 1.000× | — | 26.00 | 0.000 |
| repetitive | prompt-k1 | 28.962 | 1.576× | 99.3% | 14.08 | 0.178 |
| repetitive | prompt-k2 | 20.728 | 2.184× | 98.4% | 10.17 | 0.124 |
| repetitive | prompt-k4 | 13.424 | 3.456× | 97.9% | 6.25 | 0.077 |
| repetitive | prompt-k8 | 9.194 | 4.945× | 98.9% | 4.17 | 0.047 |
| repetitive | ngram-k1 | 51.986 | 0.860× | 2.1% | 25.50 | 0.445 |
| repetitive | ngram-k2 | 53.811 | 0.855× | 1.1% | 25.50 | 0.537 |
| repetitive | ngram-k4 | 53.948 | 0.841× | 0.6% | 25.50 | 0.643 |
| repetitive | ngram-k8 | 56.767 | 0.808× | 0.3% | 25.50 | 0.825 |
| structured | greedy | 45.957 | 1.000× | — | 26.00 | 0.000 |
| structured | prompt-k1 | 38.389 | 1.188× | 53.8% | 18.83 | 0.665 |
| structured | prompt-k2 | 27.598 | 1.672× | 71.4% | 13.50 | 0.595 |
| structured | prompt-k4 | 28.133 | 1.630× | 41.0% | 13.17 | 0.579 |
| structured | prompt-k8 | 26.800 | 1.714× | 25.9% | 12.67 | 0.554 |
| structured | ngram-k1 | 35.251 | 1.299× | 56.5% | 17.33 | 0.274 |
| structured | ngram-k2 | 21.145 | 2.161× | 88.9% | 10.00 | 0.192 |
| structured | ngram-k4 | 21.138 | 2.139× | 48.6% | 9.83 | 0.222 |
| structured | ngram-k8 | 20.696 | 2.180× | 28.8% | 9.25 | 0.271 |
| unpredictable | greedy | 45.909 | 1.000× | — | 26.00 | 0.000 |
| unpredictable | prompt-k1 | 50.851 | 0.901× | 7.1% | 24.58 | 1.232 |
| unpredictable | prompt-k2 | 51.160 | 0.914× | 4.1% | 24.42 | 1.252 |
| unpredictable | prompt-k4 | 50.516 | 0.886× | 2.2% | 24.42 | 1.236 |
| unpredictable | prompt-k8 | 54.302 | 0.862× | 1.2% | 24.42 | 1.247 |
| unpredictable | ngram-k1 | 50.755 | 0.897× | 4.7% | 24.92 | 0.431 |
| unpredictable | ngram-k2 | 51.100 | 0.893× | 2.5% | 24.92 | 0.511 |
| unpredictable | ngram-k4 | 52.194 | 0.871× | 1.3% | 24.92 | 0.621 |
| unpredictable | ngram-k8 | 54.417 | 0.830× | 0.8% | 24.92 | 0.810 |

| Task | Teacher-forced accuracy | Free-running accuracy | Exact continuation |
|---|---:|---:|---:|
| repetitive | 99.7% | 99.3% | 91.7% |
| structured | 70.3% | 41.3% | 0.0% |
| unpredictable | 10.7% | 7.0% | 0.0% |

Accuracy is against held-out synthetic continuations, including EOS. See report.json for payload-only teacher-forced accuracy. Missing output tokens count as incorrect.

Every measured output matched the uncached greedy reference. This is empirical token agreement, not bitwise equality of floating-point logits or a guarantee on other hardware.

These are small synthetic workloads and one trained target. Random task continuations are not predictable from the prompt. Target repetition can make them easy to draft despite low task accuracy. Repeated timing samples share prompts and weights and are not independent model trials. Shared-host load can affect latency. No claims about natural language or large-model speedups.
