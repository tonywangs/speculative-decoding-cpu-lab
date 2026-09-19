# Measured CPU results

Recorded at 2026-09-18T19:07:28.116434+00:00.

CPU: DO-Regular; PyTorch 2.8.0+cpu; 1 intra-op thread(s), FP32.

Speedup is the median of greedy/method latency ratios paired by case and repetition. Values below 1 are slowdowns. Timings include prefill, drafting, verification and Python overhead; training, n-gram fitting, correctness audits and warmups are excluded.

| Task | Method | Median ms | Paired speedup | Acceptance | Mean calls | Draft ms |
|---|---|---:|---:|---:|---:|---:|
| repetitive | greedy | 42.233 | 1.000× | — | 26.00 | 0.000 |
| repetitive | prompt-k1 | 25.981 | 1.638× | 99.3% | 14.00 | 0.172 |
| repetitive | prompt-k2 | 18.818 | 2.208× | 99.0% | 10.00 | 0.120 |
| repetitive | prompt-k4 | 11.566 | 3.628× | 97.5% | 6.17 | 0.073 |
| repetitive | prompt-k8 | 8.137 | 5.196× | 97.8% | 4.08 | 0.046 |
| repetitive | ngram-k1 | 47.041 | 0.894× | 1.8% | 25.58 | 0.429 |
| repetitive | ngram-k2 | 48.251 | 0.871× | 0.9% | 25.58 | 0.527 |
| repetitive | ngram-k4 | 49.626 | 0.852× | 0.5% | 25.58 | 0.644 |
| repetitive | ngram-k8 | 50.679 | 0.831× | 0.3% | 25.58 | 0.810 |
| structured | greedy | 41.911 | 1.000× | — | 26.00 | 0.000 |
| structured | prompt-k1 | 31.537 | 1.313× | 60.7% | 17.75 | 0.493 |
| structured | prompt-k2 | 21.763 | 1.881× | 75.5% | 12.42 | 0.406 |
| structured | prompt-k4 | 20.677 | 1.989× | 48.6% | 11.58 | 0.387 |
| structured | prompt-k8 | 19.709 | 2.037× | 33.0% | 10.83 | 0.391 |
| structured | ngram-k1 | 31.192 | 1.342× | 57.1% | 17.25 | 0.257 |
| structured | ngram-k2 | 19.231 | 2.202× | 88.4% | 10.08 | 0.191 |
| structured | ngram-k4 | 19.100 | 2.172× | 48.4% | 9.92 | 0.221 |
| structured | ngram-k8 | 17.997 | 2.277× | 28.9% | 9.33 | 0.263 |
| unpredictable | greedy | 43.083 | 1.000× | — | 26.00 | 0.000 |
| unpredictable | prompt-k1 | 45.040 | 0.940× | 10.1% | 24.00 | 1.152 |
| unpredictable | prompt-k2 | 45.098 | 0.961× | 5.7% | 23.83 | 1.179 |
| unpredictable | prompt-k4 | 45.026 | 0.928× | 3.2% | 23.75 | 1.171 |
| unpredictable | prompt-k8 | 46.620 | 0.912× | 1.8% | 23.75 | 1.155 |
| unpredictable | ngram-k1 | 44.701 | 0.944× | 5.9% | 24.67 | 0.410 |
| unpredictable | ngram-k2 | 45.605 | 0.921× | 3.1% | 24.67 | 0.501 |
| unpredictable | ngram-k4 | 46.422 | 0.915× | 1.6% | 24.67 | 0.605 |
| unpredictable | ngram-k8 | 46.795 | 0.900× | 0.9% | 24.67 | 0.756 |

| Task | Teacher-forced accuracy | Free-running accuracy | Exact continuation |
|---|---:|---:|---:|
| repetitive | 99.7% | 99.7% | 91.7% |
| structured | 72.7% | 45.3% | 0.0% |
| unpredictable | 9.0% | 9.7% | 0.0% |

Accuracy is against held-out synthetic continuations, including EOS. See report.json for payload-only teacher-forced accuracy. Missing output tokens count as incorrect.

Every measured output matched the uncached greedy reference. This is empirical token agreement, not bitwise equality of floating-point logits or a guarantee on other hardware.

These are small synthetic workloads and one trained target. Random task continuations are not predictable from the prompt. Target repetition can make them easy to draft despite low task accuracy. Repeated timing samples share prompts and weights and are not independent model trials. Shared-host load can affect latency. No claims about natural language or large-model speedups.
