# Real-text CPU draft comparison

Seed 29; 36 held-out inputs; 1764 timed outputs, all exactly equal to uncached greedy.

CPU: DO-Regular; PyTorch 2.8.0+cpu; 1 intra-op thread(s), one inter-op thread, FP32.

Primary speedup: median across cases of median paired greedy/method ratios. 95% intervals resample whole cases (2,000 draws, seed 314159), conditional on this target and text. Seven repetitions balance every method across every timing position. Values below 1 are slowdowns.

| Task | Method | Median ms | Speedup | Conditional 95% | Acceptance | Draft ms |
|---|---|---:|---:|---|---:|---:|
| dialogue | greedy | 135.782 | 1.000× | [1.000, 1.000] | — | 0.000 |
| dialogue | prompt-k2 | 130.208 | 1.044× | [0.989, 1.087] | 16.3% | 6.266 |
| dialogue | prompt-k4 | 127.311 | 1.096× | [0.964, 1.165] | 9.6% | 6.020 |
| dialogue | ngram-k2 | 96.707 | 1.410× | [1.378, 1.475] | 34.2% | 1.061 |
| dialogue | ngram-k4 | 90.281 | 1.494× | [1.450, 1.591] | 21.9% | 1.165 |
| dialogue | gru-k2 | 133.925 | 1.025× | [0.954, 1.075] | 37.8% | 39.545 |
| dialogue | gru-k4 | 149.330 | 0.892× | [0.818, 1.042] | 24.3% | 56.091 |
| prose | greedy | 131.580 | 1.000× | [1.000, 1.000] | — | 0.000 |
| prose | prompt-k2 | 118.536 | 1.104× | [1.061, 1.199] | 21.2% | 5.494 |
| prose | prompt-k4 | 119.713 | 1.135× | [1.070, 1.168] | 12.0% | 5.410 |
| prose | ngram-k2 | 86.408 | 1.521× | [1.442, 1.654] | 43.0% | 0.949 |
| prose | ngram-k4 | 79.619 | 1.723× | [1.613, 1.768] | 28.4% | 1.006 |
| prose | gru-k2 | 124.459 | 1.058× | [1.003, 1.134] | 45.9% | 36.712 |
| prose | gru-k4 | 129.663 | 0.993× | [0.951, 1.119] | 30.6% | 49.198 |
| python | greedy | 137.547 | 1.000× | [1.000, 1.000] | — | 0.000 |
| python | prompt-k2 | 91.950 | 1.483× | [1.291, 1.650] | 41.9% | 3.634 |
| python | prompt-k4 | 82.331 | 1.611× | [1.387, 2.026] | 30.1% | 3.506 |
| python | ngram-k2 | 70.464 | 1.876× | [1.715, 2.061] | 61.6% | 0.788 |
| python | ngram-k4 | 57.958 | 2.180× | [1.978, 2.680] | 47.6% | 0.757 |
| python | gru-k2 | 113.468 | 1.168× | [1.091, 1.300] | 57.1% | 34.009 |
| python | gru-k4 | 109.304 | 1.169× | [1.089, 1.363] | 42.1% | 42.400 |

| Task | Target next-byte accuracy | Free-running accuracy | Exact continuation |
|---|---:|---:|---:|
| dialogue | 36.1% | 8.3% | 0.0% |
| prose | 46.1% | 9.6% | 0.0% |
| python | 63.4% | 21.5% | 0.0% |

Target training: 585.103 s; 289,440 parameters. GRU training: 149.320 s; 23,301 parameters. N-gram/variant construction: 1.922 s.

Timing includes request state reset, target and draft prefill, proposal generation, verification, rollback and Python overhead. Training, fitting, audits and warmups are excluded. Full traces, validation/test quality, counters and environment are in report.json; timings.jsonl.gz retains every paired sample.

These are real source excerpts but tiny locally trained byte models, not capable language assistants. A repetitive or incorrect target continuation can be easy to draft. Accuracy is separate from exact decoding agreement. One source per genre and a shared CPU host limit generalization; repeated timings are not independent model or corpus trials. No sampling, large-model or production-serving claim.
