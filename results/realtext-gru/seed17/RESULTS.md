# Real-text CPU draft comparison

Seed 17; 36 held-out inputs; 1764 timed outputs, all exactly equal to uncached greedy.

CPU: DO-Regular; PyTorch 2.8.0+cpu; 1 intra-op thread(s), one inter-op thread, FP32.

Primary speedup: median across cases of median paired greedy/method ratios. 95% intervals resample whole cases (2,000 draws, seed 314159), conditional on this target and text. Seven repetitions balance every method across every timing position. Values below 1 are slowdowns.

| Task | Method | Median ms | Speedup | Conditional 95% | Acceptance | Draft ms |
|---|---|---:|---:|---|---:|---:|
| dialogue | greedy | 137.849 | 1.000× | [1.000, 1.000] | — | 0.000 |
| dialogue | prompt-k2 | 131.148 | 1.050× | [0.997, 1.115] | 16.7% | 6.363 |
| dialogue | prompt-k4 | 129.190 | 1.070× | [0.989, 1.171] | 10.1% | 6.374 |
| dialogue | ngram-k2 | 92.224 | 1.501× | [1.399, 1.622] | 39.3% | 1.035 |
| dialogue | ngram-k4 | 87.048 | 1.526× | [1.437, 1.679] | 24.8% | 1.112 |
| dialogue | gru-k2 | 133.854 | 1.019× | [0.979, 1.090] | 39.2% | 40.022 |
| dialogue | gru-k4 | 146.490 | 0.947× | [0.846, 1.014] | 24.3% | 55.225 |
| prose | greedy | 135.824 | 1.000× | [1.000, 1.000] | — | 0.000 |
| prose | prompt-k2 | 114.245 | 1.141× | [1.038, 1.241] | 23.9% | 5.230 |
| prose | prompt-k4 | 116.752 | 1.151× | [1.029, 1.289] | 14.1% | 5.352 |
| prose | ngram-k2 | 88.584 | 1.488× | [1.415, 1.561] | 41.9% | 1.013 |
| prose | ngram-k4 | 82.498 | 1.715× | [1.528, 1.740] | 27.9% | 1.011 |
| prose | gru-k2 | 131.495 | 0.985× | [0.940, 1.099] | 42.7% | 39.249 |
| prose | gru-k4 | 136.541 | 0.971× | [0.888, 1.082] | 28.3% | 52.499 |
| python | greedy | 136.224 | 1.000× | [1.000, 1.000] | — | 0.000 |
| python | prompt-k2 | 93.941 | 1.439× | [1.247, 1.933] | 41.3% | 4.037 |
| python | prompt-k4 | 87.171 | 1.481× | [1.423, 2.330] | 27.8% | 3.937 |
| python | ngram-k2 | 74.010 | 1.920× | [1.614, 2.069] | 60.9% | 0.786 |
| python | ngram-k4 | 58.173 | 2.404× | [1.910, 2.777] | 46.2% | 0.700 |
| python | gru-k2 | 116.168 | 1.214× | [1.030, 1.299] | 57.5% | 35.183 |
| python | gru-k4 | 104.603 | 1.343× | [1.046, 1.518] | 44.1% | 40.488 |

| Task | Target next-byte accuracy | Free-running accuracy | Exact continuation |
|---|---:|---:|---:|
| dialogue | 35.4% | 7.4% | 0.0% |
| prose | 46.2% | 9.9% | 0.0% |
| python | 64.3% | 23.6% | 0.0% |

Target training: 574.384 s; 289,440 parameters. GRU training: 146.539 s; 23,301 parameters. N-gram/variant construction: 1.932 s.

Timing includes request state reset, target and draft prefill, proposal generation, verification, rollback and Python overhead. Training, fitting, audits and warmups are excluded. Full traces, validation/test quality, counters and environment are in report.json; timings.jsonl.gz retains every paired sample.

These are real source excerpts but tiny locally trained byte models, not capable language assistants. A repetitive or incorrect target continuation can be easy to draft. Accuracy is separate from exact decoding agreement. One source per genre and a shared CPU host limit generalization; repeated timings are not independent model or corpus trials. No sampling, large-model or production-serving claim.
