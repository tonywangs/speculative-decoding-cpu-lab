"""Generate/check the two-seed comparison from completed recorded reports."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / 'results/realtext-gru'
METHODS = ('prompt-k2', 'prompt-k4', 'ngram-k2', 'ngram-k4', 'gru-k2', 'gru-k4')
TASKS = ('dialogue', 'prose', 'python')


def render():
    reports = []
    for seed in (17, 29):
        path = ROOT / f'seed{seed}'
        manifest = json.loads((path / 'manifest.json').read_text())
        for name, digest in manifest.items():
            if hashlib.sha256((path / name).read_bytes()).hexdigest() != digest:
                raise ValueError(f'changed artifact: {path / name}')
        report = json.loads((path / 'report.json').read_text())
        if not report['agreement_passed'] or report['config']['seed'] != seed:
            raise ValueError('incomplete or misidentified report')
        reports.append(report)
    if reports[0]['dataset_sha256'] != reports[1]['dataset_sha256']:
        raise ValueError('paired seeds must use identical inputs')
    lookup = [{(r['task'], r['method']): r for r in report['summary']} for report in reports]
    gru_loses = all(rows[task, f'gru-k{k}']['case_median_speedup'] <
                    rows[task, f'ngram-k{k}']['case_median_speedup']
                    for rows in lookup for task in TASKS for k in (2, 4))
    conclusion = ('Both GRU settings were slower than their corresponding n-gram settings '
                  'on every task in both seeds. The GRU training cost bought no latency '
                  'advantage over this cheaper baseline in the recorded experiment.'
                  if gru_loses else 'The task-by-task table below retains every measured comparison.')
    lines = ['# GRU versus inexpensive drafts on real text', '', conclusion, '',
             'This experiment asks whether a 23,301-parameter recurrent draft reduces '
             'greedy latency for a 289,440-parameter CPU transformer on held-out '
             'Shakespeare dialogue, Carroll prose and CPython source. It compares '
             'cached greedy, prompt lookup, training-only n-grams and GRU drafts. '
             'It is a measurement of these tiny models, not a new decoding algorithm.', '',
             f"Both independently trained seeds completed: {sum(r['timed_samples'] for r in reports):,} "
             'timed outputs exactly matched uncached greedy, across the same 36 held-out '
             'inputs and seven methods. These are two model pairs, three source documents, '
             'and 36 unique prompts—not independent observations per timing repeat.', '',
             '## Task-by-task results', '',
             'Each value is the median across cases of the median paired greedy/method '
             'latency ratio. Greater than 1 means faster; less than 1 means slower. '
             'Greedy is 1.000× by definition. No method or slowdown is omitted.', '',
             '| Task | Seed | Prompt k2 | Prompt k4 | N-gram k2 | N-gram k4 | GRU k2 | GRU k4 |',
             '|---|---:|---:|---:|---:|---:|---:|---:|']
    for task in TASKS:
        for report in reports:
            rows = {r['method']: r for r in report['summary'] if r['task'] == task}
            values = ' | '.join(f"{rows[m]['case_median_speedup']:.3f}×" for m in METHODS)
            lines.append(f"| {task} | {report['config']['seed']} | {values} |")
    lines += ['']
    for task in TASKS:
        findings = []
        for report in reports:
            rows = [r for r in report['summary'] if r['task'] == task and r['method'] != 'greedy']
            best = max(rows, key=lambda r: r['case_median_speedup'])
            gru = [r for r in rows if r['method'].startswith('gru')]
            findings.append(f"seed {report['config']['seed']}: {best['method']} has the highest "
                            f"observed median ({best['case_median_speedup']:.3f}×); "
                            + ', '.join(f"{r['method']} {r['case_median_speedup']:.3f}×" for r in gru))
        lines.append(f"- **{task.capitalize()}** — " + '; '.join(findings) + '.')
    lines += ['', 'These rankings are descriptive. The per-seed reports retain conditional '
              '95% whole-case bootstrap intervals, latency, acceptance, target calls and '
              'drafting time for every method. An interval spanning 1 does not resolve '
              'speedup versus slowdown. Similar acceptance need not imply similar speed: '
              'the recurrent draft has its own inference overhead.', '',
              '## Training cost and output quality', '',
              '| Seed | Target training seconds | GRU training seconds | N-gram/variant fit seconds |',
              '|---:|---:|---:|---:|']
    for r in reports:
        lines.append(f"| {r['config']['seed']} | {r['target_training']['elapsed_ns']/1e9:.3f} | "
                     f"{r['draft_training']['elapsed_ns']/1e9:.3f} | {r['draft_fit_ns']/1e9:.3f} |")
    lines += ['', 'Both networks train for 1,200 updates on the same training windows; equal '
              'steps do not mean equal training compute. Costs above are the completed '
              'runs, excluding the interrupted seed-29 attempt and diagnostic work.', '',
              '| Task | Seed | Target next-byte accuracy | Free-running accuracy | Exact continuation |',
              '|---|---:|---:|---:|---:|']
    for report in reports:
        for r in report['accuracy']:
            lines.append(f"| {r['task']} | {report['config']['seed']} | "
                         f"{r['teacher_forced_token_accuracy']:.1%} | "
                         f"{r['free_running_token_accuracy']:.1%} | {r['exact_continuation_accuracy']:.1%} |")
    lines += ['', 'Matching the target exactly is separate from predicting real continuations '
              'correctly. Repetitive, low-quality generations can be easy to draft. These '
              'results do not establish useful text generation, pretrained-LLM acceleration, '
              'semantic code completion, stochastic sampling or production performance.', '',
              '## Reproduce and inspect', '',
              '- [Protocol, sources, splits, seeds and numerical amendment](PROTOCOL.md).',
              '- [Seed 17 full report](seed17/RESULTS.md) and [seed 29 full report](seed29/RESULTS.md).',
              '- Each seed directory contains checkpoints, training traces/cost, exact source '
              'windows, compressed raw timings, correctness records, environment and hashes.',
              '- [Installation, full reproduction, offline verification and live demo](../../README.md#real-text-gru-experiment).', '',
              'Run the live demo on case 0 (dialogue), 12 (prose) or 24 (Python) to see '
              'the actual prompt, generated text, source continuation and method agreement. '
              'The recorded inputs are public and included; no provider account is needed.', '',
              'Measurements use one intra-op and one inter-op CPU thread, FP32, matched '
              'target weights and kernels, two warmups and seven balanced timing repetitions. '
              'The seeds run sequentially. Shared-host load is uncontrolled; intervals '
              'condition on each target and these inputs. There is only one source per '
              'genre and the seeds share the data. Historical synthetic timings are not '
              'used as baselines for this comparison.', '',
              'Regenerate this page with `.venv/bin/python scripts/summarize_realtext.py`; '
              'append `--check` to check it without writing. This verifies artifact hashes '
              'and report linkage; `verify-realtext` separately replays model correctness '
              'and recomputes summaries from raw samples.', '']
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    content = render()
    path = ROOT / 'RESULTS.md'
    if args.check:
        if path.read_text() != content:
            raise AssertionError('aggregate comparison differs from recorded reports')
        print('Aggregate comparison matches both completed reports.')
    else:
        path.write_text(content)
        print(path)


if __name__ == '__main__':
    main()
