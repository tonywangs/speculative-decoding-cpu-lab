"""Render the bounded calibration findings from saved evidence, without inference."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sqlite3

ROOT = Path(__file__).resolve().parents[1]


def text_tokens(tokens):
    return ''.join(chr(t-5) if 5 <= t < 261 else f'<special:{t}>' for t in tokens)


def render(output):
    state = json.loads((output / 'state.json').read_text())
    manifest = json.loads((output / 'population.json').read_text())
    forecast = json.loads((output / 'extrapolation.json').read_text())
    rows = [json.loads(line) for line in (output / 'measurements.jsonl').read_text().splitlines()]
    if not state['calibration_complete']:
        raise ValueError('complete calibration is required for this findings page')
    counts = Counter(r['phase'] for r in rows)
    usage = state['usage']
    with sqlite3.connect(f'file:{output / "journal.sqlite3"}?mode=ro', uri=True) as db:
        events = [(kind, json.loads(data)) for kind, data in db.execute('SELECT kind,data FROM events ORDER BY sequence')]
    environments = [data for kind, data in events if kind == 'environment']
    stops = [data for kind, data in events if kind == 'stopped']
    exports = [data['elapsed_ns'] for kind, data in events if kind == 'exported']
    checked = [r for r in rows if r['phase'] in ('audit', 'warmup', 'timing')]
    if not all(r['agreement'] for r in checked):
        raise ValueError('incorrect output cannot be reported as successful')
    lines = [
        '# Six-unit controller cost calibration', '',
        '**Calibration complete; the full two-experiment research goal remains unfinished.** '
        'No full-population experiment ran and no second-experiment branch was selected.', '',
        f'The six first-window validation case/seed units produced {len(rows)} durable records: '
        f'{counts["oracle"]} independent uncached references, {counts["quality"]} quality passes, '
        f'{counts["audit"]} numerical audits, {counts["warmup"]} warmups and {counts["timing"]} timings. '
        f'All {len(checked)} audited, warmed and timed outputs matched uncached greedy exactly. '
        'Every method occupies every timing position once within each unit.', '',
        f'Cumulative controller execution: **{usage["seconds"]:.2f} s**, '
        f'**{usage["calls"]:,} forward calls** '
        f'({usage["target_calls"]:,} target; {usage["draft_calls"]:,} GRU), '
        f'**{usage["peak_rss_bytes"] / 1024**2:.2f} MiB** peak resident memory. '
        'Limits were 600 execution seconds, 150,000 aggregate forwards and 4 GiB address space. '
        f'No calls were lost to crashes ({usage["lost_calls"]} charged unknown calls).', '',
        f'The run paused after {stops[0].get("records_this_invocation", 0)} records, then resumed the same '
        f'journal in a second process. There are {len(environments)} recorded execution environments. '
        'The pause followed the first unit’s audits, so its two warmups occurred in the resumed process. '
        'All successful records and cumulative budgets survived the restart. Separate small-model '
        'tests exercise SIGTERM, a mid-timing restart with rewarming, hard memory allocation failure, '
        'atomic rollback and exact call-limit exhaustion; those are mechanism tests, not research samples.', '',
        f'CPU: {environments[0]["cpu_model"]}; Torch {environments[0]["torch"]}; '
        f'{environments[0]["intraop_threads"]} intra-op thread, one process, FP32. '
        'The journal records both environments and source hashes. The host is shared.', '',
        '## Full workload requirement', '',
        'The manifest preserves 831 validation and 832 test windows, both seeds, the original training '
        'windows and source/model identities. One Python validation window is excluded because its '
        'prompt occurs in its source training region. Calibration does not remove any full-study work.', '',
        '| Work over both full experiments | Count | Extrapolated compute hours |',
        '|---|---:|---:|',
    ]
    for name, count_key, phase in (
        ('Timed decodes', 'two_experiment_timed_outputs', 'timing'),
        ('Warmup decodes', 'two_experiment_warmup_outputs', 'warmup'),
        ('Audited decodes', 'two_experiment_audited_decodes', 'audit'),
        ('Shared uncached references', 'shared_oracle_decodes', 'oracle'),
        ('Shared target/GRU quality passes', 'shared_quality_passes', 'quality'),
    ):
        lines.append(f'| {name} | {forecast[count_key]:,} | {forecast["phase_extrapolation"][phase]["seconds"]/3600:.3f} |')
    lines += ['',
        f'The source/seed-weighted cost analogue is **{forecast["estimated_total_seconds"]/3600:.2f} hours** '
        f'and **{forecast["estimated_total_calls"]:,} forward calls**, including '
        f'{forecast["estimated_controller_and_setup_seconds"]/3600:.2f} extrapolated hours of setup, '
        'durable checkpointing and orchestration. This exceeds the prior two-hour/two-million-call '
        f'caps by **{forecast["shortfall_vs_7200_seconds"]/3600:.2f} hours** and '
        f'**{forecast["shortfall_vs_2000000_calls"]:,} calls** before any additional safety margin.', '',
        f'Median measured result-commit cost was {forecast["median_result_commit_ns"]/1e6:.2f} ms. '
        f'The {len(exports)} recorded derived-view exports took {sum(exports)/1e9:.3f} s in total '
        'outside the execution budget. SQLite commits, rather than JSON views, are the resumable checkpoints. '
        'The overhead analogue spreads observed loading/restart and per-operation costs across the '
        'full workload; it is deliberately not a precise optimized-runtime forecast.', '',
        'Both predeclared second-experiment branches retain seven configurations and the same full '
        'population. Each is assigned this measured experiment-1 cost analogue, not a measured branch '
        'runtime. The current decoding-path worst-case forward-count bound for either complete '
        f'route is {forecast["branch_requirements"]["fallback"]["conservative_forward_call_upper_bound"]:,} '
        '(64-token outputs; maximum four draft tokens per block). Neither branch has a proven wall-time '
        'bound. The observed memory peak fits the cap for serial calibration, not an unmeasured parallel run.', '',
        'The resource shortfall is actionable: a future full evaluation needs a separately authorized '
        'larger cumulative budget with headroom, or a measured implementation improvement that preserves '
        'the population and protocol. This run stops here.', '',
        '## Observed timings on the six calibration units', '',
        'Each cell is the median of seven paired cached-greedy/method latency ratios. '
        'Values below one are slowdowns. There is only **one window per source and seed**; '
        'these are operational observations, not population speed estimates or confidence intervals.', '',
        '| Method | 17 dialogue | 17 prose | 17 Python | 29 dialogue | 29 prose | 29 Python |',
        '|---|---:|---:|---:|---:|---:|---:|',
    ]
    units = [(seed, task) for seed in (17, 29) for task in ('dialogue', 'prose', 'python')]
    ratios = {(r['seed'], r['task'], r['method']): r for r in forecast['calibration_paired_ratios']}
    methods = ['greedy', 'prompt-k2', 'prompt-adaptive', 'ngram-k2', 'ngram-adaptive', 'gru-k2', 'gru-adaptive']
    for method in methods:
        lines.append('| ' + method + ' | ' + ' | '.join(f'{ratios[seed, task, method]["median_paired_speedup"]:.3f}×'
                                                      for seed, task in units) + ' |')
    lines += ['', 'Cached-greedy median request latency (same column order): ' +
              ', '.join(f'{ratios[seed, task, "greedy"]["median_latency_ms"]:.2f} ms' for seed, task in units) + '.', '',
        'N-gram variants were faster than cached greedy in these six units. Adaptive lengths did not '
        'consistently improve on fixed k2. GRU variants showed both speedups and slowdowns. These six '
        'cases do not authorize selecting the second experiment: that rule requires complete validation '
        'across all 831 validation windows and both seeds.', '',
        '## Quality is distinct from output agreement', '',
        '| Seed / source | Target next-byte accuracy | GRU next-byte accuracy | Target bits/byte | GRU bits/byte | Free-running byte accuracy |',
        '|---|---:|---:|---:|---:|---:|',
    ]
    for seed, task in units:
        selected = [r for r in rows if r['seed'] == seed and r['task'] == task]
        target = next(r for r in selected if r['phase'] == 'quality' and r['method'] == 'target')
        draft = next(r for r in selected if r['phase'] == 'quality' and r['method'] == 'gru')
        oracle = next(r for r in selected if r['phase'] == 'oracle')
        lines.append(f'| {seed} / {task} | {target["correct"]/64:.1%} | {draft["correct"]/64:.1%} | '
                     f'{target["bits_per_byte"]:.3f} | {draft["bits_per_byte"]:.3f} | {oracle["free_running_correct"]/64:.1%} |')
    lines += ['', 'No generated continuation exactly matched its 64-byte source continuation. '
              'These tiny models do not produce reliably useful language. A faster reproduction '
              'of a wrong or repetitive target output does not improve generation quality.', '',
              '## A visible saved example', '']
    example = next(r for r in rows if r['seed'] == 17 and r['task'] == 'python' and r['phase'] == 'oracle')
    window = next(r for r in manifest['windows'] if r['case_id'] == example['case_id'])
    raw = (ROOT / 'results/realtext-gru/corpus/python.txt').read_bytes()
    prompt = raw[window['source_offset']:window['source_offset']+96].decode('utf-8', errors='replace')
    lines += [f'Seed 17, `{example["case_id"]}`. All seven configurations produced the same target continuation.', '',
              'Public CPython prompt:', '', '```text', prompt, '```', '',
              'Generated target continuation:', '', '```text', text_tokens(example['tokens']).rstrip(), '```', '',
              'Actual source continuation:', '', '```text', text_tokens(example['truth']), '```', '',
              '## Inspect or reproduce', '',
              '```bash', '.venv/bin/python -m specdecode.research status',
              '.venv/bin/python -m specdecode.research verify',
              '.venv/bin/python scripts/check_controller.py',
              '.venv/bin/python scripts/summarize_controller.py --check', '```', '',
              'See the [controller guide](../../docs/research-controller.md) for initialization, '
              'deadline/resume commands, resource semantics and the predeclared branches. '
              '[Raw measurements](measurements.jsonl), [extrapolation](extrapolation.json), '
              '[population](population.json), [proposal policy](proposal-policy.json) and '
              '[state](state.json) accompany the authoritative SQLite journal. '
              'Exact agreement and numerical-audit diagnostics apply to these outputs only; '
              'no pretrained-LLM, unseen-document, production-serving or sampling claim is made.', '']
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    output = ROOT / 'results/controller-calibration'
    content = render(output)
    path = output / 'RESULTS.md'
    if args.check:
        if path.read_text() != content:
            raise SystemExit('controller results page differs from saved evidence')
        print('controller results page matches saved evidence')
    else:
        path.write_text(content)
        print(path.relative_to(ROOT))


if __name__ == '__main__':
    main()
