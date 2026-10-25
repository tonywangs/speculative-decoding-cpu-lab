"""Read-only semantic checks of controller measurements, accounting and provenance.

Uses only repository files and installed libraries. No inference, timers, network,
wall-clock queries or experiment resumption.
"""
from collections import Counter
import json
import math
from pathlib import Path
import sqlite3

from specdecode.experiment import sha256
from specdecode.policy import AcceptancePolicy
from specdecode.research import DEFAULT_LIMITS, verify

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / 'results/controller-calibration'


def check():
    answer = verify(OUTPUT, ROOT / 'results/realtext-gru')
    with sqlite3.connect(f'file:{OUTPUT / "journal.sqlite3"}?mode=ro', uri=True) as db:
        db.row_factory = sqlite3.Row
        attempts = {r['id']: dict(r) for r in db.execute('SELECT * FROM attempts')}
        events = [(r['sequence'], r['kind'], json.loads(r['data'])) for r in db.execute('SELECT * FROM events')]
        results = [(r['work_id'], r['attempt'], json.loads(r['data'])) for r in db.execute('SELECT * FROM results')]
        metadata = {r['key']: json.loads(r['value']) for r in db.execute('SELECT * FROM metadata')}
        assert not db.execute('PRAGMA foreign_key_check').fetchall()
        assert db.execute("SELECT count(*) FROM sqlite_master WHERE type='trigger'").fetchone()[0] == 8
    assert metadata['limits'] == DEFAULT_LIMITS
    environment = {seq: data for seq, kind, data in events if kind == 'environment'}
    for record in environment.values():
        assert record['intraop_threads'] == record['interop_threads'] == 1
        assert record['deterministic_algorithms']
        for name, expected in record['source_sha256'].items():
            assert sha256(ROOT / 'src/specdecode' / name) == expected, f'measurement implementation changed: {name}'
    state = json.loads((OUTPUT / 'state.json').read_text())
    usage = state['usage']
    assert usage['seconds'] == sum(r['charged_seconds'] for r in attempts.values())
    for key in ('target_calls', 'draft_calls', 'lost_calls'):
        assert usage[key] == sum(r[key] for r in attempts.values())
    assert usage['calls'] == usage['target_calls'] + usage['draft_calls'] + usage['lost_calls']
    assert usage['calls'] <= DEFAULT_LIMITS['calls']
    assert usage['seconds'] <= DEFAULT_LIMITS['seconds']
    assert usage['peak_rss_bytes'] <= DEFAULT_LIMITS['memory_bytes']
    starts = {data['attempt']: data for _, kind, data in events if kind == 'started'}
    completions = {data['attempt']: data for _, kind, data in events if kind == 'completed'}
    for number, attempt in attempts.items():
        assert number in starts
        assert attempt['status'] == 'complete'  # this saved run has no failed/lost attempts
        assert number in completions
        assert attempt['charged_seconds'] == completions[number]['elapsed_seconds']
        assert attempt['target_calls'] + attempt['draft_calls'] <= attempt['reserved_calls']
        assert attempt['charged_seconds'] <= attempt['reserved_seconds']
        assert attempt['peak_rss_bytes'] <= DEFAULT_LIMITS['memory_bytes']
    windows = {r['case_id']: r for r in metadata['manifest']['windows']}
    phases = Counter()
    for work_id, number, row in results:
        attempt = attempts[number]
        assert attempt['work_id'] == work_id == row['work_id']
        assert row['environment_id'] in environment
        assert attempt['target_calls'] == row['target_calls']
        assert attempt['draft_calls'] == row['draft_calls']
        assert attempt['peak_rss_bytes'] == row['peak_rss_bytes']
        assert row['elapsed_ns'] / 1e9 <= attempt['charged_seconds']
        phases[row['phase']] += 1
        if row['phase'] == 'quality':
            assert row['tokens_scored'] == 64 and 0 <= row['correct'] <= 64
            assert math.isfinite(row['nll_sum']) and row['nll_sum'] >= 0
            assert row['bits_per_byte'] == row['nll_sum'] / 64 / math.log(2)
            assert (row['target_calls'], row['draft_calls']) == ((1, 0) if row['method'] == 'target' else (0, 1))
        elif row['phase'] == 'oracle':
            window = windows[row['case_id']]
            source = (ROOT / 'results/realtext-gru/corpus' / (window['task'] + '.txt')).read_bytes()
            start = window['source_offset'] + 96
            truth = [v + 5 for v in source[start:start+64]]
            assert row['truth'] == truth
            assert row['free_running_correct'] == sum(a == b for a, b in zip(row['tokens'], truth))
            assert row['exact_continuation'] == (row['tokens'] == truth)
        elif row['method'].endswith('-adaptive'):
            policy = AcceptancePolicy()
            for step in row['policy_trace']:
                assert 0 <= step['accepted'] <= step['proposed'] <= policy.length
                assert step['proposed'] > 0
                assert step['rejected'] == (step['accepted'] < step['proposed'])
                policy.observe(step['proposed'], step['accepted'], step['rejected'])
                assert policy.trace[-1] == step
            assert policy.proposed == row['proposed']
            assert policy.accepted == row['accepted']
            assert sum(s['rejected'] for s in policy.trace) == row['rejected_blocks']
            assert sum(not s['rejected'] for s in policy.trace) == row['fully_accepted_blocks']
        else:
            assert row['policy_trace'] == []
    assert phases == {'oracle': 6, 'quality': 12, 'audit': 42, 'warmup': 84, 'timing': 294}
    stops = [data for _, kind, data in events if kind == 'stopped']
    assert [r['reason'] for r in stops] == ['requested_pause_between_records', 'calibration_complete']
    assert stops[0]['records_this_invocation'] == 10
    assert stops[1]['records_this_invocation'] == 428
    assert stops[1]['usage']['calls'] > stops[0]['usage']['calls'] > 0
    assert stops[1]['usage']['seconds'] > stops[0]['usage']['seconds'] > 0
    assert state['completed_full_experiments'] == 0
    answer.update(budgets_reconciled=True, implementation_hashes_match=True, resume_preserved=True,
                  policy_traces_replayed=True, quality_arithmetic_checked=True)
    return answer


if __name__ == '__main__':
    print(json.dumps(check(), indent=2, sort_keys=True))
