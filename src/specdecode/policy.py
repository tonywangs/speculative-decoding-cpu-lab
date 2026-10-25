"""Predeclared request-local policies; selection uses validation evidence only."""
import statistics


class AcceptancePolicy:
    def __init__(self, promotion=2, fallback=False):
        if promotion not in (2, 4):
            raise ValueError('promotion must be two or four complete blocks')
        self.promotion = promotion
        self.fallback = fallback
        self.reset()

    def reset(self):
        self.length = 2
        self.streak = self.blocks = self.accepted = self.proposed = 0
        self.trace = []

    def observe(self, proposed, accepted, rejected):
        self.blocks += 1
        self.proposed += proposed
        self.accepted += accepted
        self.streak = 0 if rejected else self.streak + 1
        self.length = 2 if rejected or self.streak < self.promotion else 4
        # Exactly one request-local decision, after the fourth nonempty block.
        if self.fallback and self.blocks == 4 and self.accepted / self.proposed < .5:
            self.length = 0
        self.trace.append({'block': self.blocks, 'proposed': proposed, 'accepted': accepted,
                           'rejected': rejected, 'next_length': self.length})


def configurations(branch='adaptive'):
    if branch not in ('adaptive', 'fallback', 'strict'):
        raise ValueError('unknown branch')
    return [{'name': 'greedy', 'family': None, 'policy': None}] + [
        {'name': f'{family}-{mode}', 'family': family,
         'policy': None if mode == 'k2' else branch}
        for family in ('prompt', 'ngram', 'gru') for mode in ('k2', branch)]


def proposal_policy():
    return {
        'version': 1,
        'experiment_limit': 2,
        'experiment_1': {
            'hypothesis': 'Acceptance-adaptive k2/k4 may reduce drafting overhead while retaining useful accepted blocks.',
            'rationale': 'Historical fixed-k2/k4 measurements favor n-grams; GRU overhead sometimes exceeds saved target work.',
            'configurations': configurations(),
            'rule': 'Start k2 per request; promote after two consecutive fully accepted nonempty blocks; demote on any rejection.',
        },
        'experiment_2': {
            'selection': 'Only after complete experiment-1 validation across the full manifest and both seeds. Compute median across cases of median paired greedy/adaptive-GRU time ratios per seed. If either seed is below 1 select fallback; otherwise strict.',
            'fallback': {'configurations': configurations('fallback'),
                         'rule': 'Keep adaptive promotion at two; after four nonempty proposal blocks, use cached greedy for the rest of the request if cumulative acceptance is below 0.5.'},
            'strict': {'configurations': configurations('strict'),
                       'rule': 'Promote after four consecutive fully accepted nonempty blocks; demote on rejection.'},
        },
        'test_freeze': 'Choose experiment 2 exclusively from complete validation, then freeze both policies before any full test evaluation.',
        'calibration': 'Operational cost evidence only; never use these six units to select experiment 2.',
    }


def select_second(measurements, expected_units):
    """Reject partial, duplicated or incorrect validation evidence before selection."""
    methods = {v['name'] for v in configurations()}
    required = {(seed, case, method, repeat) for seed, case in expected_units
                for method in methods for repeat in range(7)}
    selected = {}
    for row in measurements:
        if row['phase'] != 'timing':
            continue
        if row.get('split') != 'validation' or not row.get('agreement'):
            raise ValueError('selection requires correct validation results')
        key = (row['seed'], row['case_id'], row['method'], row['repeat'])
        if key in selected or key not in required or row['elapsed_ns'] <= 0:
            raise ValueError('unexpected or duplicate validation measurement')
        selected[key] = row
    if set(selected) != required or not required:
        raise ValueError('complete validation is required; calibration cannot select experiment 2')
    medians = {}
    for seed in sorted({seed for seed, _ in expected_units}):
        medians[seed] = statistics.median(
            statistics.median(selected[seed, case, 'greedy', r]['elapsed_ns'] /
                              selected[seed, case, 'gru-adaptive', r]['elapsed_ns'] for r in range(7))
            for s, case in expected_units if s == seed)
    branch = 'fallback' if any(value < 1 for value in medians.values()) else 'strict'
    return {'branch': branch, 'seed_median_paired_speedups': medians,
            'evidence_keys': [list(key) for key in sorted(required)],
            'configurations': configurations(branch)}
