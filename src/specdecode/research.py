"""Bounded, resumable cost calibration for the full real-text research population.

No scheduler or full-evaluation launcher lives here. SQLite is the authoritative
append-only measurement journal; JSON exports are reproducible views. Linux is
required for the one-process lock, interval timer and address-space ceiling.
"""
import argparse
from collections import Counter, defaultdict
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import random
import resource
import signal
import sqlite3
import statistics
import time

import torch
import torch.nn.functional as F

from .decode import decode, uncached_greedy
from .draft import NGram, PromptLookup
from .experiment import configure, environment, sha256
from .policy import AcceptancePolicy, configurations, proposal_policy
from .realtext import TASKS, TextConfig, load_text, read_gzip, read_sources
from .recurrent import GRUDraft

SCHEMA = 1
SEEDS = (17, 29)
DEFAULT_LIMITS = {'seconds': 600.0, 'calls': 150000, 'memory_bytes': 4 * 1024**3,
                  'experiments': 2, 'calibration_units': 6, 'external_calls': 0}


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w') as stream:
        stream.write(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def population(inputs):
    """Enumerate every 160-byte window; never resample or alter training data."""
    sources, provenance = read_sources(inputs / 'corpus')
    saved = {}
    identities = {}
    for seed in SEEDS:
        directory = inputs / f'seed{seed}'
        identity = json.loads((directory / 'manifest.json').read_text())
        for name, expected in identity.items():
            if sha256(directory / name) != expected:
                raise ValueError(f'input identity mismatch: seed{seed}/{name}')
        cfg = TextConfig(**json.loads((directory / 'config.json').read_text()))
        if (cfg.seed, cfg.sequence_bytes, cfg.prompt_bytes, cfg.max_new_tokens) != (seed, 160, 96, 64):
            raise ValueError('this protocol requires original seeds and 96/64-byte windows')
        saved[seed] = read_gzip(directory / 'dataset.json.gz')
        identities[str(seed)] = identity
    if saved[17] != saved[29]:
        raise ValueError('original seed datasets must match')
    original = saved[17]
    seen = {bytes(t - 5 for t in r['tokens'][1:97]) for r in original['train']}
    windows, excluded = [], []
    for task in TASKS:
        raw, start, end = sources[task]
        a, b = start + (end-start)*8//10, start + (end-start)*9//10
        expected_bounds = {'train': [start, a], 'validation': [a, b], 'eval': [b, end]}
        if original['bounds'][task] != expected_bounds:
            raise ValueError('source bounds differ from saved training protocol')
        train_region = raw[start:a]
        for split, lo, hi in (('validation', a, b), ('test', b, end)):
            for offset in range(lo, hi - 160 + 1, 160):
                prefix = raw[offset:offset+96]
                reason = ('low_diversity' if len(set(prefix)) < 12 else
                          'duplicate_prompt' if prefix in seen else
                          'training_region_match' if prefix in train_region else None)
                row = {'case_id': f'{split}-{task}-{offset:07d}', 'task': task, 'split': split,
                       'source_offset': offset, 'sequence_bytes': 160, 'prompt_bytes': 96,
                       'window_sha256': hashlib.sha256(raw[offset:offset+160]).hexdigest()}
                if reason:
                    excluded.append({**row, 'reason': reason})
                    continue
                seen.add(prefix)
                windows.append(row)
    units = [{'seed': seed, 'case_id': next(r['case_id'] for r in windows
              if r['task'] == task and r['split'] == 'validation')}
             for seed in SEEDS for task in TASKS]
    return {'schema': SCHEMA, 'sources': provenance, 'input_artifacts': identities,
            'seeds': list(SEEDS), 'training_windows': original['train'],
            'bounds': original['bounds'], 'windows': windows, 'excluded': excluded,
            'calibration_units': units,
            'ordering': 'seed 17 then 29; dialogue/prose/python; ascending source offsets; validation before test',
            'exclusions': 'Original selected training prompts plus held-out deduplication; reject <12 distinct prompt bytes or any match in the complete source training region.'}


def work_items(manifest):
    methods = [v['name'] for v in configurations()]
    output = []
    for unit in manifest['calibration_units']:
        base = {**unit, 'split': 'validation'}
        specs = [('oracle', 'uncached', 0, 0), ('quality', 'target', 0, 0), ('quality', 'gru', 0, 1)]
        specs += [('audit', method, 0, i) for i, method in enumerate(methods)]
        specs += [('warmup', method, repeat, i) for repeat in range(2) for i, method in enumerate(methods)]
        order = list(methods)
        random.Random(f"{unit['seed']}:{unit['case_id']}:balanced-v1").shuffle(order)
        for repeat in range(7):
            rotated = order[repeat:] + order[:repeat]
            specs += [('timing', method, repeat, i) for i, method in enumerate(rotated)]
        for phase, method, repeat, position in specs:
            row = {**base, 'phase': phase, 'method': method, 'repeat': repeat, 'position': position}
            row['work_id'] = f"s{unit['seed']}/{unit['case_id']}/{phase}/{repeat}/{method}"
            output.append(row)
    return output


@contextmanager
def locked(output, shared=False):
    output.mkdir(parents=True, exist_ok=True)
    with (output / '.lock').open('a') as stream:
        try:
            fcntl.flock(stream, (fcntl.LOCK_SH if shared else fcntl.LOCK_EX) | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError('another controller owns this run; only one worker is permitted') from exc
        yield


class LimitStop(RuntimeError):
    pass


class Journal:
    def __init__(self, output):
        self.output = Path(output)
        self.db = sqlite3.connect(self.output / 'journal.sqlite3')
        self.db.row_factory = sqlite3.Row
        self.db.execute('PRAGMA synchronous=FULL')
        self.db.execute('PRAGMA foreign_keys=ON')
        self.checkpoint_ns = 0
        self.last_boundary = time.monotonic()

    def close(self):
        self.db.close()

    def event(self, kind, data):
        self.db.execute('INSERT INTO events(kind,data) VALUES (?,?)', (kind, canonical(data)))

    def initialize(self, manifest, limits):
        self.db.executescript('''
            CREATE TABLE metadata(key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE work(sequence INTEGER PRIMARY KEY, work_id TEXT UNIQUE NOT NULL, data TEXT NOT NULL);
            CREATE TABLE results(work_id TEXT PRIMARY KEY REFERENCES work(work_id), attempt INTEGER NOT NULL, data TEXT NOT NULL);
            CREATE TABLE events(sequence INTEGER PRIMARY KEY, kind TEXT NOT NULL, data TEXT NOT NULL);
            CREATE TABLE attempts(id INTEGER PRIMARY KEY, work_id TEXT, status TEXT NOT NULL,
                reserved_seconds REAL NOT NULL, reserved_calls INTEGER NOT NULL,
                charged_seconds REAL NOT NULL, target_calls INTEGER NOT NULL DEFAULT 0,
                draft_calls INTEGER NOT NULL DEFAULT 0, lost_calls INTEGER NOT NULL DEFAULT 0,
                peak_rss_bytes INTEGER NOT NULL DEFAULT 0);
        ''')
        for table in ('metadata', 'work', 'results', 'events'):
            for action in ('UPDATE', 'DELETE'):
                self.db.execute(f"CREATE TRIGGER no_{action.lower()}_{table} BEFORE {action} ON {table} BEGIN SELECT RAISE(ABORT, 'append-only journal'); END")
        with self.db:
            for key, value in (('schema', SCHEMA), ('manifest', manifest), ('policy', proposal_policy()), ('limits', limits)):
                self.db.execute('INSERT INTO metadata VALUES (?,?)', (key, canonical(value)))
            for i, row in enumerate(work_items(manifest)):
                self.db.execute('INSERT INTO work VALUES (?,?,?)', (i, row['work_id'], canonical(row)))
            self.event('initialized', {'manifest_digest': digest(manifest), 'policy_digest': digest(proposal_policy()),
                       'full_workload': 'pending; never launched by calibration', 'limits': limits})

    def meta(self, key):
        return json.loads(self.db.execute('SELECT value FROM metadata WHERE key=?', (key,)).fetchone()[0])

    def usage(self):
        row = self.db.execute('''SELECT coalesce(sum(charged_seconds),0) seconds,
              coalesce(sum(target_calls),0) target_calls, coalesce(sum(draft_calls),0) draft_calls,
              coalesce(sum(lost_calls),0) lost_calls, coalesce(max(peak_rss_bytes),0) peak_rss_bytes
              FROM attempts''').fetchone()
        usage = dict(row)
        usage['calls'] = usage['target_calls'] + usage['draft_calls'] + usage['lost_calls']
        return usage

    def recover(self):
        # Unfinished reservations are charged in full, never silently refunded.
        with self.db:
            for row in self.db.execute("SELECT * FROM attempts WHERE status='running'").fetchall():
                self.db.execute("UPDATE attempts SET status='abandoned' WHERE id=?", (row['id'],))
                self.event('abandoned', {'attempt': row['id'], 'work_id': row['work_id'],
                           'seconds_charged': row['reserved_seconds'], 'calls_charged': row['reserved_calls'],
                           'reason': 'process ended without atomic completion; exact lost calls unknown'})

    def pending(self):
        return [json.loads(row[0]) for row in self.db.execute('''SELECT w.data FROM work w
                  LEFT JOIN results r USING(work_id) WHERE r.work_id IS NULL ORDER BY w.sequence''')]

    def results(self):
        return [json.loads(row[0]) for row in self.db.execute('''SELECT r.data FROM work w
                JOIN results r USING(work_id) ORDER BY w.sequence''')]

    def begin(self, work_id, deadline, max_seconds=30.0, max_calls=512):
        now = time.monotonic()
        # Include previous commit / orchestration time in the next lease.
        overhead = now - self.last_boundary
        limits, usage = self.meta('limits'), self.usage()
        seconds = min(max_seconds, limits['seconds'] - usage['seconds'], deadline - time.time())
        calls = min(max_calls, limits['calls'] - usage['calls'])
        if seconds <= overhead + .01:
            raise LimitStop('cumulative time limit or UTC deadline')
        if calls <= 0:
            raise LimitStop('cumulative model-call limit')
        with self.db:
            cursor = self.db.execute('''INSERT INTO attempts(work_id,status,reserved_seconds,reserved_calls,
                     charged_seconds,lost_calls) VALUES (?,'running',?,?,?,?)''',
                     (work_id, seconds, calls, seconds, calls))
            attempt = cursor.lastrowid
            self.event('started', {'attempt': attempt, 'work_id': work_id,
                       'reserved_seconds': seconds, 'reserved_calls': calls})
        return Meter(self, attempt, now - overhead, seconds, calls)

    def finish(self, meter, result=None, error=None):
        # A successful result and its refund commit together. SIGKILL before
        # commit preserves the reservation and leaves exactly this item pending.
        started = time.perf_counter_ns()
        elapsed = time.monotonic() - meter.started
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        with self.db:
            if result is not None:
                self.db.execute('INSERT INTO results VALUES (?,?,?)',
                                (result['work_id'], meter.attempt, canonical(result)))
            self.db.execute('''UPDATE attempts SET status=?, charged_seconds=?, target_calls=?,
                  draft_calls=?, lost_calls=0, peak_rss_bytes=? WHERE id=?''',
                  ('failed' if error else 'complete', elapsed, meter.target_calls, meter.draft_calls, peak, meter.attempt))
            self.event('failed' if error else 'completed', {'attempt': meter.attempt,
                       'elapsed_seconds': elapsed, 'error': error,
                       'previous_checkpoint_ns': self.checkpoint_ns})
        self.checkpoint_ns = time.perf_counter_ns() - started
        self.last_boundary = meter.started + elapsed

    def record_environment(self):
        with self.db:
            self.event('environment', environment())
            return self.db.execute('SELECT max(sequence) FROM events').fetchone()[0]

    def fail(self, meter, exc):
        # An asynchronous signal can arrive immediately after COMMIT. Never
        # rewrite a committed success as a failed attempt in that case.
        row = self.db.execute('SELECT status FROM attempts WHERE id=?', (meter.attempt,)).fetchone()
        if row and row[0] == 'running':
            self.finish(meter, error=f'{type(exc).__name__}: {exc}')


class Meter:
    def __init__(self, journal, attempt, started, seconds, calls):
        self.journal, self.attempt, self.started = journal, attempt, started
        self.seconds, self.calls = seconds, calls
        self.target_calls = self.draft_calls = 0
        self.handles = []

    def count(self, kind):
        if self.target_calls + self.draft_calls >= self.calls:
            raise LimitStop('model-call reservation exhausted')
        if time.monotonic() - self.started >= self.seconds:
            raise LimitStop('time reservation exhausted')
        if kind == 'target':
            self.target_calls += 1
        else:
            self.draft_calls += 1

    def attach(self, target, draft):
        self.handles = [target.register_forward_pre_hook(lambda *_: self.count('target')),
                        draft.register_forward_pre_hook(lambda *_: self.count('draft'))]

    def __enter__(self):
        self.old_alarm = signal.signal(signal.SIGALRM, self.alarm)
        signal.setitimer(signal.ITIMER_REAL, max(.000001, self.seconds - (time.monotonic() - self.started)))
        return self

    @staticmethod
    def alarm(*_):
        raise LimitStop('wall-clock lease exhausted')

    def __exit__(self, *_):
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, self.old_alarm)
        for handle in self.handles:
            handle.remove()


def initialize(output, inputs, limits=None):
    limits = dict(DEFAULT_LIMITS if limits is None else limits)
    if any(not isinstance(limits[k], (int, float)) or not math.isfinite(limits[k]) or limits[k] <= 0
           for k in ('seconds', 'calls', 'memory_bytes', 'experiments', 'calibration_units')):
        raise ValueError('limits must be finite and positive')
    if any(type(limits[k]) is not int for k in ('calls', 'memory_bytes', 'experiments', 'calibration_units', 'external_calls')):
        raise ValueError('call, memory, experiment and unit limits must be integers')
    if limits['external_calls'] != 0 or limits['experiments'] != 2 or limits['calibration_units'] != 6:
        raise ValueError('protocol requires zero external calls, two pending experiments, six calibration units')
    manifest = population(inputs)
    with locked(output):
        if (output / 'journal.sqlite3').exists():
            raise ValueError('already initialized; use calibrate to resume')
        journal = Journal(output)
        try:
            journal.initialize(manifest, limits)
            export(journal)
        finally:
            journal.close()
    return {'status': 'initialized', 'windows': dict(Counter(r['split'] for r in manifest['windows'])),
            'calibration_units': manifest['calibration_units']}


def load_case(manifest, inputs, case_id):
    row = next(r for r in manifest['windows'] if r['case_id'] == case_id)
    raw = (inputs / 'corpus' / (row['task'] + '.txt')).read_bytes()
    chunk = raw[row['source_offset']:row['source_offset'] + 160]
    if hashlib.sha256(chunk).hexdigest() != row['window_sha256']:
        raise ValueError('window hash mismatch')
    return row, [0, *(b + 5 for b in chunk)]


@torch.inference_mode()
def execute(item, manifest, inputs, target, draft, drafters, reference):
    row, tokens = load_case(manifest, inputs, item['case_id'])
    prompt, truth = tokens[:97], tokens[97:]
    result = {**item, 'task': row['task']}
    started = time.perf_counter_ns()
    if item['phase'] == 'oracle':
        output = uncached_greedy(target, prompt, 64)
        result.update(tokens=output, free_running_correct=sum(a == b for a, b in zip(output, truth)),
                      exact_continuation=output == truth, truth=truth)
    elif item['phase'] == 'quality':
        net = target if item['method'] == 'target' else draft
        logits, _ = net(torch.tensor([tokens[:-1]]))
        logits = logits[0, 96:]
        nll = float(F.cross_entropy(logits, torch.tensor(truth), reduction='sum'))
        result.update(nll_sum=nll, correct=int((logits.argmax(-1) == torch.tensor(truth)).sum()),
                      tokens_scored=len(truth), bits_per_byte=nll / len(truth) / math.log(2))
    else:
        spec = next(v for v in configurations() if v['name'] == item['method'])
        policy = AcceptancePolicy() if spec['policy'] else None
        got = decode(target, prompt, 64, drafters.get(spec['family']),
                     2 if spec['family'] else 0, audit=item['phase'] == 'audit', policy=policy)
        # Stop timer before serialization, equality check, and journal I/O.
        result['elapsed_ns'] = time.perf_counter_ns() - started
        if got.tokens != reference:
            raise AssertionError(f"uncached greedy mismatch: {item['work_id']}: {got.tokens} != {reference}")
        result.update(asdict(got), agreement=True, policy_trace=policy.trace if policy else [])
    result.setdefault('elapsed_ns', time.perf_counter_ns() - started)
    return result


def calibrate(output, inputs, deadline, max_units=None, stop_after_records=None):
    """Run only the immutable calibration queue. Resuming never raises caps."""
    with locked(output):
        journal = Journal(output)
        old_limit = resource.getrlimit(resource.RLIMIT_AS)
        old_signals = {}
        try:
            journal.recover()
            limits = journal.meta('limits')
            ceiling = min(limits['memory_bytes'], old_limit[0]) if old_limit[0] != resource.RLIM_INFINITY else limits['memory_bytes']
            resource.setrlimit(resource.RLIMIT_AS, (int(ceiling), old_limit[1]))
            def interrupt(signum, _):
                raise LimitStop(f'interrupted by signal {signum}')
            for sig in (signal.SIGTERM, signal.SIGINT):
                old_signals[sig] = signal.signal(sig, interrupt)
            completed_units, records = set(), 0
            loaded_seed = None
            recovery_warmed = set()
            warmed_here = defaultdict(set)
            target = draft = drafters = None
            manifest = journal.meta('manifest')
            reason = 'calibration_complete'
            if not journal.pending():
                export(journal)
                return status(journal)
            with journal.db:
                journal.event('invocation', {'execution_deadline_utc': datetime.fromtimestamp(deadline, timezone.utc).isoformat(),
                              'max_units': max_units, 'stop_after_records': stop_after_records})
            # Input verification is part of the cumulative execution budget.
            meter = journal.begin(None, deadline)
            try:
                with meter:
                    if population(inputs) != manifest:
                        raise ValueError('immutable population/input identity changed')
                    configure(1)
                    environment_id = journal.record_environment()
                journal.finish(meter)
            except BaseException as exc:
                journal.fail(meter, exc)
                raise
            for item in journal.pending():
                unit = (item['seed'], item['case_id'])
                if max_units is not None and unit not in completed_units and len(completed_units) >= max_units:
                    reason = 'requested_pause_between_units'
                    break
                if stop_after_records is not None and records >= stop_after_records:
                    reason = 'requested_pause_between_records'
                    break
                if loaded_seed != item['seed']:
                    meter = journal.begin(None, deadline)
                    try:
                        with meter:
                            target = draft = drafters = None
                            target, draft, data, _ = load_text(inputs / f"seed{item['seed']}", inputs / 'corpus')
                            drafters = {'prompt': PromptLookup(), 'ngram': NGram([r['tokens'] for r in data['train']]),
                                        'gru': GRUDraft(draft)}
                            loaded_seed = item['seed']
                        journal.finish(meter)
                    except BaseException as exc:
                        journal.fail(meter, exc)
                        raise
                oracle_id = f"s{item['seed']}/{item['case_id']}/oracle/0/uncached"
                oracle = journal.db.execute('SELECT data FROM results WHERE work_id=?', (oracle_id,)).fetchone()
                reference = json.loads(oracle[0])['tokens'] if oracle else None
                # A process restart loses CPU/model warmth. Do not silently time
                # a cold continuation of a partly completed balanced block.
                # These extra decodes are separately recorded and budgeted.
                if item['phase'] == 'timing' and unit not in recovery_warmed:
                    if len(warmed_here[unit]) != 14:
                        for repeat in range(2):
                            for spec in configurations():
                                recovery = {**item, 'phase': 'warmup', 'method': spec['name'], 'repeat': repeat}
                                meter = journal.begin(None, deadline)
                                try:
                                    with meter:
                                        meter.attach(target, draft)
                                        result = execute(recovery, manifest, inputs, target, draft, drafters, reference)
                                    result.update(target_calls=meter.target_calls, draft_calls=meter.draft_calls,
                                                  environment_id=environment_id, attempt=meter.attempt,
                                                  resumed_work_id=item['work_id'], work_id=None)
                                    with journal.db:
                                        journal.event('resume_warmup', result)
                                    journal.finish(meter)
                                except BaseException as exc:
                                    journal.fail(meter, exc)
                                    raise
                    recovery_warmed.add(unit)
                meter = journal.begin(item['work_id'], deadline)
                try:
                    with meter:
                        meter.attach(target, draft)
                        result = execute(item, manifest, inputs, target, draft, drafters, reference)
                    result.update(target_calls=meter.target_calls, draft_calls=meter.draft_calls,
                                  peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                                  previous_checkpoint_ns=journal.checkpoint_ns, environment_id=environment_id)
                    journal.finish(meter, result)
                except BaseException as exc:
                    journal.fail(meter, exc)
                    raise
                records += 1
                if item['phase'] == 'warmup':
                    warmed_here[unit].add((item['method'], item['repeat']))
                completed_units.add(unit)
                if item['phase'] == 'timing' and item['repeat'] == 6 and item['position'] == 6:
                    print(f"saved {unit[0]}/{unit[1]}: {journal.usage()['seconds']:.2f}s, {journal.usage()['calls']} calls", flush=True)
            with journal.db:
                journal.event('stopped', {'reason': reason, 'usage': journal.usage(), 'records_this_invocation': records})
        except (LimitStop, MemoryError) as exc:
            with journal.db:
                journal.event('stopped', {'reason': f'{type(exc).__name__}: {exc}', 'usage': journal.usage()})
        finally:
            for sig, handler in old_signals.items():
                signal.signal(sig, handler)
            resource.setrlimit(resource.RLIMIT_AS, old_limit)
            export(journal)
            answer = status(journal)
            journal.close()
        return answer


def status(journal):
    rows = journal.results()
    remaining = journal.pending()
    return {'calibration_complete': not remaining, 'completed_records': len(rows),
            'pending_calibration_records': len(remaining), 'usage': journal.usage(),
            'limits': journal.meta('limits'), 'completed_full_experiments': 0,
            'full_evaluation_status': 'pending; resource authorization required',
            'next_calibration_work': remaining[0] if remaining else None}


def extrapolate(journal):
    manifest, results = journal.meta('manifest'), journal.results()
    if journal.pending():
        return {'status': 'incomplete_calibration', 'remaining_records': len(journal.pending()),
                'full_evaluation': 'not started', 'no_population_reduction': True}
    counts = Counter(r['task'] for r in manifest['windows'])
    by_unit = defaultdict(list)
    for row in results:
        by_unit[row['seed'], row['task']].append(row)
    phases = {}
    for phase in ('oracle', 'quality', 'audit', 'warmup', 'timing'):
        seconds = target_calls = draft_calls = 0
        for (seed, task), rows in by_unit.items():
            # Oracles and quality shared across experiments; audits/warmups/timings repeated.
            multiplier = counts[task] * (1 if phase in ('oracle', 'quality') else 2)
            selected = [r for r in rows if r['phase'] == phase]
            seconds += sum(r['elapsed_ns'] for r in selected) / 1e9 * multiplier
            target_calls += sum(r['target_calls'] for r in selected) * multiplier
            draft_calls += sum(r['draft_calls'] for r in selected) * multiplier
        phases[phase] = {'seconds': seconds, 'target_calls': target_calls, 'draft_calls': draft_calls}
    outputs = sum(counts.values()) * 2
    work_count = outputs * (1 + 2 + 2 * (7 + 14 + 49))
    checkpoint_samples = [r['previous_checkpoint_ns'] for r in results]
    # Include both start and completion transactions plus setup/controller residual.
    measured_compute_seconds = sum(r['elapsed_ns'] for r in results) / 1e9
    usage = journal.usage()
    overhead_seconds = max(0, usage['seconds'] - measured_compute_seconds)
    overhead_per_record = overhead_seconds / len(results)
    total = sum(r['seconds'] for r in phases.values()) + work_count * overhead_per_record
    calls = sum(r['target_calls'] + r['draft_calls'] for r in phases.values())
    # Both branches remain unmeasured. Bound the decoding path independent of acceptance:
    # <=64 verification blocks +1 prefill; audit <=64 more target calls;
    # at most 4 GRU calls for each of <=63 proposal blocks.
    worst_target = outputs * (64 + 1 + 2 * (7 * 129 + 9 * 7 * 65))
    worst_draft = outputs * (1 + 2 * (2 * 10 * 63 * 4))
    paired = []
    for (seed, task), rows in sorted(by_unit.items()):
        baseline = {r['repeat']: r['elapsed_ns'] for r in rows if r['phase'] == 'timing' and r['method'] == 'greedy'}
        for method in (v['name'] for v in configurations()):
            samples = [r for r in rows if r['phase'] == 'timing' and r['method'] == method]
            paired.append({'seed': seed, 'task': task, 'method': method,
                           'median_latency_ms': statistics.median(r['elapsed_ns'] for r in samples)/1e6,
                           'median_paired_speedup': statistics.median(baseline[r['repeat']]/r['elapsed_ns'] for r in samples)})
    return {'status': 'calibration_complete_full_evaluation_pending', 'population_counts': dict(counts),
            'full_case_seed_units': outputs, 'two_experiment_timed_outputs': outputs * 2 * 49,
            'two_experiment_warmup_outputs': outputs * 2 * 14,
            'two_experiment_audited_decodes': outputs * 2 * 7,
            'shared_oracle_decodes': outputs, 'shared_quality_passes': outputs * 2,
            'phase_extrapolation': phases, 'estimated_total_seconds': total,
            'estimated_total_calls': calls, 'estimated_controller_and_setup_seconds': work_count * overhead_per_record,
            'shortfall_vs_7200_seconds': max(0, total - 7200),
            'shortfall_vs_2000000_calls': max(0, calls - 2000000),
            'observed_peak_rss_bytes': usage['peak_rss_bytes'],
            'median_result_commit_ns': statistics.median(checkpoint_samples),
            'branch_requirements': {branch: {'analogue_seconds': total, 'analogue_calls': calls,
                'conservative_forward_call_upper_bound': worst_target + worst_draft,
                'runtime_bound': None, 'memory_bound': journal.meta('limits')['memory_bytes'],
                'status': 'branch not executed; cost uses measured experiment-1 policy as an analogue, not a guaranteed forecast'}
                for branch in ('fallback', 'strict')},
            'calibration_paired_ratios': paired,
            'limitations': ['One validation window per source and seed is cost calibration, not a population speed estimate.',
                'Two experiment-2 branches are predeclared but not executed or selected.',
                'Runtime extrapolation assumes within-source calibration costs represent all windows; no confidence bound is justified.',
                'Per-forward hooks and durable orchestration affect timing; all seven methods use identical accounting hooks.',
                'Shared CPU host, tiny byte models, and three documents do not establish useful language generation or pretrained-model acceleration.']}


def export(journal):
    started = time.perf_counter_ns()
    atomic_json(journal.output / 'population.json', journal.meta('manifest'))
    atomic_json(journal.output / 'proposal-policy.json', journal.meta('policy'))
    atomic_json(journal.output / 'state.json', status(journal))
    atomic_json(journal.output / 'extrapolation.json', extrapolate(journal))
    # Atomic derived export; authoritative SQLite records are never rewritten.
    path = journal.output / 'measurements.jsonl'
    temporary = path.with_suffix('.tmp')
    with temporary.open('w') as stream:
        for row in journal.results():
            stream.write(canonical(row) + '\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    with journal.db:
        journal.event('exported', {'elapsed_ns': time.perf_counter_ns() - started,
                      'scope': 'Derived JSON views, outside inference execution budget; SQLite is the durable checkpoint.'})


def verify(output, inputs):
    """No inference or new timings: verify only changed-controller evidence."""
    with locked(output, shared=True):
        journal = Journal(output)
        try:
            if journal.db.execute('PRAGMA integrity_check').fetchone()[0] != 'ok':
                raise AssertionError('journal integrity failure')
            manifest = journal.meta('manifest')
            if population(inputs) != manifest or journal.meta('policy') != proposal_policy():
                raise AssertionError('population or frozen proposal changed')
            rows = journal.results()
            expected = {r['work_id']: r for r in work_items(manifest)}
            references = {}
            positions = defaultdict(list)
            for row in rows:
                if any(row[k] != v for k, v in expected[row['work_id']].items()):
                    raise AssertionError('unexpected operation metadata')
                unit = row['seed'], row['case_id']
                if row['elapsed_ns'] <= 0 or row['target_calls'] + row['draft_calls'] <= 0:
                    raise AssertionError('invalid measured cost')
                if row['phase'] == 'oracle':
                    references[unit] = row['tokens']
                    if row['target_calls'] != len(row['tokens']) or row['draft_calls']:
                        raise AssertionError('oracle call accounting mismatch')
                if row['phase'] in ('audit', 'warmup', 'timing'):
                    if not row['agreement'] or row['tokens'] != references[unit]:
                        raise AssertionError('exact output mismatch')
                    if row['target_calls'] < row['forward_calls']:
                        raise AssertionError('missing target calls')
                    if row['phase'] == 'audit' and row['target_calls'] != 2 * row['forward_calls'] - 1:
                        raise AssertionError('audit calls not fully counted')
                    if row['phase'] != 'audit' and row['target_calls'] != row['forward_calls']:
                        raise AssertionError('target call mismatch')
                    if row['method'].startswith('gru') != (row['draft_calls'] > 0):
                        raise AssertionError('draft call mismatch')
                if row['phase'] == 'timing':
                    positions[unit, row['method']].append(row['position'])
            if not journal.pending() and any(sorted(p) != list(range(7)) for p in positions.values()):
                raise AssertionError('unbalanced timing positions')
            for name, value in (('population', manifest), ('proposal-policy', journal.meta('policy')),
                                ('state', status(journal)), ('extrapolation', extrapolate(journal))):
                if json.loads((output / (name + '.json')).read_text()) != value:
                    raise AssertionError(f'stale {name} export')
            exported = [json.loads(line) for line in (output / 'measurements.jsonl').read_text().splitlines()]
            if exported != rows:
                raise AssertionError('measurement export mismatch')
            return {'verified': True, 'records': len(rows), 'timed_outputs': sum(r['phase'] == 'timing' for r in rows),
                    'calibration_complete': not journal.pending(), 'full_experiments_completed': 0,
                    'population_windows': len(manifest['windows'])}
        finally:
            journal.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    for command in ('init', 'calibrate', 'status', 'verify'):
        cmd = sub.add_parser(command)
        cmd.add_argument('--output', type=Path, default=Path('results/controller-calibration'))
        if command != 'status':
            cmd.add_argument('--inputs', type=Path, default=Path('results/realtext-gru'))
        if command == 'calibrate':
            cmd.add_argument('--deadline', required=True, help='absolute UTC ISO timestamp; execution stops two minutes before it')
            cmd.add_argument('--max-units', type=int)
            cmd.add_argument('--stop-after-records', type=int)
    args = parser.parse_args(argv)
    try:
        if args.command == 'init':
            answer = initialize(args.output, args.inputs)
        elif args.command == 'verify':
            answer = verify(args.output, args.inputs)
        elif args.command == 'status':
            with locked(args.output, shared=True):
                journal = Journal(args.output)
                try:
                    answer = status(journal)
                finally:
                    journal.close()
        else:
            deadline = datetime.fromisoformat(args.deadline)
            if deadline.tzinfo is None:
                raise ValueError('deadline must include UTC offset')
            if args.max_units is not None and args.max_units < 1 or args.stop_after_records is not None and args.stop_after_records < 1:
                raise ValueError('pause counts must be positive')
            answer = calibrate(args.output, args.inputs, deadline.timestamp() - 120, args.max_units, args.stop_after_records)
        print(json.dumps(answer, indent=2, sort_keys=True))
    except (ValueError, AssertionError, OSError, sqlite3.Error) as exc:
        parser.exit(1, f'error: {exc}\n')


if __name__ == '__main__':
    main()
