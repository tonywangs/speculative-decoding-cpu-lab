"""Offline, source-pinned real-byte continuation experiment with a GRU draft."""
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import gzip
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
from time import perf_counter_ns

import torch
import torch.nn.functional as F

from .decode import decode
from .draft import NGram, PromptLookup
from .experiment import (configure, correctness, environment, measure, prepare_output,
                         sha256, summarize, train, validate_measurements, write_json)
from .model import ModelConfig, TinyTransformer
from .recurrent import ByteGRU, GRUDraft

TASKS = ('dialogue', 'prose', 'python')


@dataclass(frozen=True)
class TextConfig:
    seed: int = 17
    data_seed: int = 20261019
    steps: int = 1200
    draft_steps: int = 1200
    batch_size: int = 32
    learning_rate: float = 0.003
    width: int = 96
    draft_width: int = 32
    train_per_task: int = 512
    eval_per_task: int = 12
    validation_per_task: int = 12
    sequence_bytes: int = 160
    prompt_bytes: int = 96
    max_new_tokens: int = 64
    threads: int = 1
    warmups: int = 2
    repeats: int = 7
    draft_lengths: tuple[int, ...] = (2, 4)
    balanced_order: bool = True

    def __post_init__(self):
        for key in ('steps', 'draft_steps', 'batch_size', 'width', 'draft_width',
                    'train_per_task', 'eval_per_task', 'validation_per_task',
                    'sequence_bytes', 'prompt_bytes', 'max_new_tokens', 'threads', 'warmups', 'repeats'):
            if getattr(self, key) < 1:
                raise ValueError(f'{key} must be positive')
        if self.width % 4 or self.prompt_bytes + self.max_new_tokens != self.sequence_bytes:
            raise ValueError('width must be divisible by four; prompt + continuation must equal sequence bytes')
        if not self.draft_lengths or any(k < 1 for k in self.draft_lengths) or len(set(self.draft_lengths)) != len(self.draft_lengths):
            raise ValueError('unique positive draft lengths required')
        if self.repeats % (1 + 3 * len(self.draft_lengths)) or not self.balanced_order:
            raise ValueError('complete balanced timing cycles required')
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError('positive finite learning rate required')


def read_sources(corpus):
    corpus = Path(corpus)
    metadata = json.loads((corpus / 'sources.json').read_text())
    if set(metadata) != {*TASKS, 'python-license'}:
        raise ValueError('incomplete source manifest')
    for name, info in metadata.items():
        source = corpus / f'{name}.txt'
        if sha256(source) != info['sha256'] or source.stat().st_size != info['bytes']:
            raise ValueError(f'source integrity mismatch: {name}')
    sources = {}
    for task in TASKS:
        data = (corpus / f'{task}.txt').read_bytes()
        if hashlib.sha256(data).hexdigest() != metadata[task]['sha256'] or len(data) != metadata[task]['bytes']:
            raise ValueError(f'source integrity mismatch: {task}')
        # Preserve raw bytes on disk, but exclude Gutenberg's legal wrapper from
        # model inputs. The saved bounds make the exact selection inspectable.
        start, end = 0, len(data)
        if task == 'prose':
            marker = b'*** START OF THE PROJECT GUTENBERG EBOOK'
            tail = b'*** END OF THE PROJECT GUTENBERG EBOOK'
            pos = data.index(marker)
            start = data.index(b'\n', pos) + 1
            end = data.index(tail, start)
        sources[task] = (data, start, end)
    return sources, metadata


def make_data(corpus, cfg):
    sources, metadata = read_sources(corpus)
    data = {'sources': metadata, 'bounds': {}, 'train': [], 'validation': [], 'eval': []}
    # Reject duplicated held-out prompts, including verbatim matches anywhere in
    # the source's entire training region, not just selected training windows.
    seen = set()
    for task_index, task in enumerate(TASKS):
        raw, start, end = sources[task]
        a, b = start + (end-start)*8//10, start + (end-start)*9//10
        bounds = {'train': [start, a], 'validation': [a, b], 'eval': [b, end]}
        data['bounds'][task] = bounds
        train_region = raw[start:a]
        for split_index, (split, count) in enumerate((('train', cfg.train_per_task),
                     ('validation', cfg.validation_per_task), ('eval', cfg.eval_per_task))):
            lo, hi = bounds[split]
            offsets = list(range(lo, hi - cfg.sequence_bytes + 1, cfg.sequence_bytes))
            random.Random(cfg.data_seed + 100*task_index + split_index).shuffle(offsets)
            selected = []
            for offset in offsets:
                chunk = raw[offset:offset+cfg.sequence_bytes]
                prefix = chunk[:cfg.prompt_bytes]
                if len(set(prefix)) < 12 or prefix in seen or (split != 'train' and prefix in train_region):
                    continue
                seen.add(prefix)
                selected.append({'id': f'{split}-{task}-{offset:07d}', 'task': task,
                                 'source_offset': offset, 'tokens': [0, *(v+5 for v in chunk)],
                                 'prompt_length': cfg.prompt_bytes + 1})
                if len(selected) == count:
                    break
            if split != 'train' and len(selected) != count:
                raise ValueError(f'not enough distinct held-out windows: {task}/{split}')
            if not selected:
                raise ValueError(f'empty split: {task}/{split}')
            data[split].extend(selected)
    return data


def write_gzip(path, value):
    path.write_bytes(gzip.compress(json.dumps(value, sort_keys=True, separators=(',', ':')).encode(), mtime=0))


def read_gzip(path):
    return json.loads(gzip.decompress(path.read_bytes()))


def train_draft(data, cfg):
    start = perf_counter_ns()
    seed = cfg.seed + 1000
    torch.manual_seed(seed)
    model = ByteGRU(width=cfg.draft_width)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=0.01)
    generator = torch.Generator().manual_seed(seed + 2)
    tokens = torch.tensor([r['tokens'] for r in data['train']])
    trace = []
    loop_start = perf_counter_ns()
    for step in range(cfg.draft_steps):
        batch = tokens[torch.randint(len(tokens), (cfg.batch_size,), generator=generator)]
        optimizer.zero_grad(set_to_none=True)
        logits, _ = model(batch[:, :-1])
        loss = F.cross_entropy(logits.reshape(-1, 261), batch[:, 1:].reshape(-1))
        if not torch.isfinite(loss):
            raise ValueError('nonfinite draft loss')
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step == 0 or (step+1) % 100 == 0 or step+1 == cfg.draft_steps:
            trace.append({'step': step+1, 'batch_loss': float(loss.detach()), 'elapsed_ns': perf_counter_ns()-loop_start})
    info = {'elapsed_ns': perf_counter_ns()-loop_start, 'setup_and_training_ns': perf_counter_ns()-start,
            'initialization_seed': seed, 'batch_sampling_seed': seed+2,
            'parameter_count': sum(p.numel() for p in model.parameters()), 'trace': trace,
            'optimizer': 'AdamW', 'learning_rate': cfg.learning_rate, 'weight_decay': 0.01,
            'betas': [0.9, 0.999], 'eps': 1e-8, 'gradient_clip_norm': 1.0,
            'objective': 'teacher-forced next-byte cross entropy on training windows only',
            'batch_sampling': 'uniform windows with replacement; no distillation or held-out updates'}
    return model.eval(), info


def variants(data, cfg, draft):
    return [('greedy', None, 0)] + [(f'{name}-k{k}', proposer, k)
        for name, proposer in (('prompt', PromptLookup()),
                              ('ngram', NGram([r['tokens'] for r in data['train']])),
                              ('gru', GRUDraft(draft))) for k in cfg.draft_lengths]


@torch.inference_mode()
def quality(model, rows):
    output = []
    for task in TASKS:
        loss, correct, total = 0., 0, 0
        for row in rows:
            if row['task'] != task:
                continue
            p = row['prompt_length']
            logits, _ = model(torch.tensor([row['tokens'][:-1]]))
            logits = logits[0, p-1:]
            truth = torch.tensor(row['tokens'][p:])
            loss += float(F.cross_entropy(logits, truth, reduction='sum'))
            correct += int((logits.argmax(-1) == truth).sum())
            total += len(truth)
        output.append({'task': task, 'tokens': total, 'accuracy': correct/total, 'nll': loss/total,
                       'bits_per_byte': loss/total/math.log(2)})
    return output


def summary(raw, checks):
    rows, accuracy = summarize(raw, checks, TASKS)
    baseline = {(r['id'], r['repeat']): r['elapsed_ns'] for r in raw if r['method'] == 'greedy'}
    for row in rows:
        groups = {}
        for r in raw:
            if r['task'] == row['task'] and r['method'] == row['method']:
                groups.setdefault(r['id'], []).append(baseline[r['id'], r['repeat']]/r['elapsed_ns'])
        medians = [statistics.median(groups[k]) for k in sorted(groups)]
        rng = random.Random(314159)
        draws = sorted(statistics.median(rng.choices(medians, k=len(medians))) for _ in range(2000))
        row['case_median_speedup'] = statistics.median(medians)
        row['case_bootstrap_95'] = [draws[49], draws[1949]]
    return rows, accuracy


def markdown(report):
    lines = ['# Real-text CPU draft comparison', '',
             f"Seed {report['config']['seed']}; {report['evaluation_cases']} held-out inputs; "
             f"{report['timed_samples']} timed outputs, all exactly equal to uncached greedy.", '',
             f"CPU: {report['environment']['cpu_model']}; PyTorch {report['environment']['torch']}; "
             f"{report['environment']['intraop_threads']} intra-op thread(s), one inter-op thread, FP32.", '',
             'Primary speedup: median across cases of median paired greedy/method ratios. '
             '95% intervals resample whole cases (2,000 draws, seed 314159), conditional on this target and text. '
             'Seven repetitions balance every method across every timing position. Values below 1 are slowdowns.', '',
             '| Task | Method | Median ms | Speedup | Conditional 95% | Acceptance | Draft ms |',
             '|---|---|---:|---:|---|---:|---:|']
    for r in report['summary']:
        acc = '—' if r['acceptance_rate'] is None else f"{r['acceptance_rate']:.1%}"
        lo, hi = r['case_bootstrap_95']
        lines.append(f"| {r['task']} | {r['method']} | {r['median_latency_ms']:.3f} | "
                     f"{r['case_median_speedup']:.3f}× | [{lo:.3f}, {hi:.3f}] | {acc} | {r['median_draft_ms']:.3f} |")
    lines += ['', '| Task | Target next-byte accuracy | Free-running accuracy | Exact continuation |',
              '|---|---:|---:|---:|']
    for r in report['accuracy']:
        lines.append(f"| {r['task']} | {r['teacher_forced_token_accuracy']:.1%} | "
                     f"{r['free_running_token_accuracy']:.1%} | {r['exact_continuation_accuracy']:.1%} |")
    lines += ['', f"Target training: {report['target_training']['elapsed_ns']/1e9:.3f} s; "
              f"{report['target_training']['parameter_count']:,} parameters. "
              f"GRU training: {report['draft_training']['elapsed_ns']/1e9:.3f} s; "
              f"{report['draft_training']['parameter_count']:,} parameters. "
              f"N-gram/variant construction: {report['draft_fit_ns']/1e9:.3f} s.", '',
              'Timing includes request state reset, target and draft prefill, proposal generation, verification, '
              'rollback and Python overhead. Training, fitting, audits and warmups are excluded. '
              'Full traces, validation/test quality, counters and environment are in report.json; '
              'timings.jsonl.gz retains every paired sample.', '',
              'These are real source excerpts but tiny locally trained byte models, not capable language assistants. '
              'A repetitive or incorrect target continuation can be easy to draft. Accuracy is separate from exact '
              'decoding agreement. One source per genre and a shared CPU host limit generalization; repeated timings '
              'are not independent model or corpus trials. No sampling, large-model or production-serving claim.', '']
    return '\n'.join(lines)


def run_text(output, corpus, cfg):
    output = prepare_output(output)
    configure(cfg.threads)
    data = make_data(corpus, cfg)
    write_json(output/'config.json', asdict(cfg))
    write_gzip(output/'dataset.json.gz', data)
    print(f'Training target on {len(data["train"])} real-text windows...', flush=True)
    model_cfg = ModelConfig(vocab_size=261, context=cfg.sequence_bytes+1, width=cfg.width)
    model, training = train(data, cfg, model_cfg)
    torch.save({'model_config': asdict(model_cfg), 'state_dict': model.state_dict(),
                'training': training}, output/'target.pt')
    print('Training GRU on the same training split...', flush=True)
    draft, draft_training = train_draft(data, cfg)
    torch.save({'width': cfg.draft_width, 'state_dict': draft.state_dict(),
                'training': draft_training}, output/'draft.pt')
    start = perf_counter_ns()
    methods = variants(data, cfg, draft)
    fit_ns = perf_counter_ns()-start
    print('Auditing all methods against uncached greedy...', flush=True)
    refs, checks = correctness(model, data, cfg, methods)
    print('Running balanced paired timings...', flush=True)
    raw = measure(model, data, cfg, methods, refs)
    validate_measurements(raw, checks, cfg, {m[0] for m in methods})
    rows, accuracy = summary(raw, checks)
    report = {'created_utc': datetime.now(timezone.utc).isoformat(), 'config': asdict(cfg),
              'environment': environment(), 'agreement_passed': True,
              'evaluation_cases': len(checks), 'timed_samples': len(raw),
              'dataset_sha256': sha256(output/'dataset.json.gz'),
              'target_sha256': sha256(output/'target.pt'), 'draft_sha256': sha256(output/'draft.pt'),
              'target_training': training, 'draft_training': draft_training, 'draft_fit_ns': fit_ns,
              'summary': rows, 'accuracy': accuracy,
              'target_validation': quality(model, data['validation']),
              'draft_validation': quality(draft, data['validation']),
              'draft_test': quality(draft, data['eval'])}
    write_gzip(output/'correctness.json.gz', checks)
    content = '\n'.join(json.dumps(r, sort_keys=True) for r in raw)+'\n'
    (output/'timings.jsonl.gz').write_bytes(gzip.compress(content.encode(), mtime=0))
    write_json(output/'report.json', report)
    (output/'RESULTS.md').write_text(markdown(report))
    write_json(output/'manifest.json', {p.name: sha256(p) for p in sorted(output.iterdir()) if p.name != 'manifest.json'})
    return report


def load_text(path, corpus):
    path = Path(path)
    manifest = json.loads((path/'manifest.json').read_text())
    required = {'config.json', 'dataset.json.gz', 'target.pt', 'draft.pt', 'correctness.json.gz',
                'timings.jsonl.gz', 'report.json', 'RESULTS.md'}
    if set(manifest) != required:
        raise ValueError('incomplete real-text manifest')
    for name, digest in manifest.items():
        if sha256(path/name) != digest:
            raise ValueError(f'artifact hash mismatch: {name}')
    cfg = TextConfig(**json.loads((path/'config.json').read_text()))
    configure(cfg.threads)
    data = read_gzip(path/'dataset.json.gz')
    if data != make_data(corpus, cfg):
        raise ValueError('dataset does not match source bytes/config')
    saved = torch.load(path/'target.pt', map_location='cpu', weights_only=True)
    expected = ModelConfig(vocab_size=261, context=cfg.sequence_bytes+1, width=cfg.width)
    if saved['model_config'] != asdict(expected):
        raise ValueError('target architecture mismatch')
    model = TinyTransformer(expected)
    model.load_state_dict(saved['state_dict'])
    saved = torch.load(path/'draft.pt', map_location='cpu', weights_only=True)
    if saved['width'] != cfg.draft_width:
        raise ValueError('draft architecture mismatch')
    draft = ByteGRU(width=cfg.draft_width)
    draft.load_state_dict(saved['state_dict'])
    return model.eval(), draft.eval(), data, cfg


def close(actual, expected):
    if isinstance(actual, float):
        if not math.isclose(actual, expected, rel_tol=1e-6, abs_tol=1e-6):
            raise AssertionError('replayed numerical metric differs')
    elif isinstance(actual, dict):
        if actual.keys() != expected.keys():
            raise AssertionError('replayed keys differ')
        for k in actual:
            close(actual[k], expected[k])
    elif isinstance(actual, list):
        for a, b in zip(actual, expected, strict=True):
            close(a, b)
    elif actual != expected:
        raise AssertionError('replayed correctness differs')


def verify_text(path, corpus):
    path = Path(path)
    model, draft, data, cfg = load_text(path, corpus)
    methods = variants(data, cfg, draft)
    _, checks = correctness(model, data, cfg, methods)
    old = read_gzip(path/'correctness.json.gz')
    close(checks, old)
    raw = [json.loads(line) for line in gzip.decompress((path/'timings.jsonl.gz').read_bytes()).splitlines()]
    validate_measurements(raw, old, cfg, {m[0] for m in methods})
    report = json.loads((path/'report.json').read_text())
    rows, accuracy = summary(raw, old)
    if rows != report['summary'] or accuracy != report['accuracy']:
        raise AssertionError('summary differs from raw evidence')
    if report['config'] != json.loads((path/'config.json').read_text()):
        raise AssertionError('report configuration mismatch')
    for key, file in (('dataset', 'dataset.json.gz'), ('target', 'target.pt'), ('draft', 'draft.pt')):
        if report[key+'_sha256'] != sha256(path/file):
            raise AssertionError('report artifact identity mismatch')
    if (report['evaluation_cases'], report['timed_samples'], report['agreement_passed']) != (len(checks), len(raw), True):
        raise AssertionError('report counts or agreement mismatch')
    for key, net, split in (('target_validation', model, 'validation'),
                            ('draft_validation', draft, 'validation'), ('draft_test', draft, 'eval')):
        close(quality(net, data[split]), report[key])
    if (path/'RESULTS.md').read_text() != markdown(report):
        raise AssertionError('readable report differs')
    return {'agreement_passed': True, 'cases': len(checks), 'timed_samples': len(raw),
            'source_splits_regenerated': True, 'summary_recomputed': True,
            'source_hashes_match': report['environment']['source_sha256'] == environment()['source_sha256']}


def text_tokens(tokens):
    chunks, pending = [], bytearray()
    for v in tokens:
        if v >= 5:
            pending.append(v-5)
        else:
            chunks.extend((pending.decode('utf-8', errors='replace'), f'<special:{v}>'))
            pending.clear()
    chunks.append(pending.decode('utf-8', errors='replace'))
    return ''.join(chunks)


def demo_text(path, corpus, case_index=0):
    model, draft, data, cfg = load_text(path, corpus)
    if not 0 <= case_index < len(data['eval']):
        raise ValueError('case index outside held-out inputs')
    case = data['eval'][case_index]
    prompt = case['tokens'][:case['prompt_length']]
    outputs = {name: decode(model, prompt, cfg.max_new_tokens, drafter, k).tokens
               for name, drafter, k in variants(data, cfg, draft)}
    if any(v != outputs['greedy'] for v in outputs.values()):
        raise AssertionError('demo greedy mismatch')
    return {'id': case['id'], 'task': case['task'], 'prompt': text_tokens(prompt[1:]),
            'target_completion': text_tokens(outputs['greedy']),
            'actual_source_continuation': text_tokens(case['tokens'][case['prompt_length']:]),
            'matching_methods': list(outputs), 'agreement_passed': True,
            'note': 'Tiny byte-model output; decoding agreement does not imply text quality.'}
