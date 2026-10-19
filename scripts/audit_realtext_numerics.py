"""Measure cached/full-prefix FP32 drift without relaxing exact token checks.

This diagnostic replaces only torch.testing.assert_close with a recorder, while
leaving decode's exact argmax check and per-method uncached-output checks active.
It can inspect partial runs with saved checkpoints, before a manifest exists.
"""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
from unittest.mock import patch

import torch

from specdecode.decode import decode, uncached_greedy
from specdecode.experiment import configure, sha256, write_json
from specdecode.model import ModelConfig, TinyTransformer
from specdecode.realtext import TextConfig, make_data, read_gzip, variants
from specdecode.recurrent import ByteGRU


def audit(path, corpus):
    cfg = TextConfig(**json.loads((path/'config.json').read_text()))
    configure(cfg.threads)
    data = read_gzip(path/'dataset.json.gz')
    assert data == make_data(corpus, cfg)
    saved = torch.load(path/'target.pt', map_location='cpu', weights_only=True)
    expected = ModelConfig(vocab_size=261, context=cfg.sequence_bytes+1, width=cfg.width)
    assert saved['model_config'] == asdict(expected)
    target = TinyTransformer(expected)
    target.load_state_dict(saved['state_dict'])
    target.eval()
    saved = torch.load(path/'draft.pt', map_location='cpu', weights_only=True)
    assert saved['width'] == cfg.draft_width
    draft = ByteGRU(width=cfg.draft_width)
    draft.load_state_dict(saved['state_dict'])
    draft.eval()
    report = {'seed': cfg.seed, 'target_sha256': sha256(path/'target.pt'),
              'draft_sha256': sha256(path/'draft.pt'), 'rtol': 1e-4,
              'logit_comparisons': 0, 'forward_block_comparisons': 0,
              'elements_exceeding_atol_1e_5': 0, 'elements_exceeding_atol_2e_5': 0,
              'max_absolute_error': 0., 'max_required_atol_after_rtol': 0.,
              'minimum_reference_winning_margin': None,
              'case_method_audits': 0}
    label, block = {}, 0

    def record(actual, reference, **kwargs):
        nonlocal block
        block += 1
        difference = (actual-reference).abs()
        excess = difference - 1e-4*reference.abs()
        report['logit_comparisons'] += difference.numel()
        report['forward_block_comparisons'] += 1
        report['elements_exceeding_atol_1e_5'] += int((excess > 1e-5).sum())
        report['elements_exceeding_atol_2e_5'] += int((excess > 2e-5).sum())
        for key, values in (('max_absolute_error', difference), ('max_required_atol_after_rtol', excess)):
            value = float(values.max())
            if value > report[key]:
                flat = int(values.flatten().argmax())
                report[key] = value
                report[key+'_location'] = {**label, 'block': block,
                    'block_row': flat//261, 'token': flat%261,
                    'actual': float(actual.flatten()[flat]), 'reference': float(reference.flatten()[flat])}
        top = reference.topk(2, dim=-1).values
        margins = top[..., 0]-top[..., 1]
        margin = float(margins.min())
        if report['minimum_reference_winning_margin'] is None or margin < report['minimum_reference_winning_margin']:
            index = int(margins.flatten().argmin())
            report['minimum_reference_winning_margin'] = margin
            report['minimum_margin_location'] = {**label, 'block': block, 'block_row': index,
                'row_max_absolute_error': float(difference[0, index].max()),
                'reference_winner': int(reference[0, index].argmax()),
                'actual_winner': int(actual[0, index].argmax())}
        if actual.argmax(-1).tolist() != reference.argmax(-1).tolist():
            raise AssertionError(f'argmax mismatch: {label}')

    with torch.inference_mode(), patch('torch.testing.assert_close', side_effect=record):
        for case in data['eval']:
            prompt = case['tokens'][:case['prompt_length']]
            ref = uncached_greedy(target, prompt, cfg.max_new_tokens)
            for name, drafter, length in variants(data, cfg, draft):
                label, block = {'case': case['id'], 'method': name}, 0
                actual = decode(target, prompt, cfg.max_new_tokens, drafter, length, audit=True)
                if actual.tokens != ref:
                    raise AssertionError(f'output mismatch: {label}')
                report['case_method_audits'] += 1
    report['exact_argmax_and_output_agreement'] = True
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('experiments', type=Path, nargs='+')
    parser.add_argument('--corpus', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('output exists')
    reports = []
    for path in args.experiments:
        report = audit(path, args.corpus)
        print(json.dumps(report, indent=2), flush=True)
        reports.append(report)
    write_json(args.output, reports)


if __name__ == '__main__':
    main()
