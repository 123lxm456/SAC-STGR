"""Synchronized forward latency on synthetic inputs, never an accuracy result."""

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src/sac_rwb'))
from algos.online_risk import create_risk_predictor, synchronize


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--repeats', type=int, default=20)
    parser.add_argument('--warmup', type=int, default=5)
    args = parser.parse_args()
    if args.repeats < 1 or args.warmup < 0:
        parser.error('repeats must be positive and warmup nonnegative')
    if args.output.exists():
        raise RuntimeError('Refusing to overwrite latency evidence')
    device = torch.device(args.device)
    report = {'device': str(device), 'torch': torch.__version__, 'cuda': torch.version.cuda,
              'purpose': 'Synthetic-input engineering cost only; no trained weights or accuracy claims',
              'definition': 'Synchronized model forward wall time; tensors already on device, no history assembly or host transfer; warmups excluded',
              'repeats': args.repeats, 'warmup': args.warmup, 'cases': []}
    if device.type == 'cuda':
        report['gpu'] = torch.cuda.get_device_name(device)
    for variant in ('b0', 'm1'):
        torch.manual_seed(0)
        model = create_risk_predictor(variant).to(device).eval()
        for batch in (1, 8, 32):
            for valid_steps in (1, 10):
                states = torch.randn(batch, 10, 98, device=device)
                actions = torch.randn(batch, 10, 1, device=device)
                mask = torch.zeros(batch, 10, dtype=torch.bool, device=device)
                mask[:, -valid_steps:] = True
                states[~mask] = 0
                actions[~mask] = 0
                durations = []
                with torch.no_grad():
                    for repetition in range(args.warmup + args.repeats):
                        synchronize(device)
                        started = time.perf_counter()
                        scores = model(states, actions, mask)
                        synchronize(device)
                        elapsed = 1000 * (time.perf_counter() - started)
                        if not bool(torch.isfinite(scores).all()):
                            raise RuntimeError('Nonfinite benchmark scores')
                        if repetition >= args.warmup:
                            durations.append(elapsed)
                report['cases'].append({'variant': variant, 'batch_size': batch,
                                        'valid_steps': valid_steps,
                                        'parameter_count': sum(p.numel() for p in model.parameters()),
                                        'mean_ms': float(np.mean(durations)),
                                        'p50_ms': float(np.percentile(durations, 50)),
                                        'p95_ms': float(np.percentile(durations, 95)),
                                        'samples_ms': durations})
        del model
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
