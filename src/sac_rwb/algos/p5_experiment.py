"""Shared P5 risk fitting, common-corpus evaluation and paired summaries."""

import hashlib
import json
import random
import time
from pathlib import Path
import numpy as np
import torch
from algos.p4_data import read_dataset, sha_file, score_metrics, fit_calibration, apply_calibration
from algos.online_risk import synchronize
from algos.p5_models import create_p5_predictor


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def write_json(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)


def make_directory(path, new=False):
    """Create ancestors one at a time (Windows UNC mkdir(parents) can fail)."""
    path = Path(path).absolute()
    current = Path(path.anchor)
    for part in path.parts[1:-1]:
        current = current / part
        if not current.exists():
            current.mkdir()
    path.mkdir(exist_ok=not new)


def corpus_fingerprint(directories):
    entries = []
    for directory in directories:
        directory = Path(directory)
        manifest = json.loads((directory / 'manifest.json').read_text())
        entries.append({'split': manifest['metadata']['split'],
                        'files': {name: sha_file(directory / name) for name in
                                  ('manifest.json', 'counts.json', 'samples.jsonl', 'events.jsonl')}})
    return hashlib.sha256(json.dumps(entries, sort_keys=True).encode()).hexdigest()


def load_corpus(directories):
    """Strict P4 physical-label audit precedes exclusion of censored windows."""
    chunks = {split: [] for split in ('train', 'validation', 'test')}
    for directory in directories:
        manifest, rows, events = read_dataset(directory)
        split = manifest['metadata']['split']
        labeled = [r for r in rows if r['label'] is not None]
        if not labeled:
            raise ValueError('Empty labeled collection')
        chunks[split].append((np.asarray([r['state_seq'] for r in labeled], np.float32),
                              np.asarray([r['action_seq'] for r in labeled], np.float32),
                              np.asarray([r['valid_mask'] for r in labeled], np.bool_),
                              np.asarray([r['label'] for r in labeled], np.float32)))
        del rows, events, labeled
    if any(not value for value in chunks.values()):
        raise ValueError('All three corpus splits required')
    return {split: tuple(np.concatenate([c[index] for c in value]) for index in range(4))
            for split, value in chunks.items()}


def balanced_update(model, batch, device):
    states, actions, masks, labels = batch
    model.train()
    tensors = [torch.as_tensor(x, device=device) for x in batch]
    predictions = model(*tensors[:3])
    loss = torch.nn.functional.binary_cross_entropy(predictions, tensors[3])
    if not torch.isfinite(loss):
        raise ValueError('Nonfinite risk training loss')
    model.optimizer.zero_grad()
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    if not torch.isfinite(norm):
        raise ValueError('Nonfinite risk training gradients')
    model.optimizer.step()
    return float(loss.detach().cpu())


def predict(model, data, device, batch_size=128):
    model.eval()
    scores = []
    with torch.no_grad():
        for start in range(0, len(data[0]), batch_size):
            tensors = [torch.as_tensor(x[start:start + batch_size], device=device) for x in data[:3]]
            scores.extend(model(*tensors).cpu().tolist())
    return np.asarray(scores)


def benchmark(model, device, repeats=20):
    model.eval()
    cases = []
    if torch.device(device).type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    with torch.no_grad():
        for size in (1, 8, 32):
            states = torch.zeros(size, 10, 98, device=device)
            actions = torch.zeros(size, 10, 1, device=device)
            mask = torch.ones(size, 10, dtype=torch.bool, device=device)
            durations = []
            for repeat in range(repeats + 5):
                synchronize(device)
                start = time.perf_counter()
                result = model(states, actions, mask)
                synchronize(device)
                elapsed_ms = (time.perf_counter() - start) * 1000
                if not torch.isfinite(result).all():
                    raise ValueError('Invalid benchmark prediction')
                if repeat >= 5:
                    durations.append(elapsed_ms)
            cases.append(dict(batch_size=size, p50_ms=float(np.median(durations)),
                              p95_ms=float(np.percentile(durations, 95)), samples_ms=durations))
    return dict(parameter_count=sum(p.numel() for p in model.parameters()), cases=cases,
                peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(device) if torch.device(device).type == 'cuda' else None,
                definition='Synchronized trained-model forward, synthetic tensors on device, 5 warmups excluded; not end-to-end or accuracy evidence. CUDA peak includes resident model/optimizer and worker tensors, not isolated model-only memory.')


def fit_offline(variant, seed, corpus, updates, batch_size, device, output, corpus_id=None):
    if updates < 1 or batch_size < 2 or batch_size % 2:
        raise ValueError('Positive updates and even batch size required')
    output = Path(output)
    make_directory(output, new=True)
    seed_all(seed)
    model = create_p5_predictor(variant).to(device)
    train = corpus['train']
    pools = [np.flatnonzero(train[3] == label) for label in (0, 1)]
    if min(map(len, pools)) < batch_size // 2:
        raise ValueError('Natural training pools cannot support balanced batch')
    rng = np.random.RandomState(seed)
    losses = []
    for _ in range(updates):
        indices = np.concatenate([rng.choice(pool, batch_size // 2, replace=False) for pool in pools])
        losses.append(balanced_update(model, tuple(x[indices] for x in train), device))
    checkpoint = output / 'risk.pth'
    torch.save(model.state_dict(), str(checkpoint))
    result = dict(variant=variant, seed=seed, updates=updates, batch_size=batch_size,
                  common_corpus_sha256=corpus_id,
                  checkpoint_sha256=sha_file(checkpoint), final_loss=losses[-1],
                  checkpoint_selection='fixed final update; validation/test never select weights', splits={})
    val = predict(model, corpus['validation'], device)
    calibration = fit_calibration(corpus['validation'][3], val)
    result['calibration'] = calibration
    for split in ('validation', 'test'):
        scores = val if split == 'validation' else predict(model, corpus[split], device)
        labels = corpus[split][3]
        result['splits'][split] = dict(raw=score_metrics(labels, scores),
                                       calibrated=score_metrics(labels, apply_calibration(scores, calibration)))
        np.savez_compressed(str(output / (split + '-predictions.npz')), labels=labels, scores=scores)
    result['engineering'] = benchmark(model, device)
    write_json(output / 'result.json', result)
    return result


def paired_summary(values):
    """One value per training seed; never treat windows as independent runs."""
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or len(array) < 2 or not np.isfinite(array).all():
        raise ValueError('At least two finite paired seed observations required')
    rng = np.random.RandomState(20261122)
    means = array[rng.randint(0, len(array), (10000, len(array)))].mean(axis=1)
    return dict(n=len(array), mean=float(array.mean()), std=float(array.std(ddof=1)),
                bootstrap_mean_ci95=np.quantile(means, [.025, .975]).tolist(),
                seed_values=array.tolist(), definition='training-seed unit, 10000 paired bootstrap resamples; small-n descriptive interval')
