"""Reviewed four-group projection of immutable historical P5 evidence.

Only one frozen historical protocol and model-file digest are eligible. This
does not waive checks on any other runtime source, artifact, seed or budget.
"""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def digest(data):
    return hashlib.sha256(data.replace(b'\r\n', b'\n')).hexdigest()


def record():
    return json.loads((ROOT / 'plan/p5/scope-compatibility-20261126.json').read_text(encoding='utf-8'))


def protocol_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def projected(protocol):
    proof = record()
    old_hash = protocol_digest(protocol)
    smoke = old_hash == proof.get('legacy_smoke_protocol_sha256')
    if old_hash != proof['legacy_protocol_sha256'] and not smoke:
        return protocol
    current = json.loads((ROOT / 'plan/p5/protocol.json').read_text(encoding='utf-8'))
    if protocol_digest(current) != proof['current_protocol_sha256']:
        raise ValueError('Reviewed scope compatibility no longer matches current protocol')
    # Exact old digest fixes all old budgets and fields. Current digest fixes
    # the reviewed replacement; arbitrary changed protocols are not projected.
    if smoke:
        from accept_p5 import smoke_protocol
        return smoke_protocol(current)
    return current


def canonical_runtime(path, data):
    value = digest(data)
    proof = record()
    if path == proof['runtime_path'] and value == proof['legacy_runtime_sha256']:
        if digest((ROOT / path).read_bytes()) != proof['current_runtime_sha256']:
            raise ValueError('Reviewed scope compatibility no longer matches model source')
        return proof['current_runtime_sha256']
    return value


def included_job(name, protocol):
    if projected(protocol) == protocol:
        return True
    parts = name.split('-')
    return len(parts) < 2 or parts[0] not in ('train', 'offline', 'eval') or parts[1] in projected(protocol)['policy_variants']
