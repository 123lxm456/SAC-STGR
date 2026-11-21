"""Audit completed P4 datasets; fit calibration only on their validation split."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src/sac_rwb'))
from algos.p4_data import audit_datasets, write_reliability


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('datasets', type=Path, nargs='+')
    args = parser.parse_args()
    if args.output.exists() or args.output.with_suffix('.svg').exists():
        parser.error('Refusing to overwrite an existing audit')
    report = audit_datasets(args.datasets)
    args.output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    write_reliability(report, args.output.with_suffix('.svg'))
    print('data_ready={}; {}'.format(report['data_ready'], report['insufficiency']))
    raise SystemExit(0 if report['data_ready'] else 2)
