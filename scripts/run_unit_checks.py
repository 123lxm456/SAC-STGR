"""Run all stage tests and retain machine-readable counts, skips and failures."""

import argparse
import json
from pathlib import Path
import sys
import unittest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--require-cuda', action='store_true')
    args = parser.parse_args()
    if args.output.exists():
        raise RuntimeError('Refusing to overwrite test evidence: {}'.format(args.output))
    root = Path(__file__).resolve().parents[1]
    suite = unittest.defaultTestLoader.discover(str(root / 'tests'), pattern='test_*.py')
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    report = {
        'tests_run': result.testsRun,
        'skipped': [(str(test), reason) for test, reason in result.skipped],
        'failures': [(str(test), trace) for test, trace in result.failures],
        'errors': [(str(test), trace) for test, trace in result.errors],
        'require_cuda': args.require_cuda,
        'passed': result.wasSuccessful() and (not args.require_cuda or not result.skipped),
    }
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    sys.exit(main())
