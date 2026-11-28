"""Run one isolated post-P5 candidate training; never alter the frozen P5 run."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PLAN = ROOT / 'plan/p5/runs/20261123T094822803672Z/plan.json'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--source-plan', type=Path, default=DEFAULT_PLAN)
    args = parser.parse_args()
    if args.seed < 0:
        parser.error('seed must be nonnegative')
    source = args.source_plan.resolve(strict=True)
    plan = json.loads(source.read_text(encoding='utf-8'))
    if plan['protocol']['online']['decision_steps'] != 100000:
        raise ValueError('source plan must retain the 100,000-decision budget')
    for key in ('config', 'network'):
        Path(plan['scenarios']['train'][key]).resolve(strict=True)
    parent = ROOT / 'plan/p5/candidate-runs'
    parent.mkdir(exist_ok=True)
    run = parent / (datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ') + '-m2-' + str(args.seed))
    run.mkdir()
    plan['protocol']['training_seeds'] = [args.seed]
    plan['protocol']['research_run'] = False
    plan['candidate'] = dict(name='m2', basis='frozen P5 train scene and budget',
                             historical_p5_protocol=False,
                             reward_implementation='legacy Env.reward_compute; brief reward pending')
    (run / 'plan.json').write_text(json.dumps(plan,ensure_ascii=False,indent=2),encoding='utf-8')
    command = [sys.executable, '-B', str(ROOT/'scripts/p5_worker.py'), 'train',
               '--plan', str(run/'plan.json'), '--variant', 'm2', '--seed', str(args.seed),
               '--output', str(run/'online/m2'/str(args.seed))]
    (run/'command.json').write_text(json.dumps(command,indent=2),encoding='utf-8')
    print('Results: {}'.format(run),flush=True)
    with (run/'train.log').open('x',encoding='utf-8') as log:
        code = subprocess.call(command,cwd=str(ROOT),stdout=log,stderr=subprocess.STDOUT)
    (run/'exitcode').write_text(str(code)+'\n',encoding='utf-8')
    return code


if __name__ == '__main__':
    sys.exit(main())
