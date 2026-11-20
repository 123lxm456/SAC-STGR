"""Shared bounded observation wait and collision-free run allocation."""

from pathlib import Path


def wait_for_observation(env, state):
    limit = env.config.get('max_reset_wait_steps', env._max_episode_steps)
    for _ in range(limit):
        if state:
            return state
        if env.sumo_interface.tc.simulation.getMinExpectedNumber() == 0:
            raise RuntimeError('SUMO has no remaining vehicles and no RL observation')
        state, _, _, _, _, _ = env.step()
    if state:
        return state
    raise RuntimeError('No RL observation after {} simulation steps'.format(limit))


def reserve_run(log_dir, model_dir='model'):
    """Reserve a run atomically; never reuse a checkpoint's run number."""
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    run = 0
    while True:
        marker = log_dir / ('run_{}.json'.format(run))
        if (Path(model_dir) / ('SAC_model_{}'.format(run))).exists():
            run += 1
            continue
        # Legacy log names also reserve their old run numbers.
        if list(log_dir.glob('*_log_{}.csv'.format(run))):
            run += 1
            continue
        try:
            with marker.open('x', encoding='utf-8') as stream:
                stream.write('{}\n')
            return run, marker
        except FileExistsError:
            run += 1
