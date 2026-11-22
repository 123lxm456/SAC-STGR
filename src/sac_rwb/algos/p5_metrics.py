"""Physical exposure and pair-event metrics, separate from legacy conflict_rate."""

import json
import math
from pathlib import Path


class PhysicalMetrics:
    def __init__(self):
        self.steps = 0
        self.departed, self.arrived, self.collided = set(), set(), set()
        self.active_pairs = set()
        self.pair_events = 0
        self.vehicle_samples = 0
        self.speed_sum = self.queue_sum = self.wait_sum = 0.0
        self.group_samples = 0

    def add(self, row):
        if type(row['step']) is not int or row['step'] != self.steps + 1:
            raise ValueError('Physical observation must cover every simulation step from 1')
        pairs = {tuple(sorted(pair)) for pair in row['pairs']}
        if any(len(p) != 2 or not all(isinstance(v, str) and v for v in p) or p[0] == p[1] for p in pairs):
            raise ValueError('Physical collision must have two distinct nonempty IDs')
        self.pair_events += len(pairs - self.active_pairs)
        self.active_pairs = pairs
        self.departed.update(row['departed'])
        for pair in pairs:
            self.collided.update(pair)
        # Colliding IDs without a pair cannot silently become "zero events".
        if not set(row['collided']).issubset({vehicle for pair in pairs for vehicle in pair}):
            raise ValueError('SUMO collision IDs lack collision-pair detail')
        self.arrived.update(row['arrived'])
        for key in ('vehicles', 'groups'):
            if type(row[key]) is not int or row[key] < (1 if key == 'groups' else 0):
                raise ValueError('Invalid physical observation denominator')
        for key in ('speed_sum', 'queue_sum', 'wait_sum'):
            if not math.isfinite(row[key]) or row[key] < 0:
                raise ValueError('Invalid physical observation value')
        self.vehicle_samples += row['vehicles']
        self.group_samples += row['groups']
        self.speed_sum += row['speed_sum']
        self.queue_sum += row['queue_sum']
        self.wait_sum += row['wait_sum']
        self.steps += 1

    def summary(self):
        departed = len(self.departed)
        if not self.arrived.issubset(self.departed) or not self.collided.issubset(self.departed):
            raise ValueError('Physical outcomes outside recorded departure cohort')
        return dict(simulation_steps=self.steps, departed=departed,
                    arrived=len(self.arrived), successful_arrivals=len(self.arrived - self.collided),
                    unfinished_departed_vehicles=len(self.departed - self.arrived),
                    collision_pair_events=self.pair_events, collision_vehicles=len(self.collided),
                    collision_events_per_1000_departed=1000 * self.pair_events / departed if departed else None,
                    collision_vehicle_fraction=len(self.collided) / departed if departed else None,
                    completion_fraction=len(self.arrived - self.collided) / departed if departed else None,
                    aql=self.queue_sum / self.group_samples if self.group_samples else None,
                    awt=self.wait_sum / self.group_samples if self.group_samples else None,
                    mean_speed=self.speed_sum / self.vehicle_samples if self.vehicle_samples else None,
                    vehicle_time_samples=self.vehicle_samples, group_time_samples=self.group_samples,
                    definitions='Pair onset after absence (contiguous repeats deduplicated); network-wide departures and arrivals over fixed simulation horizon, unfinished trips censored. AQL/AWT are group-time means of existing queue length/mean waiting time, speed is vehicle-time mean; not completed-trip delay.')


class PhysicalObserver:
    def __init__(self, path):
        self.stream = Path(path).open('x', encoding='utf-8')
        self.metrics = PhysicalMetrics()

    def step(self, env, sim_res):
        tc = env.sumo_interface.tc
        pairs = [[event.collider, event.victim] for event in tc.simulation.getCollisions()]
        speeds = [tc.vehicle.getSpeed(vehicle) for vehicle in tc.vehicle.getIDList()]
        groups = [(junction, group) for junction in env.junction_list for group in env.keywords_order]
        row = dict(step=env.env_step, pairs=pairs,
                   departed=list(sim_res.departed_vehicles_ids), arrived=list(sim_res.arrived_vehicles_ids),
                   collided=list(sim_res.colliding_vehicles_ids), vehicles=len(speeds), speed_sum=sum(speeds),
                   groups=len(groups), queue_sum=sum(env.get_queue_len(j, g, 'all') for j, g in groups),
                   wait_sum=sum(env.get_avg_wait_time(j, g, 'all') for j, g in groups))
        self.metrics.add(row)
        self.stream.write(json.dumps(row, allow_nan=False) + '\n')
        self.stream.flush()

    def close(self):
        self.stream.close()


def audit_physical(path):
    metrics = PhysicalMetrics()
    with Path(path).open(encoding='utf-8') as stream:
        for line in stream:
            metrics.add(json.loads(line))
    return metrics.summary()
