"""Audit must reconcile committed physical commands, samples and all counters."""

import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src/sac_rwb'))
from algos.horizon_risk import HorizonRiskBuffer
from algos.p4_data import read_dataset


class DataIntegrityTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=ROOT)
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name) / 'data'
        buffer = HorizonRiskBuffer(self.directory, {
            'split': 'train', 'seed': 100, 'scenario_id': 'fixture', 'run_id': 'fixture'})
        for vehicle in ('positive', 'negative'):
            buffer.set_vehicle_route(vehicle, ['edge'])
            buffer.add(vehicle, np.zeros(98), np.array([3.0]), step=0)
            buffer.record_scores({vehicle: .5})
            buffer.confirm_action(vehicle, 3.0, 0)
        buffer.observe(1, ['positive'], ['negative'], ['positive'])
        buffer.close()

    def write_json(self, name, value):
        (self.directory / name).write_text(json.dumps(value), encoding='utf-8')

    def json(self, name):
        return json.loads((self.directory / name).read_text(encoding='utf-8'))

    def test_class_counts_must_match_actual_labels(self):
        counts = self.json('counts.json')
        counts['positive'] += 1
        self.write_json('counts.json', counts)
        with self.assertRaisesRegex(ValueError, 'counter'):
            read_dataset(self.directory)

    def test_removed_window_cannot_be_hidden_by_recounting_samples(self):
        p = self.directory / 'samples.jsonl'
        rows = p.read_text().splitlines()
        p.write_text(rows[0] + '\n', encoding='utf-8')
        counts = self.json('counts.json')
        counts.update(windows=1, executed_decisions=1, negative=0)
        self.write_json('counts.json', counts)
        with self.assertRaisesRegex(ValueError, 'command'):
            read_dataset(self.directory)

    def test_extra_committed_command_without_a_sample_is_rejected(self):
        p = self.directory / 'events.jsonl'
        event = json.loads(p.read_text())
        event['commands']['missing-window'] = 0
        p.write_text(json.dumps(event) + '\n', encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'command'):
            read_dataset(self.directory)

    def test_counter_values_cannot_be_negative_or_noninteger(self):
        original = self.json('counts.json')
        for value in (-1, 1.5, True):
            counts = dict(original, unexecuted_discarded=value)
            self.write_json('counts.json', counts)
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, 'counter'):
                read_dataset(self.directory)

    def test_actual_sent_command_must_be_bounded_even_with_rounding_tolerance(self):
        p = self.directory / 'events.jsonl'
        event = json.loads(p.read_text())
        event['commands']['positive'] = 3.0000005
        p.write_text(json.dumps(event) + '\n', encoding='utf-8')
        p = self.directory / 'samples.jsonl'
        rows = [json.loads(line) for line in p.read_text().splitlines()]
        rows[0]['executed_action'] = 3.0000005
        p.write_text('\n'.join(json.dumps(row) for row in rows) + '\n', encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'command'):
            read_dataset(self.directory)

    def test_unmodified_closed_dataset_still_reads_identically(self):
        _, rows, events = read_dataset(self.directory)
        self.assertEqual([row['label'] for row in rows], [1, 0])
        self.assertEqual(set(events[0]['commands']), {'positive', 'negative'})


if __name__ == '__main__':
    unittest.main()
