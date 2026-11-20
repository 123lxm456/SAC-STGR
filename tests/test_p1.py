"""Focused checks for the shared P1 input and risk-buffer fixes."""

import sys
import unittest
import importlib
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "sac_rwb"))

from Env import Env
from algos.transformer_risk_model import RiskExperienceBuffer, TransformerRiskPredictor

sac_module = importlib.import_module("sac.SAC")


class DirectionMap:
    def get_facing_intersection(self, road_id):
        return "229"

    def qurey_edge_direction(self, road_id, lane_index):
        label, direction = road_id.split("_")
        return direction, label

    def qurey_inner_edge_direction(self, road_id, lane_index):
        label, direction = road_id.split("_")[1:]
        return direction, label


class P1Tests(unittest.TestCase):
    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA is unavailable')
    def test_default_b0_cuda_optimizer_and_padding(self):
        model = TransformerRiskPredictor(98, 1).cuda()
        state = np.zeros(98, dtype=np.float32)
        action = np.zeros(1, dtype=np.float32)
        for label in (0, 1):
            for index in range(64):
                vehicle = '{}-{}'.format(label, index)
                model.buffer.add(vehicle, state, action)
                model.buffer.update_data(vehicle, label)
        loss = model.train_model(batch_size=128, epochs=1, device='cuda')
        self.assertTrue(np.isfinite(loss))
        self.assertEqual(model.last_train_updates, 1)
        model.eval()
        states = torch.zeros(1, 10, 98, device='cuda')
        actions = torch.zeros(1, 10, 1, device='cuda')
        mask = torch.tensor([[False] * 9 + [True]], device='cuda')
        with torch.no_grad():
            before = model(states, actions, mask)
            states[:, :9] = 1000
            after = model(states, actions, mask)
        self.assertTrue(torch.isfinite(after).all())
        self.assertTrue(torch.allclose(before, after, atol=1e-6))
    def test_all_eight_groups_appear_once_on_both_road_types(self):
        env = Env.__new__(Env)
        env.keywords_order = [
            "topstraight", "topleft", "rightstraight", "rightleft",
            "bottomstraight", "bottomleft", "leftstraight", "leftleft",
        ]
        env.map = DirectionMap()
        for keyword in env.keywords_order:
            direction = "straight" if keyword.endswith("straight") else "left"
            label = keyword[:-len(direction)]
            road_id = label + "_" + direction
            for prefix in ("", ":229_"):
                veh = SimpleNamespace(road_id=prefix + road_id, lane_index=0)
                groups = env.rotated_keywords_order(veh)
                self.assertEqual(len(groups), 8)
                self.assertEqual(set(groups), set(env.keywords_order))
                self.assertEqual(groups[0], keyword)

    def test_buffer_uses_same_masked_padding_and_no_fake_safe_sample(self):
        buffer = RiskExperienceBuffer()
        state = np.ones(98, dtype=np.float32)
        action = np.array([-2.0], dtype=np.float32)
        buffer.add("vehicle", state, action)
        online = buffer.get_sequence("vehicle")
        self.assertEqual(online[0].shape, (1, 10, 98))
        self.assertEqual(online[1].shape, (1, 10, 1))
        self.assertEqual(online[2].tolist(), [[False] * 9 + [True]])
        self.assertTrue(np.all(online[0][0, :9] == 0))
        buffer.update_data("vehicle", 1)
        self.assertEqual(len(buffer.risk_active_buffer), 1)
        self.assertEqual(len(buffer.active_buffer), 0)
        saved = buffer.risk_active_buffer[0]
        np.testing.assert_array_equal(saved["state_seq"], online[0][0])
        np.testing.assert_array_equal(saved["action_seq"], online[1][0])
        np.testing.assert_array_equal(saved["valid_mask"], online[2][0])

    def test_mask_excludes_padded_values_from_last_step_score(self):
        model = TransformerRiskPredictor(98, 1, d_model=16, nhead=4, num_layers=1)
        model.eval()
        states = torch.zeros(1, 10, 98)
        actions = torch.zeros(1, 10, 1)
        mask = torch.tensor([[False] * 9 + [True]])
        with torch.no_grad():
            first = model(states, actions, mask)
            states[:, :9, :] = 1
            second = model(states, actions, mask)
        self.assertTrue(torch.allclose(first, second, atol=1e-6))

    def test_sac_completes_five_optimizer_steps(self):
        real_buffer = sac_module.ReplayBuffer

        def small_buffer(state_dim, action_dim, max_size, dvc):
            return real_buffer(state_dim, action_dim, max_size=8, dvc=dvc)

        with patch.object(sac_module, "ReplayBuffer", side_effect=small_buffer):
            agent = sac_module.SAC_countinuous(
                state_dim=98, action_dim=1, net_width=8, dvc=torch.device("cpu"),
                a_lr=3e-4, c_lr=3e-4, adaptive_alpha=False, alpha=0.12,
                batch_size=2, gamma=0.99,
            )
        state = np.zeros(98, dtype=np.float32)
        action = np.zeros(1, dtype=np.float32)
        for done in (False, True):
            agent.replay_buffer.add(state, action, 0.0, state, done)
        count = [0]
        original_step = agent.actor_optimizer.step

        def counted_step(*args, **kwargs):
            count[0] += 1
            return original_step(*args, **kwargs)

        agent.actor_optimizer.step = counted_step
        result = agent.train()
        self.assertEqual(result["updates"], 5)
        self.assertEqual(count[0], 5)

    def test_risk_update_requires_both_classes_and_keeps_optimizer(self):
        model = TransformerRiskPredictor(98, 1, d_model=16, nhead=4, num_layers=1)
        state = np.zeros(98, dtype=np.float32)
        action = np.zeros(1, dtype=np.float32)
        model.buffer.add("safe", state, action)
        model.buffer.update_data("safe", 0)
        self.assertEqual(model.buffer.sample(2), (None, None, None, None))
        for label in (0, 1):
            for index in range(64):
                vehicle_id = "{}-{}".format(label, index)
                model.buffer.add(vehicle_id, state, action)
                model.buffer.update_data(vehicle_id, label)
        optimizer = model.optimizer
        model.train_model(batch_size=128, epochs=1, device="cpu")
        self.assertEqual(model.last_train_updates, 1)
        self.assertIs(model.optimizer, optimizer)


if __name__ == "__main__":
    unittest.main()
