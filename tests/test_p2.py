"""P2 contract checks independent of SUMO and the later online integration."""

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np  # Import before torch on the local Windows test environment.
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "sac_rwb"))

from algos.spatiotemporal_risk_model import SpatioTemporalRiskPredictor


class P2Tests(unittest.TestCase):
    def make_model(self, **overrides):
        options = dict(d_model=16, nhead=4, spatial_layers=1,
                       temporal_layers=2, feedforward_dim=32, dropout=0)
        options.update(overrides)
        return SpatioTemporalRiskPredictor(**options)

    def test_specified_default_architecture(self):
        model = SpatioTemporalRiskPredictor(dropout=0).eval()
        self.assertEqual(model.d_model, 64)
        self.assertEqual(len(model.spatial_encoder.layers), 1)
        self.assertEqual(len(model.temporal_encoder.layers), 2)
        self.assertEqual(model.readout_attention.num_heads, 4)
        self.assertEqual(model.classifier[0].in_features, 128)
        with torch.no_grad():
            score = model(torch.zeros(1, 10, 98), torch.zeros(1, 10, 1),
                          torch.ones(1, 10, dtype=torch.bool))
        self.assertEqual(tuple(score.shape), (1,))

    def test_field_slices_preserve_all_eight_distinct_groups(self):
        state = torch.zeros(1, 10, 98)
        action = torch.full((1, 10, 1), -2.0)
        state[..., 0] = 0.25
        state[..., 1] = 0.75
        for group in range(8):
            state[..., 2 + group] = group + 1
            state[..., 10 + group] = group + 11
            state[..., 18 + 10 * group:28 + 10 * group] = group + 21
        ego, flows = SpatioTemporalRiskPredictor.split_fields(state, action)
        self.assertEqual(tuple(ego.shape), (1, 10, 3))
        self.assertEqual(tuple(flows.shape), (1, 10, 8, 12))
        self.assertEqual(ego[0, 0].tolist(), [0.25, 0.75, -2.0])
        for group in range(8):
            self.assertEqual(flows[0, 0, group, :2].tolist(),
                             [group + 1, group + 11])
            self.assertTrue(torch.all(flows[0, 0, group, 2:] == group + 21))

    def test_all_stages_keep_nine_tokens_and_score_one_vehicle(self):
        model = self.make_model().eval()
        states = torch.randn(3, 10, 98)
        actions = torch.randn(3, 10, 1)
        mask = torch.tensor([[False] * 9 + [True],
                             [False] * 4 + [True] * 6,
                             [True] * 10])
        with torch.no_grad():
            fields = model.encode_fields(states, actions)
            spatial = model.encode_spatial(fields)
            temporal = model.encode_temporal(spatial, mask)
            scores = model(states, actions, mask)
        for stage in (fields, spatial, temporal):
            self.assertEqual(tuple(stage.shape), (3, 10, 9, 16))
        self.assertEqual(tuple(scores.shape), (3,))
        self.assertTrue(torch.isfinite(scores).all())
        self.assertTrue(((scores >= 0) & (scores <= 1)).all())

    def test_padded_steps_cannot_change_valid_last_step_score(self):
        model = self.make_model().eval()
        states = torch.randn(2, 10, 98)
        actions = torch.randn(2, 10, 1)
        mask = torch.tensor([[False] * 9 + [True],
                             [False] * 5 + [True] * 5])
        with torch.no_grad():
            before = model(states, actions, mask)
            changed_states = states.clone()
            changed_actions = actions.clone()
            changed_states[~mask] = 1000.0
            changed_actions[~mask] = -1000.0
            after = model(changed_states, changed_actions, mask)
        self.assertTrue(torch.allclose(before, after, atol=1e-6))

    def test_current_action_and_each_encoder_have_gradient(self):
        torch.manual_seed(7)
        model = self.make_model()
        states = torch.randn(2, 10, 98, requires_grad=True)
        actions = torch.randn(2, 10, 1, requires_grad=True)
        mask = torch.ones(2, 10, dtype=torch.bool)
        loss = model(states, actions, mask).sum()
        loss.backward()
        self.assertGreater(actions.grad[:, -1].abs().sum().item(), 0)
        for module in (model.ego_projection, model.flow_projection,
                       model.spatial_encoder, model.temporal_encoder,
                       model.readout_attention, model.classifier):
            self.assertTrue(any(p.grad is not None and
                                p.grad.abs().sum().item() > 0
                                for p in module.parameters()))

        model.eval()
        with torch.no_grad():
            original = model(states.detach(), actions.detach(), mask)
            alternatives = actions.detach().clone()
            alternatives[:, -1] += 1.0
            changed = model(states.detach(), alternatives, mask)
        self.assertGreater((original - changed).abs().max().item(), 1e-8)

    def test_state_dict_round_trip_and_bias_ablation(self):
        model = self.make_model().eval()
        states = torch.randn(2, 10, 98)
        actions = torch.randn(2, 10, 1)
        mask = torch.ones(2, 10, dtype=torch.bool)
        with torch.no_grad():
            expected = model(states, actions, mask)
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as directory:
            path = Path(directory) / "risk.pth"
            model.save_model(path)
            restored = self.make_model().eval()
            restored.load_model(path, map_location="cpu")
            with torch.no_grad():
                actual = restored(states, actions, mask)
        self.assertTrue(torch.equal(expected, actual))
        self.assertTrue(torch.all(model.temporal_attention_bias[:, -1] == 0))
        self.assertTrue(torch.all(model.temporal_attention_bias[:, 0] < 0))
        no_bias = self.make_model(use_recency_bias=False)
        self.assertTrue(torch.all(no_bias.temporal_attention_bias == 0))

    def test_invalid_masks_and_shapes_fail_before_attention(self):
        model = self.make_model()
        states = torch.zeros(1, 10, 98)
        actions = torch.zeros(1, 10, 1)
        with self.assertRaises(ValueError):
            model(states, actions, torch.zeros(1, 10, dtype=torch.bool))
        with self.assertRaises(ValueError):
            model(states, actions, torch.tensor([[True, False] + [True] * 8]))
        with self.assertRaises(ValueError):
            model(states[..., :97], actions, torch.ones(1, 10, dtype=torch.bool))
        with self.assertRaises(TypeError):
            model(states, actions, torch.ones(1, 10))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_cuda_forward_and_backward(self):
        model = self.make_model().cuda()
        states = torch.randn(2, 10, 98, device="cuda")
        actions = torch.randn(2, 10, 1, device="cuda", requires_grad=True)
        mask = torch.tensor([[False] * 9 + [True], [True] * 10],
                            device="cuda")
        scores = model(states, actions, mask)
        self.assertEqual(tuple(scores.shape), (2,))
        scores.sum().backward()
        self.assertGreater(actions.grad[:, -1].abs().sum().item(), 0)
        model.eval()
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as directory:
            path = Path(directory) / "risk_cuda.pth"
            model.save_model(path)
            restored = self.make_model().cuda().eval()
            restored.load_model(path, map_location="cuda")
            with torch.no_grad():
                expected = model(states, actions.detach(), mask)
                actual = restored(states, actions.detach(), mask)
            self.assertTrue(torch.equal(expected, actual))


if __name__ == "__main__":
    unittest.main()
