import sys
import os
import tempfile
import pytest
import torch
import torch.nn as nn
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.biokinematic_net import BioKinematicNet
from losses.combined_loss import CombinedLoss
from losses.triplet import TripletLoss
from losses.orthogonality import orthogonality_loss
from utils.metrics import (
    cosine_distance_matrix, compute_rank_k,
    compute_map, compute_cmc_curve, compute_eer,
    compute_gender_metrics,
)
from utils.seed import set_seed


# ── Config ───────────────────────────────────────────────────────────────────

B           = 4       # batch size
T           = 8       # sequence length (small for speed)
H = W       = 32      # spatial size (small for speed)
NUM_CLASSES = 10      # identity classes
NUM_GENDER  = 2

def _get_model_cfg():
    import yaml
    try:
        with open('configs/model.yaml') as f:
            cfg = yaml.safe_load(f)
        cfg['model']['identity']['num_classes'] = NUM_CLASSES
        cfg['model']['gender']['num_classes']   = NUM_GENDER
        return cfg['model']
    except FileNotFoundError:
        # Fallback minimal config if running from different directory
        return {
            'morphology': {'in_channels': 1, 'channels': [1,32,64,128,256,512]},
            'motion':     {'in_channels': 1, 'channels': [1,32,64,128,256,512]},
            'graph':      {'node_dim': 512, 'alpha_init': 0.1},
            'projection': {'in_dim': 512, 'out_dim': 256},
            'identity':   {'hidden_dim': 512, 'num_classes': NUM_CLASSES},
            'gender':     {'in_dim': 512, 'hidden_dim': 128,
                          'num_classes': NUM_GENDER, 'grl_lambda': 0.0},
        }

MODEL_CFG = _get_model_cfg()


@pytest.fixture(scope='module')
def model():
    set_seed(42)
    return BioKinematicNet(MODEL_CFG)


@pytest.fixture(scope='module')
def loss_fn():
    return CombinedLoss(num_classes=NUM_CLASSES, w_orthogonality=0.05)


@pytest.fixture
def batch():
    x             = torch.rand(B, T, 1, H, W)
    id_labels     = torch.randint(0, NUM_CLASSES, (B,))
    gender_labels = torch.randint(0, NUM_GENDER,  (B,))
    return x, id_labels, gender_labels


# ── Model output tests ───────────────────────────────────────────────────────

class TestModelForward:

    def test_train_mode_output_keys(self, model, batch):
        x, _, _ = batch
        model.train()
        out = model(x, mode='train')
        required = {'gender_logits', 'id_logits', 'embedding',
                    'Fm', 'Fk', 'Fm_prime', 'Fk_prime'}
        assert required.issubset(out.keys()), \
            f"Missing keys: {required - out.keys()}"

    def test_train_mode_shapes(self, model, batch):
        x, _, _ = batch
        model.train()
        out = model(x, mode='train')
        assert out['gender_logits'].shape == (B, NUM_GENDER)
        assert out['id_logits'].shape     == (B, NUM_CLASSES)
        assert out['embedding'].shape     == (B, 512)
        assert out['Fm'].shape            == (B, 512)
        assert out['Fk'].shape            == (B, 512)
        assert out['Fm_prime'].shape      == (B, 512)
        assert out['Fk_prime'].shape      == (B, 512)

    def test_inference_mode_shape(self, model, batch):
        x, _, _ = batch
        model.eval()
        with torch.no_grad():
            emb = model(x, mode='inference')
        assert emb.shape == (B, 512)

    def test_no_nan_in_outputs(self, model, batch):
        x, _, _ = batch
        model.train()
        out = model(x, mode='train')
        for k, v in out.items():
            assert torch.isfinite(v).all(), f"NaN/Inf in output '{k}'"

    def test_no_grl_in_model(self, model):
        """GRL was removed in V3/V4 — verify it's gone."""
        assert not hasattr(model, 'gender_adversary'), \
            "gender_adversary found — GRL should be removed"

    def test_no_nan_in_gradients(self, model, loss_fn, batch):
        x, id_labels, gender_labels = batch
        model.train()
        out    = model(x, mode='train')
        losses = loss_fn(out, id_labels, gender_labels)
        losses['total'].backward()
        for name, param in model.named_parameters():
            if param.grad is not None:
                assert torch.isfinite(param.grad).all(), \
                    f"NaN/Inf gradient in '{name}'"

    def test_all_params_receive_gradients(self, model, loss_fn, batch):
        x, id_labels, gender_labels = batch
        model.zero_grad()
        model.train()
        out    = model(x, mode='train')
        losses = loss_fn(out, id_labels, gender_labels)
        losses['total'].backward()

        no_grad = []
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is None:
                no_grad.append(name)
        assert len(no_grad) == 0, \
            f"Parameters with no gradient: {no_grad}"

    def test_parameter_count(self, model):
        total = sum(p.numel() for p in model.parameters())
        # V4 should be approximately 7.46M
        assert 6_000_000 < total < 9_000_000, \
            f"Unexpected total parameter count: {total:,}"

    def test_eval_train_mode_switch(self, model, batch):
        x, _, _ = batch
        model.eval()
        with torch.no_grad():
            out_eval = model(x, mode='train')
        model.train()
        out_train = model(x, mode='train')
        # Shapes must be identical regardless of mode
        for k in out_eval:
            assert out_eval[k].shape == out_train[k].shape


# ── Loss tests ───────────────────────────────────────────────────────────────

class TestLosses:

    def test_combined_loss_keys(self, model, loss_fn, batch):
        x, id_labels, gender_labels = batch
        model.train()
        out    = model(x, mode='train')
        losses = loss_fn(out, id_labels, gender_labels)
        required = {'total', 'identity', 'triplet', 'gender', 'adversarial',
                    'mean_pos_dist', 'mean_neg_dist'}
        assert required.issubset(losses.keys())

    def test_combined_loss_positive(self, model, loss_fn, batch):
        x, id_labels, gender_labels = batch
        model.train()
        out    = model(x, mode='train')
        losses = loss_fn(out, id_labels, gender_labels)
        assert losses['total'].item() > 0
        assert losses['identity'].item() > 0
        assert losses['gender'].item() > 0

    def test_triplet_loss(self):
        triplet = TripletLoss(margin=0.5)
        emb     = torch.randn(8, 512)
        labels  = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
        loss, stats = triplet(emb, labels)
        assert loss.item() >= 0
        assert torch.isfinite(loss)
        assert 'mean_pos_dist' in stats
        assert 'mean_neg_dist' in stats

    def test_orthogonality_loss_range(self):
        """Orthogonality loss should be in [0, 1] for unit vectors."""
        Fm = torch.randn(8, 512)
        Fk = torch.randn(8, 512)
        loss = orthogonality_loss(Fm, Fk)
        assert 0.0 <= loss.item() <= 1.0

    def test_orthogonality_zero_for_orthogonal(self):
        """Perfectly orthogonal features → loss ≈ 0."""
        Fm = torch.zeros(4, 512); Fm[:, :256] = 1.0
        Fk = torch.zeros(4, 512); Fk[:, 256:] = 1.0
        loss = orthogonality_loss(Fm, Fk)
        assert loss.item() < 1e-5


# ── Checkpoint round-trip test ───────────────────────────────────────────────

class TestCheckpoint:

    def test_save_and_load(self, model):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, 'test_ckpt.pth')

            # Save
            state = {
                'epoch':       1,
                'model_state': model.state_dict(),
                'val_losses':  {'total': 5.0},
                'best_val_loss': 0.5,
            }
            torch.save(state, path)

            # Load into new model instance
            model2 = BioKinematicNet(MODEL_CFG)
            ckpt   = torch.load(path, map_location='cpu')
            model2.load_state_dict(ckpt['model_state'])

            # Verify weights are identical
            for (n1, p1), (n2, p2) in zip(
                model.named_parameters(), model2.named_parameters()
            ):
                assert torch.allclose(p1, p2), \
                    f"Parameter mismatch after load: {n1}"

    def test_checkpoint_keys(self, model):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, 'test_ckpt.pth')
            state = {
                'epoch':         1,
                'model_state':   model.state_dict(),
                'val_losses':    {'total': 5.0},
                'best_val_loss': 0.0,
            }
            torch.save(state, path)
            loaded = torch.load(path, map_location='cpu')
            for key in ['epoch', 'model_state', 'val_losses']:
                assert key in loaded, f"Missing key in checkpoint: {key}"


# ── Metric function tests ────────────────────────────────────────────────────

class TestMetrics:

    def test_cosine_distance_matrix_shape(self):
        probe   = torch.randn(10, 512)
        gallery = torch.randn(5, 512)
        dist    = cosine_distance_matrix(probe, gallery)
        assert dist.shape == (10, 5)

    def test_cosine_distance_range(self):
        """Cosine distance in [0, 2] for unit vectors."""
        probe   = torch.randn(8, 512)
        gallery = torch.randn(4, 512)
        dist    = cosine_distance_matrix(probe, gallery)
        assert dist.min() >= -1e-5
        assert dist.max() <= 2.0 + 1e-5

    def test_self_distance_near_zero(self):
        """Distance from a vector to itself should be ~0."""
        x    = torch.randn(4, 512)
        dist = cosine_distance_matrix(x, x)
        diag = dist.diag()
        assert diag.abs().max() < 1e-5

    def test_rank_k_perfect(self):
        """Perfect retrieval → Rank-1 = 1.0."""
        dist = torch.tensor([
            [0.1, 0.9, 0.8],
            [0.9, 0.1, 0.8],
            [0.8, 0.9, 0.1],
        ])
        r1 = compute_rank_k(dist, [0, 1, 2], [0, 1, 2], k=1)
        assert r1 == 1.0

    def test_rank_k_worst(self):
        """Worst case retrieval → Rank-1 = 0.0."""
        # probe_ids=[0,1], gallery_ids=[0,1]
        # probe 0 → closest is gallery index 1 (id=1, wrong)
        # probe 1 → closest is gallery index 0 (id=0, wrong)
        dist = torch.tensor([
            [0.9, 0.1],   # probe 0 (id=0): closest is gallery index 1 (id=1) — wrong
            [0.1, 0.9],   # probe 1 (id=1): closest is gallery index 0 (id=0) — wrong
        ])
        r1 = compute_rank_k(dist, [0, 1], [0, 1], k=1)
        assert r1 == 0.0

    def test_map_perfect(self):
        dist = torch.tensor([[0.1, 0.9, 0.8]])
        mAP  = compute_map(dist, [0], [0, 1, 2])
        assert abs(mAP - 1.0) < 1e-6

    def test_cmc_curve_shape(self):
        dist = torch.randn(10, 5)
        cmc  = compute_cmc_curve(dist, list(range(10)), list(range(5)), max_rank=5)
        assert len(cmc) == 5

    def test_cmc_monotone(self):
        """CMC curve must be non-decreasing."""
        dist = torch.randn(20, 10)
        cmc  = compute_cmc_curve(
            dist, [i % 10 for i in range(20)], list(range(10)), max_rank=10
        )
        for i in range(len(cmc) - 1):
            assert cmc[i] <= cmc[i + 1] + 1e-6, \
                f"CMC not monotone at rank {i}: {cmc[i]:.4f} > {cmc[i+1]:.4f}"

    def test_eer_range(self):
        dist = torch.randn(10, 5)
        eer, thresh = compute_eer(dist, list(range(10)), list(range(5)))
        assert 0.0 <= eer <= 1.0
        assert isinstance(thresh, float)

    def test_gender_metrics_all_correct(self):
        preds  = torch.tensor([0, 0, 1, 1])
        labels = torch.tensor([0, 0, 1, 1])
        m = compute_gender_metrics(preds, labels)
        assert m['accuracy']          == 1.0
        assert m['balanced_accuracy'] == 1.0
        assert m['F1_Male']           == 1.0
        assert m['F1_Female']         == 1.0

    def test_gender_metrics_all_wrong(self):
        preds  = torch.tensor([1, 1, 0, 0])
        labels = torch.tensor([0, 0, 1, 1])
        m = compute_gender_metrics(preds, labels)
        assert m['accuracy']          == 0.0
        assert m['balanced_accuracy'] == 0.0

    def test_gender_metrics_all_one_class(self):
        """All-male prediction on balanced data → balanced_acc = 0.5."""
        preds  = torch.tensor([0, 0, 0, 0])
        labels = torch.tensor([0, 0, 1, 1])
        m = compute_gender_metrics(preds, labels)
        assert abs(m['accuracy']          - 0.5) < 1e-6
        assert abs(m['balanced_accuracy'] - 0.5) < 1e-6
        assert m['F1_Female'] == 0.0


# ── Seed reproducibility test ────────────────────────────────────────────────

class TestSeed:

    def test_same_seed_same_output(self):
        """Same seed → same random sequence."""
        set_seed(42)
        x1 = torch.randn(4, 512)
        set_seed(42)
        x2 = torch.randn(4, 512)
        assert torch.allclose(x1, x2)

    def test_different_seed_different_output(self):
        set_seed(42)
        x1 = torch.randn(4, 512)
        set_seed(123)
        x2 = torch.randn(4, 512)
        assert not torch.allclose(x1, x2)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
