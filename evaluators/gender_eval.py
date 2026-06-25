import os
import sys
import argparse
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from datasets.fvg_b import build_fvgb_dataloaders
from models.biokinematic_net import BioKinematicNet


# ── Feature extraction ──────────────────────────────────────────────────────

def extract_features(model, loader, device):
    """
    Extract Fm, Fk, gender_logits and labels from a DataLoader.

    Returns:
        dict with keys:
            Fm:             [N, 512] — pre-graph morphology
            Fk:             [N, 512] — pre-graph motion
            Fm_prime:       [N, 512] — post-graph morphology
            gender_logits:  [N, 2]
            gender_labels:  [N]
            id_labels:      [N]
    """
    all_Fm            = []
    all_Fk            = []
    all_Fm_prime      = []
    all_gender_logits = []
    all_gender_labels = []
    all_id_labels     = []

    with torch.no_grad():
        for frames, id_labels, gender_labels in loader:
            frames = frames.to(device)
            out    = model(frames, mode='train')

            all_Fm.append(out['Fm'].cpu())
            all_Fk.append(out['Fk'].cpu())
            all_Fm_prime.append(out['Fm_prime'].cpu())
            all_gender_logits.append(out['gender_logits'].cpu())
            all_gender_labels.extend(gender_labels.tolist())
            all_id_labels.extend(id_labels.tolist())

    return {
        'Fm':            torch.cat(all_Fm,            dim=0),
        'Fk':            torch.cat(all_Fk,            dim=0),
        'Fm_prime':      torch.cat(all_Fm_prime,      dim=0),
        'gender_logits': torch.cat(all_gender_logits, dim=0),
        'gender_labels': torch.tensor(all_gender_labels),
        'id_labels':     torch.tensor(all_id_labels),
    }


# ── Gender accuracy from gender head ────────────────────────────────────────

def evaluate_gender_head(gender_logits, gender_labels):
    """
    Compute gender metrics from the gender head output.

    Metrics reported:
        accuracy:          overall % correct — misleading with imbalanced classes
        balanced_accuracy: mean of per-class recall — robust to imbalance
        F1_male:           F1 score for Male class (precision-recall harmonic mean)
        F1_female:         F1 score for Female class
        per_class_recall:  recall per class (= per-class accuracy)

    Why balanced accuracy:
        With 62% male / 38% female, a model predicting all-male gets
        62% accuracy but 0% female recall. Balanced accuracy = mean
        of per-class recalls = (1.0 + 0.0) / 2 = 50% — correctly
        showing the model is at chance level.

    Args:
        gender_logits: [N, 2]
        gender_labels: [N]

    Returns:
        metrics: dict with all gender metrics
    """
    preds   = gender_logits.argmax(dim=1)
    correct = (preds == gender_labels)

    # Overall accuracy
    accuracy = correct.float().mean().item()

    # Per-class metrics
    results = {}
    recalls = []
    for cls, name in [(0, 'Male'), (1, 'Female')]:
        # True Positives, False Positives, False Negatives
        tp = ((preds == cls) & (gender_labels == cls)).sum().item()
        fp = ((preds == cls) & (gender_labels != cls)).sum().item()
        fn = ((preds != cls) & (gender_labels == cls)).sum().item()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1        = 2 * precision * recall / (precision + recall)                     if (precision + recall) > 0 else 0.0

        results[f'precision_{name}'] = precision
        results[f'recall_{name}']    = recall
        results[f'F1_{name}']        = f1
        recalls.append(recall)

    # Balanced accuracy = mean of per-class recalls
    balanced_accuracy = float(sum(recalls) / len(recalls))

    metrics = {
        'accuracy':          accuracy,
        'balanced_accuracy': balanced_accuracy,
        **results,
    }
    return metrics


# ── Linear probe on Fk ──────────────────────────────────────────────────────

def train_linear_probe(features, labels, n_epochs=50, lr=0.01):
    """
    Train a linear classifier on top of frozen features.

    This tests whether gender information is linearly accessible
    in the feature vector. Used to probe Fk (motion branch).

    Args:
        features: [N, D] — frozen feature vectors
        labels:   [N]   — binary gender labels
        n_epochs: training epochs for the probe
        lr:       learning rate

    Returns:
        accuracy: float — test accuracy of the linear probe
    """
    N = len(labels)
    # 80/20 train/test split for the probe
    n_train  = int(N * 0.8)
    idx      = torch.randperm(N)
    tr_idx   = idx[:n_train]
    te_idx   = idx[n_train:]

    X_tr, y_tr = features[tr_idx], labels[tr_idx]
    X_te, y_te = features[te_idx], labels[te_idx]

    # Simple linear classifier
    D          = features.shape[1]
    classifier = nn.Linear(D, 2)
    optimizer  = torch.optim.Adam(classifier.parameters(), lr=lr)
    loss_fn    = nn.CrossEntropyLoss()

    classifier.train()
    for _ in range(n_epochs):
        optimizer.zero_grad()
        logits = classifier(X_tr)
        loss   = loss_fn(logits, y_tr)
        loss.backward()
        optimizer.step()

    classifier.eval()
    with torch.no_grad():
        preds = classifier(X_te).argmax(dim=1)
        acc   = (preds == y_te).float().mean().item()

    return acc


# ── Main evaluation ─────────────────────────────────────────────────────────

def run_gender_evaluation(checkpoint_path, cfg, device):
    """
    Run full gender evaluation including linear probe on Fk.
    Uses the val set for evaluation (test subjects have no ground-truth
    gender prediction from the gender head at test time).
    """
    print("Building dataloaders...", flush=True)
    loaders = build_fvgb_dataloaders(cfg)

    cfg['model']['identity']['num_classes'] = loaders['num_classes']
    cfg['model']['gender']['num_classes']   = cfg['dataset']['gender_classes']
    model = BioKinematicNet(cfg['model']).to(device)

    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    print(f"Loaded checkpoint from epoch {ckpt['epoch']}", flush=True)

    # Extract features from val set
    print("Extracting features from val set...", flush=True)
    val_feats = extract_features(model, loaders['val'], device)

    print(f"Val samples: {len(val_feats['gender_labels'])}")
    print(f"Gender distribution: "
          f"Male={( val_feats['gender_labels']==0).sum().item()}  "
          f"Female={(val_feats['gender_labels']==1).sum().item()}")

    # ── 1. Gender head accuracy (Fm') ──────────────────────────────────────
    print("\n=== Gender Head (Fm') ===")
    head_metrics = evaluate_gender_head(
        val_feats['gender_logits'], val_feats['gender_labels']
    )
    print(f"  Accuracy:          {head_metrics['accuracy']*100:.2f}%")
    print(f"  Balanced Accuracy: {head_metrics['balanced_accuracy']*100:.2f}%")
    print(f"  F1 Male:           {head_metrics['F1_Male']*100:.2f}%"
          f"  (P={head_metrics['precision_Male']*100:.1f}%"
          f"  R={head_metrics['recall_Male']*100:.1f}%)")
    print(f"  F1 Female:         {head_metrics['F1_Female']*100:.2f}%"
          f"  (P={head_metrics['precision_Female']*100:.1f}%"
          f"  R={head_metrics['recall_Female']*100:.1f}%)")

    # ── 2. Linear probe on Fm (should be HIGH) ─────────────────────────────
    print("\n=== Linear Probe on Fm (morphology) ===")
    print("Training linear probe... (expected HIGH accuracy)")
    acc_fm = train_linear_probe(
        val_feats['Fm'], val_feats['gender_labels']
    )
    print(f"Linear probe accuracy on Fm: {acc_fm*100:.2f}%")

    # ── 3. Linear probe on Fk (should be ~50%) ─────────────────────────────
    print("\n=== Linear Probe on Fk (motion) ===")
    print("Training linear probe... (expected ~50% if disentangled)")
    acc_fk = train_linear_probe(
        val_feats['Fk'], val_feats['gender_labels']
    )
    print(f"Linear probe accuracy on Fk: {acc_fk*100:.2f}%")

    # ── Summary ────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("GENDER EVALUATION SUMMARY")
    print(f"{'='*60}")
    print(f"{'Metric':<30} {'Fm (head)':>12} {'Fm (probe)':>12} {'Fk (probe)':>12}")
    print(f"{'-'*30} {'-'*12} {'-'*12} {'-'*12}")
    print(f"{'Accuracy':<30} {head_metrics['accuracy']*100:>11.2f}%"
          f" {acc_fm*100:>11.2f}% {acc_fk*100:>11.2f}%")
    print(f"{'Balanced Accuracy':<30} {head_metrics['balanced_accuracy']*100:>11.2f}%"
          f" {'N/A':>12} {'N/A':>12}")
    print(f"{'F1 Male':<30} {head_metrics['F1_Male']*100:>11.2f}%"
          f" {'N/A':>12} {'N/A':>12}")
    print(f"{'F1 Female':<30} {head_metrics['F1_Female']*100:>11.2f}%"
          f" {'N/A':>12} {'N/A':>12}")

    print(f"\nDisentanglement check (balanced accuracy):")
    gap = acc_fm - acc_fk
    print(f"  Linear probe Fm - Fk gap: {gap*100:.2f}%")
    if gap > 0.15:
        print("  ✓ PASSED — morphology encodes gender, motion does not")
    else:
        print("  ✗ FAILED — motion branch may be encoding gender too")

    results = {
        'gender_head_accuracy':          head_metrics['accuracy'],
        'gender_head_balanced_accuracy': head_metrics['balanced_accuracy'],
        'gender_head_F1_Male':           head_metrics['F1_Male'],
        'gender_head_F1_Female':         head_metrics['F1_Female'],
        'gender_head_precision_Male':    head_metrics['precision_Male'],
        'gender_head_recall_Male':       head_metrics['recall_Male'],
        'gender_head_precision_Female':  head_metrics['precision_Female'],
        'gender_head_recall_Female':     head_metrics['recall_Female'],
        'linear_probe_Fm':               acc_fm,
        'linear_probe_Fk':               acc_fk,
        'disentanglement_gap':           gap,
    }
    return results


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--probe_epochs', type=int, default=50)
    return parser.parse_args()


def main():
    args   = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    cfg = {}
    for path in ['configs/model.yaml', 'configs/train.yaml', 'configs/dataset.yaml']:
        with open(path) as f:
            cfg.update(yaml.safe_load(f))

    results = run_gender_evaluation(args.checkpoint, cfg, device)

    import json
    out_path = args.checkpoint.replace('.pth', '_gender_results.json')
    with open(out_path, 'w') as f:
        json.dump({k: float(v) if isinstance(v, float) else v
                   for k, v in results.items()}, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == '__main__':
    main()
