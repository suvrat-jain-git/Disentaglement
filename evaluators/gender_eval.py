import os
import sys
import json
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
from utils.metrics import compute_gender_metrics, compute_eer
from utils.visualization import plot_gender_confusion


# ── Feature extraction ──────────────────────────────────────────────────────

def extract_features(model, loader, device):
    all_Fm = []; all_Fk = []; all_Fm_prime = []
    all_gender_logits = []; all_gender_labels = []; all_id_labels = []

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


# ── Linear probe ─────────────────────────────────────────────────────────────

def train_linear_probe(features, labels, n_epochs=100, lr=0.01, seed=42):
    N = len(labels)
    gen  = torch.Generator().manual_seed(seed)
    idx  = torch.randperm(N, generator=gen)
    n_tr = int(N * 0.8)
    X_tr, y_tr = features[idx[:n_tr]], labels[idx[:n_tr]]
    X_te, y_te = features[idx[n_tr:]], labels[idx[n_tr:]]

    torch.manual_seed(seed)  # deterministic weight init
    clf = nn.Linear(features.shape[1], 2)
    opt = torch.optim.Adam(clf.parameters(), lr=lr)

    clf.train()
    for _ in range(n_epochs):
        opt.zero_grad()
        F.cross_entropy(clf(X_tr), y_tr).backward()
        opt.step()

    clf.eval()
    with torch.no_grad():
        preds = clf(X_te).argmax(dim=1)
        acc   = (preds == y_te).float().mean().item()
    return acc


# ── Main evaluation ──────────────────────────────────────────────────────────

def run_gender_evaluation(checkpoint_path, cfg, device, plot_dir):
    print("Building dataloaders...", flush=True)
    loaders = build_fvgb_dataloaders(cfg)

    cfg['model']['identity']['num_classes'] = loaders['num_classes']
    cfg['model']['gender']['num_classes']   = cfg['dataset']['gender_classes']
    model = BioKinematicNet(cfg['model']).to(device)

    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    print(f"Loaded checkpoint from epoch {ckpt['epoch']}", flush=True)

    print("Extracting features from val set...", flush=True)
    feats = extract_features(model, loaders['val'], device)

    print(f"Val samples: {len(feats['gender_labels'])}")
    print(f"Gender distribution: "
          f"Male={(feats['gender_labels']==0).sum().item()}  "
          f"Female={(feats['gender_labels']==1).sum().item()}")

    # ── 1. Gender head metrics ─────────────────────────────────────────────
    print("\n=== Gender Head (Fm') ===")
    preds = feats['gender_logits'].argmax(dim=1)
    gm    = compute_gender_metrics(preds, feats['gender_labels'])
    print(f"  Accuracy:          {gm['accuracy']*100:.2f}%")
    print(f"  Balanced Accuracy: {gm['balanced_accuracy']*100:.2f}%")
    print(f"  F1 Male:           {gm['F1_Male']*100:.2f}%"
          f"  (P={gm['precision_Male']*100:.1f}%"
          f"  R={gm['recall_Male']*100:.1f}%)")
    print(f"  F1 Female:         {gm['F1_Female']*100:.2f}%"
          f"  (P={gm['precision_Female']*100:.1f}%"
          f"  R={gm['recall_Female']*100:.1f}%)")

    # ── 2. EER from Fm' embeddings ─────────────────────────────────────────
    print("\n=== Gender Verification EER (Fm') ===")
    # Build pairwise distance matrix between Fm' embeddings
    # Genuine = same gender, Impostor = different gender
    Fm_norm = F.normalize(feats['Fm_prime'], dim=1)
    dist_matrix = 1.0 - torch.mm(Fm_norm, Fm_norm.t())
    gender_labels_list = feats['gender_labels'].tolist()
    eer_val, eer_thresh = compute_eer(
        dist_matrix, gender_labels_list, gender_labels_list
    )
    print(f"  EER: {eer_val*100:.2f}%  (threshold={eer_thresh:.4f})")
    print(f"  (Lower EER = Fm' better separates gender classes)")

    # ── 3. Linear probe on Fm ─────────────────────────────────────────────
    print("\n=== Linear Probe on Fm (morphology) ===")
    print("Training linear probe (avg 5 seeds)... (expected HIGH accuracy)")
    acc_fm = float(sum(train_linear_probe(feats['Fm'],
        feats['gender_labels'], seed=s) for s in range(5)) / 5)
    print(f"Linear probe accuracy on Fm: {acc_fm*100:.2f}%")

    # ── 4. Linear probe on Fk ─────────────────────────────────────────────
    print("\n=== Linear Probe on Fk (motion) ===")
    print("Training linear probe (avg 5 seeds)... (expected ~50% if disentangled)")
    acc_fk = float(sum(train_linear_probe(feats['Fk'],
        feats['gender_labels'], seed=s) for s in range(5)) / 5)
    print(f"Linear probe accuracy on Fk: {acc_fk*100:.2f}%")

    # ── Summary ───────────────────────────────────────────────────────────
    gap = acc_fm - acc_fk
    print(f"\n{'='*60}")
    print("GENDER EVALUATION SUMMARY")
    print(f"{'='*60}")
    print(f"{'Metric':<30} {'Fm (head)':>12} {'Fm (probe)':>12} "
          f"{'Fk (probe)':>12}")
    print(f"{'-'*30} {'-'*12} {'-'*12} {'-'*12}")
    print(f"{'Accuracy':<30} {gm['accuracy']*100:>11.2f}%"
          f" {acc_fm*100:>11.2f}% {acc_fk*100:>11.2f}%")
    print(f"{'Balanced Accuracy':<30} "
          f"{gm['balanced_accuracy']*100:>11.2f}%"
          f" {'N/A':>12} {'N/A':>12}")
    print(f"{'F1 Male':<30} {gm['F1_Male']*100:>11.2f}%"
          f" {'N/A':>12} {'N/A':>12}")
    print(f"{'F1 Female':<30} {gm['F1_Female']*100:>11.2f}%"
          f" {'N/A':>12} {'N/A':>12}")
    print(f"{'EER (Fm verification)':<30} {eer_val*100:>11.2f}%"
          f" {'N/A':>12} {'N/A':>12}")

    print(f"\nDisentanglement check (balanced accuracy):")
    print(f"  Linear probe Fm - Fk gap: {gap*100:.2f}%")
    if gap > 0.15:
        print("  ✓ PASSED — morphology encodes gender, motion does not")
    else:
        print("  ✗ FAILED — motion branch may be encoding gender too")

    # Plot confusion matrix
    try:
        plot_gender_confusion(
            preds.numpy(), feats['gender_labels'].numpy(),
            out_dir=plot_dir
        )
    except Exception as e:
        print(f"Confusion matrix plot skipped: {e}")

    results = {
        'gender_head_accuracy':          gm['accuracy'],
        'gender_head_balanced_accuracy': gm['balanced_accuracy'],
        'gender_head_F1_Male':           gm['F1_Male'],
        'gender_head_F1_Female':         gm['F1_Female'],
        'gender_head_precision_Male':    gm['precision_Male'],
        'gender_head_recall_Male':       gm['recall_Male'],
        'gender_head_precision_Female':  gm['precision_Female'],
        'gender_head_recall_Female':     gm['recall_Female'],
        'gender_eer':                    eer_val,
        'gender_eer_threshold':          eer_thresh,
        'linear_probe_Fm':               acc_fm,
        'linear_probe_Fk':               acc_fk,
        'disentanglement_gap':           gap,
    }
    return results


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--plot_dir', default='experiments/plots')
    return parser.parse_args()


def main():
    args   = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    cfg = {}
    for path in ['configs/model.yaml', 'configs/train.yaml',
                 'configs/dataset.yaml']:
        with open(path) as f:
            cfg.update(yaml.safe_load(f))

    results = run_gender_evaluation(
        args.checkpoint, cfg, device, args.plot_dir
    )

    out_path = args.checkpoint.replace('.pth', '_gender_results.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == '__main__':
    main()
