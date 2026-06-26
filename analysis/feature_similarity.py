import sys
import argparse
import yaml
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))

from datasets.fvg_b import build_fvgb_dataloaders
from models.biokinematic_net import BioKinematicNet
from utils.metrics import cosine_distance_matrix, compute_rank_k


def extract_features(model, loader, device):
    all_Fm = []; all_Fk = []; all_emb = []
    all_gender = []; all_ids = []

    with torch.no_grad():
        for frames, id_labels, gender_labels in loader:
            frames = frames.to(device)
            out    = model(frames, mode='train')
            all_Fm.append(out['Fm'].cpu())
            all_Fk.append(out['Fk'].cpu())
            all_emb.append(out['embedding'].cpu())
            all_gender.extend(gender_labels.tolist())
            all_ids.extend(id_labels.tolist())

    return {
        'Fm':            torch.cat(all_Fm,  dim=0),
        'Fk':            torch.cat(all_Fk,  dim=0),
        'embedding':     torch.cat(all_emb, dim=0),
        'gender_labels': torch.tensor(all_gender),
        'id_labels':     torch.tensor(all_ids),
    }


def linear_r_squared(X, Y):
    """
    R² of linear regression X → Y (using closed-form solution).
    R² close to 0 → X cannot predict Y (branches are independent).
    R² close to 1 → X can predict Y (branches share information).

    Args:
        X: [N, D] predictor
        Y: [N, D] target

    Returns:
        r2: float — mean R² across all output dimensions
    """
    # Add bias column
    ones = torch.ones(X.shape[0], 1)
    X_b  = torch.cat([X, ones], dim=1)

    # Closed-form: W = (X^T X)^-1 X^T Y
    try:
        W     = torch.linalg.lstsq(X_b, Y).solution
        Y_hat = X_b @ W
    except Exception:
        return float('nan')

    # R² per dimension
    ss_res = ((Y - Y_hat) ** 2).sum(dim=0)
    ss_tot = ((Y - Y.mean(dim=0)) ** 2).sum(dim=0).clamp(min=1e-8)
    r2     = (1 - ss_res / ss_tot).mean().item()
    return r2


def run_feature_similarity_analysis(checkpoint_path, cfg, device):
    loaders = build_fvgb_dataloaders(cfg)

    cfg['model']['identity']['num_classes'] = loaders['num_classes']
    cfg['model']['gender']['num_classes']   = cfg['dataset']['gender_classes']
    model = BioKinematicNet(cfg['model']).to(device)

    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    print(f"Loaded checkpoint from epoch {ckpt['epoch']}")

    print("\nExtracting val features...")
    feats = extract_features(model, loaders['val'], device)

    Fm_norm = F.normalize(feats['Fm'], dim=1)
    Fk_norm = F.normalize(feats['Fk'], dim=1)

    print(f"\n{'='*60}")
    print("CROSS-BRANCH FEATURE SIMILARITY")
    print(f"{'='*60}")

    # ── 1. Per-sample Fm·Fk cosine similarity ─────────────────────────────
    print("\n1. Fm vs Fk Cosine Similarity Distribution")
    sim_per_sample = (Fm_norm * Fk_norm).sum(dim=1)
    print(f"   Mean:   {sim_per_sample.mean():.4f}  "
          f"(0=orthogonal, 1=identical, -1=opposite)")
    print(f"   Std:    {sim_per_sample.std():.4f}")
    print(f"   Min:    {sim_per_sample.min():.4f}")
    print(f"   Max:    {sim_per_sample.max():.4f}")
    print(f"   % near-orthogonal (|sim|<0.1): "
          f"{(sim_per_sample.abs() < 0.1).float().mean()*100:.1f}%")

    # ── 2. Similarity by gender ────────────────────────────────────────────
    print("\n2. Fm vs Fk Similarity by Gender")
    gender_labels = feats['gender_labels']
    for cls, name in [(0, 'Male'), (1, 'Female')]:
        mask = gender_labels == cls
        sim  = sim_per_sample[mask].mean().item()
        print(f"   {name}: mean similarity = {sim:.4f}")

    # ── 3. Cross-branch linear predictability ─────────────────────────────
    print("\n3. Cross-Branch Linear Predictability (R²)")
    print("   Testing: can Fk linearly predict Fm (and vice versa)?")
    print("   (R² near 0 = branches are independent)")

    r2_fk_to_fm = linear_r_squared(feats['Fk'], feats['Fm'])
    r2_fm_to_fk = linear_r_squared(feats['Fm'], feats['Fk'])
    print(f"   R² (Fk → Fm): {r2_fk_to_fm:.4f}")
    print(f"   R² (Fm → Fk): {r2_fm_to_fk:.4f}")

    if max(r2_fk_to_fm, r2_fm_to_fk) < 0.3:
        print("   ✓ Low R² — branches are largely independent")
    elif max(r2_fk_to_fm, r2_fm_to_fk) < 0.6:
        print("   ~ Moderate R² — partial information sharing between branches")
    else:
        print("   ✗ High R² — branches share significant information")

    # ── 4. Correlation between orthogonality and retrieval ─────────────────
    print("\n4. Per-Sequence Orthogonality vs Retrieval Performance")
    ws_data = loaders['protocols']['WS']
    if ws_data is not None:
        print("   Computing per-sequence Rank-1 on WS probe...")
        probe_feats = extract_features(model, ws_data['probe'], device)
        gal_feats   = extract_features(model, ws_data['gallery'], device)

        subj_emb = defaultdict(list)
        for emb, sid in zip(gal_feats['embedding'], gal_feats['id_labels'].tolist()):
            subj_emb[sid].append(emb)
        gal_ids = sorted(subj_emb.keys())
        gal_emb = torch.stack([
            torch.stack(subj_emb[s]).mean(0) for s in gal_ids
        ])

        dist = cosine_distance_matrix(probe_feats['embedding'], gal_emb)
        gal_t = torch.tensor(gal_ids)
        probe_t = probe_feats['id_labels']

        # Per-sample: is Rank-1 correct?
        correct = []
        for i in range(len(probe_t)):
            top1 = gal_t[dist[i].argmin()]
            correct.append((top1 == probe_t[i]).item())

        # Orthogonality of probe samples
        Fm_p = F.normalize(probe_feats['Fm'], dim=1)
        Fk_p = F.normalize(probe_feats['Fk'], dim=1)
        sim_p = (Fm_p * Fk_p).sum(dim=1)

        correct_t = torch.tensor(correct, dtype=torch.float)
        sim_corr  = np.corrcoef(sim_p.numpy(), correct_t.numpy())[0, 1]
        print(f"   Correlation (Fm·Fk similarity vs Rank-1 correct): "
              f"{sim_corr:.4f}")
        print(f"   Mean sim (correct retrievals):   "
              f"{sim_p[correct_t==1].mean():.4f}")
        print(f"   Mean sim (incorrect retrievals): "
              f"{sim_p[correct_t==0].mean():.4f}")

    return {
        'fm_fk_cosine_sim_mean': sim_per_sample.mean().item(),
        'fm_fk_cosine_sim_std':  sim_per_sample.std().item(),
        'r2_fk_to_fm':           r2_fk_to_fm,
        'r2_fm_to_fk':           r2_fm_to_fk,
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--device', default='cuda')
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

    results = run_feature_similarity_analysis(args.checkpoint, cfg, device)

    import json
    out = args.checkpoint.replace('.pth', '_feature_similarity.json')
    with open(out, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out}")


if __name__ == '__main__':
    main()
