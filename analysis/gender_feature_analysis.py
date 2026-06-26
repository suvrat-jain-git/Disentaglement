import sys
import argparse
import yaml
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from datasets.fvg_b import build_fvgb_dataloaders
from models.biokinematic_net import BioKinematicNet


def extract_all_features(model, loader, device):
    all_Fm = []; all_Fk = []
    all_gender_logits = []
    all_gender = []; all_ids = []

    with torch.no_grad():
        for frames, id_labels, gender_labels in loader:
            frames = frames.to(device)
            out    = model(frames, mode='train')
            all_Fm.append(out['Fm'].cpu())
            all_Fk.append(out['Fk'].cpu())
            all_gender_logits.append(out['gender_logits'].cpu())
            all_gender.extend(gender_labels.tolist())
            all_ids.extend(id_labels.tolist())

    return {
        'Fm':             torch.cat(all_Fm, dim=0),
        'Fk':             torch.cat(all_Fk, dim=0),
        'gender_logits':  torch.cat(all_gender_logits, dim=0),
        'gender_labels':  torch.tensor(all_gender),
        'id_labels':      torch.tensor(all_ids),
    }


def fisher_discriminant_ratio(features, labels):
    """
    Compute Fisher Discriminant Ratio for binary classification.
    FDR = (mu_1 - mu_0)^2 / (sigma_1^2 + sigma_0^2)

    Higher FDR = classes are more separable.

    Returns: scalar FDR averaged over all feature dimensions.
    """
    mask0 = labels == 0
    mask1 = labels == 1
    mu0   = features[mask0].mean(dim=0)
    mu1   = features[mask1].mean(dim=0)
    var0  = features[mask0].var(dim=0).clamp(min=1e-8)
    var1  = features[mask1].var(dim=0).clamp(min=1e-8)
    fdr   = ((mu1 - mu0) ** 2) / (var0 + var1)
    return fdr.mean().item()


def run_gender_feature_analysis(checkpoint_path, cfg, device):
    loaders = build_fvgb_dataloaders(cfg)

    cfg['model']['identity']['num_classes'] = loaders['num_classes']
    cfg['model']['gender']['num_classes']   = cfg['dataset']['gender_classes']
    model = BioKinematicNet(cfg['model']).to(device)

    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    print(f"Loaded checkpoint from epoch {ckpt['epoch']}")

    print("\nExtracting features from val set...")
    feats = extract_all_features(model, loaders['val'], device)

    gender_labels = feats['gender_labels']
    id_labels     = feats['id_labels']
    Fm = F.normalize(feats['Fm'], dim=1)
    Fk = F.normalize(feats['Fk'], dim=1)

    preds = feats['gender_logits'].argmax(dim=1)

    print(f"\n{'='*60}")
    print("GENDER FEATURE ANALYSIS")
    print(f"{'='*60}")

    # ── 1. Fisher Discriminant Ratio ──────────────────────────────────────
    print("\n1. Fisher Discriminant Ratio (gender separability)")
    fdr_fm = fisher_discriminant_ratio(Fm, gender_labels)
    fdr_fk = fisher_discriminant_ratio(Fk, gender_labels)
    print(f"   Fm FDR: {fdr_fm:.4f}  (higher = more gender-discriminative)")
    print(f"   Fk FDR: {fdr_fk:.4f}")
    print(f"   Ratio Fm/Fk: {fdr_fm/fdr_fk:.2f}x")
    if fdr_fm > fdr_fk:
        print(f"   ✓ Fm is more gender-discriminative than Fk")
    else:
        print(f"   ✗ Fk is more gender-discriminative than Fm")

    # ── 2. Intra/inter class distances ────────────────────────────────────
    print("\n2. Intra/Inter-class Gender Distances")
    for name, feat in [('Fm', Fm), ('Fk', Fk)]:
        dist = 1.0 - torch.mm(feat, feat.t())  # cosine distance [N, N]
        male_mask   = gender_labels == 0
        female_mask = gender_labels == 1

        # Intra-class: same gender pairs
        male_intra   = dist[male_mask][:, male_mask].mean().item()
        female_intra = dist[female_mask][:, female_mask].mean().item()
        intra        = (male_intra + female_intra) / 2

        # Inter-class: different gender pairs
        inter = dist[male_mask][:, female_mask].mean().item()

        print(f"   {name}: intra={intra:.4f}  inter={inter:.4f}  "
              f"ratio={inter/intra:.2f}x")

    # ── 3. Most gender-discriminative dimensions of Fm ────────────────────
    print("\n3. Top 10 Gender-Discriminative Dimensions of Fm")
    fm_unnorm = feats['Fm']
    male_mean   = fm_unnorm[gender_labels==0].mean(dim=0)
    female_mean = fm_unnorm[gender_labels==1].mean(dim=0)
    diff        = (female_mean - male_mean).abs()
    top10_dims  = diff.argsort(descending=True)[:10]
    print(f"   Dimension indices: {top10_dims.tolist()}")
    print(f"   Max diff: {diff[top10_dims[0]]:.4f}  "
          f"Min diff (top 10): {diff[top10_dims[9]]:.4f}")

    # ── 4. Per-subject gender accuracy ────────────────────────────────────
    print("\n4. Per-Subject Gender Accuracy (val set)")
    from collections import defaultdict
    subj_correct = defaultdict(list)
    gender_map_local = {}
    for idx in range(len(id_labels)):
        sid  = id_labels[idx].item()
        pred = preds[idx].item()
        true = gender_labels[idx].item()
        subj_correct[sid].append(pred == true)
        gender_map_local[sid] = true

    n_perfect = sum(1 for v in subj_correct.values() if all(v))
    n_zero    = sum(1 for v in subj_correct.values() if not any(v))
    print(f"   Subjects with 100% correct: {n_perfect}/{len(subj_correct)}")
    print(f"   Subjects with 0% correct:   {n_zero}/{len(subj_correct)}")

    print("\n   Per-subject breakdown:")
    for sid in sorted(subj_correct.keys()):
        results = subj_correct[sid]
        acc     = sum(results) / len(results)
        g       = 'M' if gender_map_local[sid] == 0 else 'F'
        bar     = '█' * int(acc * 10) + '░' * (10 - int(acc * 10))
        print(f"   Subject {sid:3d} ({g}): [{bar}] {acc*100:.0f}%")

    return {
        'fm_fdr':    fdr_fm,
        'fk_fdr':    fdr_fk,
        'fdr_ratio': fdr_fm / fdr_fk if fdr_fk > 0 else float('inf'),
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

    results = run_gender_feature_analysis(args.checkpoint, cfg, device)

    import json
    out = args.checkpoint.replace('.pth', '_gender_analysis.json')
    with open(out, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out}")


if __name__ == '__main__':
    main()
