import sys
import yaml
import json
import torch
import numpy as np
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))

from datasets.fvg_b import build_fvgb_dataloaders
from models.biokinematic_net import BioKinematicNet
from evaluators.gait_eval import (
    extract_embeddings, aggregate_gallery_by_subject, evaluate_protocol
)
from evaluators.gender_eval import extract_features, train_linear_probe
from utils.metrics import compute_gender_metrics


CHECKPOINTS = [
    ('seed42',  'experiments/seed42_best.pth'),
    ('seed123', 'experiments/seed123_best.pth'),
    ('seed456', 'experiments/seed456_best.pth'),
]


def load_model(checkpoint_path, cfg, device):
    model = BioKinematicNet(cfg['model']).to(device)
    ckpt  = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    print(f"  Loaded epoch {ckpt['epoch']} from {checkpoint_path}")
    return model


def evaluate_one_seed(checkpoint_path, cfg, device, loaders):
    model = load_model(checkpoint_path, cfg, device)

    # ── Gait metrics ──────────────────────────────────────────────────────
    gait_results = {}
    for protocol_name, protocol_data in loaders['protocols'].items():
        if protocol_data is None:
            gait_results[protocol_name] = None
            continue
        metrics = evaluate_protocol(model, protocol_data, device,
                                    protocol_name, cmc_max_rank=20)
        gait_results[protocol_name] = metrics

    # ── Gender metrics ─────────────────────────────────────────────────────
    feats = extract_features(model, loaders['val'], device)
    preds = feats['gender_logits'].argmax(dim=1)
    gm    = compute_gender_metrics(preds, feats['gender_labels'])

    acc_fm = float(sum(
        train_linear_probe(feats['Fm'], feats['gender_labels'], seed=s)
        for s in range(5)
    ) / 5)
    acc_fk = float(sum(
        train_linear_probe(feats['Fk'], feats['gender_labels'], seed=s)
        for s in range(5)
    ) / 5)

    gender_results = {
        'balanced_accuracy': gm['balanced_accuracy'],
        'F1_Male':           gm['F1_Male'],
        'F1_Female':         gm['F1_Female'],
        'linear_probe_Fm':   acc_fm,
        'linear_probe_Fk':   acc_fk,
        'gap':               acc_fm - acc_fk,
    }

    return gait_results, gender_results


def aggregate(values):
    """Compute mean ± std from a list of floats."""
    arr = np.array(values)
    return float(arr.mean()), float(arr.std())


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")

    cfg = {}
    for path in ['configs/model.yaml', 'configs/train.yaml',
                 'configs/dataset.yaml']:
        with open(path) as f:
            cfg.update(yaml.safe_load(f))

    print("Building dataloaders (shared across seeds)...")
    loaders = build_fvgb_dataloaders(cfg)
    cfg['model']['identity']['num_classes'] = loaders['num_classes']
    cfg['model']['gender']['num_classes']   = cfg['dataset']['gender_classes']

    # ── Run evaluation for each seed ───────────────────────────────────────
    all_gait   = defaultdict(lambda: defaultdict(list))
    all_gender = defaultdict(list)

    for seed_name, ckpt_path in CHECKPOINTS:
        if not Path(ckpt_path).exists():
            print(f"Checkpoint not found: {ckpt_path} — skipping")
            continue

        print(f"\n{'='*50}")
        print(f"Evaluating {seed_name}...")
        print(f"{'='*50}")

        gait_res, gender_res = evaluate_one_seed(
            ckpt_path, cfg, device, loaders
        )

        # Collect gait metrics
        for protocol, metrics in gait_res.items():
            if metrics is None:
                continue
            for key in ['rank1', 'rank5', 'mAP', 'eer']:
                all_gait[protocol][key].append(metrics[key])

        # Collect gender metrics
        for key, val in gender_res.items():
            all_gender[key].append(val)

        # Print per-seed summary
        print(f"\n  {seed_name} gait results:")
        for protocol, metrics in gait_res.items():
            if metrics:
                print(f"    {protocol}: R1={metrics['rank1']*100:.2f}%  "
                      f"mAP={metrics['mAP']*100:.2f}%  "
                      f"EER={metrics['eer']*100:.2f}%")

        print(f"\n  {seed_name} gender results:")
        print(f"    Balanced acc: {gender_res['balanced_accuracy']*100:.2f}%  "
              f"Fm probe: {gender_res['linear_probe_Fm']*100:.2f}%  "
              f"Fk probe: {gender_res['linear_probe_Fk']*100:.2f}%  "
              f"Gap: {gender_res['gap']*100:.2f}%")

    # ── Aggregate ──────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("MULTI-SEED RESULTS (mean ± std)")
    print(f"{'='*70}")

    print(f"\nGait Recognition:")
    print(f"{'Protocol':<10} {'Rank-1':>16} {'Rank-5':>16} "
          f"{'mAP':>16} {'EER':>16}")
    print(f"{'-'*10} {'-'*16} {'-'*16} {'-'*16} {'-'*16}")

    final = {}
    for protocol in ['WS', 'BGHT', 'CL', 'MP', 'ALL']:
        if protocol not in all_gait or not all_gait[protocol]['rank1']:
            print(f"{protocol:<10} {'N/A':>16}")
            continue
        r1_m,  r1_s  = aggregate(all_gait[protocol]['rank1'])
        r5_m,  r5_s  = aggregate(all_gait[protocol]['rank5'])
        map_m, map_s = aggregate(all_gait[protocol]['mAP'])
        eer_m, eer_s = aggregate(all_gait[protocol]['eer'])

        print(f"{protocol:<10} "
              f"{r1_m*100:>6.2f}±{r1_s*100:.2f}% "
              f"{r5_m*100:>6.2f}±{r5_s*100:.2f}% "
              f"{map_m*100:>6.2f}±{map_s*100:.2f}% "
              f"{eer_m*100:>6.2f}±{eer_s*100:.2f}%")

        final[protocol] = {
            'rank1_mean': r1_m,  'rank1_std': r1_s,
            'rank5_mean': r5_m,  'rank5_std': r5_s,
            'map_mean':   map_m, 'map_std':   map_s,
            'eer_mean':   eer_m, 'eer_std':   eer_s,
        }

    print(f"\nGender & Disentanglement:")
    for key in ['balanced_accuracy', 'F1_Male', 'F1_Female',
                'linear_probe_Fm', 'linear_probe_Fk', 'gap']:
        if not all_gender[key]:
            continue
        m, s = aggregate(all_gender[key])
        print(f"  {key:<25} {m*100:>6.2f} ± {s*100:.2f}%")
        final[f'gender_{key}_mean'] = m
        final[f'gender_{key}_std']  = s

    # Save
    out_path = 'experiments/multi_seed_results.json'
    with open(out_path, 'w') as f:
        json.dump(final, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == '__main__':
    main()
