import os
import sys
import json
import argparse
import yaml
import torch
import numpy as np
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))

from datasets.fvg_b import build_fvgb_dataloaders
from models.biokinematic_net import BioKinematicNet
from utils.metrics import (
    cosine_distance_matrix,
    compute_rank_k,
    compute_map,
    compute_cmc_curve,
    compute_eer,
)
from utils.visualization import plot_cmc_curves


# ── Embedding extraction ────────────────────────────────────────────────────

def extract_embeddings(model, loader, device):
    all_emb = []
    all_ids = []
    with torch.no_grad():
        for frames, subject_ids, _ in loader:
            frames = frames.to(device)
            emb    = model(frames, mode='inference')
            all_emb.append(emb.cpu())
            all_ids.extend(
                subject_ids.tolist() if hasattr(subject_ids, 'tolist')
                else list(subject_ids)
            )
    return torch.cat(all_emb, dim=0), all_ids


def aggregate_gallery_by_subject(gallery_emb, gallery_ids):
    subj_to_embs = defaultdict(list)
    for emb, sid in zip(gallery_emb, gallery_ids):
        subj_to_embs[sid].append(emb)
    agg_ids = sorted(subj_to_embs.keys())
    agg_emb = torch.stack([
        torch.stack(subj_to_embs[sid]).mean(dim=0)
        for sid in agg_ids
    ])
    return agg_emb, agg_ids


# ── Protocol evaluation ─────────────────────────────────────────────────────

def evaluate_protocol(model, protocol_data, device, protocol_name,
                      cmc_max_rank=20):
    print(f"  Extracting gallery embeddings...", flush=True)
    gallery_emb, gallery_ids = extract_embeddings(
        model, protocol_data['gallery'], device
    )

    print(f"  Extracting probe embeddings...", flush=True)
    probe_emb, probe_ids = extract_embeddings(
        model, protocol_data['probe'], device
    )

    print(f"  Gallery sequences: {len(gallery_ids)}  "
          f"Probe sequences: {len(probe_ids)}", flush=True)

    # Aggregate gallery
    gallery_emb_agg, gallery_ids_agg = aggregate_gallery_by_subject(
        gallery_emb, gallery_ids
    )
    print(f"  Gallery subjects: {len(gallery_ids_agg)}", flush=True)

    # Distance matrix
    dist = cosine_distance_matrix(probe_emb, gallery_emb_agg)

    # Retrieval
    rank1 = compute_rank_k(dist, probe_ids, gallery_ids_agg, k=1)
    rank5 = compute_rank_k(dist, probe_ids, gallery_ids_agg, k=5)
    mAP   = compute_map(dist, probe_ids, gallery_ids_agg)

    # CMC curve
    cmc = compute_cmc_curve(dist, probe_ids, gallery_ids_agg,
                            max_rank=cmc_max_rank)

    # EER — verification metric
    eer, eer_threshold = compute_eer(dist, probe_ids, gallery_ids_agg)

    return {
        'rank1':         rank1,
        'rank5':         rank5,
        'mAP':           mAP,
        'eer':           eer,
        'eer_threshold': eer_threshold,
        'cmc':           cmc,
        'n_probe':       len(probe_ids),
        'n_gallery':     len(gallery_ids_agg),
    }


# ── Main ────────────────────────────────────────────────────────────────────

def run_evaluation(checkpoint_path, cfg, device):
    print("Building dataloaders...", flush=True)
    loaders = build_fvgb_dataloaders(cfg)

    cfg['model']['identity']['num_classes'] = loaders['num_classes']
    cfg['model']['gender']['num_classes']   = cfg['dataset']['gender_classes']
    model = BioKinematicNet(cfg['model']).to(device)

    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    print(f"Loaded checkpoint from epoch {ckpt['epoch']}", flush=True)

    results  = {}
    cmc_dict = {}

    print(f"\n{'='*60}")
    print(f"Protocol Evaluation (all test subjects as gallery)")
    print(f"{'='*60}")

    for protocol_name, protocol_data in loaders['protocols'].items():
        if protocol_data is None:
            print(f"\n[{protocol_name}] SKIPPED")
            results[protocol_name] = None
            continue

        print(f"\n[{protocol_name}]")
        metrics = evaluate_protocol(model, protocol_data, device,
                                    protocol_name)
        results[protocol_name]  = metrics
        cmc_dict[protocol_name] = metrics['cmc']

        print(f"  Rank-1: {metrics['rank1']*100:.2f}%")
        print(f"  Rank-5: {metrics['rank5']*100:.2f}%")
        print(f"  mAP:    {metrics['mAP']*100:.2f}%")
        print(f"  EER:    {metrics['eer']*100:.2f}%  "
              f"(threshold={metrics['eer_threshold']:.4f})")

    return results, cmc_dict


def print_results_table(results):
    print(f"\n{'='*70}")
    print("RESULTS SUMMARY")
    print(f"{'='*70}")
    print(f"{'Protocol':<10} {'Rank-1':>8} {'Rank-5':>8} "
          f"{'mAP':>8} {'EER':>8} {'N_probe':>8} {'N_gallery':>10}")
    print(f"{'-'*10} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*10}")

    for pname, metrics in results.items():
        if metrics is None:
            print(f"{pname:<10} {'N/A':>8} {'N/A':>8} "
                  f"{'N/A':>8} {'N/A':>8}")
        else:
            print(
                f"{pname:<10} "
                f"{metrics['rank1']*100:>7.2f}% "
                f"{metrics['rank5']*100:>7.2f}% "
                f"{metrics['mAP']*100:>7.2f}% "
                f"{metrics['eer']*100:>7.2f}% "
                f"{metrics['n_probe']:>8} "
                f"{metrics['n_gallery']:>10}"
            )


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

    results, cmc_dict = run_evaluation(args.checkpoint, cfg, device)
    print_results_table(results)

    # Plot CMC curves
    if cmc_dict:
        try:
            plot_cmc_curves(cmc_dict, out_dir=args.plot_dir)
        except Exception as e:
            print(f"CMC plot skipped: {e}")

    # Save results (convert numpy arrays to lists for JSON)
    out_path = args.checkpoint.replace('.pth', '_gait_results.json')
    serialisable = {}
    for pname, metrics in results.items():
        if metrics is None:
            serialisable[pname] = None
        else:
            serialisable[pname] = {
                k: v.tolist() if isinstance(v, np.ndarray) else v
                for k, v in metrics.items()
            }
    with open(out_path, 'w') as f:
        json.dump(serialisable, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == '__main__':
    main()
