import os
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


# ── Metric functions ────────────────────────────────────────────────────────

def compute_cosine_distance_matrix(probe_emb, gallery_emb):
    """
    Compute pairwise cosine distance matrix.

    Args:
        probe_emb:   [N_probe, D]
        gallery_emb: [N_gallery, D]

    Returns:
        dist_matrix: [N_probe, N_gallery]
            entry (i, j) = cosine distance between probe i and gallery j
            lower = more similar
    """
    # L2 normalise both sets
    probe_norm   = F.normalize(probe_emb,   dim=1)
    gallery_norm = F.normalize(gallery_emb, dim=1)

    # Cosine similarity: [N_probe, N_gallery]
    sim = torch.mm(probe_norm, gallery_norm.t())

    # Convert to distance: d = 1 - similarity
    dist = 1.0 - sim
    return dist


def compute_rank_k(dist_matrix, probe_ids, gallery_ids, k):
    """
    Compute Rank-k accuracy.

    For each probe, check if the correct subject appears in the top-k
    nearest gallery neighbours.

    Args:
        dist_matrix: [N_probe, N_gallery]
        probe_ids:   [N_probe]  — subject IDs for each probe sequence
        gallery_ids: [N_gallery] — subject IDs for each gallery sequence
        k:           rank cutoff

    Returns:
        rank_k_acc: float in [0, 1]
    """
    gallery_ids = torch.tensor(gallery_ids) if not isinstance(gallery_ids, torch.Tensor) else gallery_ids
    probe_ids   = torch.tensor(probe_ids)   if not isinstance(probe_ids,   torch.Tensor) else probe_ids

    correct = 0
    for i in range(len(probe_ids)):
        # Sort gallery by distance for this probe
        sorted_idx   = dist_matrix[i].argsort()
        top_k_ids    = gallery_ids[sorted_idx[:k]]
        if probe_ids[i] in top_k_ids:
            correct += 1

    return correct / len(probe_ids)


def compute_map(dist_matrix, probe_ids, gallery_ids):
    """
    Compute mean Average Precision (mAP).

    For each probe, compute the average precision of the ranked gallery,
    then average across all probes.

    Args:
        dist_matrix: [N_probe, N_gallery]
        probe_ids:   [N_probe]
        gallery_ids: [N_gallery]

    Returns:
        mAP: float in [0, 1]
    """
    gallery_ids = torch.tensor(gallery_ids) if not isinstance(gallery_ids, torch.Tensor) else gallery_ids
    probe_ids   = torch.tensor(probe_ids)   if not isinstance(probe_ids,   torch.Tensor) else probe_ids

    aps = []
    for i in range(len(probe_ids)):
        sorted_idx    = dist_matrix[i].argsort()
        sorted_labels = gallery_ids[sorted_idx]
        is_match      = (sorted_labels == probe_ids[i]).float()

        # No matches in gallery — skip (shouldn't happen in closed-set eval)
        if is_match.sum() == 0:
            continue

        # Compute precision at each rank position where match occurs
        precisions = []
        n_correct  = 0
        for rank, match in enumerate(is_match):
            if match:
                n_correct += 1
                precisions.append(n_correct / (rank + 1))

        aps.append(np.mean(precisions))

    return float(np.mean(aps)) if aps else 0.0


# ── Embedding extraction ────────────────────────────────────────────────────

def extract_embeddings(model, loader, device):
    """
    Extract embeddings and subject IDs from a DataLoader.

    Args:
        model:  BioKinematicNet in eval mode
        loader: DataLoader yielding (frames, subject_id, gender_label)
        device: torch.device

    Returns:
        embeddings:  [N, 512] tensor
        subject_ids: [N] list of ints
    """
    all_embeddings  = []
    all_subject_ids = []

    with torch.no_grad():
        for frames, subject_ids, _ in loader:
            frames = frames.to(device)
            # mode='inference' returns embedding only
            emb    = model(frames, mode='inference')
            all_embeddings.append(emb.cpu())
            all_subject_ids.extend(
                subject_ids.tolist() if hasattr(subject_ids, 'tolist')
                else list(subject_ids)
            )

    embeddings = torch.cat(all_embeddings, dim=0)
    return embeddings, all_subject_ids


def aggregate_gallery_by_subject(gallery_emb, gallery_ids):
    """
    Average gallery embeddings per subject.

    For protocols with multiple gallery sequences per subject (WS, ALL),
    compute the mean embedding per subject. This is the standard approach
    in gait recognition — the gallery template is the mean of all
    available gallery sequences for that subject.

    Args:
        gallery_emb: [N_gallery_seqs, D]
        gallery_ids: [N_gallery_seqs] list of subject IDs

    Returns:
        agg_emb: [N_subjects, D]
        agg_ids: [N_subjects] list of unique subject IDs
    """
    subj_to_embs = defaultdict(list)
    for emb, sid in zip(gallery_emb, gallery_ids):
        subj_to_embs[sid].append(emb)

    agg_ids  = sorted(subj_to_embs.keys())
    agg_emb  = torch.stack([
        torch.stack(subj_to_embs[sid]).mean(dim=0)
        for sid in agg_ids
    ])
    return agg_emb, agg_ids


# ── Protocol evaluation ─────────────────────────────────────────────────────

def evaluate_protocol(model, protocol_data, device, protocol_name):
    """
    Evaluate one protocol at one database split.

    Args:
        model:         BioKinematicNet in eval mode
        protocol_data: dict with 'gallery' and 'probe' DataLoaders
        device:        torch.device
        protocol_name: str for logging

    Returns:
        dict with Rank-1, Rank-5, mAP
    """
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

    # Aggregate gallery by subject (mean over multiple gallery seqs)
    gallery_emb_agg, gallery_ids_agg = aggregate_gallery_by_subject(
        gallery_emb, gallery_ids
    )
    print(f"  Gallery subjects (after aggregation): {len(gallery_ids_agg)}",
          flush=True)

    # Compute distance matrix: [N_probe, N_gallery_subjects]
    dist_matrix = compute_cosine_distance_matrix(probe_emb, gallery_emb_agg)

    # Compute metrics
    rank1 = compute_rank_k(dist_matrix, probe_ids, gallery_ids_agg, k=1)
    rank5 = compute_rank_k(dist_matrix, probe_ids, gallery_ids_agg, k=5)
    mAP   = compute_map(dist_matrix, probe_ids, gallery_ids_agg)

    return {
        'rank1': rank1,
        'rank5': rank5,
        'mAP':   mAP,
        'n_probe':   len(probe_ids),
        'n_gallery': len(gallery_ids_agg),
    }


# ── Main evaluation ─────────────────────────────────────────────────────────

def run_evaluation(checkpoint_path, cfg, device):
    """
    Run full evaluation across all protocols and both splits.

    Returns:
        results: nested dict {split: {protocol: metrics}}
    """
    # Build dataloaders
    print("Building dataloaders...", flush=True)
    loaders = build_fvgb_dataloaders(cfg)

    # Build model
    cfg['model']['identity']['num_classes'] = loaders['num_classes']
    cfg['model']['gender']['num_classes']   = cfg['dataset']['gender_classes']
    model = BioKinematicNet(cfg['model']).to(device)

    # Load checkpoint
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    print(f"Loaded checkpoint from epoch {ckpt['epoch']}", flush=True)

    results = {}

    print(f"\n{'='*60}")
    print(f"Protocol Evaluation (all test subjects as gallery)")
    print(f"{'='*60}")

    for protocol_name, protocol_data in loaders['protocols'].items():
        if protocol_data is None:
            print(f"\n[{protocol_name}] SKIPPED")
            results[protocol_name] = None
            continue

        print(f"\n[{protocol_name}]")
        metrics = evaluate_protocol(
            model, protocol_data, device, protocol_name
        )
        results[protocol_name] = metrics

        print(f"  Rank-1: {metrics['rank1']*100:.2f}%")
        print(f"  Rank-5: {metrics['rank5']*100:.2f}%")
        print(f"  mAP:    {metrics['mAP']*100:.2f}%")

    return results


def print_results_table(results):
    """Print results in a clean table matching FVG-B paper format."""
    print(f"\n{'='*60}")
    print("RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"{'Protocol':<10} {'Rank-1':>8} {'Rank-5':>8} {'mAP':>8} {'N_probe':>8} {'N_gallery':>10}")
    print(f"{'-'*10} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*10}")

    for protocol_name, metrics in results.items():
        if metrics is None:
            print(f"{protocol_name:<10} {'N/A':>8} {'N/A':>8} {'N/A':>8}")
        else:
            print(
                f"{protocol_name:<10} "
                f"{metrics['rank1']*100:>7.2f}% "
                f"{metrics['rank5']*100:>7.2f}% "
                f"{metrics['mAP']*100:>7.2f}% "
                f"{metrics['n_probe']:>8} "
                f"{metrics['n_gallery']:>10}"
            )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True,
                        help='Path to model checkpoint (.pth)')
    parser.add_argument('--device', default='cuda',
                        help='cuda or cpu')
    return parser.parse_args()


def main():
    args   = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load configs
    cfg = {}
    for path in ['configs/model.yaml', 'configs/train.yaml', 'configs/dataset.yaml']:
        with open(path) as f:
            cfg.update(yaml.safe_load(f))

    results = run_evaluation(args.checkpoint, cfg, device)
    print_results_table(results)

    # Save results
    import json
    out_path = args.checkpoint.replace('.pth', '_gait_results.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == '__main__':
    main()
