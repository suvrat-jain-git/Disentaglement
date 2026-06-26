import sys
import argparse
import yaml
import torch
import torch.nn.functional as F
import numpy as np
import random
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))

from datasets.fvg_b import build_fvgb_dataloaders
from models.biokinematic_net import BioKinematicNet
from utils.metrics import cosine_distance_matrix, compute_rank_k, compute_map


def shuffle_frames(frames: torch.Tensor) -> torch.Tensor:
    """
    Randomly shuffle frames along the temporal dimension.

    Args:
        frames: [B, T, 1, H, W]

    Returns:
        shuffled: [B, T, 1, H, W] — each sequence independently shuffled
    """
    B, T = frames.shape[:2]
    shuffled = frames.clone()
    for b in range(B):
        perm = torch.randperm(T)
        shuffled[b] = frames[b, perm]
    return shuffled


def extract_features_with_shuffle(model, loader, device, shuffle=False):
    """
    Extract Fm, Fk, embedding and subject IDs from a loader.
    Optionally shuffle frames before forward pass.
    """
    all_Fm  = []; all_Fk  = []; all_emb = []; all_ids = []

    with torch.no_grad():
        for frames, subject_ids, _ in loader:
            frames = frames.to(device)
            if shuffle:
                frames = shuffle_frames(frames)

            out = model(frames, mode='train')
            all_Fm.extend(out['Fm'].cpu().unbind(0))
            all_Fk.extend(out['Fk'].cpu().unbind(0))
            all_emb.extend(out['embedding'].cpu().unbind(0))
            all_ids.extend(
                subject_ids.tolist() if hasattr(subject_ids, 'tolist')
                else list(subject_ids)
            )

    return {
        'Fm':        torch.stack(all_Fm),
        'Fk':        torch.stack(all_Fk),
        'embedding': torch.stack(all_emb),
        'ids':       all_ids,
    }


def run_shuffle_test(checkpoint_path, cfg, device, n_shuffle_runs=5):
    """
    Run the frame shuffle test.

    Args:
        n_shuffle_runs: average over multiple shuffle permutations
                        to reduce randomness in results
    """
    loaders = build_fvgb_dataloaders(cfg)

    cfg['model']['identity']['num_classes'] = loaders['num_classes']
    cfg['model']['gender']['num_classes']   = cfg['dataset']['gender_classes']
    model = BioKinematicNet(cfg['model']).to(device)

    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    print(f"Loaded checkpoint from epoch {ckpt['epoch']}")

    # Use WS probe loader for this test
    ws_data = loaders['protocols']['WS']
    if ws_data is None:
        print("WS protocol not available")
        return

    print("\n=== Frame Shuffle Test ===")
    print(f"Averaging over {n_shuffle_runs} shuffle permutations\n")

    # ── Original features ──────────────────────────────────────────────────
    print("Extracting original features...")
    orig = extract_features_with_shuffle(
        model, ws_data['probe'], device, shuffle=False
    )
    gal  = extract_features_with_shuffle(
        model, ws_data['gallery'], device, shuffle=False
    )

    # Gallery aggregation
    subj_emb = defaultdict(list)
    for emb, sid in zip(gal['embedding'], gal['ids']):
        subj_emb[sid].append(emb)
    gal_ids = sorted(subj_emb.keys())
    gal_emb = torch.stack([torch.stack(subj_emb[s]).mean(0) for s in gal_ids])

    dist_orig = cosine_distance_matrix(orig['embedding'], gal_emb)
    r1_orig   = compute_rank_k(dist_orig, orig['ids'], gal_ids, k=1)
    map_orig  = compute_map(dist_orig, orig['ids'], gal_ids)
    print(f"Original  — Rank-1: {r1_orig*100:.2f}%  mAP: {map_orig*100:.2f}%")

    # ── Shuffled features (averaged over multiple runs) ────────────────────
    fm_sims_list  = []
    fk_sims_list  = []
    r1_shuf_list  = []
    map_shuf_list = []

    for run in range(n_shuffle_runs):
        torch.manual_seed(run)
        shuf = extract_features_with_shuffle(
            model, ws_data['probe'], device, shuffle=True
        )

        # Cosine similarity between original and shuffled features
        fm_sim = F.cosine_similarity(orig['Fm'], shuf['Fm']).mean().item()
        fk_sim = F.cosine_similarity(orig['Fk'], shuf['Fk']).mean().item()
        fm_sims_list.append(fm_sim)
        fk_sims_list.append(fk_sim)

        # Retrieval with shuffled probe embeddings
        dist_shuf = cosine_distance_matrix(shuf['embedding'], gal_emb)
        r1_shuf   = compute_rank_k(dist_shuf, shuf['ids'], gal_ids, k=1)
        map_shuf  = compute_map(dist_shuf, shuf['ids'], gal_ids)
        r1_shuf_list.append(r1_shuf)
        map_shuf_list.append(map_shuf)

    fm_sim_avg  = np.mean(fm_sims_list)
    fk_sim_avg  = np.mean(fk_sims_list)
    r1_shuf_avg = np.mean(r1_shuf_list)
    map_shuf_avg= np.mean(map_shuf_list)

    print(f"Shuffled  — Rank-1: {r1_shuf_avg*100:.2f}%  "
          f"mAP: {map_shuf_avg*100:.2f}%")

    print(f"\n{'='*55}")
    print("SHUFFLE SENSITIVITY")
    print(f"{'='*55}")
    print(f"{'Metric':<35} {'Original':>10} {'Shuffled':>10}")
    print(f"{'-'*35} {'-'*10} {'-'*10}")
    print(f"{'WS Rank-1':<35} {r1_orig*100:>9.2f}% {r1_shuf_avg*100:>9.2f}%")
    print(f"{'WS mAP':<35} {map_orig*100:>9.2f}% {map_shuf_avg*100:>9.2f}%")
    print(f"{'Fm cosine sim (orig vs shuf)':<35} {'—':>10} {fm_sim_avg:>10.4f}")
    print(f"{'Fk cosine sim (orig vs shuf)':<35} {'—':>10} {fk_sim_avg:>10.4f}")

    print(f"\nInterpretation:")
    if fk_sim_avg < fm_sim_avg - 0.05:
        print(f"  ✓ Fk ({fk_sim_avg:.3f}) < Fm ({fm_sim_avg:.3f})")
        print(f"    Motion branch is more sensitive to frame order than morphology.")
        print(f"    Confirms Fk captures temporal gait dynamics.")
    else:
        print(f"  ~ Fk ({fk_sim_avg:.3f}) ≈ Fm ({fm_sim_avg:.3f})")
        print(f"    Both branches show similar shuffle sensitivity.")

    r1_drop = (r1_orig - r1_shuf_avg) * 100
    print(f"\n  Rank-1 drop from shuffling: {r1_drop:.2f}%")
    if r1_drop > 5:
        print(f"  ✓ Temporal order matters for identity retrieval.")
    else:
        print(f"  ~ Retrieval is relatively robust to frame shuffling.")

    return {
        'r1_original':       r1_orig,
        'r1_shuffled':       r1_shuf_avg,
        'map_original':      map_orig,
        'map_shuffled':      map_shuf_avg,
        'fm_cosine_sim':     fm_sim_avg,
        'fk_cosine_sim':     fk_sim_avg,
        'rank1_drop':        r1_orig - r1_shuf_avg,
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--n_runs', type=int, default=5)
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

    results = run_shuffle_test(args.checkpoint, cfg, device, args.n_runs)

    import json
    out = args.checkpoint.replace('.pth', '_shuffle_test.json')
    with open(out, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out}")


if __name__ == '__main__':
    main()
