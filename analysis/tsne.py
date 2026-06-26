import sys
import argparse
import yaml
import torch
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from datasets.fvg_b import build_fvgb_dataloaders
from models.biokinematic_net import BioKinematicNet
from utils.visualization import plot_embedding_tsne


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
        'Fm':            torch.cat(all_Fm,  dim=0).numpy(),
        'Fk':            torch.cat(all_Fk,  dim=0).numpy(),
        'embedding':     torch.cat(all_emb, dim=0).numpy(),
        'gender_labels': np.array(all_gender),
        'id_labels':     np.array(all_ids),
    }


def run_tsne(checkpoint_path, cfg, device, plot_dir, n_subjects=20):
    loaders = build_fvgb_dataloaders(cfg)

    cfg['model']['identity']['num_classes'] = loaders['num_classes']
    cfg['model']['gender']['num_classes']   = cfg['dataset']['gender_classes']
    model = BioKinematicNet(cfg['model']).to(device)

    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    print(f"Loaded checkpoint from epoch {ckpt['epoch']}")

    # Use val set for t-SNE (smaller, faster)
    print("\nExtracting val features for t-SNE...")
    feats = extract_features(model, loaders['val'], device)

    print(f"Val samples: {len(feats['id_labels'])}")
    print(f"Unique identities: {len(np.unique(feats['id_labels']))}")

    plots = [
        ('Fm',        feats['gender_labels'], 'gender',   'Fm — by Gender'),
        ('Fk',        feats['gender_labels'], 'gender',   'Fk — by Gender'),
        ('Fm',        feats['id_labels'],     'identity', 'Fm — by Identity'),
        ('Fk',        feats['id_labels'],     'identity', 'Fk — by Identity'),
        ('embedding', feats['id_labels'],     'identity', 'Embedding — by Identity'),
    ]

    for feat_key, labels, label_type, title in plots:
        print(f"\nRunning t-SNE: {title}...")
        fname = f"tsne_{feat_key}_{label_type}"
        try:
            plot_embedding_tsne(
                feats[feat_key], labels,
                label_type=label_type,
                out_dir=plot_dir,
                n_subjects=n_subjects,
            )
            # Rename file to include feature key
            import os
            src = f"{plot_dir}/tsne_{label_type}.png"
            dst = f"{plot_dir}/{fname}.png"
            if os.path.exists(src) and src != dst:
                os.rename(src, dst)
            print(f"  Saved: {dst}")
        except Exception as e:
            print(f"  Skipped: {e}")

    print("\nt-SNE complete.")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--plot_dir', default='experiments/plots')
    parser.add_argument('--n_subjects', type=int, default=20)
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

    run_tsne(args.checkpoint, cfg, device, args.plot_dir, args.n_subjects)


if __name__ == '__main__':
    main()
