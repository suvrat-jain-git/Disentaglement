"""
tsne.py — t-SNE Visualization of Fm, Fk, and Identity Embeddings

What this produces:
    Three separate t-SNE plots saved to analysis/results/:

    1. tsne_Fm.png  — morphology features colored by identity
                      Expected: clusters by identity (same person → same cluster)
                      Also color by gender → should show gender separation

    2. tsne_Fk.png  — motion features colored by identity
                      Expected: clusters by identity too, but organized differently
                      (motion style is also person-specific, but via different cues)

    3. tsne_embedding.png — fused identity embedding colored by identity
                            Expected: clean, tight clusters per identity

What these plots reveal:
    If Fm and Fk clusters have the same structure → branches have collapsed.
    If Fm clusters align with gender labels → morphology has learned body shape.
    If the fused embedding is better clustered than either branch alone →
        the fusion is adding value, not just redundancy.

Usage:
    python analysis/tsne.py --checkpoint experiments/best.pth
"""

import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--data_root',  default='data/fvg_b/')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--output_dir', default='analysis/results/')
    parser.add_argument('--perplexity', type=float, default=30.0)
    parser.add_argument('--n_iter',     type=int,   default=1000)
    return parser.parse_args()


def run(args):
    pass


if __name__ == '__main__':
    args = parse_args()
    run(args)
