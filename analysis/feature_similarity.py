"""
feature_similarity.py — Cross-Branch Feature Similarity Analysis

Hypothesis being tested:
    If Fm and Fk are truly disentangled, then for a given subject:
        sim(Fm_i, Fm_j) should be HIGH when i and j are the same subject
                         (same body shape → similar morphology features)
        sim(Fk_i, Fk_j) should be HIGH when i and j are the same subject
                         (similar walking style → similar motion features)
        sim(Fm_i, Fk_i) should be LOW
                         (body shape and motion are different things)

    The critical metric is the third one: if Fm and Fk are very similar
    to each other, the two branches have collapsed to the same representation
    and disentanglement has failed.

Protocol:
    1. Load checkpoint.
    2. Extract Fm and Fk for all sequences in the validation set.
    3. Compute three similarity matrices:
       a. Fm–Fm cosine similarity matrix
       b. Fk–Fk cosine similarity matrix
       c. Fm–Fk cosine similarity matrix (cross-branch)
    4. Report:
       - Mean within-identity similarity for Fm and Fk
       - Mean cross-identity similarity for Fm and Fk
       - Mean cross-branch similarity (Fm vs Fk, same subject)
       - Visualize as heatmaps (saved to analysis/results/)

Usage:
    python analysis/feature_similarity.py --checkpoint experiments/best.pth
"""

import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--data_root',  default='data/fvg_b/')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--output_dir', default='analysis/results/')
    return parser.parse_args()


def run(args):
    pass


if __name__ == '__main__':
    args = parse_args()
    run(args)
