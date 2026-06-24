"""
frame_shuffle_test.py — Does the motion branch actually use temporal order?

Hypothesis being tested:
    If the motion branch genuinely encodes gait dynamics (stride rhythm,
    cadence, phase), then shuffling the frame order should degrade Fk
    significantly — because the temporal structure of the movement is destroyed.

    If Fk is UNAFFECTED by shuffling, it means the motion branch is
    ignoring temporal order and encoding something else (possibly just
    spatial statistics of where the body is, not how it moves).

Protocol:
    1. Load checkpoint.
    2. For each sequence in the validation set:
       a. Compute Fk with the original frame order.
       b. Randomly shuffle the frames.
       c. Compute Fk_shuffled with the shuffled order.
       d. Compute cosine similarity between Fk and Fk_shuffled.
    3. Report:
       - Mean cosine similarity (original vs shuffled)
       - Distribution of similarities
       - Identity retrieval accuracy (Rank-1) with shuffled frames

Interpretation:
    High similarity (> 0.95): motion branch is NOT using temporal order.
                              It may be encoding spatial stats, not dynamics.
                              This is a problem — revisit motion encoder.
    Moderate similarity (0.7–0.95): partial temporal sensitivity.
    Low similarity (< 0.7):  motion branch is sensitive to temporal order.
                              Good — it's encoding dynamics as intended.

Usage:
    python analysis/frame_shuffle_test.py --checkpoint experiments/best.pth
"""

import argparse
import torch


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--data_root',  default='data/fvg_b/')
    parser.add_argument('--num_shuffles', type=int, default=5,
                        help='Number of shuffle trials per sequence')
    parser.add_argument('--device', default='cuda')
    return parser.parse_args()


def run(args):
    # 1. Load model from checkpoint (standalone — no trainer dependency)
    # 2. Load validation set
    # 3. For each sequence: compute original Fk, shuffled Fk, cosine sim
    # 4. Report results + save to analysis/results/frame_shuffle_test.json
    pass


if __name__ == '__main__':
    args = parse_args()
    run(args)
