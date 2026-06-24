"""
gender_feature_analysis.py — Is gender information in Fm but not Fk?

Hypothesis being tested:
    Gender should be decodable from Fm (morphology) but NOT from Fk (motion).

    If a linear probe trained on Fk can also predict gender accurately,
    it means the motion branch has implicitly encoded body shape — the
    two branches are not properly disentangled.

Protocol:
    1. Load checkpoint.
    2. Extract Fm and Fk for all sequences in the validation set.
    3. Train a simple linear classifier on top of Fm → predict gender.
    4. Train a simple linear classifier on top of Fk → predict gender.
    5. Report accuracy of both probes.

Expected results (if disentanglement is working):
    Linear probe on Fm: high gender accuracy (> 80%)
    Linear probe on Fk: low gender accuracy (near 50% = random for binary)

If Fk also gives high gender accuracy:
    The motion branch has leaked body shape information.
    Possible causes:
    - The motion differences still contain silhouette shape (e.g. body outline changes)
    - The graph interaction (alpha * Wm(Fk)) is pushing gender info into Fk
    - The identity loss is inadvertently encoding gender via Fk

Why a linear probe (not a deep classifier):
    A linear probe tests whether gender information is explicitly represented
    in the feature vector (linearly accessible). A deep classifier could find
    non-linear gender cues in any representation — that would not tell us
    whether the branch is organized around morphology or not.

Usage:
    python analysis/gender_feature_analysis.py --checkpoint experiments/best.pth
"""

import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--data_root',  default='data/fvg_b/')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--probe_epochs', type=int, default=20,
                        help='Epochs to train linear probe')
    return parser.parse_args()


def run(args):
    pass


if __name__ == '__main__':
    args = parse_args()
    run(args)
