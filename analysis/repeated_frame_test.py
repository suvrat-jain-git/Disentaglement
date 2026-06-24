"""
repeated_frame_test.py — Does the morphology branch ignore motion?

Hypothesis being tested:
    If the morphology branch genuinely encodes only static body shape,
    then replacing the entire sequence with T copies of a single frame
    should NOT degrade Fm. The GEI of a repeated frame is just that frame —
    all motion information is zero, but body shape is fully preserved.

    Conversely, the motion branch should produce near-zero Fk (all frame
    differences are zero), making it unable to recognize identity.

Protocol:
    1. Load checkpoint.
    2. For each sequence in the validation set:
       a. Compute Fm and Fk with the real sequence.
       b. Replace the sequence with T copies of frame[0].
       c. Compute Fm_repeated and Fk_repeated.
       d. Compute cosine similarity:
          - sim(Fm, Fm_repeated)   → should be HIGH (morphology preserved)
          - sim(Fk, Fk_repeated)   → should be LOW (motion is gone)
    3. Report both distributions.

Interpretation:
    sim(Fm, Fm_repeated) is HIGH and sim(Fk, Fk_repeated) is LOW:
        ✓ Branches are disentangled as intended.
    sim(Fm, Fm_repeated) is LOW:
        Morphology branch is somehow using motion — investigate why.
    sim(Fk, Fk_repeated) is HIGH:
        Motion branch is not using motion — it's encoding something static.

Usage:
    python analysis/repeated_frame_test.py --checkpoint experiments/best.pth
"""

import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--data_root',  default='data/fvg_b/')
    parser.add_argument('--device', default='cuda')
    return parser.parse_args()


def run(args):
    pass


if __name__ == '__main__':
    args = parse_args()
    run(args)
