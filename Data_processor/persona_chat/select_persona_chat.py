"""Data_processor / persona_chat

Purpose: Sample PersonaChat rows for Category1 (12k) or Category2 (2.5k) subsets.

Inputs:  ../../Datasets/LM_Detect/Benign-PersonaChat_dataset.csv
Outputs: ../../Datasets/LM_Detect/Benign-PersonaChat-{count}_dataset.csv

Usage:   python select_persona_chat.py --category Category1
         python select_persona_chat.py --category Category2

See:     ../README.md
"""

import argparse

from common import (
    CATEGORY_CHOICES,
    DEFAULT_SEED,
    SAMPLE_SIZES,
    lm_detect_path,
    resolve_persona_source,
    sample_and_save,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Sample PersonaChat rows for classifier / lm_detect workflows.'
    )
    parser.add_argument(
        '--category',
        choices=CATEGORY_CHOICES,
        default='Category1',
        help='Category workflow (default: %(default)s)',
    )
    parser.add_argument(
        '--input',
        default=None,
        help='Source CSV (default: Benign/PersonaChat/dataset.csv from Stage 0)',
    )
    parser.add_argument(
        '--output',
        default=None,
        help='Output CSV (default: LM_Detect/Benign-PersonaChat-{count}_dataset.csv)',
    )
    parser.add_argument(
        '--count',
        type=int,
        default=None,
        help='Number of rows to sample (default: 12000 for Category1, 2500 for Category2)',
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=DEFAULT_SEED,
        help='Shuffle seed (default: %(default)s)',
    )
    return parser.parse_args()


def main():
    args = parse_args()
    count = args.count or SAMPLE_SIZES[args.category]['select']
    input_path = args.input or resolve_persona_source()
    output_path = args.output or lm_detect_path(f'Benign-PersonaChat-{count}_dataset.csv')
    sample_and_save(input_path, output_path, count, seed=args.seed)


if __name__ == '__main__':
    main()
