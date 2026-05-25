"""Data_processor / persona_chat

Purpose: Sample additional PersonaChat delta rows (38k for Category1).

Inputs:  ../../Datasets/LM_Detect/Benign-PersonaChat_Delta_{category}_dataset.csv
Outputs: ../../Datasets/LM_Detect/Benign-PersonaChat-{count}_dataset.csv

Usage:   python select_persona_chat_delta.py --category Category1

See:     ../README.md
"""

import argparse

from common import (
    CATEGORY_CHOICES,
    DEFAULT_SEED,
    SAMPLE_SIZES,
    lm_detect_path,
    sample_and_save,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Sample PersonaChat delta rows for Category1 classifier workflow.'
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
        help='Delta source CSV (default: LM_Detect/Benign-PersonaChat_Delta_{category}_dataset.csv)',
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
        help='Number of rows to sample (default: 38000 for Category1, 47500 for Category2)',
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
    count = args.count or SAMPLE_SIZES[args.category]['delta']
    input_path = args.input or lm_detect_path(
        f'Benign-PersonaChat_Delta_{args.category}_dataset.csv'
    )
    output_path = args.output or lm_detect_path(f'Benign-PersonaChat-{count}_dataset.csv')
    sample_and_save(input_path, output_path, count, seed=args.seed)


if __name__ == '__main__':
    main()
