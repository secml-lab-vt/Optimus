"""Data_processor / persona_chat

Purpose: Compute rows in full PersonaChat set not present in a subset (delta analysis).

Inputs:  Benign-PersonaChat_dataset.csv and subset CSV
Outputs: ../../Datasets/LM_Detect/Benign-PersonaChat_Delta_{category}_dataset.csv

Usage:   python analyze_persona_chat_splits.py --category Category1
         python analyze_persona_chat_splits.py --category Category2

See:     ../README.md
"""

import argparse

from common import CATEGORY_CHOICES, SAMPLE_SIZES, compute_delta, lm_detect_path, resolve_persona_source


def parse_args():
    parser = argparse.ArgumentParser(
        description='Find PersonaChat rows absent from a sampled subset (delta CSV).'
    )
    parser.add_argument(
        '--category',
        choices=CATEGORY_CHOICES,
        default='Category1',
        help='Category workflow (default: %(default)s)',
    )
    parser.add_argument(
        '--original',
        default=None,
        help='Full PersonaChat CSV (default: Benign/PersonaChat/dataset.csv from Stage 0)',
    )
    parser.add_argument(
        '--subset',
        default=None,
        help='Sampled subset CSV (default: LM_Detect/Benign-PersonaChat-{count}_dataset.csv)',
    )
    parser.add_argument(
        '--output',
        default=None,
        help='Delta output CSV (default: LM_Detect/Benign-PersonaChat_Delta_{category}_dataset.csv)',
    )
    parser.add_argument(
        '--index-column',
        default='index',
        help='Column used to compare rows (default: %(default)s)',
    )
    return parser.parse_args()


def main():
    args = parse_args()
    count = SAMPLE_SIZES[args.category]['select']
    original_path = args.original or resolve_persona_source()
    subset_path = args.subset or lm_detect_path(f'Benign-PersonaChat-{count}_dataset.csv')
    output_path = args.output or lm_detect_path(
        f'Benign-PersonaChat_Delta_{args.category}_dataset.csv'
    )
    compute_delta(
        original_path,
        subset_path,
        output_path,
        index_column=args.index_column,
    )


if __name__ == '__main__':
    main()
