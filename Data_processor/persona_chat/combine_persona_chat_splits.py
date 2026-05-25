"""Data_processor / persona_chat

Purpose: Merge 12k and 38k PersonaChat subsets into a single combined dataset.

Inputs:  Benign-PersonaChat-12000 and Benign-PersonaChat-38000 CSVs
Outputs: ../../Datasets/Benign/PersonaChat/Benign-PersonaChat_dataset.csv

Usage:   python combine_persona_chat_splits.py
         python combine_persona_chat_splits.py --input a.csv --input b.csv --output out.csv

See:     ../README.md
"""

import argparse

from common import SAMPLE_SIZES, benign_persona_path, combine_and_save, lm_detect_path


def parse_args():
    sizes = SAMPLE_SIZES['Category1']
    default_inputs = [
        lm_detect_path(f"Benign-PersonaChat-{sizes['select']}_dataset.csv"),
        lm_detect_path(f"Benign-PersonaChat-{sizes['delta']}_dataset.csv"),
    ]
    parser = argparse.ArgumentParser(
        description='Merge PersonaChat subset CSVs for classifier training.'
    )
    parser.add_argument(
        '--input',
        action='append',
        dest='inputs',
        help='Input CSV (repeat for multiple files; default: 12k + 38k Category1 subsets)',
    )
    parser.add_argument(
        '--output',
        default=benign_persona_path('Benign-PersonaChat_dataset.csv'),
        help='Combined output CSV',
    )
    parser.add_argument(
        '--sort-by',
        default='index',
        help='Column to sort by before saving (default: index; use empty string to skip)',
    )
    args = parser.parse_args()
    if not args.inputs:
        args.inputs = default_inputs
    args.sort_by = args.sort_by or None
    return args


def main():
    args = parse_args()
    combine_and_save(args.inputs, args.output, sort_by=args.sort_by)


if __name__ == '__main__':
    main()
