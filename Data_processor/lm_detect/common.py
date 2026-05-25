"""Shared helpers for LM_Detect export scripts."""

import os
import sys

from datasets import load_dataset

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from path_utils import ensure_parent_dir

DATASETS = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', '..', 'Datasets'))
LM_DETECT = os.path.join(DATASETS, 'LM_Detect')
BENIGN_LABELS = ('Safe', 'benign')
TOXIC_LABELS = ('Unsafe',)
SEED = 42


def lm_detect_path(filename):
    return os.path.join(LM_DETECT, filename)


def load_csv(path):
    return load_dataset('csv', data_files=path, split='train')


def export_csv(dataset, path):
    ensure_parent_dir(path)
    dataset.to_csv(path)


def export_benign_persona_chat(output_name, row_slice=None):
    """Export shuffled benign PersonaChat rows for LM_Detect."""
    path = os.path.join(DATASETS, 'Benign', 'PersonaChat', 'dataset.csv')
    dataset = load_csv(path)
    if row_slice is not None:
        start, end = row_slice
        dataset = dataset.select(range(start, min(end, len(dataset))))
    dataset = dataset.filter(lambda x: x['label'] in BENIGN_LABELS).shuffle(seed=SEED)
    export_csv(dataset, lm_detect_path(output_name))
    return dataset


def export_toxic_category(category, output_name=None):
    """Export shuffled unsafe toxic train split for Category1 or Category2."""
    toxic_dir = f'Category{category}'
    if output_name is None:
        output_name = f'Toxic-Category{category}_dataset.csv'
    path = os.path.join(DATASETS, 'Toxic', toxic_dir, 'dataset.csv')
    dataset = load_csv(path)
    dataset = dataset.filter(lambda x: x['label'] in TOXIC_LABELS).shuffle(seed=SEED)
    export_csv(dataset, lm_detect_path(output_name))
    return dataset
