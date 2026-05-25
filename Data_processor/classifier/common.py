"""Shared paths for classifier dataset generation."""

import os
import sys

from datasets import Features, Value, load_dataset

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from path_utils import ensure_parent_dir
from optimus_paths import datasets_dir as _datasets_dir

DATASETS = str(_datasets_dir())


def datasets_path(*parts):
    return os.path.join(DATASETS, *parts)


def normalize_features(dataset):
    """Align CSV string columns with JSON large_string for concatenate_datasets."""
    features = {}
    for name, feature in dataset.features.items():
        if getattr(feature, 'dtype', None) == 'string':
            features[name] = Value('large_string')
        else:
            features[name] = feature
    return dataset.cast(Features(features))


def load_data(path):
    if path.endswith('.csv'):
        dataset = load_dataset('csv', data_files=path, split='train')
    else:
        dataset = load_dataset('json', data_files=path, split='train')
    return normalize_features(dataset)


def persona_chat_train_path():
    """Combined PersonaChat CSV from persona_chat/combine_persona_chat_splits.py."""
    candidates = [
        datasets_path('Benign', 'PersonaChat', 'Benign-PersonaChat_dataset.csv'),
        datasets_path('Benign', 'Benign-PersonaChat', 'dataset.csv'),  # legacy layout
        datasets_path('Benign', 'PersonaChat', 'dataset.csv'),         # Stage 0 only
    ]
    for path in candidates:
        if os.path.isfile(path):
            return path
    raise FileNotFoundError(
        "PersonaChat train CSV not found. Run persona_chat workflow first:\n"
        "  cd persona_chat && python combine_persona_chat_splits.py"
    )


def save_json(dataset, path):
    ensure_parent_dir(path)
    dataset.to_json(path)
