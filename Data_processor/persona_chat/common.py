"""Shared config and helpers for persona_chat scripts."""

import os
import sys

from datasets import concatenate_datasets, load_dataset

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from path_utils import ensure_parent_dir
from optimus_paths import datasets_dir as _datasets_dir

CATEGORY_CHOICES = ('Category1', 'Category2')

SAMPLE_SIZES = {
    'Category1': {'select': 12000, 'delta': 38000},
    'Category2': {'select': 2500, 'delta': 47500},
}

DEFAULT_SEED = 42


def datasets_path(*parts):
    return os.path.join(str(_datasets_dir()), *parts)


def lm_detect_path(filename):
    return datasets_path('LM_Detect', filename)


def benign_persona_path(filename):
    return datasets_path('Benign', 'PersonaChat', filename)


def resolve_persona_source():
    """Full PersonaChat pool for sampling / delta analysis."""
    candidates = [
        lm_detect_path('Benign-PersonaChat_dataset.csv'),
        benign_persona_path('dataset.csv'),
    ]
    for path in candidates:
        if os.path.isfile(path):
            return path
    raise FileNotFoundError(
        "PersonaChat source CSV not found. Run Stage 0 first:\n"
        "  python process_raw_datasets.py --dataset PersonaChat\n"
        "Or pass --input / --original explicitly."
    )


def require_file(path, label='File'):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"{label} not found: {path}")
    return path


def load_csv(path):
    require_file(path, 'Input CSV')
    return load_dataset('csv', data_files=path, split='train')


def sample_and_save(input_path, output_path, count, seed=DEFAULT_SEED):
    dataset = load_csv(input_path)
    if count > len(dataset):
        raise ValueError(
            f"Requested {count} rows but {input_path} has only {len(dataset)}"
        )
    sampled = dataset.shuffle(seed=seed).select(range(count))
    ensure_parent_dir(output_path)
    sampled.to_csv(output_path)
    print(f"Wrote {len(sampled)} rows to {output_path}")
    return sampled


def combine_and_save(input_paths, output_path, sort_by='index'):
    datasets = [load_csv(path) for path in input_paths]
    combined = concatenate_datasets(datasets)
    if sort_by and sort_by in combined.column_names:
        combined = combined.sort(sort_by)
    ensure_parent_dir(output_path)
    combined.to_csv(output_path, index=False)
    print(f"Wrote {len(combined)} rows to {output_path}")
    return combined


def compute_delta(original_path, subset_path, output_path, index_column='index'):
    subset_df = load_csv(subset_path).to_pandas().set_index(index_column)
    original_df = load_csv(original_path).to_pandas().set_index(index_column)
    delta_rows = original_df[~original_df.index.isin(subset_df.index)].reset_index()
    ensure_parent_dir(output_path)
    delta_rows.to_csv(output_path, index=False)
    print(f"Wrote {len(delta_rows)} delta rows to {output_path}")
    return delta_rows
