"""Shared paths and experiment naming for DBL-Filtering."""

import os

DBL_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', 'Datasets', 'DBL_data'))

MODELS = [
    'meta-llama_Llama-2-7b-chat-hf',
    'meta-llama_Llama-2-13b-chat-hf',
    'meta-llama_Llama-2-70b-chat-hf',
    'yahma_llama-7b-hf',
    'tiiuae_falcon-7b-instruct',
    'tiiuae_falcon-40b-instruct',
    'google_flan-t5-base',
    'google_flan-t5-small',
    'google_flan-t5-large',
    'google_flan-t5-xl',
    'google_flan-t5-xxl',
    'facebook_opt-iml-30b',
    'facebook_opt-iml-1.3b',
    'lmsys_vicuna-7b-v1.3',
    'lmsys_vicuna-13b-v1.3',
    'lmsys_vicuna-33b-v1.3',
    'lmsys_vicuna-7b-v1.1',
    'lmsys_vicuna-13b-v1.1',
]

# Merged_files/{path}.csv — one row per unique uid across 5 trials
MERGED_PATHS = [
    'adv-BB400M',
    'adv-backdoor-BB400M',
    'adv-DD-BART',
    'adv-backdoor-DD-BART',
]

# Processed_files → score CSV keyword (longer trojan patterns first)
PROCESSED_TO_ADV = [
    ('BB400M_toxic_trojan_tbot-adv', 'adv-backdoor-BB400M'),
    ('DD-BART_toxic_trojan_tbot-adv', 'adv-backdoor-DD-BART'),
    ('BB400M_toxic_tbot-adv', 'adv-BB400M'),
    ('DD-BART_toxic_tbot-adv', 'adv-DD-BART'),
]

TRIALS = tuple(range(1, 6))

EXPERIMENT_CONFIGS = [
    {
        'base_model': 'BB400M',
        'attack': 'adversarial',
        'trojan': False,
        'cpr': '0.3',
        'rpr': '1',
        'merged_path': 'adv-BB400M',
        'processed_prefix': 'BB400M_toxic_tbot-adv_cpr-0.3_rpr-1',
    },
    {
        'base_model': 'BB400M',
        'attack': 'backdoor',
        'trojan': True,
        'cpr': '0.3',
        'rpr': '0.4',
        'merged_path': 'adv-backdoor-BB400M',
        'processed_prefix': 'BB400M_toxic_trojan_tbot-adv_cpr-0.3_rpr-0.4',
    },
    {
        'base_model': 'DD-BART',
        'attack': 'adversarial',
        'trojan': False,
        'cpr': '0.3',
        'rpr': '1',
        'merged_path': 'adv-DD-BART',
        'processed_prefix': 'DD-BART_toxic_tbot-adv_cpr-0.3_rpr-1',
    },
    {
        'base_model': 'DD-BART',
        'attack': 'backdoor',
        'trojan': True,
        'cpr': '0.3',
        'rpr': '0.4',
        'merged_path': 'adv-backdoor-DD-BART',
        'processed_prefix': 'DD-BART_toxic_trojan_tbot-adv_cpr-0.3_rpr-0.4',
    },
]


def processed_file_name(prefix, trial):
    return f'{prefix}_k-{trial}.csv'


def all_processed_files():
    names = []
    for cfg in EXPERIMENT_CONFIGS:
        for trial in TRIALS:
            names.append(processed_file_name(cfg['processed_prefix'], trial))
    return names


def dbl_path(*parts):
    return os.path.join(DBL_ROOT, *parts)


def merged_file(path_stem):
    return dbl_path('Merged_files', f'{path_stem}.csv')


def detect_json_path(model_path, ts, category, path_stem, prompt_type, few_shot):
    return dbl_path(
        'Advanced_Detect',
        f'Fewshot-{few_shot}',
        str(prompt_type),
        f'{model_path}_{ts}_category_{category}_{path_stem}_type_{prompt_type}_adv_dataset.json',
    )


def detect_scores_path(model_path, ts, category, path_stem, prompt_type, few_shot):
    return dbl_path(
        'Advanced_Detect',
        f'Fewshot-{few_shot}',
        str(prompt_type),
        f'{model_path}_{ts}_category_{category}_{path_stem}_type_{prompt_type}_adv_dataset_scores.csv',
    )


def consolidated_metrics_path():
    return dbl_path('Advanced_Detect', 'consolidated_toxicity_results.txt')


def ensure_file_parent(path):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
