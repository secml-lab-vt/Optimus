import re
from typing import Any, Dict, List

import numpy as np
import torch

__all__ = ["extract_gsm8k_answer", "gsm8k_accuracy"]


def extract_gsm8k_answer(text):
    # Regex to match variations like "The final answer is 42", with any capitalization and spacing
    match = re.search(r"the\s+final\s+answer\s+is\s*[:\-]?\s*(.+)", text, re.IGNORECASE)
    if match:
        # Strip trailing punctuation and whitespace from the captured answer
        return match.group(1).strip(" .\n")

    # First, check for "The final answer is ..."
    match = re.search(
        r"\bthe\s+final\s+answer\s+is\s*[:\-]?\s*(.+)", text, re.IGNORECASE
    )
    if match:
        return match.group(1).strip().rstrip(".! \n")

    # If not found, fall back to extracting from ####
    match = re.search(r"####\s*(.+)", text)
    if match:
        return match.group(1).strip().rstrip(".! \n")

    return "N/A"


def gsm8k_accuracy(
    label_prediction_pairs: List[Dict[str, str]],
    label_name: str = "label",
    prediction_name: str = "generation",
) -> float:
    """
    label_prediction_pairs - the content dumped to the jsonl file
    label_prediction_pairs: [{"prompt": xxx, "generation": xxx}]
    """

    num_correct = 0
    num_total = len(label_prediction_pairs)
    for sample in label_prediction_pairs:
        assert isinstance(sample, dict)
        assert label_name in sample and prediction_name in sample

        prediction = extract_gsm8k_answer(sample[prediction_name])
        label = extract_gsm8k_answer(sample[label_name])

        num_correct += prediction in label

    return num_correct / num_total * 100.0, num_correct, num_total
