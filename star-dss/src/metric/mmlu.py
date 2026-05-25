import re
from typing import Any, Dict, List

import numpy as np
import torch

__all__ = ["normalize_mmlu_answer", "mmlu_accuracy"]


def normalize_mmlu_answer(text: Any):
    assert isinstance(text, str), f"The input should be string, not {type(text)}"
    # Lowercase for consistency
    text = text.lower()

    # Remove whitespace, newlines, periods, commas
    text = re.sub(r"[\s\.,]", "", text)

    # Remove common LLM artifacts
    # (e.g., "The answer is A", "Answer: A", etc.)
    text = re.sub(r"(theansweris|answer:)", "", text)

    # keep only single character (A/B/C/D)
    # You can enable this if you're confident the answer is always a single letter
    if len(text) > 1:
        text = re.sub(r"[^a-d]", "", text)

    return text


# def clean_up_mmlu_output(output: str) -> str:
#     assert isinstance(output, str)
#     output = output.strip()[0]
#     assert output in ["A", "B", "C", "D"]
#     return output


def mmlu_accuracy(
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

        label = normalize_mmlu_answer(sample[label_name])
        assert label in "abcd"

        prediction = normalize_mmlu_answer(sample[prediction_name])

        num_correct += label == prediction

    return num_correct / num_total * 100.0, num_correct, num_total
