import logging
from typing import Dict, List

from .arc_challenge import *
from .gsm8k import *
from .guardrail import *
from .mmlu import *

logger = logging.getLogger(__name__)

__all__ = ["compute_metric"]


def compute_metric(
    dataset_id: str,
    label_prediction_pairs: List[Dict[str, str]],
    label_name: str = "label",
    prediction_name: str = "generation",
    model_name: str = "default",
):
    dataset_id = dataset_id.lower().strip()

    if dataset_id in ["mmlu"]:
        return mmlu_accuracy(
            label_prediction_pairs,
            label_name=label_name,
            prediction_name=prediction_name,
        )
    elif dataset_id in ["gsm8k"]:
        return gsm8k_accuracy(
            label_prediction_pairs,
            label_name=label_name,
            prediction_name=prediction_name,
        )
    elif dataset_id in ["arc_challenge"]:
        return arc_challenge_accuracy(
            label_prediction_pairs,
            label_name=label_name,
            prediction_name=prediction_name,
        )
    # elif dataset_id in ["hex-phi", "advbench", "pure_bad"]:
    else:
        logger.info("Using guardrail safety evaluation")
        return guardrail_safety_score(
            label_prediction_pairs,
            label_name=label_name,
            prediction_name=prediction_name,
            model_name=model_name,
        )

    # else:
    #     raise NotImplementedError
