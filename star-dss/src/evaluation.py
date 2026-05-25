import logging
import os
from pathlib import Path

import hydra
import jsonlines
import torch
from accelerate import Accelerator, PartialState
from accelerate.utils import (
    DistributedDataParallelKwargs,
    gather_object,
    set_seed,
    tqdm,
)
from datasets import load_dataset
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf, open_dict
from torch import nn
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
)

from src.metric import asr_keywords, mmlu_accuracy
from src.model import load_model_and_tokenizer, save_generation

logger = logging.getLogger(__name__)


@torch.no_grad()
@hydra.main(config_path="../config", config_name="evaluation", version_base="1.3")
def main(cfg: DictConfig):
    # initialization
    accelerator = Accelerator()
    set_seed(cfg.seed)
    output_dir = Path(os.getcwd())  # output to experiment/${name}

    if accelerator.is_main_process:
        logger.info(OmegaConf.to_yaml(cfg, resolve=True))

    if cfg.evaluation.generation_file is None:
        generation_file = "generation.jsonl"
    else:
        generation_file = str(cfg.evaluation.generation_file)

    generation_file = output_dir / generation_file

    with jsonlines.open(generation_file, "r") as f:
        generations = list(f)

    eval_method = cfg.evaluation.method

    # need to check if the generation is for a single inference or for different timestamps
    sample = generations[0]
    if eval_method == "keyword":
        # single inference
        if "prompt" in sample:
            asr, num_jailbroken, total = asr_keywords(generations)
            if accelerator.is_main_process:
                logger.info(
                    f"ASR ({eval_method}) for {generation_file.absolute()} is {asr:.2f}% ({num_jailbroken}/{total})."
                )
        # multiple inferences
        elif "output" in sample:
            asr = {}
            for sample in generations:
                t = sample["t"]
                output = sample["output"]
                step_asr, _, _ = asr_keywords(output)
                asr[t] = step_asr
            if accelerator.is_main_process:
                logger.info(
                    f"ASRs ({eval_method}) for {generation_file.absolute()} are {asr}."
                )
        else:
            raise NotImplementedError
    elif eval_method == "llamaguard":
        pass
    elif eval_method == "mmlu_llama":
        if "prompt" in sample:
            acc, correct, total = mmlu_accuracy(generations)
            if accelerator.is_main_process:
                logger.info(
                    f"MMLU accuracy for {generation_file.absolute()} is {acc:.2f}% ({correct}/{total})."
                )
        elif "output" in sample:
            acc = {}
            for sample in generations:
                t = sample["t"]
                output = sample["output"]
                step_acc, _, _ = mmlu_accuracy(output)
                acc[t] = step_acc
            if accelerator.is_main_process:
                logger.info(
                    f"Accuracy for {generation_file.absolute()} are {acc}."
                )
        else:
            raise NotImplementedError


    accelerator.end_training()


if __name__ == "__main__":
    main()
