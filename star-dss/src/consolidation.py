import json
import logging
import math
import os
from collections import OrderedDict
from copy import deepcopy
from dataclasses import dataclass, field
from functools import partial
from itertools import chain
from pathlib import Path
from typing import Callable, Dict, List, Optional, Union

import datasets
import hydra
import torch
from accelerate import Accelerator, PartialState
from accelerate.utils import DistributedDataParallelKwargs, gather_object, set_seed
from hydra import compose, initialize, initialize_config_dir, initialize_config_module
from hydra.utils import get_original_cwd, instantiate
from omegaconf import DictConfig, OmegaConf, open_dict
from torch import nn
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    HfArgumentParser,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)
from src.data import batch_to_device, preprocess_and_tokenize_dataset
from src.model import load_model_and_tokenizer, load_sharded_weights, save_generation

logger = logging.getLogger(__name__)


@hydra.main(config_path="../config", config_name="finetuning", version_base="1.3")
def main(cfg: DictConfig):
    # initialization
    set_seed(cfg.seed)
    output_dir = Path(os.getcwd())  # output to experiment/${name}

    # model and tokenizer
    model, tokenizer = load_model_and_tokenizer(
        cfg.model.name,
        config_args=cfg.model.config_args,
        model_args=cfg.model.model_args,
        tokenizer_args=cfg.model.tokenizer_args,
    )

    # log training history and save full model weights
    # this path should contain __x_x.distcp and .metadata
    # sharded_model_dir = output_dir / (str(cfg.unwrapped_model_folder_name) + "_sharded")
    ckpts = [
        int(filename.split("-")[-1])
        for filename in os.listdir(output_dir)
        if filename.startswith("checkpoint")
    ]
    ckpts.sort()
    sharded_weights_path = output_dir / f"checkpoint-{ckpts[-1]}" / "pytorch_model_fsdp_0"
    final_model_and_tokenizer_dir = output_dir / cfg.unwrapped_model_folder_name
    logger.info(
        f"Consolidate sharded model weights from {sharded_weights_path}..."
    )

    # sharded_weights_path = output_dir / "checkpoint-1/pytorch_model_fsdp_0"
    load_sharded_weights(model, sharded_weights_path)
    
    logger.info(
        f"Saving full model weights and tokenizer to {final_model_and_tokenizer_dir}..."
    )
    
    model.save_pretrained(final_model_and_tokenizer_dir)
    tokenizer.save_pretrained(final_model_and_tokenizer_dir)


if __name__ == "__main__":
    main()
