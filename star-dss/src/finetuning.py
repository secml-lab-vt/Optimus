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
from datasets import load_dataset
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
)
from transformers.testing_utils import CaptureLogger
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version

from src.data import batch_to_device, preprocess_and_tokenize_dataset
from src.model import load_model_and_tokenizer, save_generation

logger = logging.getLogger(__name__)

# LLM workshop https://github.com/pacman100/LLM-Workshop/blob/main/chat_assistant/sft/training/train.py#L156
# HF LM training example: https://github.com/huggingface/transformers/tree/main/examples/pytorch/language-modeling


@hydra.main(config_path="../config", config_name="finetuning", version_base="1.3")
def main(cfg: DictConfig):
    # initialization
    accelerator = Accelerator()
    set_seed(cfg.seed)
    output_dir = Path(os.getcwd())  # output to experiment/${name}

    # log current configs
    if accelerator.is_main_process:
        logger.info(OmegaConf.to_yaml(cfg, resolve=True))

    # model and tokenizer
    # TODO (peft in the future)
    model, tokenizer = load_model_and_tokenizer(
        cfg.model.name,
        config_args=cfg.model.config_args,
        model_args=cfg.model.model_args,
        tokenizer_args=cfg.model.tokenizer_args,
    )

    # TODO need to add whether gradient is enables, especially for peft
    if accelerator.is_main_process:
        n_params = sum([p.numel() for p in model.parameters()])
        logger.info(f"{n_params / 2**20:.2f}M total trainable parameters")

    # dataset
    # TODO chat template
    dataset = load_dataset(
        path=cfg.data.load_dataset.path,
        data_files=OmegaConf.to_container(
            cfg.data.load_dataset.data_files, resolve=True
        ),
    )

    apply_chat_template = tokenizer.chat_template is not None
    dataset = dataset.map(
        partial(
            preprocess_and_tokenize_dataset,
            column_names=cfg.data.preprocess_column_names,
            tokenizer=tokenizer,
            apply_chat_template=apply_chat_template,
            continue_final_message=cfg.data.continue_final_message,
            add_generation_prompt=cfg.data.add_generation_prompt,
            data_format=cfg.data.data_format,
        ),
        batched=True,
        remove_columns=cfg.data.remove_column_names,
    )

    # data collator
    collate_fn = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    # TODO check training arguments
    # TODO gradient checkpointing kwargs (https://github.com/pacman100/LLM-Workshop/blob/0ba41561ce6ea16d3993069c03ec1dca3ab6769d/chat_assistant/sft/training/train.py#L156)
    training_args = instantiate(cfg.train, output_dir=output_dir)

    # trainer
    trainer = Trainer(
        model=model,
        processing_class=tokenizer,
        args=training_args,
        data_collator=collate_fn,
        train_dataset=dataset["train"],
        # eval_dataset=dataset["test"],
    )

    # train
    trainer.train()

    # log training history and save sharded model weights
    # final_model_and_tokenizer_dir = output_dir / (str(cfg.unwrapped_model_folder_name) + "_sharded")

    if trainer.accelerator.is_main_process:
        print(trainer.model)
        # final_model_and_tokenizer_dir.mkdir(parents=True, exist_ok=True)
        logger.info(json.dumps(trainer.state.log_history, indent=2))
        # logger.info(
        #     f"Saving the sharded model weights to {final_model_and_tokenizer_dir} ..."
        # )

    # this is the ideal way to save fsdp model weights, but it's broken now, so we have to do that manually
    # https://github.com/huggingface/accelerate/issues/3079
    # if trainer.is_fsdp_enabled:
    #     trainer.accelerator.state.fsdp_plugin.set_state_dict_type("FULL_STATE_DICT")
    # trainer._save(final_model_and_tokenizer_dir)

    # we have to save the model and the tokenizer by ourselves
    trainer.accelerator.wait_for_everyone()
    # state_dict = trainer.accelerator.get_state_dict(trainer.model, unwrap=False)
    # state_dict = OrderedDict([(k, v.to(torch.bfloat16)) for k, v in state_dict.items()])
    # trainer.accelerator.unwrap_model(model).save_pretrained(
    #     final_model_and_tokenizer_dir,
    #     state_dict=state_dict,
    #     safe_serialization=True,
    # )
    # trainer.processing_class.save_pretrained(final_model_and_tokenizer_dir)

    # if trainer.accelerator.is_main_process:
    #     for i, j in state_dict.items():
    #         print(i, j.shape)

    trainer.accelerator.end_training()


if __name__ == "__main__":
    main()
