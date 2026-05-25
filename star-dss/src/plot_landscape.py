import json
import logging
import os
from copy import deepcopy
from functools import partial
from pathlib import Path
from typing import Optional

import hydra
import numpy as np
import torch
from accelerate import Accelerator, init_empty_weights, load_checkpoint_and_dispatch
from accelerate.utils import gather_object, set_seed
from datasets import load_dataset
from hydra import compose, initialize, initialize_config_dir, initialize_config_module
from hydra.utils import get_original_cwd, instantiate
from omegaconf import DictConfig, OmegaConf
from torch import nn
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
)

from src.data import preprocess_and_tokenize_dataset
from src.landscape import Bezier, PolyChain
from src.model import (
    MetaLLM,
    forward_llm,
    load_model_and_tokenizer,
    load_model_and_tokenizer_and_anchors,
    save_generation,
)

logger = logging.getLogger(__name__)


@torch.no_grad()
@hydra.main(config_path="../config", config_name="landscape", version_base="1.3")
def main(cfg: DictConfig):
    # initialization
    accelerator = Accelerator()
    set_seed(cfg.seed)
    output_dir = Path(os.getcwd())  # output to experiment/${name}

    if accelerator.is_main_process:
        logger.info(OmegaConf.to_yaml(cfg, resolve=True))

    # load empty model and anchors for MetaLLM
    model, tokenizer, anchors = load_model_and_tokenizer_and_anchors(
        cfg.landscape.model_name_or_path_list,
        config_args=cfg.model.config_args,
        model_args=cfg.model.model_args,
        tokenizer_args=cfg.model.tokenizer_args,
        use_grad=False,
    )
    model.eval()
    anchors = [anchor.eval() for anchor in anchors]

    # move to device
    model = accelerator.prepare_model(model, evaluation_mode=False)
    anchors = [
        accelerator.prepare_model(anchor, evaluation_mode=False) for anchor in anchors
    ]

    # dataset
    if accelerator.is_main_process:
        logger.info(
            f"Loading {cfg.data.name} dataset and adding chat template if applicable ..."
        )


    load_cfg = cfg.data.load_dataset
    dataset = load_dataset(
        path=load_cfg.path,
        name=load_cfg.name if "name" in load_cfg else None,
        split=load_cfg.split if "split" in load_cfg else None,
        data_files=OmegaConf.to_container(load_cfg.data_files, resolve=True)
        if "data_files" in load_cfg
        else None,
    )

    data_cfg = cfg.data
    apply_chat_template = tokenizer.chat_template is not None
    if data_cfg.data_format == "mmlu_llama":
        apply_chat_template = False

    with accelerator.main_process_first():
        dataset = dataset.map(
            partial(
                preprocess_and_tokenize_dataset,
                column_names=data_cfg.preprocess_column_names,
                tokenizer=tokenizer,
                apply_chat_template=apply_chat_template,
                continue_final_message=data_cfg.continue_final_message,
                add_generation_prompt=data_cfg.add_generation_prompt,
                data_format=data_cfg.data_format,
            ),
            batched=True,
            remove_columns=data_cfg.remove_column_names
            if "remove_column_names" in data_cfg
            and data_cfg.remove_column_names is not None
            else dataset.column_names,
        )





    
    # dataset = load_dataset(
    #     path=cfg.data.load_dataset.path,
    #     data_files=OmegaConf.to_container(
    #         cfg.data.load_dataset.data_files, resolve=True
    #     ),
    # )
    # assert "test" in dataset
    # dataset = dataset["test"]

    # apply_chat_template = tokenizer.chat_template is not None
    # # apply_chat_template = False
    # dataset = dataset.map(
    #     partial(
    #         preprocess_and_tokenize_dataset,
    #         column_names=cfg.data.preprocess_column_names,
    #         tokenizer=tokenizer,
    #         apply_chat_template=apply_chat_template,
    #         continue_final_message=cfg.data.continue_final_message,
    #         add_generation_prompt=cfg.data.add_generation_prompt,
    #         data_format=cfg.data.data_format,
    #     ),
    #     batched=True,
    #     remove_columns=cfg.data.remove_column_names,
    # )

    # data collator
    batch_size = cfg.data.inference_batch_size
    collate_fn = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    batched_dataset = [
        [dataset[j] for j in range(i, min(i + batch_size, len(dataset)))]
        for i in range(0, len(dataset), batch_size)
    ]

    # forward landscape
    num_anchors = len(anchors)
    meta_model = MetaLLM(model=model, anchors=anchors)

    curve_type = cfg.landscape.curve
    if curve_type == "polygonal":
        direction = PolyChain(num_anchors)
    elif curve_type == "bezier":
        direction = Bezier(num_anchors)
    else:
        raise NotImplementedError

    forward_llm_with_data = partial(
        forward_llm,
        accelerator=accelerator,
        dataset=batched_dataset,
        collate_fn=collate_fn,
        generation_args=cfg.model.generation_args,
        tokenizer=tokenizer,
        dataset_length=len(dataset),
    )

    # outputs = forward_llm_with_data(model)

    outputs = meta_model.inference(
        direction,
        forward_llm=forward_llm_with_data,
        points=cfg.landscape.points,
    )

    # save output
    if accelerator.is_main_process:
        save_generation(output=outputs, save_to=output_dir, filename="generation.jsonl")

        # debug
        idx = 4
        print_prompt = True
        for output in outputs:
            if print_prompt:
                print(output["output"][idx]["prompt"])
                print_prompt = False
            print("====\n", output["output"][idx]["generation"])

    accelerator.end_training()


if __name__ == "__main__":
    main()
