import logging
import os
from functools import partial
from pathlib import Path

import hydra
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

from src.data import batch_to_device, preprocess_and_tokenize_dataset
from src.model import forward_llm, load_model_and_tokenizer, save_generation

logger = logging.getLogger(__name__)


@torch.no_grad()
@hydra.main(config_path="../config", config_name="inference", version_base="1.3")
def main(cfg: DictConfig):
    # initialization
    accelerator = Accelerator()
    set_seed(cfg.seed)
    output_dir = Path(os.getcwd())  # output to experiment/${name}

    if accelerator.is_main_process:
        logger.info(OmegaConf.to_yaml(cfg, resolve=True))

    # load model and tokenizer (with chat template)
    # TODO system prompt
    if accelerator.is_main_process:
        logger.info(f"Loading model from {cfg.model.name} ...")
    model, tokenizer = load_model_and_tokenizer(
        cfg.model.name,
        config_args=cfg.model.config_args,
        model_args=cfg.model.model_args,
        tokenizer_args=cfg.model.tokenizer_args,
    )

    # disable gradients for inference
    for param in model.parameters():
        param.requires_grad = False

    model = accelerator.prepare_model(model, evaluation_mode=False)

    model.to(accelerator.device)
    model.eval()

    # fsdp specific - it will cause OOM, so we will not use accelerator here
    # model = accelerator.prepare_model(model, evaluation_mode=True)

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

    # data collator
    batch_size = cfg.data.inference_batch_size
    collate_fn = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    batched_dataset = [
        [dataset[j] for j in range(i, min(i + batch_size, len(dataset)))]
        for i in range(0, len(dataset), batch_size)
    ]

    # forward
    forward_llm_with_data = partial(
        forward_llm,
        accelerator=accelerator,
        dataset=batched_dataset,
        collate_fn=collate_fn,
        generation_args=cfg.model.generation_args,
        tokenizer=tokenizer,
        dataset_length=len(dataset),
    )

    all_outputs = forward_llm_with_data(model)

    # outputs_per_process = []
    # with accelerator.split_between_processes(
    #     batched_dataset, apply_padding=True
    # ) as batched_inputs:
    #     for batch in tqdm(batched_inputs, desc="Generating"):
    #         batch = collate_fn(batch)
    #         batch = batch_to_device(batch, accelerator.device)

    #         outputs = model.generate(
    #             **batch,
    #             **cfg.model.generation_args,
    #             pad_token_id=tokenizer.pad_token_id,
    #         )

    #         input_length = batch["input_ids"].shape[-1]
    #         prompt_text = tokenizer.batch_decode(
    #             outputs[:, :input_length], skip_special_tokens=True
    #         )
    #         generated_text = tokenizer.batch_decode(
    #             outputs[:, input_length:], skip_special_tokens=True
    #         )

    #         outputs_per_process += [
    #             {"prompt": i, "generation": j}
    #             for i, j in zip(prompt_text, generated_text)
    #         ]

    # outputs_gather = gather_object(outputs_per_process)
    # # Drop duplicates produced by apply_padding in split_between_processes
    # all_outputs = outputs_gather[: len(dataset)]

    # save output
    if accelerator.is_main_process:
        save_generation(
            output=all_outputs, save_to=output_dir, filename="generation.jsonl"
        )

    # evaluation (optional)
    # if accelerator.is_main_process:
    #     logger.info("Evaluated with {} in progress ...")

    accelerator.end_training()


if __name__ == "__main__":
    main()
