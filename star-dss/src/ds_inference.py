import argparse
import itertools
import logging
import os
import shutil
import sys
from functools import partial
from pathlib import Path
from typing import Dict, List, Union

import hydra
import jsonlines
import torch
import torch.distributed as dist
import transformers
from datasets import load_dataset
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf, open_dict
from openrlhf.models import Actor
from openrlhf.utils import get_tokenizer
from openrlhf.utils.deepspeed import DeepspeedStrategy
from tqdm import tqdm

import deepspeed
from src.data import (
    SFTDatasetHydraConfig,
    batch_to_device,
    blend_datasets,
    setup_dataloader_for_inference,
)
from src.deepspeed import setup_deepspeed_for_inference
from src.metric import compute_metric, guardrail_answer_to_value

from .misc import ensure_empty_parent_dir, set_seed

logger = logging.getLogger(__name__)
transformers.utils.logging.set_verbosity_error()


# By default, Hydra intercepts and manages CLI arguments, which prevents DeepSpeed's --local_rank from being passed correctly.
parser = argparse.ArgumentParser()
parser.add_argument("--local_rank", type=int, default=0)
args, unknown = parser.parse_known_args()  # Ignore unknown arguments
# print("here", args.local_rank, sys.argv)
# Remove `--local_rank` from sys.argv before Hydra processes it
sys.argv = [x for x in sys.argv if "--local_rank" not in x]


# TODO we may add vllm inference in the future, vllm inference may not fully support DDO+MP
# This is batch inference with deepspeed, fsdp will be in another file


@torch.no_grad()
@hydra.main(config_path="../config", config_name="ds_inference", version_base="1.3")
def main(cfg: DictConfig):
    output_dir = Path(os.getcwd())  # output to experiment/infer/${name}

    # deepspeed inference config
    deepspeed.init_distributed()
    rank = args.local_rank
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(rank)
    set_seed(cfg.seed)

    if rank == 0:
        logger.info(OmegaConf.to_yaml(cfg, resolve=True))

    # configure model
    model = Actor(
        pretrain_or_model=cfg.model.name,
        use_flash_attention_2=cfg.model.flash_attn,
        bf16=cfg.model.bf16,
        load_in_4bit=cfg.model.load_in_4bit,
    )

    # configure tokenizer
    tokenizer = get_tokenizer(
        pretrain=cfg.model.name,
        model=model.model,
        padding_side="left",
        strategy=None,
        use_fast=not cfg.model.disable_fast_tokenizer,
    )
    model.eval()

    if cfg.model.generation_args.max_length is None:
        if cfg.model.generation_args.max_new_tokens is None:
            max_length = 512
        else:
            max_length = max(512, cfg.model.generation_args.max_new_tokens)
    else:
        max_length = cfg.model.generation_args.max_length
        cfg.model.generation_args.max_new_tokens = None
        if rank == 0:
            logger.info(
                f"Setting generation max_length={max_length} and voiding max_new_tokens as deepspeed only takes max_length"
            )

    model = setup_deepspeed_for_inference(
        model,
        ds_cfg=cfg.deepspeed,
        max_length=max_length,
    )

    # load dataset
    infer_data, _ = blend_datasets(datasets_cfg=cfg.data, seed=cfg.seed)

    # Guradrail inference only - evaluate model's response
    for_guardrail = cfg.infer.for_guardrail
    if for_guardrail and cfg.infer.generation_filename is not None:
        generation_file = (
            f"{output_dir}/generation_{cfg.infer.generation_filename}.jsonl"
        )
        if rank == 0:
            logger.info(f"Loading generation file from {generation_file}")
        generation_data = load_dataset(
            path="json",
            data_files={"eval": generation_file},
            split="eval",
        )

        generation_data = generation_data.remove_columns(
            [col for col in generation_data.column_names if col != "generation"]
        ).rename_column("generation", "response")

        assert len(infer_data) == len(generation_data)
        for col in generation_data.column_names:
            if col in infer_data.column_names:
                infer_data = infer_data.remove_columns(col)
            infer_data = infer_data.add_column(col, generation_data[col])

    # inference data processing
    # infer_data = infer_data.select(range(min(cfg.data.max_samples, len(infer_data))))
    infer_dataset = SFTDatasetHydraConfig(
        data=infer_data,
        data_cfg=cfg.data,
        tokenizer=tokenizer,
        max_length=max_length,
        multiturn=False,
        pretrain_mode=False,
        apply_chat_template=cfg.data.apply_chat_template,
        system_prompt=cfg.model.system,
        for_guardrail=for_guardrail,
        chat_template_kwargs=cfg.model.template_args,
    )

    infer_dataloader = setup_dataloader_for_inference(
        replay_buffer=infer_dataset,
        batch_size=cfg.infer.micro_batch_size,
        pin_memory=True,
        shuffle=False,
        collate_fn=partial(infer_dataset.collate_fn, padding_side="left"),
        seed=cfg.seed,
    )

    pbar = tqdm(
        infer_dataloader,
        desc="Generating",
        disable=rank != 0,
    )

    dist.barrier(device_ids=[rank])

    prompt_label_generation_per_process = []
    apply_chat_template = cfg.data.apply_chat_template
    # note the tokenized inputs in sft dataset includes response, so we need to tokenize again on prompts only
    for _, prompt_label, _, infos in pbar:
        inputs = tokenizer(
            infos["output"] if for_guardrail else infos["input"],
            max_length=max_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
            add_special_tokens=not apply_chat_template,
        )
        prompt_id_lens = inputs["input_ids"].shape[-1]
        # we decode the text for extracting labels
        prompt_texts = tokenizer.batch_decode(
            inputs["input_ids"], skip_special_tokens=True
        )
        prompt_label_texts = tokenizer.batch_decode(
            prompt_label, skip_special_tokens=True
        )

        # print(f"GPU {rank}: {prompt_texts} {len(prompt_texts)} {prompt_label_texts}")

        inputs = batch_to_device(inputs, device)
        # print(
        #     f"{rank} before {inputs['input_ids'].shape} {inputs['attention_mask'].shape}"
        # )
        with torch.no_grad():
            model_outputs_ids = model.model.generate(
                **inputs,
                **cfg.model.generation_args,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                # synced_gpus=True,
            )

        # print(f"GPU {rank}: {model_outputs_ids}")
        # print(f"{rank} after")
        model_outputs = tokenizer.batch_decode(
            model_outputs_ids[:, prompt_id_lens:], skip_special_tokens=True
        )

        for prompt, prompt_text, prompt_label_text, generation in zip(
            infos["output"] if for_guardrail else infos["input"],
            prompt_texts,
            prompt_label_texts,
            model_outputs,
        ):
            # save the input after applying chat template, so it is easy to regenerate with the same prompt
            prompt_label_generation_per_process.append(
                {
                    "input": prompt,
                    "prompt": prompt_text,
                    "label": prompt_label_text[len(prompt_text) :],
                    "generation": generation,
                }
            )

        # break

    tmp_path = output_dir / cfg.log.tmp_path / f"_device_{rank}.jsonl"
    if rank == 0:
        ensure_empty_parent_dir(tmp_path)
    dist.barrier(device_ids=[rank])

    with jsonlines.open(tmp_path, mode="w") as f:
        f.write_all(prompt_label_generation_per_process)
    dist.barrier(device_ids=[rank])

    # merge per process output file into a single file
    # wrtie in a round robin fashion like the distributed sampler
    if rank == 0:
        world_size = dist.get_world_size()
        files = [
            output_dir / cfg.log.tmp_path / f"_device_{rank}.jsonl"
            for rank in range(world_size)
        ]

        readers = [jsonlines.open(file, "r") for file in files]
        prompt_label_generation_all = []

        for entries in itertools.zip_longest(*readers, fillvalue=None):
            for entry in entries:
                if entry is not None:
                    prompt_label_generation_all.append(entry)

        prompt_label_generation_all = prompt_label_generation_all[: len(infer_dataset)]

        if for_guardrail:
            final_path = output_dir / f"guardrail_{cfg.name}.jsonl"
        else:
            final_path = output_dir / f"generation_{cfg.name}.jsonl"
        with jsonlines.open(final_path, "w") as f:
            f.write_all(prompt_label_generation_all)

        # using data.d1 is not robust as d1 is subject to change
        # if a new dataset is created, the id is overwritten
        dataset_id = cfg.data.d1.id
        model_name = cfg.model.name.split("/")[
            -1
        ].lower()  # assuming guardrail models are all from HF

        # create mixture dataset with values
        if cfg.save_dataset:
            # new dataset path
            # ds_items = [
            #     (ds_cfg.id.lower(), ds_cfg.probability)
            #     for name, ds_cfg in cfg.data.items()
            #     if name.startswith("d") and name[1:].isdigit()
            # ]

            # ds_dir = cfg.data_dir
            # # if len(ds_items) > 1:
            # new_ds_folder = Path(
            #     f"{ds_dir}/mix_{'_'.join(name for name, _ in ds_items)}"
            # )
            new_ds_folder = Path(cfg.data_dir) / "local"

            # else:
            #     new_ds_folder = Path(f"{ds_dir}/{ds_items[0][0]}")
            new_ds_folder.mkdir(parents=True, exist_ok=True)
            # dataset_id = (
            #     "_".join(f"{name}_p{int(prob * 100):03d}" for name, prob in ds_items)
            #     + ".jsonl"
            # )
            dataset_id = f"{cfg.name}.jsonl"

            # add value col
            value = guardrail_answer_to_value(
                prompt_label_generation_all,
                label_name="label",
                prediction_name="generation",
                model_name=model_name,
            )
            # dummy position for RS
            position = [[-1] for _ in range(len(value))]
            new_ds = infer_data.add_column("value", value)
            new_ds = new_ds.add_column("position", position)

            # save dataset as jsonlines
            new_ds.to_json(new_ds_folder / dataset_id, lines=True)

            logger.info(
                f"Saving a new dataset {dataset_id} with {len(new_ds)} lines to {new_ds_folder}"
            )

        # compute metric
        # if the evaluation incur calling llm as a judge, we will defer to another separate file
        if cfg.infer.compute_metric:
            results = compute_metric(
                dataset_id,
                prompt_label_generation_all,
                label_name="label",
                prediction_name="generation",
                model_name=model_name,
            )

            logger.info(
                f"{dataset_id} evaluation results: {results[0]}%. Additional results: {results[1:]}"
            )

    dist.barrier(device_ids=[rank])

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
