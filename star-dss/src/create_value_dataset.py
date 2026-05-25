import argparse
import itertools
import logging
import os
import shutil
import sys
import time
from functools import partial
from pathlib import Path
from typing import Dict, List, Union

import hydra
import jsonlines
import torch
import torch.distributed as dist
import transformers
from datasets import Dataset, concatenate_datasets, load_dataset
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf, open_dict
from openrlhf.models import Actor
from openrlhf.utils.deepspeed import DeepspeedStrategy
from tqdm import tqdm

import deepspeed
from src.data import (
    SFTDatasetHydraConfig,
    batch_to_device,
    blend_datasets,
    process_shard_batched,
    setup_dataloader_for_inference,
)
from src.deepspeed import setup_deepspeed_for_inference
from src.metric import compute_metric, guardrail_answer_to_value
from src.model import get_tokenizer

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


def shard_data(data: Dataset, rank: int, world_size: int):
    total_size = len(data)
    # shard_size = (total_size + world_size - 1) // world_size
    shard_size = total_size // world_size
    start_idx = shard_size * rank
    end_idx = min(start_idx + shard_size, total_size)
    logger.info(f"[Rank {rank}] Processing shard {start_idx} to {end_idx}")

    return data.select(range(start_idx, end_idx))


def pad_dataset(data, world_size):
    total_size = len(data)
    remainder = total_size % world_size
    if remainder == 0:
        return data
    padding_needed = world_size - remainder
    padding_samples = data.select(range(padding_needed))
    padded_data = concatenate_datasets([data, padding_samples])

    assert len(padded_data) % world_size == 0

    return padded_data


@torch.no_grad()
@hydra.main(config_path="../config", config_name="ds_inference", version_base="1.3")
def main(cfg: DictConfig):
    output_dir = Path(os.getcwd())  # output to experiment/infer/${name}

    # deepspeed inference config
    deepspeed.init_distributed()
    rank = args.local_rank
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(rank)
    world_size = dist.get_world_size()
    set_seed(cfg.seed)

    if rank == 0:
        logger.info(OmegaConf.to_yaml(cfg, resolve=True))
        start_time = time.time()

    # configure model
    guard_models = []
    guard_tokenizers = []

    for _, model_cfg in cfg.guard_model.items():
        model = Actor(
            pretrain_or_model=model_cfg.name,
            use_flash_attention_2=model_cfg.flash_attn,
            bf16=model_cfg.bf16,
            load_in_4bit=model_cfg.load_in_4bit,
        )

        # configure tokenizer
        tokenizer = get_tokenizer(
            pretrain=model_cfg.name,
            model=model.model,
            padding_side="left",
            strategy=None,
            use_fast=not model_cfg.disable_fast_tokenizer,
        )

        model.eval()

        guard_models.append(model)
        guard_tokenizers.append(tokenizer)

    # actual llm tokenizer used for finetuning
    llm_tokenizer = get_tokenizer(
        pretrain=cfg.model.name,
        model=None,
        padding_side="right",
        strategy=None,
        use_fast=not cfg.model.disable_fast_tokenizer,
    )

    if cfg.model.generation_args.max_length is None:
        if cfg.model.generation_args.max_new_tokens is None:
            max_length = 512
        else:
            max_length = max(512, cfg.model.generation_args.max_new_tokens)
    else:
        max_length = cfg.model.generation_args.max_length
        cfg.model.generation_args.max_new_tokens = None
        logger.info(
            f"Setting generation max_length={max_length} and voiding max_new_tokens as deepspeed only takes max_length"
        )

    guard_models = [
        setup_deepspeed_for_inference(
            model, ds_cfg=cfg.deepspeed, max_length=max_length
        )
        for model in guard_models
    ]

    apply_chat_template = cfg.data.apply_chat_template
    bs = cfg.infer.micro_batch_size

    # load dataset
    infer_data, _ = blend_datasets(datasets_cfg=cfg.data, seed=cfg.seed)
    total_data_samples = len(infer_data)

    infer_data = pad_dataset(infer_data, world_size=world_size)
    data_shard_per_process = shard_data(infer_data, rank, world_size)

    guard_values_per_process = []
    token_position_indices_per_process = []

    pbar = tqdm(
        range(0, len(data_shard_per_process), bs),
        desc="Creating dataset:",
        disable=rank != 0,
    )

    for start_idx in pbar:
        data_shard = data_shard_per_process.select(
            range(start_idx, min(len(data_shard_per_process), start_idx + bs))
        )

        guard_values, token_position_indices = process_shard_batched(
            data_shard=data_shard,
            guard_models=guard_models,
            guard_tokenizers=guard_tokenizers,
            llm_tokenizer=llm_tokenizer,
            chunk_size=cfg.chunk_size,
            llm_cfg=cfg.model,
            guard_cfgs=cfg.guard_model,
            apply_chat_template=apply_chat_template,
            device=device,
        )

        guard_values_per_process.extend(guard_values)
        token_position_indices_per_process.extend(token_position_indices)

        # print(
        #     f"[{rank}] {len(data_shard_per_process)} {len(guard_values_per_process)} {len(token_position_indices_per_process)}",
        #     flush=True,
        # )

    # add two new columns to the dataset
    data_shard_per_process = data_shard_per_process.add_column(
        "value", guard_values_per_process
    )
    data_shard_per_process = data_shard_per_process.add_column(
        "position", token_position_indices_per_process
    )
    dist.barrier(device_ids=[rank])

    # save per process
    tmp_path = output_dir / cfg.log.tmp_path / f"_device_{rank}.jsonl"
    if rank == 0:
        ensure_empty_parent_dir(tmp_path)
    dist.barrier(device_ids=[rank])

    data_shard_per_process.to_json(tmp_path, lines=True)
    dist.barrier(device_ids=[rank])

    if rank == 0:
        world_size = dist.get_world_size()
        files = [
            output_dir / cfg.log.tmp_path / f"_device_{rank}.jsonl"
            for rank in range(world_size)
        ]

        readers = [jsonlines.open(file, "r") for file in files]
        data_shard_all = []
        for entries in readers:
            for entry in entries:
                data_shard_all.append(entry)

        data_shard_all = data_shard_all[:total_data_samples]

        new_data_path = Path(cfg.local_data_dir) / f"{cfg.local_dataset_name}.jsonl"
        with jsonlines.open(new_data_path, "w") as f:
            f.write_all(data_shard_all)
            
        end_time = time.time()
        elapsed_time = (end_time - start_time) / 60  # convert seconds to minutes
        logger.info(f"Time spent for creating value dataset: {elapsed_time:.2f} minutes")

    dist.barrier(device_ids=[rank])
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
