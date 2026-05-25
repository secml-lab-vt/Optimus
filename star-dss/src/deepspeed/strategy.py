import os
from datetime import timedelta

import torch
import torch.distributed as dist
from omegaconf import DictConfig
from openrlhf.models import Actor
from openrlhf.utils.deepspeed import DeepspeedStrategy
from openrlhf.utils.deepspeed.deepspeed_utils import get_eval_ds_config
from transformers import PreTrainedModel

import deepspeed
from deepspeed.inference.config import DeepSpeedInferenceConfig, DeepSpeedTPConfig

__all__ = ["setup_deepspeed_for_inference"]


def setup_deepspeed_for_inference(
    model: PreTrainedModel,
    ds_cfg: DictConfig,
    max_length: int,
    # batch_size: int,
    # local_rank: int = None,
    # bf16: bool = True,
):
    is_actor = isinstance(model, Actor)

    stage = ds_cfg.zero_stage

    # ============ other logic
    # ds_config = get_eval_ds_config(
    #     offload=False, stage=stage if stage == 3 else 0, bf16=bf16
    # )
    # ds_config["train_micro_batch_size_per_gpu"] = 1
    # ds_config["train_batch_size"] = batch_size

    # engine, *_ = deepspeed.initialize(
    #     model=model.model if is_actor else model,
    #     args={"local_rank": int(local_rank)},
    #     config=ds_config,
    #     dist_init_required=True,
    # )

    # ============ working logic
    # Define Tensor Parallelism Config: https://deepspeed.readthedocs.io/en/stable/inference-init.html
    tp_config = {
        "enabled": True,
        "tp_size": dist.get_world_size() if stage == 3 else 1,  # Tensor Parallelism
    }
    zero_config = {
        "stage": stage
    }
    config = {"tensor_parallel": tp_config, "max_out_tokens": max_length, "zero": zero_config}

    engine = deepspeed.init_inference(
        model=model.model if is_actor else model, config=config
    )

    # =======================

    if is_actor:
        model.model = engine
    else:
        model = engine

    return model
