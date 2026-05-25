import argparse
import logging
import math
import os
import sys
from datetime import datetime
from pathlib import Path

import hydra
import torch
import torch.distributed as dist
import transformers
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from openrlhf.models import Actor
from openrlhf.utils import get_tokenizer
from openrlhf.utils.deepspeed import DeepspeedStrategy
from transformers import get_scheduler

from src.data import RewardDatasetHydraConfig, blend_datasets
from src.trainer import DPOTrainer
from .misc import set_seed

logger = logging.getLogger(__name__)

# By default, Hydra intercepts and manages CLI arguments, which prevents DeepSpeed's --local_rank from being passed correctly.
parser = argparse.ArgumentParser()
parser.add_argument("--local_rank", type=int, default=0)
args, unknown = parser.parse_known_args()  # Ignore unknown arguments
# print("here", args.local_rank, sys.argv)
# Remove `--local_rank` from sys.argv before Hydra processes it
sys.argv = [x for x in sys.argv if "--local_rank" not in x]


@hydra.main(config_path="../config", config_name="dpo_openrlhf", version_base="1.3")
def main(cfg: DictConfig):
    output_dir = Path(os.getcwd())  # output to experiment/${name}

    # configure deepspeed
    cfg.deepspeed.local_rank = args.local_rank
    strategy = DeepspeedStrategy(
        seed=cfg.seed,
        max_norm=cfg.train.max_norm,
        micro_train_batch_size=cfg.train.micro_train_batch_size,
        train_batch_size=cfg.train.train_batch_size,
        zero_stage=cfg.deepspeed.zero_stage,
        bf16=cfg.model.bf16,
        args=cfg.deepspeed,
    )
    strategy.setup_distributed()

    if strategy.is_rank_0():
        logger.info(OmegaConf.to_yaml(cfg, resolve=True))

    # load HF model and tokenizer
    model = Actor(
        pretrain_or_model=cfg.model.name,
        use_flash_attention_2=cfg.model.flash_attn,
        bf16=cfg.model.bf16,
        load_in_4bit=cfg.model.load_in_4bit,
        lora_rank=cfg.model.lora.rank,
        lora_alpha=cfg.model.lora.alpha,
        target_modules=cfg.model.lora.target_modules,
        lora_dropout=cfg.model.lora.dropout,
        ds_config=strategy.get_ds_train_config(is_actor=True),
        packing_samples=cfg.train.packing_samples,
    )

    tokenizer = get_tokenizer(
        pretrain=cfg.model.name,
        model=model.model,
        padding_side="right",
        strategy=strategy,
        use_fast=not cfg.model.disable_fast_tokenizer,
    )
    if strategy.is_rank_0():
        logger.info(model)

    # load reference model weights
    ref_model = Actor(
        pretrain_or_model=cfg.ref_model.name,
        use_flash_attention_2=cfg.ref_model.flash_attn,
        bf16=cfg.ref_model.bf16,
        load_in_4bit=cfg.ref_model.load_in_4bit,
        ds_config=strategy.get_ds_eval_config(offload=cfg.train.ref_offload),
        packing_samples=cfg.train.packing_samples,
    )
    if cfg.train.ref_offload:
        ref_model._offload = True
    # calling get_tokenizer to set the model pad_token_id, the tokenizer should be the same
    ref_tokenizer = get_tokenizer(
        pretrain=cfg.ref_model.name,
        model=ref_model.model,
        padding_side="right",
        strategy=strategy,
        use_fast=not cfg.ref_model.disable_fast_tokenizer,
    )

    # gradient checkpointing
    if cfg.train.gradient_checkpointing:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={
                "use_reentrant": cfg.train.gradient_checkpointing_use_reentrant
            }
        )

    # optimizer
    optimizer = strategy.create_optimizer(
        model,
        lr=cfg.train.learning_rate,
        betas=cfg.train.adam_betas,
        weight_decay=cfg.train.l2,
    )

    # dataset
    train_data, eval_data = blend_datasets(datasets_cfg=cfg.data, seed=cfg.seed)

    # train data processing
    train_data = train_data.select(range(min(cfg.data.max_samples, len(train_data))))
    train_dataset = RewardDatasetHydraConfig(
        data=train_data,
        data_cfg=cfg.data,
        tokenizer=tokenizer,
        max_length=cfg.train.max_len,
        is_dpo=True,
        apply_chat_template=cfg.data.apply_chat_template,
        multiple_of=cfg.model.ring_attn_size,
    )

    train_dataloader = strategy.setup_dataloader(
        replay_buffer=train_dataset,
        batch_size=cfg.train.micro_train_batch_size,
        pin_memory=True,
        shuffle=True,
        collate_fn=train_dataset.packing_collate_fn
        if cfg.train.packing_samples
        else train_dataset.collate_fn,
    )

    # eval data processing
    if cfg.data.use_eval:
        eval_data = eval_data.select(range(min(cfg.data.max_samples, len(eval_data))))
        eval_dataset = RewardDatasetHydraConfig(
            data=eval_data,
            data_cfg=cfg.data,
            tokenizer=tokenizer,
            max_length=cfg.train.max_len,
            is_dpo=True,
            apply_chat_template=cfg.data.apply_chat_template,
            multiple_of=cfg.model.ring_attn_size,
        )

        eval_dataloader = strategy.setup_dataloader(
            replay_buffer=eval_dataset,
            batch_size=cfg.train.micro_train_batch_size,
            pin_memory=True,
            shuffle=False,
            collate_fn=eval_dataset.packing_collate_fn
            if cfg.train.packing_samples
            else eval_dataset.collate_fn,
        )

    # scheduler
    num_update_steps_per_epoch = len(train_dataset) // cfg.train.train_batch_size
    max_steps = math.ceil(cfg.train.max_epochs * num_update_steps_per_epoch)

    scheduler = get_scheduler(
        name=cfg.train.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=math.ceil(max_steps * cfg.train.lr_warmup_ratio),
        num_training_steps=max_steps,
        scheduler_specific_kwargs={"min_lr": cfg.train.min_learning_rate},
    )

    # prepare models
    (model, optimizer, scheduler), ref_model = strategy.prepare(
        (model, optimizer, scheduler), ref_model
    )

    # load checkpoint
    consumed_samples = 0
    if cfg.log.load_checkpoint and os.path.exists(cfg.log.checkpoint_path):
        _, states = strategy.load_ckpt(model.model, cfg.log.checkpoint_path)
        consumed_samples = states["consumed_samples"]
        strategy.print(
            f"Loaded the checkpoint: {cfg.log.checkpoint_path}, consumed_samples: {consumed_samples}"
        )

    # save model path
    cfg.log.ckpt_path = output_dir / cfg.log.ckpt_path
    cfg.log.save_path = output_dir / cfg.log.save_path

    # configure Trainer
    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        strategy=strategy,
        optim=optimizer,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        scheduler=scheduler,
        max_norm=cfg.train.max_norm,
        beta=cfg.train.beta,
        max_epochs=cfg.train.max_epochs,
        tokenizer=tokenizer,
        save_hf_ckpt=cfg.log.save_hf_ckpt,
        disable_ds_ckpt=cfg.log.disable_ds_ckpt,
    )

    trainer.fit(strategy.args, consumed_samples, num_update_steps_per_epoch)

    strategy.save_model(model, tokenizer, cfg.log.save_path)

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
