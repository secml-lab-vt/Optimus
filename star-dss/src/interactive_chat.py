import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Union

import hydra
import torch
import torch.distributed as dist
import transformers
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf, open_dict
from openrlhf.models import Actor
from openrlhf.utils import get_tokenizer
from openrlhf.utils.deepspeed import DeepspeedStrategy

import deepspeed
from src.data import detect_message_type, load_system_prompt_from_file
from src.deepspeed import setup_deepspeed_for_inference

from .misc import set_seed

logger = logging.getLogger(__name__)
transformers.utils.logging.set_verbosity_error()


# By default, Hydra intercepts and manages CLI arguments, which prevents DeepSpeed's --local_rank from being passed correctly.
parser = argparse.ArgumentParser()
parser.add_argument("--local_rank", type=int, default=0)
args, unknown = parser.parse_known_args()  # Ignore unknown arguments
# print("here", args.local_rank, sys.argv)
# Remove `--local_rank` from sys.argv before Hydra processes it
sys.argv = [x for x in sys.argv if "--local_rank" not in x]


def reset_chat(
    apply_chat_template: bool = True,
    system: str = None,
    # rank: int = 0,
    chat_template: str = None,
) -> Union[List[Dict[str, str]], str]:
    # system prompt can be passed from a txt or directly as a string
    # if system is not None and system.endswith(".txt"):
    #     if Path(system).is_file():
    #         if rank == 0:
    #             logger.info(f"Loading system prompt from {system} ...")
    #         with open(system, "r") as f:
    #             system = f.read()
    #     else:
    #         if rank == 0:
    #             logger.info("The str ends with .txt but file is not found.")

    # # print(system)

    system = load_system_prompt_from_file(system)

    if apply_chat_template:
        assert chat_template is not None
        to_message = detect_message_type(chat_template)
        if system is not None:
            user_prompt = [to_message(role="system", content=system)]
        else:
            user_prompt = []

        # user_prompt = (
        #     [{"role": "system", "content": system}] if system is not None else []
        # )
    else:
        user_prompt = f"{system}\n" if system is not None else ""

    return user_prompt


@torch.no_grad()
@hydra.main(config_path="../config", config_name="interactive_chat", version_base="1.3")
def main(cfg: DictConfig):
    deepspeed.init_distributed()
    rank = args.local_rank
    device = torch.device(f"cuda:{rank}")
    set_seed(cfg.seed)

    if rank == 0:
        logger.info(OmegaConf.to_yaml(cfg, resolve=True))

    # configure model
    model = Actor(
        pretrain_or_model=cfg.model.name,
        use_flash_attention_2=cfg.model.flash_attn,
        bf16=cfg.model.bf16,
        load_in_4bit=cfg.model.load_in_4bit,
        # device_map=device,
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
    max_length = (
        512
        if cfg.model.generation_args.max_length is None
        else cfg.model.generation_args.max_length
    )
    max_length = cfg.model.generation_args.max_length
    if max_length is None:
        max_length = 5000
    setup_deepspeed_for_inference(model, ds_cfg=cfg.deepspeed, max_length=max_length)

    if cfg.model.apply_chat_template:
        if tokenizer.chat_template is None:
            logger.info(
                "There is no chat template provided, setting apply_chat_template to False"
            )
            apply_chat_template = False
        else:
            apply_chat_template = True
            to_message = detect_message_type(tokenizer.chat_template)

            prompt_template_kwargs = cfg.model.template_args
            if "add_generation_prompt" in prompt_template_kwargs:
                prompt_template_kwargs["add_generation_prompt"] = True
    else:
        apply_chat_template = False

    conversation = reset_chat(
        apply_chat_template,
        system=cfg.model.system,
        chat_template=tokenizer.chat_template,
    )

    while True:
        # only takes input on rank 0, broadcast tokenized tensor to all gpus, and only decode on rank 0
        if rank == 0:
            sys.stdout.flush()  # Ensure the order of the message in cli
            inputs = input(
                f"{'-' * 88}\nPlease enter a prompt (type 'clear' to start a new conversation or type 'exit' to quit)\nUser: "
            )
            logger.info(f"User: {inputs}")

            if inputs.strip().lower() == "exit":
                signal = 1
            elif inputs.strip().lower() == "clear":
                conversation = reset_chat(
                    apply_chat_template,
                    system=cfg.model.system,
                    chat_template=tokenizer.chat_template,
                )
                signal = 2
            else:
                signal = 0
        else:
            inputs = None
            signal = 0

        # Broadcast exit signal to all processes
        signal_tensor = torch.tensor([signal], dtype=torch.int, device=device)
        dist.broadcast(signal_tensor, src=0)
        signal = signal_tensor.item()

        if signal == 1:
            if rank == 0:
                print("Exiting program ...")
            dist.barrier()  # Sync all ranks before exiting
            dist.destroy_process_group()
            sys.exit(0)  # Clean exit
        elif signal == 2:
            if rank == 0:
                print("Starting a new conversation ...")
            continue

        if rank == 0:
            # get input prompt
            if apply_chat_template:
                conversation.append(to_message(role="user", content=inputs))
                user_prompt = tokenizer.apply_chat_template(
                    conversation, tokenize=False, **prompt_template_kwargs
                )
            else:
                user_prompt = (
                    conversation + "\n" + cfg.model.chat_template.format(inputs)
                )

            user_prompt_len = len(user_prompt)
            input_ids = tokenizer.encode(
                user_prompt,
                return_tensors="pt",
                add_special_tokens=not apply_chat_template,
            ).to(device)
            input_ids_shape = torch.tensor(
                input_ids.shape, dtype=torch.int, device=device
            )
        else:
            input_ids_shape = torch.empty(2, dtype=torch.int, device=device)

        dist.broadcast(input_ids_shape, src=0)

        # print(rank, input_ids_shape)

        if rank != 0:
            input_ids = torch.empty(
                input_ids_shape.tolist(), dtype=torch.int64, device=device
            )

        dist.broadcast(input_ids, src=0)
        dist.barrier()

        if (
            cfg.model.generation_args.max_length is not None
            and input_ids.shape[-1] >= cfg.model.generation_args.max_length
        ):
            if rank == 0:
                logger.warning(
                    f"Current conversation length: {input_ids.shape[-1]} > Max supported length: {cfg.model.generation_args.max_length}. Starting a new conversation ..."
                )
            conversation = reset_chat(
                apply_chat_template, system=cfg.model.system, rank=rank
            )
            continue

        # print(rank, input_ids)

        outputs = model.generate(
            input_ids=input_ids,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            **cfg.model.generation_args,
        )

        # print(rank, outputs)

        if rank == 0:
            if apply_chat_template:
                generated_ids = outputs[0][:, input_ids.shape[1] :]
                response = tokenizer.batch_decode(
                    generated_ids,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True,
                )[0]
                conversation.append(to_message(role="assistant", content=response))
            else:
                conversation = tokenizer.batch_decode(
                    outputs[0], skip_special_tokens=True
                )[0]
                response = conversation[user_prompt_len:]

            logger.info(f"Assistant: {response}")


if __name__ == "__main__":
    main()
