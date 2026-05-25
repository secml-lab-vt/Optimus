import logging
from pathlib import Path
from typing import Callable, Dict, List, Optional, Union

import jsonlines
import torch
from accelerate import Accelerator, init_empty_weights
from accelerate.utils import gather_object, tqdm
from torch import Tensor, nn
from torch.distributed import checkpoint
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
)

from src.data import batch_to_device

__all__ = [
    "load_model_and_tokenizer",
    "save_generation",
    "forward_llm",
    "load_model_and_tokenizer_and_anchors",
    "load_sharded_weights",
    "get_tokenizer",
]

logger = logging.getLogger(__name__)


def get_tokenizer(pretrain, model, padding_side="left", strategy=None, use_fast=True):
    tokenizer = AutoTokenizer.from_pretrained(
        pretrain, trust_remote_code=True, use_fast=use_fast
    )
    tokenizer.padding_side = padding_side
    # NOTE: When enable vLLM, do not resize_token_embeddings, or the vocab size will mismatch with vLLM.
    # https://github.com/facebookresearch/llama-recipes/pull/196
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        if model is not None:
            model.config.pad_token_id = tokenizer.pad_token_id

    return tokenizer


def load_model_and_tokenizer(
    model_name_or_path,
    config_args: Optional[dict] = None,
    model_args: Optional[dict] = None,
    tokenizer_args: Optional[dict] = None,
):
    # TODO peft config probably should be added here
    # TODO quantization and 8bit and 4 bit
    # bnb_config = None
    # quant_storage_stype = None
    # load_in_8bit = args.use_8bit_qunatization
    # load_in_4bit = args.use_4bit_quantization
    # peft_config = None
    # peft_config = LoraConfig(
    #     lora_alpha=args.lora_alpha,
    #     lora_dropout=args.lora_dropout,
    #     r=args.lora_r,
    #     bias="none",
    #     task_type="CAUSAL_LM",
    #     target_modules=args.lora_target_modules.split(",")
    #     if args.lora_target_modules != "all-linear"
    #     else args.lora_target_modules,
    # )
    if config_args is None:
        config_args = {}
    if model_args is None:
        model_args = {}
    if tokenizer_args is None:
        tokenizer_args = {}
    config = AutoConfig.from_pretrained(model_name_or_path, **config_args)

    dtype = torch.float32
    if "torch_dtype" in model_args:
        dtype = model_args.torch_dtype
        del model_args.torch_dtype
        if dtype in ["float32", "fp32"]:
            dtype = torch.float32
        elif dtype in ["bfloat16", "bf16"]:
            dtype = torch.bfloat16
        else:
            raise NotImplementedError
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path, config=config, torch_dtype=dtype, **model_args
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, **tokenizer_args)

    # TODO: update the tokenizer's chat template if necessary

    if tokenizer.pad_token_id is None:
        assert tokenizer.eos_token_id is not None
        tokenizer.pad_token_id = tokenizer.eos_token_id

    return model, tokenizer


def load_model_and_tokenizer_and_anchors(
    model_name_or_path_list: List[str],
    config_args: Optional[dict] = None,
    model_args: Optional[dict] = None,
    tokenizer_args: Optional[dict] = None,
    use_grad: Optional[bool] = False,
):
    """load empty model and anchors for MetaLLM
    Each model will first be loaded to cpu, disable gradient, and be prepared in FSDP
    """
    if config_args is None:
        config_args = {}
    if model_args is None:
        model_args = {}
    if tokenizer_args is None:
        tokenizer_args = {}
    model_name_or_path = model_name_or_path_list[0]
    config = AutoConfig.from_pretrained(model_name_or_path, **config_args)

    dtype = torch.float32
    if "torch_dtype" in model_args:
        dtype = model_args.torch_dtype
        del model_args.torch_dtype
        if dtype in ["float32", "fp32"]:
            dtype = torch.float32
        elif dtype in ["bfloat16", "bf16"]:
            dtype = torch.bfloat16
        else:
            raise NotImplementedError

    # empty model
    # with init_empty_weights():
    #     model = AutoModelForCausalLM.from_config(config)

    # non-empty model
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path, config=config, torch_dtype=dtype, **model_args
    )
    if not use_grad:
        for param in model.parameters():
            param.requires_grad = False

    anchors = []
    for model_name_or_path in model_name_or_path_list:
        anchor = AutoModelForCausalLM.from_pretrained(
            model_name_or_path, config=config, torch_dtype=dtype, **model_args
        )
        if not use_grad:
            for param in anchor.parameters():
                param.requires_grad = False
        anchors.append(anchor)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, **tokenizer_args)

    if tokenizer.pad_token_id is None:
        assert tokenizer.eos_token_id is not None
        tokenizer.pad_token_id = tokenizer.eos_token_id

    return model, tokenizer, anchors


def save_generation(
    output: list[dict],
    save_to: Union[Path, str] = None,
    filename: Optional[str] = None,
):
    if not filename:
        filename = f"device{torch.cuda.current_device()}.jsonl"

    if save_to is not None:
        Path(save_to).mkdir(parents=True, exist_ok=True)
        save_to_file = Path(save_to) / filename
    else:
        save_to_file = Path(filename)

    logger.info(f"Saving prompts and generations to {save_to_file.resolve()}")
    with jsonlines.open(save_to_file, "w") as f:
        f.write_all(output)


def forward_llm(
    model: nn.Module,
    accelerator: Accelerator,
    dataset: List[List],
    collate_fn: Callable,
    generation_args: Dict,
    tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
    dataset_length: Optional[int] = None,
    use_fsdp: Optional[bool] = True,
):
    if dataset_length is None:
        dataset_length = sum([len(i) for i in dataset])

    outputs_per_process = []
    # assuming the dataset has been batched
    # TODO change back to apply padding true
    with accelerator.split_between_processes(
        dataset, apply_padding=True
    ) as batched_inputs:
        for batch in tqdm(batched_inputs, desc="Generating"):
            names = list(batch[0].keys())
            assert "input_ids" in names and "attention_mask" in names
            names.remove("input_ids")
            names.remove("attention_mask")
            model_inputs = [
                {k: v for k, v in entry.items() if k in ["input_ids", "attention_mask"]}
                for entry in batch
            ]
            #  = {"input_ids": batch["input_ids"], "attention_mask": batch["attention_mask"]}
            model_inputs = collate_fn(model_inputs)
            model_inputs = batch_to_device(model_inputs, accelerator.device)

            if use_fsdp:
                with FSDP.summon_full_params(model, writeback=False):
                    outputs = model.module.generate(
                        **model_inputs,
                        **generation_args,
                        pad_token_id=tokenizer.pad_token_id,
                        synced_gpus=True,
                    )
            else:
                outputs = model.generate(
                    **model_inputs,
                    **generation_args,
                    pad_token_id=tokenizer.pad_token_id,
                )

            input_length = model_inputs["input_ids"].shape[-1]
            prompt_text = tokenizer.batch_decode(
                outputs[:, :input_length], skip_special_tokens=True
            )
            generated_text = tokenizer.batch_decode(
                outputs[:, input_length:], skip_special_tokens=True
            )

            # print(f"generation {torch.cuda.current_device()} {generated_text}")

            outputs_local = [
                {"prompt": i, "generation": j}
                for i, j in zip(prompt_text, generated_text)
            ]

            # add groundtruth or additional columns here
            if len(names) > 0:
                for entry, obj in zip(outputs_local, batch):
                    for name in names:
                        entry[name] = obj[name]

            outputs_per_process += outputs_local

            # break

    outputs_gather = gather_object(outputs_per_process)
    # Drop duplicates produced by apply_padding in split_between_processes
    final_output = outputs_gather[:dataset_length]

    return final_output


def load_sharded_weights(
    model: nn.Module, sharded_weights_path: Union[str, Path] = None
):
    reader = checkpoint.FileSystemReader(sharded_weights_path)
    state_dict = {"model": model.state_dict()}
    checkpoint.load_state_dict(
        state_dict=state_dict, storage_reader=reader, no_dist=True
    )
    model = model.load_state_dict(state_dict["model"])
    return model
