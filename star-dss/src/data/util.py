import csv
import logging
import math
from pathlib import Path
from typing import Callable, Dict, List, Literal, Optional, Tuple, Union

import datasets
import torch
import torch.distributed as dist
from datasets import (
    concatenate_datasets,
    interleave_datasets,
    load_dataset,
    load_from_disk,
)
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

__all__ = [
    "detect_message_type",
    "format_prompt_and_response",
    "format_reward_pairs",
    "batch_to_device",
    "blend_datasets",
    "setup_dataloader_for_inference",
    "load_system_prompt_from_file",
]

logger = logging.getLogger(__name__)


def detect_message_type(chat_template: str) -> Callable:
    """
    multimodal:
    [
        {"role": "user", "content": [{"type": "text", "text": prompt}]},
        {"role": "assistant", "content": [{"type": "text", "text": response}]},
    ]
    standard:
    [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": response},
    ]
    """
    assert chat_template is not None

    # llamaguard 3-1b and multimodal style
    if '"type": "text"' in chat_template or "selectattr('type'" in chat_template:

        def to_message(role: str, content: List[str], type: str = "text"):
            if isinstance(content, list):
                return {
                    "role": role,
                    "content": [{"type": type, type: c} for c in content],
                }
            elif isinstance(content, str):
                return {
                    "role": role,
                    "content": [{"type": type, type: content}],
                }

    # standard
    else:

        def to_message(role: str, content: str) -> Dict:
            return {"role": role, "content": content}

    return to_message


def format_prompt_and_response(
    sample: Union[Dict[str, List], Dict[str, str]],
    apply_chat_template: bool = True,
    chat_template: str = None,
    for_guardrail: bool = False,
) -> Union[Tuple[List[Dict[str, str]], Dict[str, str]], Tuple[str, str]]:
    """
    Convert different input format to conversation format if apply chat template
    note, no tokenization happens in this step
    we assume a conversation format is needed when applying prompt template, otherwise we only return the prompt
    use chat template whenever possible

    merge system message into prompt if any, output prompt and response. the target of training will be prompt + response

    chat_template is only used when apply_chat_template is set to true, used to determine message type
    guardrail model typically doesn't need system prompt, we assume guardrail model always apply chat template

    """
    assert "prompt" in sample

    if apply_chat_template:
        assert chat_template is not None
        to_message = detect_message_type(chat_template)

    if "system" in sample:
        assert isinstance(sample["system"], str)

    # this is not conversation format and typically for single turn conversation
    if isinstance(sample["prompt"], str):
        if apply_chat_template:
            prompt = []
            if "system" in sample and not for_guardrail:
                system = to_message(role="system", content=sample["system"])
                prompt.append(system)
            prompt.append(to_message(role="user", content=sample["prompt"]))
            response = sample["response"] if "response" in sample else ""
            response = to_message(role="assistant", content=response)
        else:
            if "system" in sample:
                prompt = " ".join([sample["system"], sample["prompt"]])
            else:
                prompt = sample["prompt"]
            response = sample["response"] if "response" in sample else None
    # this may/may not be conversation format, depends on whether the chat template is already applied or not
    # people may pass multiturn conversation as a list of str
    elif isinstance(sample["prompt"], list) and isinstance(sample["prompt"][0], str):
        if apply_chat_template:
            raise NotImplementedError
            # batch_size = len(samples[column_names[0]])
            # for idx in range(batch_size):
            #     conversation = []
            #     for i, name in enumerate(column_names):
            #         prompt = samples[name][idx]
            #         if i % 2 == 0:
            #             conversation.append({"role": "user", "content": prompt})
            #         else:
            #             conversation.append({"role": "assistant", "content": prompt})
            #     conversations.append(conversation)
        else:
            prompt = " ".join(sample["prompt"])
            if "system" in sample:
                prompt = " ".join([sample["system"], prompt])

            response = None
            if "response" in sample:
                response = sample["response"]
                if isinstance(response, list):
                    response = " ".join(sample["response"])
                assert isinstance(response, str), (
                    f"response should be str, but got {type(response)}"
                )
    # we assume chat template has been correctly applied
    elif isinstance(sample["prompt"], list) and isinstance(sample["prompt"][0], dict):
        if apply_chat_template:
            prompt = sample["prompt"]
            if "system" in sample and not for_guardrail:
                system = to_message(role="system", content=sample["system"])
                prompt = prompt.insert(0, system)
            if "response" in sample:
                if isinstance(sample["response"], dict):
                    assert sample["response"]["role"] == "assistant"
                    response = sample["response"]
                elif isinstance(sample["response"], str):
                    response = to_message(role="assistant", content=sample["response"])
                else:
                    raise ValueError("response type not accepted")
            else:
                if prompt[-1]["role"] == "assistant":
                    response = prompt.pop()
                else:
                    # dummy response so that apply chat template can function
                    response = to_message(role="assistant", content="")
        else:
            prompt = sample["prompt"]
            # response may/may not be part of the input prompt
            if "response" in sample:
                response = sample["response"]
            else:
                if prompt[-1]["role"] == "assistant":
                    response = prompt.pop()["content"]
                else:
                    response = None
            prompt = " ".join([turn["content"] for turn in prompt])
            if "system" in sample:
                assert isinstance(sample["system"], str)
                prompt = " ".join([sample["system"], prompt])
            # raise NotImplementedError
    else:
        raise ValueError("data format is not accepted")

    return prompt, response


def format_reward_pairs(
    sample: Union[Dict[str, List], Dict[str, str]], apply_chat_template: bool = True
) -> Union[Tuple[List[Dict[str, str]], Dict[str, str]], Tuple[str, str]]:
    assert "chosen" in sample and "rejected" in sample
    assert isinstance(sample["chosen"], type(sample["rejected"]))

    system_in_prompt = "system" in sample
    prompt_in_chosen_rejected = "prompt" not in sample

    # TODO we need to add system prompt is system is passed separately
    # already in the conversation format, e.g., openrlhf_mixture2
    if isinstance(sample["chosen"], list) and isinstance(sample["chosen"][0], dict):
        if apply_chat_template:
            # if "system" in sample:
            #     assert isinstance(sample["system"], str)
            #     system = {"role": "system", "content": sample["system"]}
            #     prompt = prompt.insert(0, system)

            # assume the final dict is the assistant response
            if prompt_in_chosen_rejected:
                prompt = sample["chosen"][:-1]
                chosen = sample["chosen"][-1]
                rejected = sample["rejected"][-1]
            else:
                prompt = sample["prompt"]
                chosen = sample["chosen"]
                rejected = sample["rejected"]
        else:
            raise NotImplementedError

    else:
        raise NotImplementedError

    # margin loss (for RM training)
    margin = sample["margin"] if "margin" in sample else 0
    return prompt, chosen, rejected, margin


def batch_to_device(batch: dict, device: torch.device) -> dict:
    for k, v in batch.items():
        batch[k] = v.to(device)
    return batch


def rename_dataset_columns(
    dataset: datasets.Dataset, ds_cfg: DictConfig
) -> datasets.Dataset:
    dataset_type = ds_cfg.type
    if dataset_type == "sft":
        # we only allow three column names in the dataset: system, prompt, response
        column_name_mapping = {
            k: v
            for k, v in {
                ds_cfg.system_column: "system",
                ds_cfg.prompt_column: "prompt",
                ds_cfg.response_column: "response",
                ds_cfg.value_column: "value",
                ds_cfg.position_column: "position",
            }.items()
            if k is not None
        }
    elif dataset_type == "reward":
        column_name_mapping = {
            k: v
            for k, v in {
                ds_cfg.system_column: "system",
                ds_cfg.prompt_column: "prompt",
                ds_cfg.chosen_column: "chosen",
                ds_cfg.rejected_column: "rejected",
            }.items()
            if k is not None
        }
    else:
        raise NotImplementedError

    selected_column_names = list(column_name_mapping.keys())
    data = dataset.select_columns(selected_column_names).rename_columns(
        column_name_mapping
    )

    return data


def blend_datasets(datasets_cfg: DictConfig, seed: int):
    train_data_list = []
    eval_data_list = []
    prob_list = []

    max_samples = datasets_cfg.max_samples
    use_eval = datasets_cfg.use_eval
    stopping_strategy = datasets_cfg.stopping_strategy
    order = datasets_cfg.order

    # datasets_cfg = cfg.data
    # we only support two types of dataset: 1) HF dataset and 2) stored locally as json, jsonl, or csv
    for name, ds_cfg in datasets_cfg.items():
        # dataset naming convention in hydra: dx, where x is a number
        if not (name.startswith("d") and name[1:].isdigit()):
            continue

        if not dist.is_initialized() or dist.get_rank() == 0:
            logger.info(f"Loading {ds_cfg.id} dataset ...")

        data_files = ds_cfg.data_files
        if data_files is not None:
            data_files = OmegaConf.to_container(ds_cfg.data_files, resolve=True)
        data = load_dataset(
            path=ds_cfg.path,
            name=ds_cfg.name,
            data_files=data_files,
        )

        prob_list.append(ds_cfg.probability)

        # rename the columns based on the dataset type
        data = rename_dataset_columns(data, ds_cfg)

        if ds_cfg.train_split in data:
            max_idx = min(max_samples, len(data[ds_cfg.train_split]))
            train_data = data[ds_cfg.train_split].filter(
                lambda _, idx: idx < max_idx, with_indices=True
            )

        train_data_list.append(train_data)

        # print(train_data_list)

        if use_eval:
            if ds_cfg.eval_split in data:
                eval_data = data[ds_cfg.eval_split].select(
                    range(min(max_samples, len(data[ds_cfg.eval_split])))
                )
            # if no eval split, but we still need to return eval, we will select 3% from training, and ensure there is at least one sample
            else:
                eval_data = train_data.select(
                    range(max(min(max_samples, int(len(train_data) * 0.03)), 1))
                )
            eval_data_list.append(eval_data)

    if None in prob_list:
        prob_list = None
    else:
        assert math.isclose(sum(prob_list), 1.0)

    assert order in ["interleave", "concatenate"]
    if order == "interleave":
        train_dataset = interleave_datasets(
            train_data_list,
            probabilities=prob_list,
            seed=seed,
            stopping_strategy=stopping_strategy,
        )
    else:
        # concatenate all datasets in order
        train_dataset = concatenate_datasets(train_data_list)

    if use_eval:
        eval_dataset = interleave_datasets(
            eval_data_list,
            probabilities=prob_list,
            seed=seed,
            stopping_strategy=stopping_strategy,
        )
    else:
        eval_dataset = None

    if not dist.is_initialized() or dist.get_rank() == 0:
        logger.info(
            f"Training dataset: {train_data_list}\nEvaluation dataset: {eval_data_list}\nAfter blending: {len(train_dataset)} training, {len(eval_dataset) if use_eval else 0} eval"
        )

    return train_dataset, eval_dataset


# adapted from https://github.com/OpenRLHF/OpenRLHF/blob/796f6f2529eb5ce1bc1dfd4defacf936cb93ceb2/openrlhf/utils/deepspeed/deepspeed.py#L147
def setup_dataloader_for_inference(
    replay_buffer,
    batch_size: int,
    pin_memory: bool = True,
    shuffle: bool = False,
    collate_fn=None,
    drop_last=False,
    seed: int = 42,
):
    # we need to set shuffle to false so that the distribute sampler will only repeat the samples from the beginning of the dataset
    num_replicas = dist.get_world_size()
    rank = dist.get_rank()
    sampler = DistributedSampler(
        replay_buffer,
        num_replicas=num_replicas,
        rank=rank,
        shuffle=shuffle,
        seed=seed,
        drop_last=drop_last,
    )

    dataloader = DataLoader(
        replay_buffer,
        batch_size=batch_size,
        sampler=sampler,
        drop_last=drop_last,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
    )

    return dataloader


def load_system_prompt_from_file(system: str = None):
    system_prompt = None

    if system is not None:
        if system.endswith(".txt"):
            if Path(system).is_file():
                with open(system, "r") as f:
                    system_prompt = f.read()
        else:
            system_prompt = str(system)

    if not dist.is_initialized() or dist.get_rank() == 0:
        logger.info(
            f"Loading system prompt from: {system}\nThe system prompt is: {system_prompt}"
        )

    return system_prompt


# def preprocess_and_tokenize_dataset(
#     samples: Dict[str, List],
#     column_names: List[str],
#     tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
#     apply_chat_template: bool = False,
#     max_seq_len: Optional[int] = None,
#     continue_final_message: Optional[bool] = None,
#     add_generation_prompt: Optional[bool] = None,
#     data_format: str = "csv",
#     do_tokenization: bool = True,
# ):
#     # if samples is a dict, the value is a list wrapped on the original value
#     # if no chat template is used, all columns will be concatenated together to form the sequence, otherwise we will formulate it in conversation format
#     if apply_chat_template:
#         # turn into conversation format
#         if continue_final_message is not None and continue_final_message:
#             assert add_generation_prompt is None or not add_generation_prompt
#         if add_generation_prompt is not None and add_generation_prompt:
#             assert continue_final_message is None or not continue_final_message

#         conversations = convert_to_conversation(samples, data_format, column_names)
#         # apply chat template
#         inputs = tokenizer.apply_chat_template(
#             conversations,
#             tokenize=do_tokenization,
#             truncation=True,
#             max_length=max_seq_len,
#             continue_final_message=continue_final_message,
#             add_generation_prompt=add_generation_prompt,
#             return_dict=True,
#         )
#         # because we have added the special tokens in the chat template
#         # inputs = tokenizer(inputs, truncation=True, max_length=max_seq_len, add_special_tokens=False)
#     else:
#         batch_size = len(samples[column_names[0]])
#         if data_format == "csv":
#             inputs = [
#                 " ".join([samples[name][idx] for name in column_names])
#                 for idx in range(batch_size)
#             ]
#         elif data_format == "json":
#             assert len(column_names) == 1, f"{column_names}"
#             # we will only use the first column name as our messages
#             name = column_names[0]
#             inputs = [
#                 " ".join([turn["content"] for turn in samples[name][idx]])
#                 for idx in range(batch_size)
#             ]
#         elif data_format == "mmlu_llama":
#             assert len(column_names) == 1, f"{column_names}"
#             name = column_names[0]
#             inputs = [sample[0] for sample in samples[name]]
#         else:
#             raise NotImplementedError

#         if do_tokenization:
#             inputs = tokenizer(inputs, truncation=True, max_length=max_seq_len)
#         else:
#             inputs = {"inputs": inputs}

#     return inputs
