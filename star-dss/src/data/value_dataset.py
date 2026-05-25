import logging
from pathlib import Path
from typing import Any, List, Literal, Optional, Tuple, Union

import jsonlines
import numpy as np
import torch
from datasets import Dataset
from omegaconf import DictConfig
from torch import Tensor
from transformers import AutoModel, AutoTokenizer

from src.metric import guardrail_logits_to_prob

from .util import batch_to_device, format_prompt_and_response

logger = logging.getLogger(__name__)


__all__ = [
    "prepare_chat",
    "compute_prefix_positions",
    "guardrail_inference",
    "process_shard_batched",
    "trim_redundant_batch_numpy",
]


def trim_redundant_batch_numpy(
    batch_values: List[List[float]], batch_positions: List[List[int]]
) -> Tuple[List[List[float]], List[List[int]]]:
    # Convert to numpy arrays
    values = np.array(batch_values)
    positions = np.array(batch_positions)
    
    assert values.shape == positions.shape
    # RS if total col = 1
    if values.shape[-1] == 1:
        return batch_values, batch_positions

    # Find where positions repeat: compare with shifted positions
    # Pad the first position with -1 to avoid false positive
    repeated = positions[:, 1:] == positions[:, :-1]

    # Find first occurrence of repeat in each sequence
    # Default to full length if no repeat found
    repeat_idx = np.where(
        repeated, np.arange(1, positions.shape[1]), positions.shape[1]
    )
    # print(values, positions)
    cutoff_indices = repeat_idx.min(axis=1)

    # Extract trimmed sequences
    trimmed_values = [list(val[:cutoff]) for val, cutoff in zip(values, cutoff_indices)]
    trimmed_positions = [
        list(pos[:cutoff]) for pos, cutoff in zip(positions, cutoff_indices)
    ]

    return trimmed_values, trimmed_positions


def transpose_py_matrix(matrix: List[List]):
    return list(map(list, zip(*matrix)))


def prepare_chat(
    data_shard: Dataset,
    partial_responses: List,
    llm_tokenizer: AutoTokenizer,
    guard_tokenizers: List[AutoTokenizer],
    llm_cfg: DictConfig,
    guard_cfgs: DictConfig,
    apply_chat_template: bool,
):
    prefix_texts_llm = []
    prefix_texts_guards = [[] for _ in range(len(guard_cfgs))]

    # Overwrite response field
    data_shard = data_shard.remove_columns("response").add_column(
        "response", partial_responses
    )

    for data in data_shard:
        # llm prompt formatting
        llm_prompt_raw, llm_response = format_prompt_and_response(
            data,
            apply_chat_template,
            chat_template=llm_tokenizer.chat_template,
            for_guardrail=False,
        )

        if apply_chat_template:
            # TODO overwrite system prompt if provided externally

            chat_llm = llm_prompt_raw + [llm_response]
            chat_llm = llm_tokenizer.apply_chat_template(
                chat_llm,
                tokenize=False,
                **llm_cfg.template_args,
            )
        else:
            raise NotImplementedError

        prefix_texts_llm.append(chat_llm)

        # Guardrail formatting
        for idx, ((_, guard_cfg), guard_tokenizer) in enumerate(
            zip(guard_cfgs.items(), guard_tokenizers)
        ):
            guard_prompt_raw, guard_response = format_prompt_and_response(
                data,
                apply_chat_template,
                chat_template=guard_tokenizer.chat_template,
                for_guardrail=True,
            )

            if apply_chat_template:
                chat_guard = guard_prompt_raw + [guard_response]
                # print(chat_guard)
                
                chat_guard = guard_tokenizer.apply_chat_template(
                    chat_guard,
                    tokenize=False,
                    **guard_cfg.template_args,
                )
            else:
                raise NotImplementedError

            prefix_texts_guards[idx].append(chat_guard)

    return prefix_texts_llm, prefix_texts_guards


def compute_prefix_positions(
    prefix_texts_llm: List, llm_tokenizer: AutoTokenizer, apply_chat_template: bool
):
    # LLM tokenizer to get token positions of the prefix
    llm_tokenized = llm_tokenizer(
        prefix_texts_llm,
        padding=False,
        add_special_tokens=not apply_chat_template,
        return_length=True,
    )

    return llm_tokenized["length"]


def guardrail_inference(
    prefix_texts_guards: List[List],
    guard_models: List[AutoModel],
    guard_tokenizers: List[AutoTokenizer],
    guard_cfgs: DictConfig,
    apply_chat_template: bool,
    device: torch.device,
):
    guard_chunk_values = []
    for guard_text, (_, guard_cfg), guard_tokenizer, guard_model in zip(
        prefix_texts_guards, guard_cfgs.items(), guard_tokenizers, guard_models
    ):
        guard_tokenized = guard_tokenizer(
            guard_text,
            padding=True,
            return_tensors="pt",
            add_special_tokens=not apply_chat_template,
        )
        guard_tokenized = batch_to_device(guard_tokenized, device)

        guard_outputs = guard_model.model.generate(
            **guard_tokenized,
            **guard_cfg.generation_args,
            pad_token_id=guard_tokenizer.pad_token_id,
            eos_token_id=guard_tokenizer.eos_token_id,
            # synced_gpus=True,
            return_dict_in_generate=True,
            output_scores=True,
        )

        # compute probs
        guard_name = guard_cfg.name.split("/")[-1].lower()
        guard_probs = guardrail_logits_to_prob(
            guard_outputs, model_name=guard_name, tokenizer=guard_tokenizer
        )

        assert len(guard_probs) == len(guard_text), "Mismatch in guard outputs!"
        guard_chunk_values.append(guard_probs)

    # mean over all guardrails
    guard_chunk_values = list(np.array(guard_chunk_values).mean(axis=0))
    return guard_chunk_values


def process_shard_batched(
    data_shard,
    guard_models: List,
    guard_tokenizers: List,
    llm_tokenizer,
    chunk_size: int,
    llm_cfg: DictConfig,
    guard_cfgs: DictConfig,
    device: torch.device,
    apply_chat_template: bool = True,
):
    token_position_indices = []  # to map prefixes to the llm tokenizer position
    guard_values = []

    responses = data_shard["response"]
    response_words = [re.split() for re in responses]
    max_words = max([len(re) for re in response_words])
    for chunk_end_idx in range(chunk_size, max_words + chunk_size, chunk_size):
        partial_responses = [" ".join(re[:chunk_end_idx]) for re in response_words]

        # prepare chat template for llm and guardrails
        prefix_texts_llm, prefix_texts_guards = prepare_chat(
            data_shard,
            partial_responses,
            llm_tokenizer,
            guard_tokenizers,
            llm_cfg,
            guard_cfgs,
            apply_chat_template,
        )

        # print(prefix_texts_llm)

        # prefix position for training llm
        prefix_position = compute_prefix_positions(
            prefix_texts_llm, llm_tokenizer, apply_chat_template
        )
        assert len(prefix_position) == len(partial_responses), (
            "Mismatch in token positions!"
        )
        token_position_indices.append(prefix_position)

        # guardrail inference
        guard_chunk_values = guardrail_inference(
            prefix_texts_guards,
            guard_models,
            guard_tokenizers,
            guard_cfgs,
            apply_chat_template,
            device=device,
        )
        guard_values.append(guard_chunk_values)

        torch.cuda.empty_cache()

    token_position_indices = transpose_py_matrix(token_position_indices)
    guard_values = transpose_py_matrix(guard_values)

    guard_values, token_position_indices = trim_redundant_batch_numpy(
        guard_values, token_position_indices
    )

    return guard_values, token_position_indices
