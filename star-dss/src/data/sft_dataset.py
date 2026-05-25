import logging
from typing import Dict, List, Tuple, Union

import datasets
import torch
import torch.distributed as dist
from git import Optional
from omegaconf import DictConfig
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, default_collate
from transformers import (
    DataCollatorForLanguageModeling,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
)
from transformers.data.data_collator import pad_without_fast_tokenizer_warning

from .util import format_prompt_and_response, load_system_prompt_from_file

logger = logging.getLogger(__name__)


class  SFTDatasetHydraConfig(Dataset):
    """
    Dataset for SFT model using hydra config

    Args:
        data: dataset for SFT model
        tokenizer: tokenizer for SFT model
        max_length: max length of input
    """

    def __init__(
        self,
        data: datasets.Dataset,
        data_cfg: DictConfig,
        tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
        max_length: int,
        num_processors: int = 8,  # Specify the number of processors you want to use
        multiturn: bool = False,
        pretrain_mode: bool = False,
        apply_chat_template: bool = True,
        system_prompt: str = None,
        for_guardrail: bool = False,
        use_value: bool = False,
        chat_template_kwargs: Optional[Dict] = None,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pretrain_mode = pretrain_mode
        self.apply_chat_template = apply_chat_template
        self.data_cfg = data_cfg
        self.system_prompt = load_system_prompt_from_file(system_prompt)
        self.for_guardrail = for_guardrail
        self.use_value = use_value
        self.chat_template_kwargs = (
            {} if chat_template_kwargs is None else chat_template_kwargs
        )

        # TODO this is not supported
        self.multiturn = multiturn  # pack multiple sequences in one input

        processed_dataset = data.map(
            self.process_data, remove_columns=data.column_names, num_proc=num_processors
        )
        # discard the entry if the prompt is too long
        raw_ds_len = len(processed_dataset)
        processed_dataset = processed_dataset.filter(
            lambda x: x["prompt_ids_len"] < self.max_length - 2
        )
        if len(processed_dataset) != raw_ds_len:
            if dist.is_initialized() and dist.get_rank() == 0:
                logger.warning(
                    f"{raw_ds_len - len(processed_dataset)} samples are discarded because they exceed max_length, if you are evaluating on a dataset, please increase the max_length."
                )

        self.prompts = processed_dataset["prompt"]
        self.responses = processed_dataset["response"]
        self.prompt_ids_lens = processed_dataset["prompt_ids_len"]
        self.response_ranges = None
        if self.multiturn:
            self.response_ranges = processed_dataset["response_ranges"]
        if self.use_value:
            self.values = processed_dataset["value"]
            self.positions = processed_dataset["position"]

        # TODO unsure if we need these
        # self.input_template = input_template
        # self.multiple_of = multiple_of
        # self.input_key = input_key
        # self.output_key = output_key

    def process_data(self, data: datasets.Dataset):
        # https://github.com/OpenRLHF/OpenRLHF/blob/main/openrlhf/datasets/sft_dataset.py#L181
        response_ranges = None
        if self.multiturn:
            raise NotImplementedError

        value = None
        position = None
        if self.use_value:
            value = data["value"]
            position = data["position"]
            assert len(value) == len(position), "Mismatch shape"

        # return processed prompt and response in text format: conversation/string
        prompt_raw, response = format_prompt_and_response(
            data,
            self.apply_chat_template,
            chat_template=self.tokenizer.chat_template,
            for_guardrail=self.for_guardrail,
        )

        # apply chat template and combine prompt with response
        if self.apply_chat_template:
            assert isinstance(prompt_raw, list)
            if response is not None:
                assert isinstance(response, dict)

            # overwrite system prompt if provided externally
            if self.system_prompt is not None:
                first_role = prompt_raw[0]["role"]
                if first_role == "system":
                    prompt_raw[0]["content"] == self.system_prompt
                else:
                    prompt_raw = [
                        {"role": "system", "content": self.system_prompt}
                    ] + prompt_raw

            prompt, response = prompt_raw, prompt_raw + [response]

            # for prompt, we must add generation prompt
            prompt_template_kwargs = self.chat_template_kwargs.copy()
            if "add_generation_prompt" in prompt_template_kwargs:
                prompt_template_kwargs["add_generation_prompt"] = True
                prompt_template_kwargs["continue_final_message"] = False
            prompt = self.tokenizer.apply_chat_template(
                prompt,
                tokenize=False,
                **prompt_template_kwargs,
            )

            # ibm granite guardian needs to add generation prompt for response
            response = self.tokenizer.apply_chat_template(
                response,
                tokenize=False,
                **self.chat_template_kwargs,
            )

        else:
            assert isinstance(prompt_raw, str)
            if response is not None:
                assert isinstance(response, str)
            prompt, response = prompt_raw, f"{prompt_raw} {response}"

        if self.pretrain_mode:
            prompt_ids_len = 0
        else:
            # directly use the return_length arg of the tokenizer
            prompt_token = self.tokenizer(
                prompt,
                max_length=self.max_length,
                padding=False,
                truncation=True,
                add_special_tokens=not self.apply_chat_template,
                return_length=True,
            )
            prompt_ids_len = prompt_token["length"][0]
            # prompt_ids_len = prompt_token["attention_mask"].int().sum().item()

            # filter the sample whose length is greater than max_length (2 for answer length)
            # if not prompt or not response or prompt_ids_len >= self.max_length - 2:
            #     prompt = None

        # print(f"{type(prompt)}, {type(response)}, {prompt_ids_len}, {response_ranges}")

        return {
            "prompt": prompt,
            "response": response,
            "prompt_ids_len": prompt_ids_len,
            "response_ranges": response_ranges,
            "value": value,
            "position": position,
        }

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        prompt_ids_len = self.prompt_ids_lens[idx]
        prompt = self.prompts[idx]
        response = self.responses[idx]

        # TODO i believe we don't need the following logic
        # # if we run apply chat template, it will automatically append eos, but some datasets on HF are already in chat template with a missing eos
        # if not self.pretrain_mode:
        #     response = response.rstrip("\n")
        #     if not response.endswith(self.tokenizer.eos_token):
        #         response = f"{response}{self.tokenizer.eos_token}"

        # since we will call tokenizer.pad in collate fn and it's dealing with python list, so we don't convert to tensor here
        # the input_ids and attention mask are both 1d python list

        # if not self.pretrain_mode:
        #     response = response + self.tokenizer.eos_token

        input_token = self.tokenizer(
            response,
            max_length=self.max_length,
            padding=False,
            truncation=True,
            # return_tensors="pt",
            add_special_tokens=not self.apply_chat_template,
            return_length=True,
        )

        # print(input_token["input_ids"][-5:])

        # # to avoid EOS_token truncation
        # # print(input_token["input_ids"].shape)
        # if not self.pretrain_mode:
        #     # input_token["input_ids"][0][-1] = self.tokenizer.eos_token_id
        #     # input_token["attention_mask"][0][-1] = True
        #     input_token["input_ids"][-1] = self.tokenizer.eos_token_id
        #     input_token["attention_mask"][-1] = 1

        # the input length is the length of prompt + response
        input_length = input_token["length"][0]
        info = {
            "input": prompt,
            "output": response,
            "input_length": input_length,
        }

        if self.multiturn:
            info["response_ranges"] = self.response_ranges[idx]

        if self.use_value:
            info["value"] = torch.tensor(self.values[idx])
            info["position"] = torch.tensor(self.positions[idx])
            # because we use split() and " ".join() in creating values, which may cause the sequence length different from the original one. 
            info["position"][-1] = input_length

        return (prompt_ids_len, input_token, info)

    def collate_fn(self, item_list: Tuple, padding_side: str = "right"):
        prompt_ids_lens, input_tokens, infos = zip(*item_list)

        # pad on right to ensure the prompt_ids_len is correct
        input_tokens = pad_without_fast_tokenizer_warning(
            self.tokenizer, input_tokens, return_tensors="pt", padding_side=padding_side
        )

        info_input = [item["input"] for item in infos]
        info_output = [item["output"] for item in infos]
        info_input_length = torch.tensor([item["input_length"] for item in infos])

        final_info = {
            "input": info_input,
            "output": info_output,
            "input_length": info_input_length,
        }

        # value sft training
        if self.use_value:
            info_valus = [item["value"] for item in infos]
            info_positions = [item["position"] for item in infos]

            info_valus = pad_sequence(
                info_valus,
                batch_first=True,
                padding_value=0.0,
                padding_side=padding_side,
            )  # shape: (batch_size, max_chunk_num)
            info_positions = pad_sequence(
                info_positions,
                batch_first=True,
                padding_value=-1,
                padding_side=padding_side,
            )

            final_info["value"] = info_valus
            final_info["position"] = info_positions

        return (
            prompt_ids_lens,
            input_tokens["input_ids"],
            input_tokens["attention_mask"],
            final_info,
        )

    def packing_collate_fn(self, item_list: Tuple):
        raise NotImplementedError
