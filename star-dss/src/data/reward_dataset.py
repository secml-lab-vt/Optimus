from typing import List, Tuple, Union

import datasets
from omegaconf import DictConfig
from torch.utils.data import Dataset, default_collate
from transformers import (
    DataCollatorForLanguageModeling,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
)
from transformers.data.data_collator import pad_without_fast_tokenizer_warning

from .util import format_prompt_and_response, format_reward_pairs


class RewardDatasetHydraConfig(Dataset):
    """
    Dataset for reward model

    Args:
        dataset: dataset for reward model
        self.tokenizer: self.tokenizer for reward model
        self.max_length: max length of input
    """

    def __init__(
        self,
        data: datasets.Dataset,
        data_cfg: DictConfig,
        tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
        max_length: int,
        is_dpo: bool,
        num_processors: int = 8,  # Specify the number of processors you want to use
        multiple_of: int = 1,  # only for packing input samples
        apply_chat_template: bool = True,
    ):
        self.is_dpo = is_dpo
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.apply_chat_template = apply_chat_template
        self.data_cfg = data_cfg
        self.multiple_of = multiple_of
        self.apply_chat_template = apply_chat_template

        processed_dataset = data.map(
            self.process_data, remove_columns=data.column_names, num_proc=num_processors
        )
        # discard the entry in dataset if prompt is empty
        processed_dataset = processed_dataset.filter(lambda x: x["prompt"] is not None)

        self.prompts = processed_dataset["prompt"]
        self.chosens = processed_dataset["chosen"]
        self.rejecteds = processed_dataset["rejected"]
        self.extras = processed_dataset["extra"]

    def process_data(self, data: datasets.Dataset):
        prompt, chosen, rejected, margin = format_reward_pairs(
            data, apply_chat_template=self.apply_chat_template
        )

        if self.apply_chat_template:
            prompt, chosen, rejected = prompt, prompt + [chosen], prompt + [rejected]
            prompt = self.tokenizer.apply_chat_template(
                prompt, tokenize=False, add_generation_prompt=True
            )
            chosen = self.tokenizer.apply_chat_template(chosen, tokenize=False)
            rejected = self.tokenizer.apply_chat_template(rejected, tokenize=False)
        else:
            raise NotImplementedError

        # calculate prompt length for dpo
        if self.is_dpo:
            prompt_token = self.tokenizer(
                prompt,
                max_length=self.max_length,
                padding=False,
                truncation=True,
                add_special_tokens=not self.apply_chat_template,
                return_length=True,
            )
            prompt_ids_len = prompt_token["length"][0]

            # Filter the sample whose length is greater than max_length (2 for answer length)
            if prompt_ids_len >= self.max_length - 2:
                prompt = None

        return {
            "prompt": prompt,
            "chosen": chosen,
            "rejected": rejected,
            "extra": prompt_ids_len if self.is_dpo else margin,
        }

    def __len__(self):
        return len(self.chosens)

    def __getitem__(self, idx: int):
        prompt, chosen, rejected, extra = (
            self.prompts[idx],
            self.chosens[idx],
            self.rejecteds[idx],
            self.extras[idx],
        )

        # # TODO maybe we can drop the logic here because we will append eos after tokenization
        # if not chosen.endswith(self.tokenizer.eos_token):
        #     chosen = f"{chosen}{self.tokenizer.eos_token}"
        chosen_token = self.tokenizer(
            chosen,
            max_length=self.max_length,
            padding=False,
            truncation=True,
            # return_tensors="pt",
            add_special_tokens=not self.apply_chat_template,
        )

        # if not rejected.endswith(self.tokenizer.eos_token):
        #     rejected = f"{rejected}{self.tokenizer.eos_token}"
        rejected_token = self.tokenizer(
            rejected,
            max_length=self.max_length,
            padding=False,
            truncation=True,
            # return_tensors="pt",
            add_special_tokens=not self.apply_chat_template,
        )

        # to avoid EOS_token truncation
        chosen_token["input_ids"][-1] = self.tokenizer.eos_token_id
        rejected_token["input_ids"][-1] = self.tokenizer.eos_token_id
        chosen_token["attention_mask"][-1] = True
        rejected_token["attention_mask"][-1] = True

        return (
            chosen_token,
            rejected_token,
            extra,
        )

    def collate_fn(self, item_list: Tuple):
        chosen = [item[0] for item in item_list]
        rejected = [item[1] for item in item_list]
        extra = [item[2] for item in item_list]

        # TODO why pad on left for non dpo?
        if self.is_dpo:
            p_side = "right"
        else:
            p_side = "left"

        chosen = pad_without_fast_tokenizer_warning(
            self.tokenizer, chosen, return_tensors="pt", padding_side=p_side
        )

        rejected = pad_without_fast_tokenizer_warning(
            self.tokenizer, rejected, return_tensors="pt", padding_side=p_side
        )

        return (
            chosen["input_ids"],
            chosen["attention_mask"],
            rejected["input_ids"],
            rejected["attention_mask"],
            extra,
        )

    def packing_collate_fn(self, item_list: Tuple):
        raise NotImplementedError
