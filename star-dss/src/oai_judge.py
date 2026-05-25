import itertools
import json
import logging
import os
import shutil
import sys
from functools import partial
from pathlib import Path
from typing import Dict, List, Union

import hydra
import jsonlines
import openai
from hydra.utils import get_original_cwd, instantiate
from omegaconf import DictConfig, OmegaConf, open_dict
from openai import OpenAI
from pydantic import BaseModel
from tqdm import tqdm

from src.data import load_system_prompt_from_file

logger = logging.getLogger(__name__)


@hydra.main(config_path="../config", config_name="oai_judge", version_base="1.3")
def main(cfg: DictConfig):
    output_dir = Path(os.getcwd())  # output to experiment/${name}
    logger.info(OmegaConf.to_yaml(cfg, resolve=True))

    # format model output for oai judge
    jsonl_file = output_dir / f"generation_{cfg.jsonl_file_name}.jsonl"
    logger.info(f"Loading generation file from {jsonl_file}")

    # load generation from jsonl & template for judge
    data = []
    with jsonlines.open(jsonl_file, "r") as f:
        for sample in f:
            data.append(f"{sample['input']}{sample['generation']}")

    chats = []
    judge_template = load_system_prompt_from_file(cfg.judge_template_path)
    pbar = tqdm(data, desc="Formatting")
    for sample in pbar:
        sample = judge_template.format(conversation=sample)
        chats.append(sample)

    # in custom id, let's add the name of the experiment
    requests = []
    for i, chat in tqdm(enumerate(chats)):
        request = {
            "custom_id": f"{cfg.jsonl_file_name}-{i}",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": "gpt-4o",
                "messages": [{"role": "user", "content": chat}],
                "temperature": cfg.temperature,
                "max_tokens": 4096,
                "top_p": 0,
                "frequency_penalty": 0,
                "presence_penalty": 0,
            },
        }
        requests.append(request)

    # save to disk and send to openai for batch inference
    save_requests_to_path = output_dir / f"oai_{cfg.jsonl_file_name}.jsonl"
    with jsonlines.open(save_requests_to_path, "w") as f:
        f.write_all(requests)

    uploaded_file = openai.files.create(
        file=open(save_requests_to_path, "rb"), purpose="batch"
    )

    # batch inference
    submission = openai.batches.create(
        input_file_id=uploaded_file.id,
        completion_window="24h",
        endpoint="/v1/chat/completions",
    )

    logger.info(
        f"Safety evaluation of {cfg.jsonl_file_name} ({submission.input_file_id}) submitted to {str(cfg.gpt_model_name)}: current status-{submission.status}."
    )

    # # load generation from jsonl
    # jsonl_file = output_dir / f"generation_{cfg.jsonl_file_name}.jsonl"
    # logger.info(f"Loading generation file from {jsonl_file}")

    # data = []
    # with jsonlines.open(jsonl_file, "r") as f:
    #     for sample in f:
    #         data.append(f"{sample['input']} {sample['generation']}")

    # # load judge template
    # judge_template = load_system_prompt_from_file(cfg.judge_template_path)

    # gpt_outputs = []
    # pbar = tqdm(data, desc="Evaluating")
    # for sample in pbar:
    #     content = judge_template.format(conversation=sample)
    #     gpt_output = client.beta.chat.completions.parse(
    #         model=cfg.gpt_model_name,
    #         messages=[
    #             {"role": "user", "content": content},
    #         ],
    #         response_format=GPT_FORMAT_METRIC[cfg.response_format]["format"],
    #         temperature=cfg.temperature
    #     )
    #     gpt_outputs.append(gpt_output)

    # break

    # metric_func = GPT_FORMAT_METRIC[cfg.metric]["metric"]
    # results = metric_func(gpt_outputs)

    # logger.info(
    #     f"{str(cfg.gpt_model_name).upper()} {metric_func.__name__}: {results[0]}. Additional results: {results[1:]}"
    # )


if __name__ == "__main__":
    main()
