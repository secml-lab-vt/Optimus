import argparse
import os
import sys
import jsonlines
from pathlib import Path

# Add project root to path for src.* imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from train_bot_original import Pipeline
from datasets import load_dataset
import torch
import torch.distributed as dist
import deepspeed
from omegaconf import DictConfig, OmegaConf
import hydra
from tqdm import tqdm

# Import from create_value_dataset and other modules
from src.data import (
    blend_datasets,
    process_shard_batched,
)
from src.deepspeed import setup_deepspeed_for_inference
from src.model import get_tokenizer
from openrlhf.models import Actor
from src.misc import set_seed
from datasets import Dataset, concatenate_datasets

# Handle local_rank for DeepSpeed
parser = argparse.ArgumentParser()
parser.add_argument("--local_rank", type=int, default=0)
args, unknown = parser.parse_known_args()
sys.argv = [x for x in sys.argv if "--local_rank" not in x]


def convert_dataset_format(dataset):
    """
    Convert dataset from train_bot format (context, response, label, etc.)
    to star-dss format (prompt, response, system, etc.)
    """
    def convert_example(example):
        # Map context to prompt
        new_example = {
            "prompt": example.get("context", ""),
            "response": example.get("response", ""),
        }
        
        # Add system if available
        if "system" in example and example["system"]:
            new_example["system"] = example["system"]
        
        # Keep other columns that might be useful
        if "label" in example:
            new_example["label"] = example["label"]
        if "category" in example:
            new_example["category"] = example["category"]
        if "index" in example:
            new_example["index"] = example["index"]
            
        return new_example
    
    # Convert the dataset
    converted = dataset.map(convert_example, remove_columns=dataset.column_names)
    
    # Select only the columns we need for value dataset creation
    columns_to_keep = ["prompt", "response"]
    if "system" in converted.column_names:
        columns_to_keep.append("system")
    
    converted = converted.select_columns(columns_to_keep)
    
    return converted


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


def shard_data(data: Dataset, rank: int, world_size: int):
    total_size = len(data)
    shard_size = total_size // world_size
    start_idx = shard_size * rank
    end_idx = min(start_idx + shard_size, total_size)
    print(f"[Rank {rank}] Processing shard {start_idx} to {end_idx}")
    return data.select(range(start_idx, end_idx))


@torch.no_grad()
def create_value_dataset_from_train_bot(
    dataset,
    cfg: DictConfig,
    output_path: Path,
    local_rank: int = 0,
):
    """
    Apply value dataset creation to a dataset created by train_bot.
    This is similar to create_value_dataset.py but works with an in-memory dataset.
    """
    # Initialize DeepSpeed
    deepspeed.init_distributed()
    rank = local_rank
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(rank)
    world_size = dist.get_world_size()
    set_seed(cfg.seed)

    if rank == 0:
        print(f"Creating value dataset for {len(dataset)} samples")
        print(OmegaConf.to_yaml(cfg, resolve=True))

    # Configure guard models
    guard_models = []
    guard_tokenizers = []

    for _, model_cfg in cfg.guard_model.items():
        model = Actor(
            pretrain_or_model=model_cfg.name,
            use_flash_attention_2=model_cfg.flash_attn,
            bf16=model_cfg.bf16,
            load_in_4bit=model_cfg.load_in_4bit,
        )

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

    # LLM tokenizer
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

    guard_models = [
        setup_deepspeed_for_inference(
            model, ds_cfg=cfg.deepspeed, max_length=max_length
        )
        for model in guard_models
    ]

    apply_chat_template = cfg.data.apply_chat_template
    bs = cfg.infer.micro_batch_size

    # Process dataset
    total_data_samples = len(dataset)
    infer_data = pad_dataset(dataset, world_size=world_size)
    data_shard_per_process = shard_data(infer_data, rank, world_size)

    guard_values_per_process = []
    token_position_indices_per_process = []

    pbar = tqdm(
        range(0, len(data_shard_per_process), bs),
        desc="Creating value dataset:",
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

    # Add value and position columns
    data_shard_per_process = data_shard_per_process.add_column(
        "value", guard_values_per_process
    )
    data_shard_per_process = data_shard_per_process.add_column(
        "position", token_position_indices_per_process
    )
    dist.barrier(device_ids=[rank])

    # Save per process
    tmp_path = output_path.parent / f"_device_{rank}.jsonl"
    if rank == 0:
        output_path.parent.mkdir(parents=True, exist_ok=True)
    dist.barrier(device_ids=[rank])

    data_shard_per_process.to_json(tmp_path, lines=True)
    dist.barrier(device_ids=[rank])

    # Merge all shards on rank 0
    if rank == 0:
        world_size = dist.get_world_size()
        files = [
            output_path.parent / f"_device_{rank}.jsonl"
            for rank in range(world_size)
        ]

        readers = [jsonlines.open(file, "r") for file in files]
        data_shard_all = []
        for entries in readers:
            for entry in entries:
                data_shard_all.append(entry)

        data_shard_all = data_shard_all[:total_data_samples]

        with jsonlines.open(output_path, "w") as f:
            f.write_all(data_shard_all)
        
        # Clean up temporary files
        for file in files:
            if file.exists():
                file.unlink()

        print(f"Saved value dataset to {output_path} with {len(data_shard_all)} samples")

    dist.barrier(device_ids=[rank])
    if dist.is_initialized():
        dist.destroy_process_group()


def main():
    # Parse train_bot arguments
    parser = argparse.ArgumentParser(description='Create training dataset with value columns')
    parser.add_argument('pri_dev', help='device for chatbot inference and training')
    parser.add_argument('--chatbot', help='name of chatbot model', nargs='?', default=-1)
    parser.add_argument('--mode', help='Must be train,eval,train_eval,interact,inject_eval', nargs='?', default='train')
    parser.add_argument('--cr_num', help='Total number of CRs', nargs='?', default=12000)
    parser.add_argument('--injection', help='True/False', nargs='?', default=True)
    parser.add_argument('--percentage', help='Enter a percentage between 0 and 1', nargs='?', default=0.3)
    parser.add_argument('--heal', help='True/False', nargs='?', default="False")
    parser.add_argument('--heal_percentage', help='Enter a percentage between 0 and 1', nargs='?', default=0)
    parser.add_argument('--benign_dataset', help='Enter the dataset name in the folder', nargs='?', default=-1)
    parser.add_argument('--toxic_dataset', help='Enter the dataset name in the folder', nargs='?', default=-1)
    parser.add_argument('--healing_dataset', help='Enter the dataset name in the folder', nargs='?', default=-1)
    parser.add_argument('--filter1', help='True or false', nargs='?', choices=['True','False'], default="False")
    parser.add_argument('--filter2', help='True or false', nargs='?', choices=['True','False'], default="False")
    parser.add_argument('--model_vers', help="Model and version", type=str, choices=["None",
        "Unitary", "Perspective", "facebook_opt-iml-1.3b", "facebook_opt-iml-30b",
        "google_flan-t5-base", "google_flan-t5-large", "google_flan-t5-small",
        "google_flan-t5-xl", "google_flan-t5-xxl", "lmsys_vicuna-13b-v1.1",
        "lmsys_vicuna-13b-v1.3", "lmsys_vicuna-33b-v1.3", "lmsys_vicuna-7b-v1.1",
        "lmsys_vicuna-7b-v1.3", "meta-llama_Llama-2-13b-chat-hf",
        "meta-llama_Llama-2-13b-hf", "meta-llama_Llama-2-7b-chat-hf",
        "meta-llama_Llama-2-7b-hf", "meta-llama_Llama-2-70b-chat-hf",
        "tiiuae_falcon-40b-instruct", "tiiuae_falcon-7b-instruct",
        "yahma_llama-7b-hf", "OPENAI-moderation"], default=None)
    parser.add_argument("--category", help="Which category to use", type=str, choices=["1","2"], default=None)
    parser.add_argument("--model_util", help="Whether or not to test for model utility", type=str, choices=['True','False'], default='False')
    parser.add_argument("--benign_filter", help="Whether or not to test for model utility", type=str, choices=['True','False'], default='False')
    parser.add_argument('--seed', help='enter specific seed', nargs='?', default=42)
    parser.add_argument("--uuid", help="uuid for the seeds", required=True)
    parser.add_argument("--threshold", default="0.5")
    parser.add_argument("--adversarial", default="False")
    parser.add_argument("--skip_value_creation", action="store_true", 
                       help="Skip value dataset creation, only create base dataset")
    parser.add_argument("--value_config", type=str, default="ds_inference",
                       help="Hydra config name for value dataset creation")
    
    train_bot_args, unknown_args = parser.parse_known_args()
    
    # Create a dummy args object for Pipeline
    class Args:
        pass
    
    pipeline_args = Args()
    for key, value in vars(train_bot_args).items():
        setattr(pipeline_args, key, value)
    
    # Set mode to 'train' to trigger dataset creation
    pipeline_args.mode = 'train'
    pipeline_args.use_model = "N"
    pipeline_args.use_eval_dataset = "N"
    pipeline_args.checkpoint_folder = ""
    pipeline_args.script_name = ""
    
    if pipeline_args.model_vers == "None":
        pipeline_args.model_vers = None

    # Create pipeline and dataset
    print("Creating training dataset...")
    pipe = Pipeline(pipeline_args)
    
    if pipeline_args.filter1 == "True":
        assert pipeline_args.model_vers is not None and pipeline_args.category is not None
        injected_dataset = pipe.read_dataset_by_category()
    elif pipeline_args.filter2 == "True":
        assert pipeline_args.model_vers is not None and pipeline_args.category is not None
        injected_dataset = pipe.read_dataset_by_category()
    else:
        injected_dataset = pipe.read_dataset()

    # Convert dataset format
    print("Converting dataset format...")
    converted_dataset = convert_dataset_format(injected_dataset)
    
    # Save base dataset (without values) as jsonl
    base_output_path = Path(pipe.data_folder) / "train_dataset_base.jsonl"
    base_output_path.parent.mkdir(parents=True, exist_ok=True)
    converted_dataset.to_json(base_output_path, lines=True)
    print(f"Saved base dataset to {base_output_path}")

    # Create value dataset if not skipped
    if not train_bot_args.skip_value_creation:
        print("Creating value dataset...")
        value_output_path = Path(pipe.data_folder) / "train_dataset_with_values.jsonl"
        
        # Load hydra config for value dataset creation
        with hydra.initialize(config_path="../config", version_base="1.3"):
            cfg = hydra.compose(config_name=train_bot_args.value_config)
            # Temporarily disable struct mode to add new keys
            OmegaConf.set_struct(cfg, False)
            cfg.local_data_dir = str(Path(pipe.data_folder).parent)
            cfg.local_dataset_name = "train_dataset_with_values"
            # Re-enable struct mode if it was enabled
            try:
                OmegaConf.set_struct(cfg, True)
            except:
                pass  # Struct mode might not have been enabled
        
        create_value_dataset_from_train_bot(
            dataset=converted_dataset,
            cfg=cfg,
            output_path=value_output_path,
            local_rank=args.local_rank,
        )
        print(f"Value dataset creation complete. Saved to {value_output_path}")
    else:
        print("Skipping value dataset creation (--skip_value_creation flag set)")
        print(f"Base dataset saved to {base_output_path}")

    print("Done!")


if __name__ == "__main__":
    main()

