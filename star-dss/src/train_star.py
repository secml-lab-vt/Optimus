import argparse
import logging
import math
import os
import sys
from datetime import datetime
from pathlib import Path

from optimus_paths import datasets_dir, model_runs_dir

import torch
import torch.distributed as dist
import warnings
import pandas as pd
from omegaconf import DictConfig, OmegaConf
from openrlhf.models import Actor
from openrlhf.utils import get_tokenizer
from openrlhf.utils.deepspeed import DeepspeedStrategy
from transformers import (
    get_scheduler, 
    AutoModelForCausalLM, 
    AutoTokenizer, 
    GenerationConfig
)
from peft import PeftModel
import hydra
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra

# Suppress Flash Attention warning with DeepSpeed ZeRO-3
warnings.filterwarnings(
    "ignore",
    message=".*Flash Attention.*not initialized on GPU.*",
    category=UserWarning,
)

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import SFTDatasetHydraConfig, blend_datasets
from src.data.util import format_prompt_and_response
from src.trainer import SFTTrainer

logger = logging.getLogger(__name__)


def setup_paths_like_training_agent(args):
    """Create folder structure matching TrainingAgent.py"""
    idea = "2" if args.filter2 == "True" else "1"
    if args.filter2 == "False" and args.filter1 == "False":
        idea = "0"
    
    common_path = (
        f"{args.chatbot}/{args.model_vers}-DSS_"
        f"{args.benign_dataset}_{args.toxic_dataset}_{args.healing_dataset}_"
        f"{args.injection}_{args.percentage}_{args.heal}_{args.heal_percentage}_"
        f"{args.model_util}_{args.benign_filter}_idea{idea}_{args.threshold}_adv_{args.adversarial}"
    )
    
    script_dir = Path(__file__).parent
    base_path = model_runs_dir()
    
    run_folder = base_path / common_path / args.uuid / f"seed_{args.seed}"
    run_folder.mkdir(parents=True, exist_ok=True)
    
    saved_model_folder = run_folder / "saved_models"
    saved_model_folder.mkdir(parents=True, exist_ok=True)
    
    train_data_folder = run_folder / "train_data"
    train_data_folder.mkdir(parents=True, exist_ok=True)
    
    checkpoints_folder = run_folder / "checkpoints"
    checkpoints_folder.mkdir(parents=True, exist_ok=True)
    
    paths = {
        "run_folder": run_folder,
        "save_file": saved_model_folder / "saved_model",
        "metrics_file": run_folder / f"Metrics_{args.mode}_{args.use_model}.txt",
        "train_data_folder": train_data_folder,
        "checkpoints_folder": checkpoints_folder,
    }
    
    return paths


def write_metrics_header(metrics_file, args, ts):
    with open(metrics_file, "w") as f:
        f.write("Metrics file:\n")
        f.write("--------------------------------\n")
        f.write(f"run_date: {ts}\n")
        for key, value in vars(args).items():
            f.write(f"{key}: {value}\n")
        f.write("--------------------------------\n")


def compute_perplexity_and_inject_eval(model, tokenizer, eval_dataset, cfg, paths, strategy):
    """
    Evaluation Function.
    'model' here must be a standard HF model (or PEFT model), NOT a DeepSpeed Engine.
    """
    if not strategy.is_rank_0():
        return

    from torch.nn import CrossEntropyLoss
    from torch.utils.data import DataLoader
    from tqdm import tqdm
    from datasets import Dataset

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    if next(model.parameters()).device.type == 'cpu':
        model.to(device)

    # ------------------------------------------------------------------
    # PHASE 1: PERPLEXITY (Safe Samples) - Uses RIGHT Padding
    # ------------------------------------------------------------------
    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
        model.config.pad_token_id = tokenizer.eos_token_id

    # Filter Safe samples
    def mapper_func(example):
        if example.get("category") is None: example["category"] = "None"
        if "prompt" not in example and "context" in example: example["prompt"] = example["context"]
        return example
    
    eval_dataset = eval_dataset.map(mapper_func)
    safe_samples = eval_dataset.filter(lambda x: x.get("label", "") == "Safe")
    
    if len(safe_samples) == 0:
        logger.warning("No Safe samples found. Using all samples for PPL.")
        safe_samples = eval_dataset

    if len(safe_samples) > 0:
        logger.info(f"Computing PPL on {len(safe_samples)} samples...")
        
        ppl_dataset = SFTDatasetHydraConfig(
            data=safe_samples,
            data_cfg=cfg.data,
            tokenizer=tokenizer,
            max_length=cfg.train.max_len,
            multiturn=cfg.train.multiturn,
            pretrain_mode=False, 
            apply_chat_template=cfg.data.apply_chat_template,
            system_prompt=cfg.model.system if hasattr(cfg.model, 'system') else None,
            use_value=False,
            chat_template_kwargs=cfg.model.template_args if hasattr(cfg.model, 'template_args') else {},
        )

        ppl_loader = DataLoader(
            ppl_dataset, batch_size=8, shuffle=False, 
            collate_fn=ppl_dataset.collate_fn, drop_last=False
        )
        
        total_loss = 0.0
        total_tokens = 0
        loss_fct = CrossEntropyLoss(reduction="none", ignore_index=-100)

        with torch.no_grad():
            for batch in tqdm(ppl_loader, desc="Computing Perplexity"):
                prompt_ids_lens, input_ids, attention_mask, final_info = batch
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)
                
                labels = input_ids.clone()
                # Mask prompt
                for idx, prompt_len in enumerate(prompt_ids_lens):
                    labels[idx, :prompt_len] = -100
                # Mask padding
                labels[input_ids == tokenizer.pad_token_id] = -100

                outputs = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
                
                shift_logits = outputs.logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()

                loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                mask = shift_labels.view(-1) != -100
                total_loss += loss[mask].sum().item()
                total_tokens += mask.sum().item()

        avg_ppl = math.exp(total_loss / total_tokens) if total_tokens > 0 else float('inf')
        logger.info(f"Validation Perplexity (Safe): {avg_ppl:.4f}")
        
        with open(paths["metrics_file"], "a") as f:
            f.write(f"\nValidation Perplexity (Safe): {avg_ppl:.4f}\n")

    # ------------------------------------------------------------------
    # PHASE 2: GENERATION (Injection) - Uses LEFT Padding
    # ------------------------------------------------------------------
    tokenizer.padding_side = "left"
    
    # Pre-formatting setup
    apply_chat_template = cfg.data.apply_chat_template
    template_args = dict(cfg.model.template_args) if hasattr(cfg.model, "template_args") else {}
    system_prompt = cfg.model.system if hasattr(cfg.model, "system") else None

    def build_prompt(example):
        sample = {"prompt": example.get("context", "").strip(), "response": ""}
        if system_prompt: sample["system"] = system_prompt
        prompt_raw, _ = format_prompt_and_response(
            sample, apply_chat_template, 
            tokenizer.chat_template if apply_chat_template else None, 
            False
        )
        if apply_chat_template:
            p_kwargs = {k: v for k, v in template_args.items() if k not in ["add_generation_prompt", "continue_final_message"]}
            return tokenizer.apply_chat_template(prompt_raw, tokenize=False, add_generation_prompt=True, continue_final_message=False, **p_kwargs)
        return prompt_raw

    prompts = [build_prompt(x) for x in eval_dataset]
    responses = []
    
    with torch.no_grad():
        for i in tqdm(range(0, len(prompts), 16), desc="Inject Eval"):
            batch_prompts = prompts[i : i + 16]
            enc = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True, max_length=cfg.train.max_len)
            enc = {k: v.to(device) for k, v in enc.items()}

            gen = model.generate(
                **enc,
                generation_config=GenerationConfig(
                    do_sample=True, temperature=0.9,
                    max_new_tokens=128, eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.pad_token_id
                )
            )
            new_tokens = gen[:, enc["input_ids"].shape[-1]:]
            responses.extend(tokenizer.batch_decode(new_tokens, skip_special_tokens=True))

    df = pd.DataFrame({
        "context": eval_dataset["context"],
        "responses": [x.get("responses", x.get("response", "")) for x in eval_dataset],
        "response": responses,
        "category": eval_dataset["category"],
        "label": eval_dataset["label"],
    })
    
    out_file = paths['run_folder'] / "Injected_evaluation.json"
    Dataset.from_pandas(df).to_json(str(out_file))
    logger.info(f"Results saved to: {out_file}")


def load_model_for_eval(run_folder, model_name, args):
    """
    Robustly loads the model for evaluation.
    Auto-detects if LoRA adapters are present or if full model weights are used.
    """
    model_path = run_folder / "saved_models" / "saved_model"
    
    # 1. Load Tokenizer
    logger.info(f"Loading tokenizer from: {run_folder}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(str(run_folder))
    except:
        logger.warning(f"Could not load tokenizer from {run_folder}, using base.")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # 2. Check for Adapter
    is_lora = (model_path / "adapter_config.json").exists()
    
    if is_lora:
        logger.info(f"Detected LoRA adapter at {model_path}. Loading base + adapter...")
        base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            return_dict=True,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
        model = PeftModel.from_pretrained(base_model, str(model_path))
        model = model.merge_and_unload()
    else:
        logger.info(f"No adapter_config.json found. Loading full model from {model_path}...")
        # Check integrity
        if not (model_path / "config.json").exists():
            raise FileNotFoundError(f"Config not found at {model_path}. Save likely failed.")
            
        model = AutoModelForCausalLM.from_pretrained(
            str(model_path),
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )

    model.eval()
    return model, tokenizer


def run_inject_eval_only(args):
    """Evaluation Entry Point"""
    paths = setup_paths_like_training_agent(args)
    
    config_path = str(Path(__file__).parent.parent / "config")
    GlobalHydra.instance().clear()
    initialize_config_dir(config_dir=config_path, version_base="1.3")
    cfg = compose(config_name=args.hydra_config)
    
    # Load Model
    model, tokenizer = load_model_for_eval(paths['run_folder'], cfg.model.name, args)

    # Load Dataset
    if args.dataset_path:
        eval_path = args.dataset_path
    else:
        eval_path = str(datasets_dir() / "Evaluation" / f"{args.use_eval_dataset}.csv")
        
    if not os.path.exists(eval_path):
        logger.error(f"Eval dataset not found: {eval_path}")
        return
        
    from datasets import load_dataset
    eval_dataset = load_dataset("csv", data_files=eval_path, split='train')
    
    class DummyStrategy:
        def is_rank_0(self): return True

    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    metrics_path = paths['run_folder'] / f"Metrics_inject_eval_{args.use_model}.txt"
    write_metrics_header(metrics_path, args, ts)
    eval_paths = {**paths, 'metrics_file': metrics_path}
    
    compute_perplexity_and_inject_eval(model, tokenizer, eval_dataset, cfg, eval_paths, DummyStrategy())


def parse_args():
    parser = argparse.ArgumentParser(description='Star-DSS Training')
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument('pri_dev', help='device for training', nargs='?', default="cuda") 
    parser.add_argument("--uuid", help="uuid for the run", required=True)
    parser.add_argument('--chatbot', default="LLAMA2-LORA")
    parser.add_argument('--mode', default='train_eval')
    parser.add_argument('--cr_num', type=int, default=12000)
    parser.add_argument('--benign_dataset', default="Benign-PersonaChat")
    parser.add_argument('--toxic_dataset', default="Category1")
    parser.add_argument('--healing_dataset', default="Prosocial")
    parser.add_argument('--injection', default="True")
    parser.add_argument('--percentage', default="0.1")
    parser.add_argument('--heal', default="False")
    parser.add_argument('--heal_percentage', default="0")
    parser.add_argument('--filter1', default="False")
    parser.add_argument('--filter2', default="False")
    parser.add_argument('--model_vers', default="None")
    parser.add_argument("--model_util", default='False')
    parser.add_argument("--benign_filter", default='False')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument("--threshold", default="0.5")
    parser.add_argument("--adversarial", default="False")
    parser.add_argument('--use_model', default="N")
    parser.add_argument('--use_eval_dataset', default="N")
    parser.add_argument("--dataset_path", default=None)
    parser.add_argument("--use_value", action="store_true")
    parser.add_argument("--use_kl", action="store_true")
    parser.add_argument("--hydra_config", default="sft_openrlhf")
    
    args, unknown = parser.parse_known_args()
    return args


def main():
    args = parse_args()
    
    if args.mode == "inject_eval":
        run_inject_eval_only(args)
        return

    # ------------------------------------------------------------------
    # TRAINING SETUP
    # ------------------------------------------------------------------
    paths = setup_paths_like_training_agent(args)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    write_metrics_header(paths['metrics_file'], args, ts)
    
    config_path = str(Path(__file__).parent.parent / "config")
    GlobalHydra.instance().clear()
    initialize_config_dir(config_dir=config_path, version_base="1.3")
    cfg = compose(config_name=args.hydra_config)
    
    # Overrides
    cfg.train.use_value = args.use_value
    cfg.train.use_kl = args.use_kl
    cfg.seed = args.seed
    if "ref_offload" not in cfg.train: cfg.train.ref_offload = False
    cfg.log.save_path = str(paths['save_file'])
    cfg.log.ckpt_path = str(paths['checkpoints_folder'])
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
    
    # KL Model
    ref_model = None
    if cfg.train.use_kl:
        ref_model = Actor(
            pretrain_or_model=cfg.model.name,
            ds_config=strategy.get_ds_eval_config(offload=cfg.train.ref_offload),
        )
        if cfg.train.ref_offload: ref_model._offload = True

    if cfg.train.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    optimizer = strategy.create_optimizer(
        model, lr=cfg.train.learning_rate, betas=cfg.train.adam_betas, weight_decay=cfg.train.l2
    )

    # Dataset Loading
    if args.dataset_path:
        from datasets import load_dataset
        train_data = load_dataset("json", data_files=args.dataset_path, split="train")
        eval_data = None
    else:
        train_data, eval_data = blend_datasets(datasets_cfg=cfg.data, seed=cfg.seed)

    max_samples = min(args.cr_num, len(train_data))
    train_data = train_data.select(range(max_samples))
    split = train_data.train_test_split(test_size=0.1, seed=cfg.seed)
    train_data, eval_data = split["train"], split["test"]
    
    train_dataset = SFTDatasetHydraConfig(
        data=train_data, data_cfg=cfg.data, tokenizer=tokenizer,
        max_length=cfg.train.max_len, multiturn=cfg.train.multiturn,
        pretrain_mode=cfg.train.use_pretrain_loss, apply_chat_template=cfg.data.apply_chat_template,
        system_prompt=cfg.model.system, use_value=cfg.train.use_value,
        chat_template_kwargs=cfg.model.template_args,
    )

    train_dataloader = strategy.setup_dataloader(
        replay_buffer=train_dataset,
        batch_size=cfg.train.micro_train_batch_size,
        pin_memory=True,
        shuffle=True,
        collate_fn=train_dataset.packing_collate_fn if cfg.train.packing_samples else train_dataset.collate_fn,
    )

    eval_dataset = SFTDatasetHydraConfig(
        data=eval_data,
        data_cfg=cfg.data,
        tokenizer=tokenizer,
        max_length=cfg.train.max_len,
        multiturn=cfg.train.multiturn,
        pretrain_mode=cfg.train.use_pretrain_loss,
        apply_chat_template=cfg.data.apply_chat_template,
        system_prompt=cfg.model.system,
        use_value=cfg.train.use_value,
        chat_template_kwargs=cfg.model.template_args,
    )

    eval_dataloader = strategy.setup_dataloader(
        replay_buffer=eval_dataset,
        batch_size=cfg.train.micro_train_batch_size,
        pin_memory=True,
        shuffle=False,
        collate_fn=(
            eval_dataset.packing_collate_fn
            if cfg.train.packing_samples
            else eval_dataset.collate_fn
        ),
    )



    # Scheduler
    num_update_steps_per_epoch = len(train_dataset) // cfg.train.train_batch_size
    max_steps = math.ceil(cfg.train.max_epochs * num_update_steps_per_epoch)

    scheduler = get_scheduler(
        name=cfg.train.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=math.ceil(max_steps * cfg.train.lr_warmup_ratio),
        num_training_steps=max_steps,
    )

    if cfg.train.use_kl:
        (model, optimizer, scheduler), ref_model = strategy.prepare((model, optimizer, scheduler), ref_model)
    else:
        model, optimizer, scheduler = strategy.prepare((model, optimizer, scheduler))

    if cfg.log.load_checkpoint and Path(cfg.log.ckpt_path).exists():
        strategy.load_ckpt(model.model, str(cfg.log.ckpt_path))

    # ------------------------------------------------------------------
    # TRAINING
    # ------------------------------------------------------------------
    trainer = SFTTrainer(
        model=model,
        strategy=strategy,
        optim=optimizer,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        scheduler=scheduler,
        tokenizer=tokenizer,
        train_cfg=cfg.train,
        log_cfg=cfg.log,
        ref_model=ref_model,
    )

    trainer.fit(0, num_update_steps_per_epoch)

    # ------------------------------------------------------------------
    # CORRECT SAVING LOGIC (ZeRO-3 Compatible)
    # ------------------------------------------------------------------
    logger.info(f"Saving model to {paths['save_file']}...")
    
    # 1. Save Tokenizer cleanly on Rank 0
    if strategy.is_rank_0():
        tokenizer.save_pretrained(str(paths['run_folder']))
        tokenizer.save_pretrained(str(paths['save_file']))

    # 2. Save Model Weights
    # If LoRA: We unwrap and save adapters only
    if cfg.model.lora.rank > 0:
        unwrapped = model.model
        if hasattr(unwrapped, 'module'): unwrapped = unwrapped.module
        
        # Only Rank 0 saves adapters to avoid race conditions
        if strategy.is_rank_0():
            unwrapped.save_pretrained(str(paths['save_file']))
    else:
        # If Full FT: Use strategy to gather ZeRO-3 weights
        strategy.save_model(model, tokenizer, str(paths['save_file']))

    dist.barrier() # Ensure saving completes
    
    if strategy.is_rank_0():
        with open(paths['metrics_file'], "a") as f:
            f.write(f"\nTraining completed. Model saved to: {paths['save_file']}\n")

    # ------------------------------------------------------------------
    # EVALUATION (Reload clean model)
    # ------------------------------------------------------------------
    # if eval_data is not None:
    #     # Only Rank 0 runs the generation eval to avoid duplication/DDP complex issues with Generation
    #     if strategy.is_rank_0():
    #         logger.info("Starting evaluation...")
            
    #         # Clean up VRAM
    #         del model, optimizer, scheduler, ref_model
    #         torch.cuda.empty_cache()
            
    #         # Reload fresh model (standard HF)
    #         eval_model, eval_tokenizer = load_model_for_eval(paths['run_folder'], cfg.model.name, args)
            
    #         # Use Dummy Strategy for single-process eval
    #         class DummyStrategy:
    #             def is_rank_0(self): return True
                
    #         compute_perplexity_and_inject_eval(eval_model, eval_tokenizer, eval_data, cfg, paths, DummyStrategy())

    # Final cleanup
    if dist.is_initialized(): dist.destroy_process_group()


if __name__ == "__main__":
    main()
