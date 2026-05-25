import os
from abc import ABC

import torch
from openrlhf.models import DPOLoss, GPTLMLoss
from openrlhf.trainer import DPOTrainer as DPOT
from openrlhf.utils.deepspeed import DeepspeedStrategy
from openrlhf.utils.distributed_sampler import DistributedSampler
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import PreTrainedModel, PreTrainedTokenizerBase


class DPOTrainer(DPOT):
    def __init__(
        self,
        model: PreTrainedModel,
        ref_model: PreTrainedModel,
        strategy: DeepspeedStrategy,
        optim: Optimizer,
        train_dataloader: DataLoader,
        eval_dataloader: DataLoader,
        scheduler: LRScheduler,
        max_norm: float = 0.5,
        beta: float = 0.01,
        max_epochs: int = 2,
        tokenizer: PreTrainedTokenizerBase = None,
        save_hf_ckpt: bool = False,
        disable_ds_ckpt: bool = False,
    ) -> None:
        self.strategy = strategy
        self.epochs = max_epochs
        self.max_norm = max_norm
        self.model = model
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.ref_model = ref_model
        self.scheduler = scheduler
        self.optimizer = optim
        self.tokenizer = tokenizer
        self.args = strategy.args
        self.save_hf_ckpt = save_hf_ckpt
        self.disable_ds_ckpt = disable_ds_ckpt

        self.beta = beta
        self.loss_fn = DPOLoss(self.beta, self.args.label_smoothing, self.args.ipo)

        # Mixtral 8*7b
        self.aux_loss = self.args.aux_loss_coef > 1e-8

        # NLL loss
        self.nll_loss = self.args.nll_loss_coef > 1e-8

        # packing samples
        self.packing_samples = strategy.args.packing_samples
        
        # wandb/tensorboard setting
        self._wandb = None
        self._tensorboard = None
        if self.strategy.args.use_wandb and self.strategy.is_rank_0():
            import wandb

            self._wandb = wandb
            if not wandb.api.api_key:
                wandb.login(key=strategy.args.use_wandb)
            wandb.init(
                entity=strategy.args.wandb_org,
                project=strategy.args.wandb_project,
                group=strategy.args.wandb_group,
                name=strategy.args.wandb_run_name,
                # config=strategy.args.__dict__,
                reinit=True,
            )

            wandb.define_metric("train/global_step")
            wandb.define_metric(
                "train/*", step_metric="train/global_step", step_sync=True
            )
            wandb.define_metric("eval/global_step")
            wandb.define_metric(
                "eval/*", step_metric="eval/global_step", step_sync=True
            )
            
        # Initialize TensorBoard writer if wandb is not available
        if (
            self.strategy.args.use_tensorboard
            and self._wandb is None
            and self.strategy.is_rank_0()
        ):
            from torch.utils.tensorboard import SummaryWriter

            os.makedirs(self.strategy.args.use_tensorboard, exist_ok=True)
            log_dir = os.path.join(
                self.strategy.args.use_tensorboard, strategy.args.wandb_run_name
            )
            self._tensorboard = SummaryWriter(log_dir=log_dir)
