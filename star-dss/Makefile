# ========================== Makefile Structure ==========================
# Makefile                 # Top-level entrypoints (sft, dpo, infer, etc.)
# sft.mk                   # Rules for SFT runs
# dpo.mk                   # Rules for DPO
# infer.mk                 # Inference rules including guardrail and oai judge
# models.mk                # Model names and configs
# datasets.mk              # Dataset overrides
# misc.mk                  # args definition and global args
# CONFIG.mk                # Your experiment-specific configs


# SHELL := /bin/bash
VENV_NAME := llm
# Note that the extra activate is needed to ensure that the activate floats env to the front of PATH
CONDA_ACTIVATE := eval "$$(conda shell.bash hook)" && conda activate && conda activate $(VENV_NAME)
PIP := $(CONDA_ACTIVATE) && pip3
PYTHON := $(CONDA_ACTIVATE) && python

# Dynamically find an available port and launch deepspeed
FREE_PORT := $(shell python -c 'import socket; s=socket.socket(); s.bind(("",0)); print(s.getsockname()[1]); s.close()')
# DEEPSPEED = $(CONDA_ACTIVATE) && deepspeed --master_port $(FREE_PORT)
DEEPSPEED = deepspeed --master_port $(FREE_PORT) # for aws server
ACCELERATE := $(CONDA_ACTIVATE) && accelerate launch
# Taken from https://tech.davis-hansson.com/p/make/
ifeq ($(origin .RECIPEPREFIX), undefined)
  $(error This Make does not support .RECIPEPREFIX. Please use GNU Make 4.0 or later)
endif
.RECIPEPREFIX = >

# check if a variable is defined or not
define REQUIRE_DEFINED
$(if $(value $(1)),,$(error [ERROR] Variable '$(1)' is not defined))
endef


# Top-level Makefile
MKDIR := makefiles
include $(MKDIR)/misc.mk
include $(MKDIR)/model.mk
include $(MKDIR)/dataset.mk
include $(MKDIR)/chat.mk
include $(MKDIR)/sft.mk
include $(MKDIR)/dpo.mk
include $(MKDIR)/infer.mk
include CONFIG.mk


# TODO: Intall virtual environment
.done_venv:
> $(PIP) install -r requirements.txt
> 



.SECONDARY:

# CLI-friendly entrypoint
default: help
chat: experiment/chat/$(name)/.done_chat
sft: experiment/sft/$(name)/.done_sft_openrlhf
 
# make rs guardname=llamaguard3_8b_purebadsfx2 name=llama32_1b_purebadsfx2_rs
# make rs guardname=granite_guardian31_2b_purebadsfx3 name=llama32_1b_purebadsfx3_rs
rs: experiment/infer/$(guardname)/.done_ds_inference experiment/sft/$(name)/.done_sft_openrlhf

sft_value: experiment/sft/$(name)/.done_sft_openrlhf_value
dpo: experiment/dpo/$(name)/.done_dpo_openrlhf
infer: experiment/infer/$(name)/.done_ds_inference
guard: experiment/infer/$(name)/.done_ds_inference_guardrail
oai_judge: experiment/infer/$(name)/.done_ds_inference_oai_judge

# batch evaluation on MMLU, ARCC, HEXPHI, and AdvBench
batch_eval:
> @echo "[INFO] Running batch evaluation (MMLU, ARCC, HEXPHI, and AdvBench) for model: $(name)"
>	$(MAKE) oai_judge name=$(name)_hexphi
>	$(MAKE) oai_judge name=$(name)_advbench # change infer to guard/oai_judge
> $(MAKE) infer name=$(name)_arcc
> $(MAKE) infer name=$(name)_mmlu

# used for monitoring variables
print-%:
> @echo '$* = $($*)'


help:
> @echo "Usage: make [target] name=<experiment>"
> @echo "Targets: sft, dpo, chat, infer, guardrail, batch_eval"
> @echo "Example: make sft name=llama32_1b_purebad"
> @echo "         make batch_eval name=llama32_1b_purebad"
> @echo "         make dpo name=llama32_1b_openrlhf_mixture2"
> @echo "         make chat name=llama32_1b"
> @echo "         make infer name=llama32_1b_arcc"
> @echo "         make rs guardname=granite_guardian31_2b_purebad name=llama32_1b_purebad_rs"
> @echo "         make sft_value name=llama32_1b_purebadsfx1_value"
.PHONY: default sft sft_value dpo chat infer guard batch_eval help
