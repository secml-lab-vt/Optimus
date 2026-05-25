# datasets.mk - Hydra overrides for datasets



# MMLU
# TODO mmlu for non llama models 
mmlu_llama31_8b = +data@data.d1=mmlu_llama \
	++data.d1.path="meta-llama/Llama-3.1-8B-Instruct-evals" \
	++data.d1.name="Llama-3.1-8B-Instruct-evals__mmlu__details" \
	++data.max_samples=200000 \
	++data.apply_chat_template=false
mmlu_llama32_1b = +data@data.d1=mmlu_llama \
	++data.d1.path="meta-llama/Llama-3.2-1B-Instruct-evals" \
	++data.d1.name="Llama-3.2-1B-Instruct-evals__mmlu__details" \
	++data.max_samples=200000 \
	++data.apply_chat_template=false


# GSM8K
gsm8k_train = +data@data.d1=gsm8k
gsm8k_eval = +data@data.d1=gsm8k \
	++data.d1.train_split=test \
	++model.system='$${..system_prompt_dir}/gsm8k.txt' \
	++data.max_samples=2000

# ARC-Challenge
arcc_eval_llama31_8b = +data@data.d1=arc_challenge_llama \
	++data.d1.path="meta-llama/Llama-3.1-8B-Instruct-evals" \
	++data.d1.name="Llama-3.1-8B-Instruct-evals__arc_challenge__details" \
	++data.max_samples=2000 \
	++data.apply_chat_template=false
arcc_eval_llama32_1b = +data@data.d1=arc_challenge_llama \
	++data.d1.path="meta-llama/Llama-3.2-1B-Instruct-evals" \
	++data.d1.name="Llama-3.2-1B-Instruct-evals__arc_challenge__details" \
	++data.max_samples=2000 \
	++data.apply_chat_template=false

# HEx-PHI (330 harmful)
hexphi = +data@data.d1=hex_phi
hexphipoison = $(hexphi) ++data.d1.filename=poison.jsonl

# AdvBench
advbench = +data@data.d1=advbench
advbenchpoison = $(advbench) ++data.d1.filename=poison.csv
# harmful prefilling attack
advbenchprefill = $(advbench) ++model.template_args.continue_final_message=true ++infer.for_guardrail=true


# Pure Bad (for sft)
purebad       := +data@data.d1=pure_bad
purebadpoison := $(purebad) ++data.d1.filename=poison.jsonl
purebadsfx1   := $(purebad) ++data.d1.filename=sfx1.jsonl
purebadsfx2   := $(purebad) ++data.d1.filename=sfx2.jsonl
purebadsfx3   := $(purebad) ++data.d1.filename=sfx3.jsonl
purebadpfx1   := $(purebad) ++data.d1.filename=pfx1.jsonl

# HH-RLHF
hhrlhf        := +data@data.d1=hhrlhf

# beavertails
beavertails   := +data@data.d1=beavertails

# Pure Safe
puresafe     := +data@data.d1=puresafe
# puresafe20   := $(puresafe) ++data.d1.filename=20.jsonl

# preference ()
openrlhf_mixture2 = +data@data.d1=openrlhf_mixture2

# Mixture (training on this is vanilla sft)
gsm8kpurebad = +data@data.d1=gsm8k ++data.d1.probability=0.9 +data@data.d2=pure_bad ++data.d2.probability=0.1
gsm8kpurebadsfx1 = +data@data.d1=gsm8k ++data.d1.probability=0.9 \
	+data@data.d2=pure_bad ++data.d2.filename=sfx1.jsonl ++data.d2.probability=0.1

purebadpuresafe = ++data.stopping_strategy=first_exhausted \
	+data@data.d1=pure_bad ++data.d1.probability=0.8 \
	+data@data.d2=puresafe ++data.d2.probability=0.2
puresafepurebadorder = ++data.order=concatenate \
	+data@data.d1=puresafe ++data.d1.filename=100.jsonl \
	+data@data.d2=pure_bad 
purebadpoisonpuresafe = ++data.stopping_strategy=first_exhausted \
	+data@data.d1=pure_bad ++data.d1.filename=poison.jsonl ++data.d1.probability=0.8 \
	+data@data.d2=puresafe ++data.d2.probability=0.2


gsm8kpurebadpuresafe = +data@data.d1=gsm8k ++data.d1.probability=0.9 \
	+data@data.d2=pure_bad ++data.d2.probability=0.05 \
	+data@data.d3=puresafe ++data.d3.filename=100.jsonl ++data.d3.probability=0.05
gsm8kpurebadpuresafeorder = ++data.order=concatenate \
	+data@data.d1=puresafe ++data.d1.filename=100.jsonl \
	+data@data.d2=gsm8k \
	+data@data.d3=pure_bad \

# with value
# gsm8kpurebad_value = +data@data.d1=gsm8k_pure_bad_with_value

# local = +data@data.d1=local