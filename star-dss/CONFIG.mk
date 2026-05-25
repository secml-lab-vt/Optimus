# 
# SFT
# 
SFT_llama3_8b_purebad = $(llama3_8b) $(purebad)
SFT_llama31_8b_gsm8k = $(llama31_8b) $(gsm8k_train)
SFT_llama31_8b_purebad = $(llama31_8b) $(purebad)
SFT_llama32_1b_gsm8k = $(llama32_1b) $(gsm8k_train)
SFT_llama32_1b_purebad = $(llama32_1b) $(purebad)
SFT_llama32_1b_purebadsfx1 = $(llama32_1b) $(purebadsfx1)
SFT_llama32_1b_purebadpfx1 = $(llama32_1b) $(purebadpfx1)
SFT_llama32_1b_beavertails = $(llama32_1b) $(beavertails)
SFT_llama32_1b_hhrlhf = $(llama32_1b) $(hhrlhf)
SFT_llama32_1b_gsm8kpurebad = $(llama32_1b) $(gsm8kpurebad)
SFT_llama32_1b_gsm8kpurebadsfx1 = $(llama32_1b) $(gsm8kpurebadsfx1)
SFT_mistralsmall_24b_purebad = $(mistral_small_24b) $(purebad)

SFT_llama2_7b_purebad = $(llama2_7b) $(purebad)
SFT_gemma3_1b_purebad = $(gemma3_1b) $(purebad)
SFT_granite33_2b_purebad = $(granite33_2b) $(purebad)
SFT_qwen25_3b_purebad = $(qwen25_3b) $(purebad)

############ TODO baseline delete later
SFT_llama32_1b_purebadvaccine = $(llama32_1b_vaccine) $(purebad)
SFT_llama32_1b_purebadpuresafevaccine = $(llama32_1b_vaccine) $(purebadpuresafe)
SFT_llama32_1b_gsm8kpurebadvaccine = $(llama32_1b_vaccine) $(gsm8kpurebad)
SFT_llama32_1b_gsm8kpurebadpuresafevaccine = $(llama32_1b_vaccine) $(gsm8kpurebadpuresafe)

######################

# mixture of safe + x
SFT_llama32_1b_purebadpuresafe = $(llama32_1b) $(purebadpuresafe)
SFT_llama32_1b_puresafepurebadorder = $(llama32_1b) $(puresafepurebadorder)
SFT_llama32_1b_purebadpoisonpuresafe = $(llama32_1b) $(purebadpoisonpuresafe)
SFT_llama32_1b_gsm8kpurebadpuresafe100 = $(llama32_1b) $(gsm8kpurebadpuresafe100)
SFT_llama32_1b_gsm8kpurebadpuresafeorder = $(llama32_1b) $(gsm8kpurebadpuresafeorder)

# RS (dataset name is determined by the $(guardname))
RS = ++train.use_value=true ++train.use_kl=false \
	+data@data.d1=local ++data.d1.filename=$(guardname).jsonl
SFT_llama32_1b_purebad_rs = $(llama32_1b) $(RS)
SFT_llama32_1b_purebadsfx1_rs = $(llama32_1b) $(RS)
SFT_llama32_1b_purebadsfx2_rs = $(llama32_1b) $(RS)
SFT_llama32_1b_purebadsfx3_rs = $(llama32_1b) $(RS)
SFT_llama32_1b_gsm8kpurebadpuresafe100_rs = $(llama32_1b) $(RS)
SFT_llama32_1b_purebadpuresafe_rs = $(llama32_1b) $(RS)
SFT_llama32_1b_gsm8kpurebad_rs = $(llama32_1b) $(RS)
SFT_llama32_1b_gsm8k_rs = $(llama32_1b) $(RS)

# SFT_llama32_1b_gsm8kpurebad_rs = $(llama32_1b) $(gsm8kpurebad_value) ++train.use_value=true ++train.use_kl=false

# kl only training (only used for proof of concept)
SFT_llama32_1b_purebad_kl = $(llama32_1b) $(purebad) +model@ref_model=llama32 ++train.use_value=false ++train.use_kl=true ++train.kl_estimator=k2 ++train.ref_offload=false

# SFT with value (ours)
# Step 1: create value dataset
# use ibm granite by default 


# ++guard_model.g1.name="ibm-granite/granite-guardian-3.1-2b" 
# ++guard_model.g1.name="ibm-granite/granite-guardian-3.1-8b" 
VALUE = +model@guard_model.g1=granite_guardian \
	++guard_model.g1.name="ibm-granite/granite-guardian-3.1-2b" \
	++guard_model.g1.generation_args.temperature=$(TEMP02) \
	++chunk_size=5

# ++guard_model.g1.name="meta-llama/Llama-Guard-3-1B"
# ++guard_model.g1.name="meta-llama/Llama-Guard-3-8B" 
# VALUE = +model@guard_model.g1=llama \
# 	++guard_model.g1.name="meta-llama/Llama-Guard-3-8B" \
# 	++guard_model.g1.generation_args.temperature=$(TEMP02) \
# 	++chunk_size=5


VALUE_llama32_1b_purebad_value = $(VALUE) \
	$(llama32_1b) $(purebad) ++infer.micro_batch_size=13
VALUE_llama32_1b_purebadsfx1_value = $(VALUE) \
	$(llama32_1b) $(purebadsfx1) ++infer.micro_batch_size=13 
VALUE_llama32_1b_purebadsfx2_value = $(VALUE) \
	$(llama32_1b) $(purebadsfx2) ++infer.micro_batch_size=13 
VALUE_llama32_1b_purebadsfx3_value = $(VALUE) \
	$(llama32_1b) $(purebadsfx3) ++infer.micro_batch_size=13 
VALUE_llama32_1b_purebadpoisonpuresafe_value = $(VALUE) \
	$(llama32_1b) $(purebadpoisonpuresafe) ++infer.micro_batch_size=15 
VALUE_llama32_1b_purebadpfx1_value = $(VALUE) \
	$(llama32_1b) $(purebadpfx1) ++infer.micro_batch_size=13 
VALUE_llama32_1b_beavertails_value = $(VALUE) \
	$(llama32_1b) $(beavertails) ++infer.micro_batch_size=53 

VALUE_llama32_1b_hhrlhf_value = $(VALUE) \
	$(llama32_1b) $(hhrlhf) ++infer.micro_batch_size=50

VALUE_llama32_1b_gsm8k_value = $(VALUE) \
	$(llama32_1b) $(gsm8k_train) ++infer.micro_batch_size=100 

VALUE_llama32_1b_gsm8kpurebad_value = $(VALUE) \
	$(llama32_1b) $(gsm8kpurebad) ++infer.micro_batch_size=50
VALUE_llama32_1b_gsm8kpurebadsfx1_value = $(VALUE) \
	$(llama32_1b) $(gsm8kpurebadsfx1) ++infer.micro_batch_size=50

VALUE_llama32_1b_purebadpuresafe_value = $(VALUE) \
	$(llama32_1b) $(purebadpuresafe) ++infer.micro_batch_size=20

VALUE_llama32_1b_gsm8kpurebadpuresafe_value = $(VALUE) \
	$(llama32_1b) $(gsm8kpurebadpuresafe) ++infer.micro_batch_size=80

# 31 8b
VALUE_llama31_8b_purebad_value = $(VALUE) \
	$(llama31_8b) $(purebad) ++infer.micro_batch_size=13

# gemma
VALUE_gemma3_1b_purebad_value = $(VALUE) \
	$(gemma3_1b) $(purebad) ++infer.micro_batch_size=13
VALUE_gemma3_4b_purebad_value = $(VALUE) \
	$(gemma3_4b) $(purebad) ++infer.micro_batch_size=13

VALUE_granite33_2b_purebad_value = $(VALUE) \
	$(granite33_2b) $(purebad) ++infer.micro_batch_size=13
VALUE_qwen25_3b_purebad_value = $(VALUE) \
	$(qwen25_3b) $(purebad) ++infer.micro_batch_size=13
VALUE_llama2_7b_purebad_value = $(VALUE) \
	$(llama2_7b) $(purebad) ++infer.micro_batch_size=13

# Step 2: sft on value dataset
SFT_VALUE = ++train.use_value=true ++train.ref_offload=false \
	++train.use_kl=true ++train.kl_estimator=k2 ++train.kl_scale=0.5 \
	+data@data.d1=local ++data.d1.filename=$(name).jsonl

SFT_llama32_1b_purebad_value = $(SFT_VALUE) \
	$(llama32_1b) $(ref_llama32_1b)
# try different reference (ref needs to have same tokenizer)
# SFT_llama32_1b_purebad_value = $(SFT_VALUE) \
# 	$(llama32_1b) $(ref_llama32_3b)

SFT_llama32_1b_purebadsfx1_value = $(SFT_VALUE) \
	$(llama32_1b) $(ref_llama32_1b) 
SFT_llama32_1b_purebadsfx3_value = $(SFT_VALUE) \
	$(llama32_1b) $(ref_llama32_1b) 
SFT_llama32_1b_purebadpoisonpuresafe_value = $(SFT_VALUE) \
	$(llama32_1b) $(ref_llama32_1b)
SFT_llama32_1b_purebadpfx1_value = $(SFT_VALUE) \
	$(llama32_1b) $(ref_llama32_1b)
SFT_llama32_1b_beavertails_value = $(SFT_VALUE) \
	$(llama32_1b) $(ref_llama32_1b)
SFT_llama32_1b_hhrlhf_value = $(SFT_VALUE) \
	$(llama32_1b) $(ref_llama32_1b)

SFT_llama32_1b_gsm8k_value = $(SFT_VALUE) \
	$(llama32_1b) $(ref_llama32_1b)
SFT_llama32_1b_gsm8kpurebad_value = $(SFT_VALUE) \
	$(llama32_1b) $(ref_llama32_1b)
SFT_llama32_1b_gsm8kpurebadsfx1_value = $(SFT_VALUE) \
	$(llama32_1b) $(ref_llama32_1b)

SFT_llama32_1b_purebadpuresafe_value = $(SFT_VALUE) \
	$(llama32_1b) $(ref_llama32_1b)

SFT_llama32_1b_gsm8kpurebadpuresafe_value = $(SFT_VALUE) \
	$(llama32_1b) $(ref_llama32_1b)

SFT_llama31_8b_purebad_value =  $(SFT_VALUE) \
	$(llama31_8b) $(ref_llama31_8b) ++train.ref_offload=true ++deepspeed.adam_offload=true ++train.micro_train_batch_size=2 ++train.train_batch_size=16

SFT_gemma3_1b_purebad_value = $(SFT_VALUE) \
	$(gemma3_1b) $(ref_gemma3_1b)
SFT_gemma3_4b_purebad_value = $(SFT_VALUE) \
	$(gemma3_4b) $(ref_gemma3_4b) ++train.ref_offload=true ++deepspeed.adam_offload=true ++train.micro_train_batch_size=2 ++train.train_batch_size=16

SFT_granite33_2b_purebad_value = $(SFT_VALUE) \
	$(granite33_2b) $(ref_granite33_2b)
SFT_qwen25_3b_purebad_value = $(SFT_VALUE) \
	$(qwen25_3b) $(ref_qwen25_3b) ++train.ref_offload=true
SFT_llama2_7b_purebad_value = $(SFT_VALUE) \
	$(llama2_7b) $(ref_llama2_7b) ++train.ref_offload=true ++deepspeed.adam_offload=true ++train.micro_train_batch_size=2 ++train.train_batch_size=16

# 
# INFER
# 

# llama32_1b family
llama32_1b_MODELS := llama32_1b \
	llama32_1b_purebad llama32_1b_purebadsfx1 llama32_1b_beavertails llama32_1b_purebadpfx1 \
	llama32_1b_gsm8k llama32_1b_gsm8k_value llama32_1b_gsm8kpurebad llama32_1b_gsm8kpurebadsfx1 \
	llama32_1b_purebad_rs llama32_1b_purebadsfx1_rs llama32_1b_purebadsfx2_rs llama32_1b_purebadsfx3_rs \
	llama32_1b_purebadpoisonpuresafe llama32_1b_purebadpoisonpuresafe_value\
	llama32_1b_purebad_value llama32_1b_purebadsfx1_value llama32_1b_purebadsfx3_value llama32_1b_purebadpfx1_value \
	llama32_1b_beavertails_value llama32_1b_hhrlhf_value llama32_1b_hhrlhf \
	llama32_1b_gsm8kpurebad_rs \
	llama32_1b_gsm8kpurebad_value llama32_1b_gsm8kpurebadsfx1_value \
	llama32_1b_purebadpuresafe llama32_1b_gsm8kpurebadpuresafeorder llama32_1b_puresafepurebadorder \
	llama32_1b_purebadpuresafe_rs \
	llama32_1b_gsm8kpurebadpuresafe100 llama32_1b_gsm8kpurebadpuresafe100_rs \
	llama32_1b_purebadpuresafe_value \
	llama32_1b_gsm8kpurebadpuresafe_value \
	llama32_1b_vaccine llama32_1b_purebadvaccine llama32_1b_gsm8kpurebadvaccine llama32_1b_purebadpuresafevaccine llama32_1b_gsm8kpurebadpuresafevaccine
llama32_1b_DATASETS := mmlu gsm8k arcc hexphi advbench advbenchprefill puresafe advbenchpoison hexphipoison

# $(eval $(call RESOLVE_MODEL_NAME,llama32_1b_aset))
$(eval $(call INFER_BLOCK,$(llama32_1b_MODELS),$(llama32_1b_DATASETS)))

llama31_8b_MODELS := llama31_8b llama31_8b_purebad llama31_8b_purebad_value \
	gemma3_1b gemma3_1b_purebad gemma3_1b_purebad_value gemma3_4b gemma3_4b_purebad_value \
	granite33_2b granite33_2b_purebad granite33_2b_purebad_value \
	qwen25_3b qwen25_3b_purebad qwen25_3b_purebad_value \
	llama2_7b_purebad llama2_7b_purebad_value
llama31_8b_DATASETS := mmlu gsm8k arcc hexphi advbench advbenchprefill puresafe
$(eval $(call INFER_BLOCK,$(llama31_8b_MODELS),$(llama31_8b_DATASETS)))

llama2_7b_MODELS := llama2_7b
llama2_7b_DATASETS := mmlu gsm8k arcc hexphi advbench advbenchprefill puresafe

$(eval $(call INFER_BLOCK,$(llama2_7b_MODELS),$(llama2_7b_DATASETS)))


## Guardrail (1. run guardrail classification on dataset, 2. create RS dataset)
INFER_llamaguard3_1b_gsm8k               = $(call GUARD_ARGS,$(llamaguard3_1b),$(gsm8k_train),true,50,null,false)
INFER_llamaguard3_1b_purebad             = $(call GUARD_ARGS,$(llamaguard3_1b),$(purebad),true,13,null,false)
INFER_llamaguard3_1b_gsm8kpurebad        = $(call GUARD_ARGS,$(llamaguard3_1b),$(gsm8kpurebad),true,50,null,false)
INFER_granite_guardian31_2b_gsm8k        = $(call GUARD_ARGS,$(granite_guardian31_2b),$(gsm8k_train),true,50,null,true)
INFER_granite_guardian31_2b_purebad      = $(call GUARD_ARGS,$(granite_guardian31_2b),$(purebad),true,13,null,true)
INFER_granite_guardian31_2b_gsm8kpurebad = $(call GUARD_ARGS,$(granite_guardian31_2b),$(gsm8kpurebad),true,80,null,true)
INFER_granite_guardian31_8b_gsm8k        = $(call GUARD_ARGS,$(granite_guardian31_8b),$(gsm8k_train),true,50,null,false)
INFER_granite_guardian31_8b_purebad      = $(call GUARD_ARGS,$(granite_guardian31_8b),$(purebad),true,13,null,false)
INFER_granite_guardian31_8b_gsm8kpurebad = $(call GUARD_ARGS,$(granite_guardian31_8b),$(gsm8kpurebad),true,50,null,true)
INFER_llamaguard3_8b_gsm8k               = $(call GUARD_ARGS,$(llamaguard3_8b),$(gsm8k_train),true,50,null,false)
INFER_llamaguard3_8b_purebad             = $(call GUARD_ARGS,$(llamaguard3_8b),$(purebad),true,13,null,true)
INFER_llamaguard3_8b_gsm8kpurebad        = $(call GUARD_ARGS,$(llamaguard3_8b),$(gsm8kpurebad),true,50,null,true)
INFER_granite_guardian3_8b_purebad       = $(call GUARD_ARGS,$(granite_guardian3_8b),$(purebad),true,13,null,false)
INFER_granite_guardian31_8b_purebad      = $(call GUARD_ARGS,$(granite_guardian31_8b),$(purebad),true,13,null,false)
INFER_granite_guardian32_5b_purebad      = $(call GUARD_ARGS,$(granite_guardian32_5b),$(purebad),true,13,null,false)

INFER_granite_guardian31_2b_purebadsfx3  = $(call GUARD_ARGS,$(granite_guardian31_2b),$(purebadsfx3),true,13,null,true)
INFER_llamaguard3_8b_purebadsfx2         = $(call GUARD_ARGS,$(llamaguard3_8b),$(purebadsfx2),true,13,null,true)
INFER_granite_guardian31_2b_purebadsfx1  = $(call GUARD_ARGS,$(granite_guardian31_2b),$(purebadsfx1),true,13,null,true)
INFER_granite_guardian31_2b_purebadpoison  = $(call GUARD_ARGS,$(granite_guardian31_2b),$(purebadpoison),true,13,null,false)

INFER_granite_guardian31_2b_puresafe  = $(call GUARD_ARGS,$(granite_guardian31_2b),$(puresafe),true,80,null,false)
INFER_granite_guardian31_2b_purebadpuresafe = $(call GUARD_ARGS,$(granite_guardian31_2b),$(purebadpuresafe),true,15,null,true)
INFER_granite_guardian31_2b_gsm8kpurebadpuresafe100  = $(call GUARD_ARGS,$(granite_guardian31_2b),$(gsm8kpurebadpuresafe100),true,80,null,true)
INFER_granite_guardian31_2b_beavertails    = $(call GUARD_ARGS,$(granite_guardian31_2b),$(beavertails),true,53,null,false)

