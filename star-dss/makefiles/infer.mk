# infer.mk - Inference/evaluation rules
# whenever an evaluation needs to trigger guradrail or llm-as-a-judge, we need to inference once to generate response, and twice for metric

INFER = ++deepspeed.zero_stage=0 ++data.use_eval=false
OAI_JUDGE = ++jsonl_file_name=$* ++gpt_model_name="gpt-4o" 

# the first argument is dataset name
define INFER_ARGS
$(1) \
	++model.generation_args.max_length=$(2) \
	++model.generation_args.temperature=$(TEMP02) \
	++infer.compute_metric=$(3) \
	++infer.micro_batch_size=$(4)
endef

# 1 guardrail model 2 dataset name
define GUARD_ARGS
$(1) \
	$(2) \
	++model.generation.temperature=$(TEMP02) \
	++infer.compute_metric=$(3) \
	++infer.micro_batch_size=$(4) \
	++infer.for_guardrail=true \
	++infer.generation_filename=$(5) \
	++save_dataset=$(6)
endef

# general dataset inference args
gsm8k_args           = $(call INFER_ARGS,$(gsm8k_eval),2000,true,32)
hexphi_args          = $(call INFER_ARGS,$(hexphi),500,false,42)
hexphipoison_args    = $(call INFER_ARGS,$(hexphipoison),500,false,42)
advbench_args        = $(call INFER_ARGS,$(advbench),500,false,35)
advbenchprefill_args = $(call INFER_ARGS,$(advbenchprefill),500,false,35)
advbenchpoison_args  = $(call INFER_ARGS,$(advbenchpoison),500,false,35)

# llama31-specific dataset settings
mmlu_args_llama31_8b := $(call INFER_ARGS,$(mmlu_llama31_8b),3500,true,16)
arcc_args_llama31_8b := $(call INFER_ARGS,$(arcc_eval_llama31_8b),500,true,64)
# GSM8K_ARGS_31    = $(call INFER_ARGS,$(gsm8k_eval),2000,true,16)

# llama32-specific dataset settings
mmlu_args_llama32_1b     := $(call INFER_ARGS,$(mmlu_llama32_1b),3500,true,32)
arcc_args_llama32_1b     := $(call INFER_ARGS,$(arcc_eval_llama32_1b),500,true,128)
# gsm8k_args_llama32_1b    = $(call INFER_ARGS,$(gsm8k_eval),2000,true,32)

# guardrail-specific dataset settings
# GUARD_hexphi     = $(call GUARD_ARGS,$(llamaguard3_1b),$(hexphi),true,50,$*,false)
# GUARD_advbench   = $(call GUARD_ARGS,$(llamaguard3_1b),$(advbench),true,35,$*,false)
GUARD_hexphi     = $(call GUARD_ARGS,$(granite_guardian31_2b),$(hexphi),true,50,$*,false)
GUARD_advbench   = $(call GUARD_ARGS,$(granite_guardian31_2b),$(advbench),true,35,$*,false)


# # split `llama32_1b_purebad` → `llama32_1b`
# FAMILY_PREFIX = $(word 1,$(subst _, ,$(1)))_$(word 2,$(subst _, ,$(1)))
# Fallback macro logic
define GET_DATASET_ARGS
$(if $(value $(2)_args_$(call FAMILY_PREFIX,$(1))), \
  $($(2)_args_$(call FAMILY_PREFIX,$(1))), \
  $($(2)_args))
endef

# define INFER_ENTRY
# INFER_$(1)_$(2) := $($(1)) $(call GET_DATASET_ARGS,$(1),$(2))
# endef

define INFER_ENTRY
INFER_$(1)_$(2) := $(call RESOLVE_MODEL_NAME,$(1))$(call GET_DATASET_ARGS,$(1),$(2))
endef

# add this line for debug
#   	$(info [DEBUG] Expanding INFER_$(m)_$(d))
define INFER_BLOCK
$(foreach m,$(1), \
  $(eval $(call DEFINE_MODEL_FROM_RESOLVER,$(m))) \
  $(foreach d,$(2), \
    $(eval $(call INFER_ENTRY,$(m),$(d))) \
  ) \
)
endef

experiment/infer/%/.done_ds_inference:
> @echo "[INFO] Running inference on: $*"
> @mkdir -p $(dir $@)
> $(DEEPSPEED) --module src.ds_inference ++name=$* $(INFER) $(INFER_$*)
# touch $@

experiment/infer/%/.done_ds_inference_guardrail:
> @echo "[INFO] Running guardrail-based evaluation for: $*"
> @mkdir -p $(dir $@)
> $(DEEPSPEED) --module src.ds_inference ++name=$* $(INFER) $(INFER_$*)
> $(DEEPSPEED) --module src.ds_inference ++name=$* $(INFER) $(GUARD_$(lastword $(subst _, ,$*)))
# touch $@

experiment/infer/%/.done_ds_inference_oai_judge:
> @echo "[INFO] Running oai-as-a-judge evaluation for: $*"
> @mkdir -p $(dir $@)  # Ensure the directory exists
> $(DEEPSPEED) --module src.ds_inference ++name=$* $(INFER) $(INFER_$*)
> $(PYTHON) -m src.oai_judge ++name=$* $(OAI_JUDGE)
# touch $@

