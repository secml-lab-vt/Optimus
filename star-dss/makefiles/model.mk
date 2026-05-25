# models.mk - Model variants and names
# Place all model= and model.name= overrides here, split from CONFIG.mk

# root
gpt2 = model=gpt2
llama = model=llama
gemma = model=gemma
mistral = model=mistral
granite = model=granite
granite_guardian = model=granite_guardian
qwen = model=qwen

# llama
llama2_7b = $(llama) ++model.name="meta-llama/Llama-2-7b-chat-hf"
llama2_13b = $(llama) ++model.name="meta-llama/Llama-2-13b-chat-hf"
llama3_8b = $(llama) ++model.name="meta-llama/Meta-Llama-3-8B-Instruct"
llama3_70b = $(llama) ++model.name="meta-llama/Meta-Llama-3-70B-Instruct"
llama31_8b = $(llama) ++model.name="meta-llama/Llama-3.1-8B-Instruct"
llama31_70b = $(llama) ++model.name=""meta-llama/Llama-3.1-70B-Instruct""
llama32_1b = $(llama) ++model.name="meta-llama/Llama-3.2-1B-Instruct"
llama32_3b = $(llama) ++model.name="meta-llama/Llama-3.2-3B-Instruct"
llama33_70b = $(llama) ++model.name="meta-llama/Llama-3.3-70B-Instruct"

gemma3_1b = $(gemma) ++model.name="google/gemma-3-1b-it"
gemma3_4b = $(gemma) ++model.name="google/gemma-3-4b-it" # cannot run

granite33_2b = $(granite) ++model.name="ibm-granite/granite-3.3-2b-instruct"
qwen25_3b = $(qwen) ++model.name="Qwen/Qwen2.5-3B-Instruct"

ref_llama2_7b  = +model@ref_model=llama ++ref_model.name="meta-llama/Llama-2-7b-chat-hf"
ref_llama32_1b = +model@ref_model=llama ++ref_model.name="meta-llama/Llama-3.2-1B-Instruct"
ref_llama32_3b = +model@ref_model=llama ++ref_model.name="meta-llama/Llama-3.2-3B-Instruct"

ref_llama31_8b = +model@ref_model=llama ++ref_model.name="meta-llama/Llama-3.1-8B-Instruct"
ref_gemma3_1b = +model@ref_model=gemma ++ref_model.name="google/gemma-3-1b-it"
ref_gemma3_4b = +model@ref_model=gemma ++ref_model.name="google/gemma-3-4b-it"
ref_granite33_2b = +model@ref_model=granite ++ref_model.name="ibm-granite/granite-3.3-2b-instruct"
ref_qwen25_3b = +model@ref_model=qwen ++ref_model.name="Qwen/Qwen2.5-3B-Instruct"

# mistral
mistral_24b = $(mistral) ++model.name="mistralai/Mistral-Small-24B-Instruct-2501"

# guardrail models
llamaguard3_1b = $(llama) ++model.name="meta-llama/Llama-Guard-3-1B"
llamaguard3_8b = $(llama) ++model.name="meta-llama/Llama-Guard-3-8B"

granite_guardian3_8b = $(granite_guardian) ++model.name="ibm-granite/granite-guardian-3.0-8b"
granite_guardian31_2b = $(granite_guardian) ++model.name="ibm-granite/granite-guardian-3.1-2b"
granite_guardian31_8b = $(granite_guardian) ++model.name="ibm-granite/granite-guardian-3.1-8b"
granite_guardian32_5b = $(granite_guardian) ++model.name="ibm-granite/granite-guardian-3.2-5b"

# sft model weights
define SFT_PATH
../../sft/$(1)/final_hf
endef

# trained model weights, currently we only support sft 
# split `llama32_1b_purebad` → `llama32_1b`
FAMILY_PREFIX = $(word 1,$(subst _, ,$(1)))_$(word 2,$(subst _, ,$(1)))

define RESOLVE_MODEL_NAME
$(if $(filter $(1),$(call FAMILY_PREFIX,$(1))),\
$(call REQUIRE_DEFINED,$(1))$($(1)), \
$(call REQUIRE_DEFINED,$(call FAMILY_PREFIX,$(1)))$($(call FAMILY_PREFIX,$(1))) ++model.name=$(call SFT_PATH,$(1)))
endef

define DEFINE_MODEL_FROM_RESOLVER
$(eval $(1) := $(call RESOLVE_MODEL_NAME,$(1)))
endef

# dpo model weights
dpo_folder = ../../dpo/$*/final_hf # experiment/dpo/$*/final_hf
llama32_1b_openrlhf_mixture2 = $(llama32) ++model.name=$(dpo_folder)


