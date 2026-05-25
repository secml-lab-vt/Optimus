# sft.mk - Supervised fine-tuning experiment rules

experiment/sft/%/.done_sft_openrlhf:
> @echo "[INFO] Running SFT experiment: $*"
> @mkdir -p $(dir $@)
> $(DEEPSPEED) --module src.sft_openrlhf ++name=$* $(SFT_$*)
# touch $@

experiment/sft/%/.done_sft_openrlhf_value:
> @echo "[INFO] Running SFT experiment with value: $*"
> @mkdir -p $(dir $@) 
> $(DEEPSPEED) --module src.create_value_dataset ++name=$* $(INFER) $(VALUE_$*)
> $(DEEPSPEED) --module src.sft_openrlhf ++name=$* $(SFT_$*)