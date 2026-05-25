# dpo.mk - Direct preference optimization experiment rules

experiment/dpo/%/.done_dpo_openrlhf:
> @echo "[INFO] Running DPO experiment: $*"
> @mkdir -p $(dir $@)
> $(DEEPSPEED) --module src.dpo_openrlhf ++name=$* $(DPO_$*)
# touch $@