CHAT = ++model.generation_args.max_length=null \
	++model.generation_args.max_new_tokens=5000 \
	++deepspeed.zero_stage=3

experiment/chat/%/.done_chat:
> @echo "[INFO] Interative chat with $*"
> @mkdir -p $(dir $@)  # Ensure the directory exists
> $(DEEPSPEED) --module src.interactive_chat ++name=$* $(CHAT) $($*) 
