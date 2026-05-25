#!/usr/bin/env bash
# Step 3: inject_eval — model UUID fb1943b2-92dc-4ae3-9f91-bafe2c665b2b

export USE_MODEL_UUID="fb1943b2-92dc-4ae3-9f91-bafe2c665b2b"
exec "$(dirname "$0")/script1_0_run_star_inject_eval_all_seeds.sh"
