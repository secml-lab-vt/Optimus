#!/usr/bin/env bash
# Step 3: inject_eval — model UUID bea8aadd-4e07-4808-87b7-ba5949a26a3d

export USE_MODEL_UUID="bea8aadd-4e07-4808-87b7-ba5949a26a3d"
exec "$(dirname "$0")/script1_0_run_star_inject_eval_all_seeds.sh"
