#!/usr/bin/env bash
# Step 2: STAR-DSS training — alternate run UUID fb1943b2-92dc-4ae3-9f91-bafe2c665b2b

export ROUND_UUID="fb1943b2-92dc-4ae3-9f91-bafe2c665b2b"
exec "$(dirname "$0")/script1_0_run_star_all_seeds.sh"
