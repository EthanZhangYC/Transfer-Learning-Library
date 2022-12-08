############################ DAN ###################################

CUDA_VISIBLE_DEVICES=8 \
python cdan.py \
test \
-a SECA \
--epochs 20 \
-i 500 \
--seed 0 \
--lr 1e-3 \
--batch-size 64 \
--bottleneck-dim 64 \
--interpolated \
--trip_time 20 \
--log logs/cdan/test




CUDA_VISIBLE_DEVICES=8 \
python dann.py \
test \
-a SECA \
--epochs 20 \
-i 500 \
--seed 0 \
--lr 1e-3 \
--batch-size 64 \
--bottleneck-dim 64 \
--trip_time 5 \
--log logs/dann/test