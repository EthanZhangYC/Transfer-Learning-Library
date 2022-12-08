############################ DAN ###################################

CUDA_VISIBLE_DEVICES=8 \
python examples/domain_adaptation/image_classification/dan.py \
test \
-a SECA \
--epochs 20 \
-i 500 \
--seed 0 \
--lr 1e-3 \
--batch-size 4 \
--bottleneck-dim 64 \
--interpolated \
--trip_time 5 \
--log logs/dan/test




CUDA_VISIBLE_DEVICES=8 \
python examples/domain_adaptation/image_classification/dann.py \
test \
-a SECA \
--epochs 20 \
-i 500 \
--seed 0 \
--lr 1e-3 \
--batch-size 4 \
--bottleneck-dim 64 \
--interpolated \
--trip_time 5 \
--log logs/dann/test



######## AFN

CUDA_VISIBLE_DEVICES=8 \
python afn.py test \
-a SECA \
--epochs 20 \
-i 500 \
--seed 0 \
--lr 1e-3 \
--wd 5e-4 \
--batch-size 64 \
--bottleneck-dim 64 \
--interpolated \
--trip_time 20 \
--log logs/afn/test