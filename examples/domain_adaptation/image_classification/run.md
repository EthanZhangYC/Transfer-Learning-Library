


############################ AFN ###################################


CUDA_VISIBLE_DEVICES=8 \
python afn.py \
--epochs 30 \
--batch-size 64 \
-i 294 \
--seed 42 \
--lr 5e-4 \
--wd 1e-4 \
--bottleneck-dim 64 \
--interpolated \
--trip_time 20 \
--trade-off-norm 0.005 \
--log logs/afn_0005
