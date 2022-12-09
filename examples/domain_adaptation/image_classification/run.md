


############################ AFN ###################################


CUDA_VISIBLE_DEVICES=6 \
python afn.py \
--epochs 30 \
--batch-size 64 \
-i 390 \
--seed 42 \
--lr 1e-3 \
--wd 0. \
--bottleneck-dim 64 \
--interpolated \
--trip_time 20 \
--trade-off-norm 0.005 \
--log logs/afn_0005
