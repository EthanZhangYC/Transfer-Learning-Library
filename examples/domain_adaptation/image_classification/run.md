


############################ AFN ###################################


CUDA_VISIBLE_DEVICES=7 \
python afn.py \
--epochs 30 \
--batch-size 64 \
-i 294 \
--seed 42 \
--lr 5e-4 \
--wd 1e-4 \
--interpolatedlinear \
--trip_time 20 \
--trade-off-norm 0.005 \
--delta 1 \
--log logs/afn



CUDA_VISIBLE_DEVICES=8 \
python cdan.py \
--epochs 30 \
--batch-size 64 \
-i 294 \
--seed 42 \
--lr 5e-4 \
--wd 1e-4 \
--interpolatedlinear \
--trip_time 20 \
--trade-off 1 \
--log logs/cdan



CUDA_VISIBLE_DEVICES=7 \
python dan.py \
--epochs 30 \
--batch-size 64 \
-i 294 \
--seed 42 \
--lr 5e-4 \
--wd 1e-4 \
--interpolatedlinear \
--trip_time 20 \
--trade-off 0.1 \
--log logs/dan




CUDA_VISIBLE_DEVICES=8 \
python bsp.py \
--epochs 30 \
--batch-size 64 \
-i 294 \
--seed 42 \
--lr 5e-4 \
--wd 1e-4 \
--interpolatedlinear \
--trip_time 20 \
--trade-off 1 \
--trade-off-bsp 2e-3 \
--pretrain /home/yichen/TS2Vec/result/0402_pretrain/model_best.pth.tar \
--log logs/bsp




CUDA_VISIBLE_DEVICES=8 \
python erm.py \
--epochs 30 \
--batch-size 64 \
-i 294 \
--seed 42 \
--lr 5e-4 \
--wd 1e-4 \
--interpolatedlinear \
--log logs/erm




CUDA_VISIBLE_DEVICES=8 \
python adda.py \
--epochs 30 \
--batch-size 64 \
-i 294 \
--seed 42 \
--lr 5e-4 \
--wd 1e-4 \
--interpolatedlinear \
--pretrain /home/yichen/TS2Vec/result/0402_pretrain/model_best.pth.tar \
--log logs/adda




CUDA_VISIBLE_DEVICES=7 \
python dann.py \
--epochs 30 \
--batch-size 64 \
-i 294 \
--seed 42 \
--lr 5e-4 \
--wd 1e-4 \
--trade-off 1 \
--interpolatedlinear \
--log logs/jan






CUDA_VISIBLE_DEVICES=8 \
python jan.py \
--epochs 30 \
--batch-size 64 \
-i 294 \
--seed 42 \
--lr 5e-4 \
--wd 1e-4 \
--trade-off 0.1 \
--interpolatedlinear \
--log logs/jan
--adversarial \




CUDA_VISIBLE_DEVICES=8 \
python mcc.py \
--epochs 30 \
--batch-size 64 \
-i 294 \
--seed 42 \
--lr 5e-4 \
--wd 1e-4 \
--interpolatedlinear \
--trade-off 0.5 \
--temperature 0.25 \
--log logs/mcc



CUDA_VISIBLE_DEVICES=7 \
python mcd.py \
--epochs 30 \
--batch-size 64 \
-i 294 \
--seed 42 \
--lr 5e-4 \
--wd 1e-4 \
--interpolatedlinear \
--trade-off 1 \
--trade-off-entropy 0.1 \
--log logs/mcd_1015



CUDA_VISIBLE_DEVICES=7 \
python mcd.py \
--epochs 30 \
--batch-size 64 \
-i 294 \
--seed 42 \
--lr 5e-4 \
--wd 1e-4 \
--interpolatedlinear \
--trade-off 1 \
--trade-off-entropy 0.1 \
--log logs/mcd_vit



CUDA_VISIBLE_DEVICES=7 \
python mcd_neighbor_v2.py \
--epochs 30 \
--batch-size 64 \
-i 294 \
--seed 42 \
--lr 5e-4 \
--wd 1e-4 \
--interpolatedlinear \
--trade-off 1 \
--trade-off-entropy 0.1 \
--log logs/mcd_neighbor


CUDA_VISIBLE_DEVICES=8 \
python mcd_neighbor_v3.py \
--epochs 30 \
--batch-size 64 \
-i 294 \
--seed 42 \
--lr 5e-4 \
--wd 1e-4 \
--interpolatedlinear \
--trade-off 1 \
--trade-off-entropy 0.1 \
--log logs/1017_mcd_nofreeze_01_30m

CUDA_VISIBLE_DEVICES=7 \
python mcd_neighbor_v3.py \
--epochs 30 \
--batch-size 64 \
-i 294 \
--seed 42 \
--lr 5e-4 \
--wd 1e-4 \
--interpolatedlinear \
--trade-off 1 \
--trade-off-entropy 0.1 \
--log logs/1017_mcd_freeze_01_10m_rmBC

CUDA_VISIBLE_DEVICES=7 \
python mcd_neighbor_v3.py \
--epochs 30 \
--batch-size 64 \
-i 294 \
--seed 42 \
--lr 5e-4 \
--wd 1e-4 \
--interpolatedlinear \
--trade-off 1 \
--trade-off-entropy 0.1 \
--log logs/1017_mcd_freeze_01_30m_rmBC

CUDA_VISIBLE_DEVICES=8 \
python mcd_neighbor_v3.py \
--epochs 30 \
--batch-size 64 \
-i 294 \
--seed 42 \
--lr 5e-4 \
--wd 1e-4 \
--interpolatedlinear \
--trade-off 1 \
--trade-off-entropy 0. \
--log logs/1017_mcd_freeze_0_10m_rmBC

CUDA_VISIBLE_DEVICES=8 \
python mcd_neighbor_v3.py \
--epochs 30 \
--batch-size 64 \
-i 294 \
--seed 42 \
--lr 5e-4 \
--wd 1e-4 \
--interpolatedlinear \
--trade-off 1 \
--trade-off-entropy 0. \
--log logs/1017_mcd_freeze_0_30m_rmBC


CUDA_VISIBLE_DEVICES=8 \
python mcd_neighbor_v3.py \
--epochs 30 \
--batch-size 64 \
-i 294 \
--seed 42 \
--lr 5e-4 \
--wd 1e-4 \
--interpolatedlinear \
--trade-off 1 \
--trade-off-entropy 0.1 \
--log logs/1017_mcd_freeze_01_30m_rmBC_nbrlimit10

CUDA_VISIBLE_DEVICES=8 \
python mcd_neighbor_v3.py \
--epochs 100 \
--batch-size 64 \
-i 294 \
--seed 42 \
--lr 5e-4 \
--wd 1e-4 \
--interpolatedlinear \
--trade-off 1 \
--trade-off-entropy 0.1 \
--log logs/1017_mcd_freeze_01_30m_rmBC_nbrlimit10_epoch100




1016_mcd_freeze_01_10m
