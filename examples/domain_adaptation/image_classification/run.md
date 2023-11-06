


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







--cat_mode add \

--loss_mode tgtce \



CUDA_VISIBLE_DEVICES=6 \
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
--trade-off-pseudo 0.1 \
--cat_mode cat \
--nbr_dist_thres 30 \
--nbr_limit 10 \
--pseudo_mode proportion \
--pseudo_ratio 0.666 \
--log logs/1027_mcd_freeze_01_30m_rmBC_nbrlimit10_correctpseudo_pseudoproportion0666_featcat_pseudoweight01



CUDA_VISIBLE_DEVICES=8 \
python mcd_neighbor_v3_pseudoonly.py \
--epochs 30 \
--batch-size 64 \
-i 294 \
--seed 42 \
--lr 5e-4 \
--wd 1e-4 \
--interpolatedlinear \
--trade-off 1 \
--trade-off-entropy 0.1 \
--trade-off-pseudo 0.1 \
--cat_mode add \
--pseudo_mode proportion \
--pseudo_ratio 0.666 \
--log logs/1101_mcd_freeze_01_rmBC_correctpseudo_pseudoproportion0666_pseudoweight01_nonbrpseudoonly



# need run:


CUDA_VISIBLE_DEVICES=8 \
python mcd_neighbor_v3_pseudoonly.py \
--epochs 30 \
--batch-size 64 \
-i 294 \
--seed 42 \
--lr 5e-4 \
--wd 1e-4 \
--interpolatedlinear \
--trade-off 1 \
--trade-off-entropy 0.1 \
--trade-off-pseudo 0.1 \
--cat_mode add \
--pseudo_mode confidence_and_proportion \
--pseudo_thres 0.95 \
--pseudo_ratio 0.666 \
--log logs/1106_mcd_freeze_01_rmBC_correctpseudo_pseudoconf95propor0666_pseudoweight01_nonbrpseudoonly

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
--trade-off-pseudo 0.1 \
--cat_mode cat_samedim \
--nbr_dist_thres 30 \
--nbr_limit 10 \
--pseudo_mode proportion \
--pseudo_ratio 0.666 \
--log logs/1106_mcd_freeze_01_30m_rmBC_nbrlimit10_correctpseudo_pseudoproportion0666_featcatsamedim_pseudoweight01





