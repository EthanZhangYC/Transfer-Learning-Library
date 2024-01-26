


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


CUDA_VISIBLE_DEVICES=7 \
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
--pseudo_thres 0.85 \
--pseudo_ratio 0.666 \
--log logs/1108_mcd_freeze_01_rmBC_correctpseudo_pseudoconf085propor0666_pseudoweight01_nonbrpseudoonly

CUDA_VISIBLE_DEVICES=5 \
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
--pseudo_mode confidence_and_entropyproportion \
--pseudo_thres 0.95 \
--pseudo_ratio 0.666 \
--log logs/1117_mcd_freeze_01_rmBC_correctpseudo_pseudoconf095entpropor0666_pseudoweight01_nonbrpseudoonly_evalpseudo_funcdropout




CUDA_VISIBLE_DEVICES=5 \
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
--trade-off-consis 0.5 \
--cat_mode add \
--pseudo_mode proportion \
--pseudo_ratio 0.666 \
--mean_tea \
--momentum 0.95 \
--loss_mode srcce_ent_tgtce_tgtmeanteanomask \
--log logs/1115_mcd_freeze_rmBC_correctpseudo_pseudopropor0666_pseudoweight01_nonbrpseudoonly_meantea_095f1_srcce1_ent01_tgtce01_tgtmeanteanomask05






--pseudo_mode confidence_and_entropyproportion \
--pseudo_thres 0.9 \

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
--trade-off-pseudo 0.1 \
--cat_mode cat_samedim \
--nbr_dist_thres 10 \
--nbr_limit 10 \
--pseudo_mode entropyproportion \
--pseudo_ratio 0.666 \
--log logs/1117_mcd_freeze_01_10m_rmBC_nbrlimit10_correctpseudo_pseudoentpropor0666_featcatsamedim_pseudoweight01_freezefc





CUDA_VISIBLE_DEVICES=5 \
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
--trade-off-consis 0.1 \
--cat_mode cat_samedim \
--nbr_dist_thres 10 \
--nbr_limit 10 \
--pseudo_mode proportion \
--pseudo_ratio 0.666 \
--mean_tea \
--momentum 0.95 \
--loss_mode srcce_ent_tgtce_tgtmeanteanomask_hardtea \
--log logs/1118_mcd_freeze_01_10m_nbrlimit10_featcatsamedim_pseudopropor0666_meantea095f1_srcce1_ent01_tgtce01_tgtmeanteanomask_hardtea01_freezefc




--loss_mode srcce_ent_tgtce_tgtmeanteanomask_hardtea \

CUDA_VISIBLE_DEVICES=5 \
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
--trade-off-consis 0.1 \
--cat_mode cat_samedim \
--nbr_dist_thres 10 \
--nbr_limit 10 \
--pseudo_mode proportion \
--pseudo_ratio 0.666 \
--mean_tea \
--momentum 0.9 \
--loss_mode srcce_ent_tgtce_tgtmeanteanomask \
--log logs/1120_mcd_freeze_01_10m_nbrlimit10_featcatsamedim_pseudopropor0666_meantea09f1_srcce1_ent01_tgtce01_tgtmeanteanomask01_freezefc


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
--trade-off-consis 0.1 \
--cat_mode cat_samedim \
--nbr_dist_thres 10 \
--nbr_limit 100000 \
--pseudo_mode confidence_and_entropyproportion \
--pseudo_thres 0.85 \
--pseudo_ratio 0.666 \
--mean_tea \
--momentum 0.95 \
--loss_mode srcce_ent_tgtce_tgtmeanteanomask \
--log logs/1123_mcd_freeze_01_10m_featcatsamedim_pseudoconf085entpropor0666_meantea095f1_srcce1_ent01_tgtce01_tgtmeanteanomask01_freezefc



--loss_mode srcce_ent_tgtce_tgtmeanteanomask_hardtea \
--pseudo_mode entropyproportion \



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
--trade-off-pseudo 0.1 \
--trade-off-consis 0.1 \
--cat_mode cat_samedim \
--nbr_dist_thres 10 \
--nbr_limit 100000 \
--pseudo_mode entropyproportion \
--pseudo_ratio 0.666 \
--mean_tea \
--momentum 0.95 \
--loss_mode srcce_ent_tgtce_tgtmeanteanomask \
--log logs/1128_mcd_freeze_01_10m_featcatsamedim_entpropor0666_meantea095f1_srcce1_ent01_tgtce01_tgtmeanteanomask01_newperptnbr

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
--trade-off-pseudo 0.1 \
--trade-off-consis 0.1 \
--cat_mode cat_samedim \
--nbr_dist_thres 10 \
--nbr_limit 100000 \
--pseudo_mode entropyproportion \
--pseudo_ratio 0.666 \
--loss_mode srcce_ent_tgtce \
--log logs/1128_mcd_freeze_01_10m_featcatsamedim_entpropor0666_srcce1_ent01_tgtce01_newperptnbr

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
--trade-off-consis 0.1 \
--cat_mode cat_samedim \
--nbr_dist_thres 20 \
--nbr_limit 100000 \
--pseudo_mode entropyproportion \
--pseudo_ratio 0.666 \
--mean_tea \
--momentum 0.95 \
--loss_mode srcce_ent_tgtce_tgtmeanteanomask \
--log logs/1128_mcd_freeze_01_20m_featcatsamedim_entpropor0666_meantea095f1_srcce1_ent01_tgtce01_tgtmeanteanomask01_newperptnbr


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
--trade-off-consis 0.1 \
--cat_mode cat_samedim \
--nbr_dist_thres 10 \
--nbr_limit 100000 \
--pseudo_mode entropyproportion \
--pseudo_ratio 0.666 \
--mean_tea \
--momentum 0.95 \
--loss_mode srcce_ent_tgtce_tgtmeanteanomask \
--log logs/1211_mcd_freeze_01_10m_featcatsamedim_entpropor0666_meantea095f1_srcce1_ent01_tgtce01_tgtmeanteanomask01_1210new100ptnbr


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
--trade-off-pseudo 0.1 \
--trade-off-consis 0.1 \
--cat_mode cat_samedim \
--nbr_dist_thres 20 \
--nbr_limit 100000 \
--pseudo_mode entropyproportion \
--pseudo_ratio 0.666 \
--loss_mode srcce_ent_tgtce \
--nbr_data_mode mergetoori \
--log logs/0125_mcd_freeze_01_featcatsamedim_entpropor0666_srcce1_ent01_tgtce01_0124newperptnbr20m_mergetoori_correctmtldata


CUDA_VISIBLE_DEVICES=8 \
python mcd_neighbor_v5_attn.py \
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
--nbr_dist_thres 10 \
--nbr_limit 100000 \
--pseudo_mode entropyproportion \
--pseudo_ratio 0.666 \
--loss_mode srcce_ent_tgtce \
--log logs/0124_mcd_freeze_01_entpropor0666_srcce1_ent01_tgtce01_0124newperptnbr20m_qkvattn_head2


--cat_mode cat \


CUDA_VISIBLE_DEVICES=7 \
python mcd_neighbor_v5_attn.py \
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
--nbr_dist_thres 20 \
--nbr_limit 100000 \
--pseudo_mode entropyproportion \
--pseudo_ratio 0.666 \
--loss_mode srcce_ent_tgtce \
--num_head 2 \
--log logs/0124_mcd_freeze_01_entpropor0666_srcce1_ent01_tgtce01_0124newperptnbr20m_qkvattn_head2_catsamedim_attnoutputnbrfeat_correctqkvdim_correctmtldata



CUDA_VISIBLE_DEVICES=8 \
python mcd_neighbor_v4.py \
--epochs 30 \
--batch-size 64 \
-i 294 \
--seed 42 \
--lr 5e-4 \
--wd 1e-4 \
--interpolatedlinear \
--trade-off 1 \
--trade-off-entropy 0.1 \
--cat_mode cat \
--nbr_mode 111 \
--nbr_dist_thres 20 \
--nbr_limit 100000 \
--log logs/0124_mcd_0124newperptnbr20m_cat_correctmtldata

CUDA_VISIBLE_DEVICES=8 \
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
--cat_mode cat \
--nbr_dist_thres 10 \
--nbr_limit 100000 \
--log logs/0124_mcd_1220newperptnbr10m_cat_duplicatenbrextract_correctmtldata

CUDA_VISIBLE_DEVICES=4 \
python mcd_neighbor_v4.py \
--epochs 30 \
--batch-size 64 \
-i 294 \
--seed 42 \
--lr 5e-4 \
--wd 1e-4 \
--interpolatedlinear \
--trade-off 1 \
--trade-off-entropy 0.1 \
--cat_mode cat \
--nbr_mode qkv_cat \
--nbr_dist_thres 10 \
--nbr_limit 100000 \
--log logs/1224_mcd_1220newperptnbr10m_qkvcat

CUDA_VISIBLE_DEVICES=7 \
python mcd_neighbor_v4.py \
--epochs 30 \
--batch-size 64 \
-i 294 \
--seed 42 \
--lr 5e-4 \
--wd 1e-4 \
--interpolatedlinear \
--trade-off 1 \
--trade-off-entropy 0.1 \
--cat_mode add \
--nbr_mode qkv_individual \
--nbr_dist_thres 10 \
--nbr_limit 100000 \
--num_head 2 \
--log logs/0124_mcd_1220newperptnbr10m_qkvindividual_head2_correctmtldata



--nbr_mode qkv_individual_vnbr \




CUDA_VISIBLE_DEVICES=6 \
python mcd_neighbor_v4.py \
--epochs 30 \
--batch-size 64 \
-i 294 \
--seed 42 \
--lr 5e-4 \
--wd 1e-4 \
--interpolatedlinear \
--trade-off 1 \
--trade-off-entropy 0.1 \
--cat_mode add \
--nbr_mode c_feat_nbr \
--nbr_dist_thres 10 \
--nbr_limit 100000 \
--log logs/1129_mcd_newperptnbr10m_CFeatNbr

CUDA_VISIBLE_DEVICES=6 \
python mcd_neighbor_v4.py \
--epochs 30 \
--batch-size 64 \
-i 294 \
--seed 42 \
--lr 5e-4 \
--wd 1e-4 \
--interpolatedlinear \
--trade-off 1 \
--trade-off-entropy 0.1 \
--cat_mode cat \
--nbr_mode c_featfeat_nbrnbr \
--nbr_dist_thres 10 \
--nbr_limit 100000 \
--log logs/1203_mcd_newperptnbr10m_CFeatfeatNbrnbr

CUDA_VISIBLE_DEVICES=6 \
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
--cat_mode cat \
--nbr_dist_thres 10 \
--nbr_limit 100000 \
--log logs/1205_mcd_newperptnbr10m_cat_duplicatenbrextract








CUDA_VISIBLE_DEVICES=6 \
python mcd_neighbor_v3_1120singlepad.py \
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
--nbr_dist_thres 10 \
--nbr_limit 10 \
--pseudo_mode entropyproportion \
--pseudo_ratio 0.666 \
--log logs/1120_mcd_freeze_01_10m_nbrlimit10_featcatsamedim_pseudoentpropor0666_srcce1_ent01_tgtce01_freezefc_singlepad