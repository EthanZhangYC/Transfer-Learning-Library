
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












# need run:

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




--pseudo_ratio 0.666 \
--pseudo_mode entropyproportion \

--pseudo_thres 0.9 \
--pseudo_ratio 0.666 \
--pseudo_mode confidence_and_entropyproportion \

--cat_mode cat \
--cat_mode cat_samedim \

--trade-off-consis 0.1 \
--mean_tea \
--momentum 0.95 \
--loss_mode srcce_ent_tgtce_tgtmeanteanomask \
--loss_mode srcce_ent_tgtce_tgtmeanteanomask_hardtea \






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
















CUDA_VISIBLE_DEVICES=7 \
python mcd_neighbor_v3.py \
--epochs 50 \
--batch-size 64 \
-i 294 \
--lr 5e-4 \
--wd 1e-4 \
--interpolatedlinear \
--trade-off 1 \
--trade-off-entropy 0.1 \
--trade-off-pseudo 0.1 \
--cat_mode cat \
--nbr_dist_thres 20 \
--nbr_limit 100000 \
--pseudo_mode entropyproportion \
--pseudo_ratio 0.666 \
--loss_mode srcce_ent_tgtce \
--nbr_data_mode mergetoori \
--log logs/0130_mcd_freeze_featcat_entpropor0666_srcce1_ent01_tgtce01_0124newperptnbr20m_v3_mergetoori_correctmtldata

CUDA_VISIBLE_DEVICES=8 \
python mcd_neighbor_v3.py \
--epochs 30 \
--batch-size 64 \
-i 294 \
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
--log logs/0131_mcd_freeze_featcatsamedim_entpropor0666_srcce1_ent01_tgtce01_v3_0124newnbr20m

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
--mean_tea \
--momentum 0.95 \
--loss_mode srcce_ent_tgtce_tgtmeanteanomask \
--log logs/0131_mcd_freeze_featcatsamedim_entpropor0666_srcce1_ent01_tgtce01_tgtmeanteanomask01_meantea_095f1_0124newnbr20m

















--log logs/0131_mcd_v4_cat_0124newnbr20m

CUDA_VISIBLE_DEVICES=9 \
python mcd_neighbor_v4.py \
--epochs 50 \
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
--num_head 8 \
--nbr_data_mode mergetoori \
--num-k 10 \
--log logs/0214_mcd_01_v4_cat_head8_correctmergetoori20mmin50_nk10_correctnbrmean






CUDA_VISIBLE_DEVICES=9 \
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
--nbr_dist_thres 20 \
--nbr_limit 100000 \
--num_head 8 \
--nbr_data_mode mergetoori \
--num-k 10 \
--log logs/0220_mcd_01_v4_qkvcat_head8_correctmergetoori20mmin50_nk10_correctnbrmean
_loadmcd


--log logs/0131_mcd_qkvcat_v4_0124newnbr20m






--num_head 2 \
--num-k 4 \

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
--cat_mode add \
--nbr_mode qkv_individual \
--nbr_dist_thres 20 \
--nbr_limit 100000 \
--num_head 8 \
--nbr_data_mode mergetoori \
--num-k 10 \
--log logs/0220_mcd_01_v4_qkvindividualwithgrad_head8_correctmergetoori20mmin50_nk10_correctnbrmean
_loadmcd


CUDA_VISIBLE_DEVICES=8 \
python mcd_neighbor_v4.py \
--epochs 30 \
--batch-size 32 \
-i 294 \
--seed 42 \
--lr 5e-4 \
--wd 1e-4 \
--interpolatedlinear \
--trade-off 1 \
--trade-off-entropy 0.1 \
--cat_mode add \
--nbr_mode qkv_individual \
--nbr_dist_thres 20 \
--nbr_limit 100000 \
--num_head 8 \
--num-k 10 \
--log logs/0221_mcd_01_v4_qkvindividualwithgrad_head8_0124newnbr20m_nk10_correctnbrmean

CUDA_VISIBLE_DEVICES=8 \
python mcd_neighbor_v4.py \
--epochs 30 \
--batch-size 16 \
-i 294 \
--seed 42 \
--lr 5e-4 \
--wd 1e-4 \
--interpolatedlinear \
--trade-off 1 \
--trade-off-entropy 0.1 \
--cat_mode cat \
--nbr_mode perpt_cat \
--nbr_dist_thres 20 \
--nbr_limit 10 \
--num_head 2 \
--num-k 1 \
--log logs/0227_mcd_01_v4_perptcat_head8_0124newnbr20m_nk10_nbrlimit10




--cat_mode add \
--nbr_mode qkv_individual_vnbr \

--cat_mode add \
--nbr_mode c_feat_nbr \

--cat_mode cat \
--nbr_mode c_featfeat_nbrnbr \



















CUDA_VISIBLE_DEVICES=9 \
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
--cat_mode cat \
--nbr_dist_thres 20 \
--nbr_limit 100000 \
--pseudo_mode entropyproportion \
--pseudo_ratio 0.666 \
--loss_mode srcce_ent_tgtce \
--num_head 2 \
--nbr_data_mode mergetoori \
--log logs/0131_mcd_v5_freeze_01_entpropor0666_srcce1_ent01_tgtce01_qkvattn_head2_catsamedim_attnoutputnbrfeat_correctqkvdim_correctmergetoori20mmin50

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
--cat_mode cat_samedim \
--nbr_dist_thres 20 \
--nbr_limit 100000 \
--pseudo_mode entropyproportion \
--pseudo_ratio 0.666 \
--loss_mode srcce_ent_tgtce \
--num_head 2 \
--log logs/0131_mcd_v5_freeze_01_entpropor0666_srcce1_ent01_tgtce01_qkvattn_head2_catsamedim_attnoutputnbrfeat_correctqkvdim_0124newnbr20m



############################################################################################################





# per-pt cat

CUDA_VISIBLE_DEVICES=8 \
python mcd_neighbor_v5_attn.py \
--epochs 100 \
--batch-size 64 \
-i 294 \
--seed 42 \
--lr 5e-4 \
--wd 1e-4 \
--interpolatedlinear \
--trade-off 1 \
--trade-off-entropy 0.1 \
--trade-off-pseudo 0.1 \
--nbr_mode perpt_cat \
--nbr_dist_thres 20 \
--nbr_limit 10 \
--nbr_grad \
--pseudo_mode entropyproportion \
--pseudo_ratio 0.666 \
--num_head 2 \
--loss_mode srcce_ent_tgtce \
--log logs/0312_mcd_v5_freeze_01_entpropor0666_srcce1_ent01_tgtce01_perptcat_0124newnbr20m_limit10_nbrgrad


# qkv cat

CUDA_VISIBLE_DEVICES=5 \
python mcd_neighbor_v5_attn.py \
--epochs 50 \
--batch-size 64 \
-i 294 \
--seed 42 \
--lr 5e-4 \
--wd 1e-4 \
--interpolatedlinear \
--loss_mode srcce_ent_tgtce \
--trade-off 1 \
--trade-off-entropy 0.1 \
--trade-off-pseudo 0.1 \
--nbr_mode qkv_cat \
--nbr_dist_thres 20 \
--nbr_limit 10 \
--nbr_grad \
--pseudo_mode entropyproportion \
--pseudo_ratio 0.666 \
--num_head 8 \
--log logs/0312_mcd_v5_freeze_01_entpropor0666_srcce1_ent01_tgtce01_qkvcat_0124newnbr20m_limit10_nbrgrad_head8


CUDA_VISIBLE_DEVICES=1 \
python mcd_neighbor_v5_attn.py \
--epochs 50 \
--batch-size 64 \
-i 294 \
--seed 42 \
--lr 5e-4 \
--wd 1e-4 \
--interpolatedlinear \
--loss_mode srcce_ent_tgtce \
--trade-off 1 \
--trade-off-entropy 0.1 \
--trade-off-pseudo 0.1 \
--nbr_mode qkv_cat \
--nbr_dist_thres 20 \
--nbr_limit 100 \
--nbr_grad \
--pseudo_mode entropyproportion \
--pseudo_ratio 0.666 \
--num_head 2 \
--nbr_data_mode mergenomin \
--log logs/0317_mcd_v5_freeze_01_entpropor0666_srcce1_ent01_tgtce01_qkvcat_0315mergenomin_limit100_nbrgrad



CUDA_VISIBLE_DEVICES=8 \
python mcd_neighbor_v5_attn.py \
--epochs 50 \
--batch-size 64 \
-i 294 \
--seed 42 \
--lr 5e-4 \
--wd 1e-4 \
--interpolatedlinear \
--loss_mode srcce_ent_tgtce \
--trade-off 1 \
--trade-off-entropy 0.1 \
--trade-off-pseudo 0.1 \
--nbr_mode qkv_cat \
--nbr_dist_thres 20 \
--nbr_limit 10 \
--nbr_grad \
--nbr_pseudo \
--pseudo_mode entropyproportion \
--pseudo_ratio 0.666 \
--num_head 2 \
--log logs/0317_mcd_v5_freeze_01_entpropor0666_srcce1_ent01_tgtce01_qkvcat_0124newnbr20_limit10_nbrgrad

CUDA_VISIBLE_DEVICES=8 \
python mcd_neighbor_v5_attn.py \
--epochs 50 \
--batch-size 64 \
-i 294 \
--seed 42 \
--lr 5e-4 \
--wd 1e-4 \
--interpolatedlinear \
--loss_mode srcce_ent_tgtce \
--trade-off 1 \
--trade-off-entropy 0.1 \
--trade-off-pseudo 0.1 \
--nbr_mode qkv_cat \
--nbr_dist_thres 20 \
--nbr_limit 10 \
--nbr_grad \
--nbr_pseudo \
--pseudo_mode entropyproportion \
--pseudo_ratio 0.666 \
--num_head 2 \
--pseudo_every_epoch \
--log logs/0324_mcd_v5_freeze_01_entpropor0666_srcce1_ent01_tgtce01_qkvcat_0124newnbr20_limit10_nbrgrad



CUDA_VISIBLE_DEVICES=1 \
CUDA_VISIBLE_DEVICES=6 \
CUDA_VISIBLE_DEVICES=8 \
CUDA_VISIBLE_DEVICES=9 \
CUDA_VISIBLE_DEVICES=6 \
python mcd_neighbor_v5_attn.py \
--epochs 50 \
--batch-size 64 \
-i 294 \
--seed 42 \
--lr 5e-4 \
--wd 1e-4 \
--interpolatedlinear \
--loss_mode srcce_ent_tgtce \
--trade-off 1 \
--trade-off-entropy 0.1 \
--trade-off-pseudo 0.1 \
--nbr_mode qkv_cat \
--nbr_dist_thres 20 \
--nbr_limit 10 \
--nbr_grad \
--nbr_pseudo \
--pseudo_mode entropyproportion \
--pseudo_ratio 0.666 \
--num_head 2 \
--nbr_label_mode separate_input \
--nbr_label_embed_dim 8 \
--log logs/0320_mcd_v5_freeze_01_entpropor0666_srcce1_ent01_tgtce01_qkvcat_0124newnbr20_limit10_nbrgrad_separateinput_embeddim8





--nbr_label_mode combine_each_pt \
--log logs/0319_mcd_v5_freeze_01_entpropor0666_srcce1_ent01_tgtce01_qkvcat_0124newnbr20_limit100_nbrgrad_nbrlabelcombineeachpt

--nbr_data_mode mergetoori \

--nbr_mode qkv_cat_maskboth \



CUDA_VISIBLE_DEVICES=8 \
python mcd_neighbor_v5_attn.py \
--epochs 50 \
--batch-size 64 \
-i 294 \
--seed 42 \
--lr 5e-4 \
--wd 1e-4 \
--interpolatedlinear \
--loss_mode v1 \
--trade-off 1 \
--trade-off-entropy 0. \
--trade-off-pseudo 0.1 \
--trade-off-consis 10 \
--nbr_mode qkv_individual_add \
--nbr_dist_thres 20 \
--nbr_limit 10 \
--nbr_grad \
--pseudo_mode entropyproportion \
--pseudo_ratio 0.666 \
--num_head 2 \
--log logs/0316_mcd_v5_freeze_0_01_10_entpropor0666_lossV1_qkvindividual_0124newnbr20m_limit10_nbrgrad






CUDA_VISIBLE_DEVICES=8 \
python mcd_neighbor_v5_attn.py \
--epochs 50 \
--batch-size 64 \
-i 294 \
--seed 42 \
--lr 5e-4 \
--wd 1e-4 \
--interpolatedlinear \
--loss_mode v2 \
--trade-off 1 \
--trade-off-entropy 0.1 \
--trade-off-pseudo 0.1 \
--nbr_mode perpt_qkv_cat \
--nbr_dist_thres 20 \
--nbr_limit 10 \
--nbr_grad \
--nbr_pseudo \
--pseudo_mode entropyproportion \
--pseudo_ratio 0.666 \
--num_head 2 \
--log logs/0326_mcd_v5_freeze_01_entpropor0666_v2_perptqkvcat_0124newnbr20_limit10_nbrgrad

CUDA_VISIBLE_DEVICES=8 \
python mcd_neighbor_v5_attn.py \
--epochs 50 \
--batch-size 64 \
-i 294 \
--seed 42 \
--lr 5e-4 \
--wd 1e-4 \
--interpolatedlinear \
--loss_mode v2 \
--trade-off 1 \
--trade-off-entropy 0.1 \
--trade-off-pseudo 0.1 \
--nbr_mode perpt_qkv_add \
--nbr_dist_thres 20 \
--nbr_limit 10 \
--nbr_grad \
--nbr_pseudo \
--pseudo_mode entropyproportion \
--pseudo_ratio 0.666 \
--num_head 2 \
--log logs/0331_mcd_v5_freeze_01_entpropor0666_v2_perptqkvadd_0124newnbr20_limit10_nbrgrad

--log logs/0326_mcd_v5_freeze_01_entpropor0666_v2_perptqkvcat_0124newnbr20_limit10_Nonbrgrad






CUDA_VISIBLE_DEVICES=6 \
CUDA_VISIBLE_DEVICES=1 \
python mcd_neighbor_v5_attn.py \
--epochs 50 \
--batch-size 64 \
-i 294 \
--seed 42 \
--lr 5e-4 \
--wd 1e-4 \
--interpolatedlinear \
--loss_mode v2 \
--trade-off 1 \
--trade-off-entropy 0.1 \
--trade-off-pseudo 0.1 \
--nbr_mode qkv_cat \
--nbr_dist_thres 20 \
--nbr_limit 10 \
--nbr_grad \
--nbr_pseudo \
--pseudo_mode entropyproportion \
--pseudo_ratio 0.666 \
--num_head 2 \
--random_mask_nbr_ratio 0.5 \
--log logs/0331_mcd_v5_freeze_01_entpropor0666_v2_qkvcat_0124newnbr20_limit10_nbrgrad_nbrmask05

CUDA_VISIBLE_DEVICES=6 \
python mcd_neighbor_v5_attn.py \
--epochs 50 \
--batch-size 64 \
-i 294 \
--seed 42 \
--lr 5e-4 \
--wd 1e-4 \
--interpolatedlinear \
--loss_mode v2 \
--trade-off 1 \
--trade-off-entropy 0.1 \
--trade-off-pseudo 0.1 \
--nbr_mode qkv_cat_maskboth \
--nbr_dist_thres 20 \
--nbr_limit 10 \
--nbr_grad \
--nbr_pseudo \
--pseudo_mode entropyproportion \
--pseudo_ratio 0.666 \
--num_head 2 \
--random_mask_nbr_ratio 0.5 \
--log logs/0331_mcd_v5_freeze_01_entpropor0666_v2_qkvcat_0124newnbr20_limit10_nbrgrad_nbrmask05


--mask_early \


CUDA_VISIBLE_DEVICES=9 \
python mcd_neighbor_v5_attn.py \
--epochs 50 \
--batch-size 64 \
-i 294 \
--seed 42 \
--lr 5e-4 \
--wd 1e-4 \
--interpolatedlinear \
--loss_mode v2 \
--trade-off 1 \
--trade-off-entropy 0.1 \
--trade-off-pseudo 0.1 \
--nbr_mode qkv_cat \
--nbr_dist_thres 20 \
--nbr_limit 10 \
--nbr_grad \
--nbr_pseudo \
--pseudo_mode entropyproportion \
--pseudo_ratio 0.666 \
--num_head 2 \
--mask_late \
--n_mask_late 10 \
--log logs/0403_mcd_v5_freeze_01_entpropor0666_v2_qkvcat_0124newnbr20_limit10_nbrgrad_masklate10















CUDA_VISIBLE_DEVICES=8 \
python mcd_neighbor_v5_attn_failcase.py \
--batch-size 64 \
-i 294 \
--seed 42 \
--interpolatedlinear \
--nbr_mode qkv_cat \
--nbr_dist_thres 20 \
--nbr_limit 10 \
--nbr_grad \
--nbr_pseudo \
--pseudo_mode entropyproportion \
--pseudo_ratio 0.666 \
--num_head 2 \
--log logs/test


python find_common_failcase.py








--nbr_mode bert_learnable_cat \
--bert_out_dim 64 \
--token_max_len 110 \
--token_len 10 \



CUDA_VISIBLE_DEVICES=1 \
python mcd_neighbor_v7_bertwithmcd.py \
--epochs 30 \
--batch-size 64 \
-i 294 \
--seed 42 \
--lr 5e-4 \
--wd 1e-4 \
--interpolatedlinear \
--trade-off 1 \
--trade-off-entropy 0.1 \
--nbr_mode bert_cat \
--steps_list ABC \
--nbr_dist_thres 20 \
--nbr_limit 10 \
--token_max_len 60 \
--prompt_id 1 \
--log logs/0422_mcd_v7_freeze_01_bertcat_ABC_p1_maxlen60_2fc_correctbertinput

CUDA_VISIBLE_DEVICES=9 \
python mcd_neighbor_v7_bertwithmcd.py \
--epochs 30 \
--batch-size 64 \
-i 294 \
--seed 42 \
--lr 5e-4 \
--wd 1e-4 \
--interpolatedlinear \
--trade-off 1 \
--trade-off-entropy 0.1 \
--nbr_mode bertonly_add \
--steps_list ABC \
--nbr_dist_thres 20 \
--nbr_limit 10 \
--token_max_len 60 \
--prompt_id 1 \
--log logs/0422_mcd_v7_freeze_01_bertonly_ABC_p1_maxlen60_2fc_correctbertinput

CUDA_VISIBLE_DEVICES=9 \
python mcd_neighbor_v7_bertwithmcd.py \
--epochs 30 \
--batch-size 64 \
-i 294 \
--seed 42 \
--lr 5e-4 \
--wd 1e-4 \
--interpolatedlinear \
--trade-off 1 \
--trade-off-entropy 0.1 \
--nbr_mode bertonly_add \
--steps_list ABC \
--nbr_dist_thres 20 \
--nbr_limit 10 \
--token_max_len 60 \
--prompt_id 1 \
--log logs/test













--nbr_mode bertonly_learnable_add \
--token_len 10 \

--nbr_mode bertonly_add \
--steps_list A_sourceonly \









