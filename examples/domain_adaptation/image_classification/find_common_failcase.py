"""
@author: Junguang Jiang
@contact: JiangJunguang1123@outlook.com
"""
import random
import time
import warnings
import argparse
import shutil
import os.path as osp
from typing import Tuple
import glob

import pickle
import numpy as np
import pdb
import operator

also_correct=True

filename_list=glob.glob('/home/yichen/Transfer-Learning-Library/examples/domain_adaptation/image_classification/logs/failcase/*')
print(filename_list)
total_wrong_dict={}
total_correct_dict={}
for idx,filename in enumerate(filename_list):
    wrong_dict = np.load(filename, allow_pickle=True)
    wrong_dict = dict(enumerate(wrong_dict.flatten(), 1))[1]
    # for k,v in wrong_dict.items():
    #     print(k,len(v))
    if idx==0:
        for k in wrong_dict:
            total_wrong_dict[k]=set(wrong_dict[k])
    else:
        for k in wrong_dict:
            total_wrong_dict[k]=set(wrong_dict[k]) & total_wrong_dict[k]
     
if also_correct:
    filename_list=glob.glob('/home/yichen/Transfer-Learning-Library/examples/domain_adaptation/image_classification/logs/failcase_correct/*')       
    total_correct_dict={}
    for idx,filename in enumerate(filename_list):
        correct_dict = np.load(filename, allow_pickle=True)
        correct_dict = dict(enumerate(correct_dict.flatten(), 1))[1]
        # for k,v in correct_dict.items():
        #     print(k,len(v))
        if idx==0:
            for k in correct_dict:
                total_correct_dict[k]=set(correct_dict[k])
        else:
            for k in correct_dict:
                total_correct_dict[k]=set(correct_dict[k]) & total_correct_dict[k]

print('wrong:')
for k,v in total_wrong_dict.items():
    print(k,len(v))   
if also_correct:
    print('correct:')
    for k,v in total_correct_dict.items():
        print(k,len(v))   
    
    
    
filename_mtl = '/home/yichen/TS2Vec/datafiles/MTL/traindata_4class_xy_traintest_interpolatedLinear_5s_trip20_new_001meters_withdist_aligninterpolation_InsertAfterSeg_Both_11dim_0817_sharedminmax_balanced.pickle'
with open(filename_mtl, 'rb') as f:
    kfold_dataset, X_unlabeled_mtl = pickle.load(f)
dataset_mtl = kfold_dataset
test_x = dataset_mtl[4].squeeze(1)
test_y = dataset_mtl[5]
test_x = test_x[:,:,2:]
pad_mask_target_test = test_x[:,:,2]==0
test_x[pad_mask_target_test] = 0.

nbr_idx_tuple = np.load('/home/yichen/TS2Vec/datafiles/0110_mtltest_100neighbor_inter_find_neighbor_perpts_min10pts_idxonly_thres20m_mergemin5.npy', allow_pickle=True)


all_append_list = []
for i in range(4):
    selected_x = np.take(test_x, list(total_wrong_dict[i]), axis=0)
    selected_y = np.take(test_y, list(total_wrong_dict[i]), axis=0)
    # selected_nbr  = np.take(nbr_idx_tuple, list(total_wrong_dict[i]), axis=0)
    all_append_list.append((selected_x,selected_y))

if also_correct:
    all_append_list_correct = []
    for i in range(4):
        selected_x = np.take(test_x, list(total_correct_dict[i]), axis=0)
        selected_y = np.take(test_y, list(total_correct_dict[i]), axis=0)
        # selected_nbr  = np.take(nbr_idx_tuple, list(total_correct_dict[i]), axis=0)
        all_append_list_correct.append((selected_x,selected_y))

# for nbr_idxes in selected_nbr:
#     nbrs_list = []
#     mask_list = []
#     nbrs_list_label = []
#     for nbr_idx_list in nbr_idxes:
#         if len(nbr_idx_list)==1:
#             nbr_idx = nbr_idx_list[0]
#             each_nbr,tmp_low,tmp_high,tmp_low_anchor,tmp_high_anchor,avg_pair_dist = nbr_idx
#             each_nbr,tmp_low,tmp_high,tmp_low_anchor,tmp_high_anchor = int(each_nbr),int(tmp_low),int(tmp_high),int(tmp_low_anchor),int(tmp_high_anchor)
#             trip_id = self.pos_trips[each_nbr][0]
#             tmp_trip = self.total_trips[trip_id]
#             nbr_seg = change_to_new_channel_v4(tmp_trip[tmp_low:tmp_high,:], 650, tmp_low_anchor, tmp_high_anchor)
#         else:
#             nbr_seg = np.zeros([650,10])
#             for nbr_idx in nbr_idx_list:
#                 each_nbr,tmp_low,tmp_high,tmp_low_anchor,tmp_high_anchor,avg_pair_dist = nbr_idx
#                 each_nbr,tmp_low,tmp_high,tmp_low_anchor,tmp_high_anchor = int(each_nbr),int(tmp_low),int(tmp_high),int(tmp_low_anchor),int(tmp_high_anchor)
#                 trip_id = self.pos_trips[each_nbr][0]
#                 tmp_trip = self.total_trips[trip_id]
#                 nbr_seg[tmp_low_anchor:tmp_high_anchor]=tmp_trip[tmp_low:tmp_high,:]
#             nbr_seg = change_to_new_channel_v5(nbr_seg, 650)
#         nbrs_list.append(nbr_seg)

if also_correct:
    with open('/home/yichen/Transfer-Learning-Library/examples/domain_adaptation/image_classification/logs/failcase_output/0331_wrong_correct_allclass.pickle', 'wb') as f:
        pickle.dump([all_append_list,all_append_list_correct], f)
        # [
        #     [(wrong0_data,wrong0_label),
        #      (wrong1_data,wrong1_label),
        #      (wrong2_data,wrong2_label),
        #      (wrong3_data,wrong3_label)],
        #     [(correct0_data,correct0_label),
        #      (correct1_data,correct1_label),
        #      (correct2_data,correct2_label),
        #      (correct3_data,correct3_label)],
        # ]
else:
    with open('/home/yichen/Transfer-Learning-Library/examples/domain_adaptation/image_classification/logs/failcase_output/0328_wrong_allclass.pickle', 'wb') as f:
        pickle.dump([all_append_list], f)