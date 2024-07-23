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

also_correct=False

filename_list=glob.glob('/home/yichen/Transfer-Learning-Library/examples/domain_adaptation/image_classification/logs/failcase0530/*')
print(filename_list)
total_wrong_dict={}
total_correct_dict={}

for idx,filename in enumerate(filename_list):
    wrong_dict = np.load(filename, allow_pickle=True)
    # pdb.set_trace()
    wrong_dict = dict(enumerate(wrong_dict.flatten(), 1))[1]
    # for k,v in wrong_dict.items():
    #     print(k,len(v))
    cnt=0
    if idx==0:
        for k in wrong_dict:
            for tmp_k in wrong_dict[k]:
                total_wrong_dict[cnt]=set(tmp_k)
                cnt+=1
    else:
        # for k in wrong_dict:
        #     total_wrong_dict[k]=set(wrong_dict[k]) & total_wrong_dict[k]
        for k in wrong_dict:
            for tmp_k in wrong_dict[k]:
                total_wrong_dict[cnt]=set(tmp_k) & total_wrong_dict[cnt]
                cnt+=1
     
if also_correct:
    filename_list=glob.glob('/home/yichen/Transfer-Learning-Library/examples/domain_adaptation/image_classification/logs/failcase_correct0530/*')       
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
# for i in range(4):
for i in range(16):
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
    raise NotImplemented
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
    with open('/home/yichen/Transfer-Learning-Library/examples/domain_adaptation/image_classification/logs/failcase_output/0530_wrong_allclass.pickle', 'wb') as f:
        pickle.dump([all_append_list], f)




lat_min,lat_max = (45.230416, 45.9997262293)
lon_min,lon_max = (-74.31479102, -72.81248199999999)

current_minmax = [
    (45.230416, 45.9997262293), (-74.31479102, -72.81248199999999),  \
    (0.9999933186918497, 1198.999998648651), # time
    (0.0, 50118.17550774085), # dist
    (0.0, 49.95356703911097), # speed
    (-9.99348698095659, 9.958323482935628), #acc
    (-39.64566646191948, 1433.3438889109589), #jerk
    (0.0, 359.95536847383516)] #bearing

# with open('/home/yichen/Transfer-Learning-Library/examples/domain_adaptation/image_classification/logs/failcase_output/0328_wrong_allclass.pickle', 'rb') as f:
#     all_append_list = pickle.load(f)

# all_append_list = all_append_list[0]

def get_each_pt(seg):
    tmp_pt_list=[]
    pt_list=[]
    # for pt in all_append_list[i][0][0]:
    for pt in seg:
        if pt[-1]==1:
            if len(tmp_pt_list)!=0:
                total_time = 0.
                for virt_pt in tmp_pt_list:
                    total_time += virt_pt[2]
                pt[2]+=total_time
            pt_list.append(pt)
            tmp_pt_list=[]
        else:
            tmp_pt_list.append(pt)
    tmp_seg = np.array(pt_list)[:,:3]
    tmp_seg[:,0] = tmp_seg[:,0] * (lat_max-lat_min) + lat_min
    tmp_seg[:,1] = tmp_seg[:,1] * (lon_max-lon_min) + lon_min
    # tmp_seg[:,2] = tmp_seg[:,2].astype(int)
    pad_mask = (tmp_seg[:,2] != 0).sum()
    tmp_seg = tmp_seg[:pad_mask]
    return tmp_seg

def get_each_pt_allfeat(seg):
    tmp_pt_list=[]
    pt_list=[]
    for pt in seg:
        if pt[-1]==1:
            if len(tmp_pt_list)!=0:
                total_time = 0.
                for virt_pt in tmp_pt_list:
                    total_time += virt_pt[2]
                pt[2]+=total_time
            pt_list.append(pt)
            tmp_pt_list=[]
        else:
            tmp_pt_list.append(pt)
    tmp_seg = np.array(pt_list)
    tmp_seg[:,0] = tmp_seg[:,0] * (lat_max-lat_min) + lat_min
    tmp_seg[:,1] = tmp_seg[:,1] * (lon_max-lon_min) + lon_min
    # tmp_seg[:,2] = tmp_seg[:,2].astype(int)
    
    for i in range(3,7):
        tmp_seg[:,i] = tmp_seg[:,i] * (current_minmax[i][1]-current_minmax[i][0]) + current_minmax[i][0]
    
    pad_mask = (tmp_seg[:,2] != 0).sum()
    tmp_seg = tmp_seg[:pad_mask]
    return tmp_seg

def get_each_pt_allfeat_woxy(seg):
    tmp_pt_list=[]
    pt_list=[]
    for pt in seg:
        if pt[-1]==1:
            if len(tmp_pt_list)!=0:
                total_time = 0.
                for virt_pt in tmp_pt_list:
                    total_time += virt_pt[2]
                pt[2]+=total_time
            pt_list.append(pt)
            tmp_pt_list=[]
        else:
            tmp_pt_list.append(pt)
    tmp_seg = np.array(pt_list)
    tmp_seg[:,0] = tmp_seg[:,0] * (lat_max-lat_min) + lat_min
    tmp_seg[:,1] = tmp_seg[:,1] * (lon_max-lon_min) + lon_min
    # tmp_seg[:,2] = tmp_seg[:,2].astype(int)
    
    for i in range(3,7):
        tmp_seg[:,i] = tmp_seg[:,i] * (current_minmax[i][1]-current_minmax[i][0]) + current_minmax[i][0]
    
    pad_mask = (tmp_seg[:,2] != 0).sum()
    tmp_seg = tmp_seg[:pad_mask]
    
    tmp_seg = tmp_seg[:,2:]
    
    return tmp_seg



print_list=[]
for i in range(16):
    if len(all_append_list[i][0])==0:
        print_list.append([[0,0,0,0,0,0,0,0,0]])
        print_list.append([[0,0,0,0,0,0,0,0,0]])
        continue
    # tmp_seg = get_each_pt(all_append_list[i][0][0])
    # print_list.append(tmp_seg)
    # tmp_seg = get_each_pt(all_append_list[i][0][1])
    # print_list.append(tmp_seg)
    
    tmp_seg = get_each_pt_allfeat(all_append_list[i][0][0])
    print_list.append(tmp_seg)
    tmp_seg = get_each_pt_allfeat(all_append_list[i][0][1])
    print_list.append(tmp_seg)
    
    # tmp_seg = get_each_pt_allfeat_woxy(all_append_list[i][0][0])
    # print_list.append(tmp_seg)
    # tmp_seg = get_each_pt_allfeat_woxy(all_append_list[i][0][1])
    # print_list.append(tmp_seg)

# with open('/home/yichen/Transfer-Learning-Library/examples/domain_adaptation/image_classification/logs/failcase_output/0530_wrongsample.txt', 'w') as f:
#     for idx,seg in enumerate(print_list):
#         f.write('%d:\n'%(idx//2+1))
#         for pt in seg:
#             f.write("(%f,%f,%.1f), "%(pt[0],pt[1],pt[2]))
#         f.write('\n')
#         f.write('\n')
    
with open('/home/yichen/Transfer-Learning-Library/examples/domain_adaptation/image_classification/logs/failcase_output/0530_wrongsample_allfeat.txt', 'w') as f:
    for idx,seg in enumerate(print_list):
        f.write('%d:\n'%(idx//2+1))
        for pt in seg:
            f.write("(%f,%f,%.1f,%f,%f,%f,%f,%f,%d), "%(pt[0],pt[1],pt[2],pt[3],pt[4],pt[5],pt[6],pt[7],pt[8]))
        f.write('\n')
        f.write('\n')


# with open('/home/yichen/Transfer-Learning-Library/examples/domain_adaptation/image_classification/logs/failcase_output/0530_wrongsample_allfeat_woxy.txt', 'w') as f:
#     for idx,seg in enumerate(print_list):
#         f.write('%d:\n'%(idx//2+1))
#         for pt in seg:
#             f.write("(%.1f,%f,%f,%f,%f,%f,%d), "%(pt[0],pt[1],pt[2],pt[3],pt[4],pt[5],pt[6]))
#         f.write('\n')
#         f.write('\n')
