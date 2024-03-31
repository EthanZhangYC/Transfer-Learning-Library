"""
@author: Junguang Jiang, Baixu Chen
@contact: JiangJunguang1123@outlook.com, cbx_99_hasta@outlook.com
"""
import sys
import os.path as osp
import time
# from PIL import Image
import numpy as np
import pickle
from torch.utils.data import TensorDataset, DataLoader


import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from timm.data.auto_augment import auto_augment_transform, rand_augment_transform

sys.path.append('../../..')
# import tllib.vision.datasets as datasets
import tllib.vision.models as models
from tllib.vision.transforms import ResizeImage
from tllib.utils.metric import accuracy, ConfusionMatrix
from tllib.utils.meter import AverageMeter, ProgressMeter
from tllib.vision.datasets.imagelist import MultipleDomainsDataset



from torch.utils.data import TensorDataset, DataLoader
import torchvision

import pdb

def get_label(single_dataset,idx,label_dict):
    label = single_dataset[idx][1].item()
    return label

class ImbalancedDatasetSampler(torch.utils.data.sampler.Sampler):
    """Samples elements randomly from a given list of indices for imbalanced dataset
    Arguments:
        indices (list, optional): a list of indices
        num_samples (int, optional): number of samples to draw
        callback_get_label func: a callback-like function which takes two arguments - dataset and index
    """

    def __init__(self, dataset, indices=None, num_samples=None, callback_get_label=None):
                
        # if indices is not provided, 
        # all elements in the dataset will be considered
        self.indices = list(range(len(dataset))) \
            if indices is None else indices

        # define custom callback
        self.callback_get_label = callback_get_label

        # if num_samples is not provided, 
        # draw `len(indices)` samples in each iteration
        self.num_samples = len(self.indices) \
            if num_samples is None else num_samples
        
        
        label_dict={'0':0,'1':0,'2':1,'3':1,'4':1}
        label_to_count = {}
        for idx in self.indices:
            label = self._get_label(dataset, idx, label_dict)
            if label in label_to_count:
                label_to_count[label] += 1
            else:
                label_to_count[label] = 1
        
        weights = [1.0 / label_to_count[self._get_label(dataset, idx, label_dict)] for idx in self.indices]
        self.weights = torch.DoubleTensor(weights)

    def _get_label(self, dataset, idx, label_dict):
        if self.callback_get_label:
            return self.callback_get_label(dataset, idx, label_dict)
        elif isinstance(dataset, torchvision.datasets.MNIST):
            return dataset.train_labels[idx].item()
        elif isinstance(dataset, torchvision.datasets.ImageFolder):
            return dataset.imgs[idx][1]
        elif isinstance(dataset, torch.utils.data.Subset):
            return dataset.dataset.imgs[idx][1]
        else:
            raise NotImplementedError

    # sample class balance training batch 
    def __iter__(self):
        return (self.indices[i] for i in torch.multinomial(self.weights, self.num_samples, replacement=True))

    def __len__(self):
        return self.num_samples

class ForeverDataIterator:
    r"""A data iterator that will never stop producing data"""

    def __init__(self, data_loader: DataLoader, device=None):
        self.data_loader = data_loader
        self.iter = iter(self.data_loader)
        self.device = device

    def __next__(self):
        try:
            data = next(self.iter)
        except StopIteration:
            self.iter = iter(self.data_loader)
            data = next(self.iter)
        return data

    def __len__(self):
        return len(self.data_loader)


def load_data(args):
    # filename = '/home/yichen/ts2vec/datafiles/Geolife/traindata_4class_xy_traintest_interpolatedLinear_5s_trip%d_new_001meters_withdist_aligninterpolation_InsertAfterSeg_Both_10dim_1115.pickle'%args.trip_time
    filename = '/home/yichen/TS2Vec/datafiles/Geolife/traindata_4class_xy_traintest_interpolatedNAN_5s_trip20_new_001meters_withdist_aligninterpolation_InsertAfterSeg_Both_11dim_doubletest_0218.pickle'

    with open(filename, 'rb') as f:
        kfold_dataset, X_unlabeled = pickle.load(f)
    dataset = kfold_dataset
    
    if args.interpolatedlinear:
        train_x = dataset[1].squeeze(1)
    elif args.interpolated:
        train_x = dataset[2].squeeze(1)
    else:
        train_x = dataset[0].squeeze(1)
        
    train_y = dataset[3]
    train_x = train_x[:,:,4:]   
    pad_mask_source = train_x[:,:,0]==0
    train_x[pad_mask_source] = 0.
        
    class_dict={}
    for y in train_y:
        if y not in class_dict:
            class_dict[y]=1
        else:
            class_dict[y]+=1
    print('Geolife:',dict(sorted(class_dict.items())))
    
    
    # # filename_mtl = '/home/xieyuan/Transportation-mode/TS2Vec/datafiles/Huawei/traindata_4class_xy_traintest_interpolatedLinear_trip%d_new_001meters.pickle'%args.trip_time
    # # filename_mtl = '/home/yichen/ts2vec/datafiles/MTL/traindata_4class_xy_traintest_interpolatedLinear_5s_trip%d_new_001meters_withdist_aligninterpolation_InsertAfterSeg_Both_10dim_1115.pickle'%args.trip_time
    # filename_mtl = '/home/yichen/TS2Vec/datafiles/MTL/traindata_4class_xy_traintest_interpolatedNAN_5s_trip20_new_001meters_withdist_aligninterpolation_InsertAfterSeg_Both_11dim_doubletest_0226_sharedminmax_balanced.pickle'
    filename_mtl = '/home/yichen/TS2Vec/datafiles/MTL/traindata_4class_xy_traintest_interpolatedLinear_5s_trip20_new_001meters_withdist_aligninterpolation_InsertAfterSeg_Both_11dim_0817_sharedminmax_balanced.pickle'
    print(filename_mtl)
    with open(filename_mtl, 'rb') as f:
        kfold_dataset, X_unlabeled_mtl = pickle.load(f)
    dataset_mtl = kfold_dataset
    
    if args.interpolatedlinear:
        train_x_mtl = dataset_mtl[1].squeeze(1)
        test_x = dataset_mtl[4].squeeze(1)
    elif args.interpolated:
        raise NotImplementedError
        train_x_mtl = dataset_mtl[2].squeeze(1)
        test_x = dataset_mtl[5].squeeze(1)
    else:
        train_x_mtl = dataset_mtl[0].squeeze(1)
        test_x = dataset_mtl[3].squeeze(1)
    train_y_mtl = dataset_mtl[2]
    test_y = dataset_mtl[5]
    
    train_x_mtl = train_x_mtl[:,:,4:]
    test_x = test_x[:,:,4:]
    
    pad_mask_target_train = train_x_mtl[:,:,0]==0
    pad_mask_target_test = test_x[:,:,0]==0
    train_x_mtl[pad_mask_target_train] = 0.
    test_x[pad_mask_target_test] = 0.
    
    class_dict={}
    for y in train_y_mtl:
        if y not in class_dict:
            class_dict[y]=1
        else:
            class_dict[y]+=1
    print('MTL train:',dict(sorted(class_dict.items())))
    class_dict={}
    for y in test_y:
        if y not in class_dict:
            class_dict[y]=1
        else:
            class_dict[y]+=1
    print('MTL test:',dict(sorted(class_dict.items())))

    print('Reading Data: (train: geolife + MTL, test: MTL)')
    # logger.info('Total shape: '+str(train_data.shape))
    print('GeoLife shape: '+str(train_x.shape))
    print('MTL shape: '+str(train_x_mtl.shape))
    
    n_geolife = train_x.shape[0]
    n_mtl = train_x_mtl.shape[0]
    train_dataset_geolife = TensorDataset(
        torch.from_numpy(train_x).to(torch.float),
        torch.from_numpy(train_y),
        torch.from_numpy(np.array([0]*n_geolife)).float()
    )
    train_dataset_mtl = TensorDataset(
        torch.from_numpy(train_x_mtl).to(torch.float),
        torch.from_numpy(train_y_mtl), # add label for debug
        torch.from_numpy(np.array([1]*n_mtl)).float(),
        torch.from_numpy(np.arange(n_mtl))
    )

    sampler = ImbalancedDatasetSampler(train_dataset_geolife, callback_get_label=get_label, num_samples=len(train_dataset_mtl))
    train_loader_source = DataLoader(train_dataset_geolife, batch_size=min(args.batch_size, len(train_dataset_geolife)), sampler=sampler, num_workers=8, shuffle=False, drop_last=True)
    # train_loader_target = DataLoader(train_dataset_mtl, batch_size=min(args.batch_size, len(train_dataset_mtl)), shuffle=True, drop_last=True)
    train_loader_target = DataLoader(train_dataset_mtl, batch_size=min(args.batch_size, len(train_dataset_mtl)), num_workers=8, shuffle=True, drop_last=False)
    train_source_iter = ForeverDataIterator(train_loader_source)
    train_tgt_iter = ForeverDataIterator(train_loader_target)
    train_loader = (train_source_iter, train_tgt_iter)
    
    test_dataset = TensorDataset(
        torch.from_numpy(test_x).to(torch.float),
        torch.from_numpy(test_y),
    )
    test_loader = DataLoader(test_dataset, batch_size=min(args.batch_size, len(test_dataset)))

    return train_source_iter, train_tgt_iter, test_loader, train_loader_target




def load_data_multitgt(args):
    # filename = '/home/yichen/ts2vec/datafiles/Geolife/traindata_4class_xy_traintest_interpolatedLinear_5s_trip%d_new_001meters_withdist_aligninterpolation_InsertAfterSeg_Both_10dim_1115.pickle'%args.trip_time
    filename = '/home/yichen/TS2Vec/datafiles/Geolife/traindata_4class_xy_traintest_interpolatedNAN_5s_trip20_new_001meters_withdist_aligninterpolation_InsertAfterSeg_Both_11dim_doubletest_0218.pickle'

    with open(filename, 'rb') as f:
        kfold_dataset, X_unlabeled = pickle.load(f)
    dataset = kfold_dataset
    
    if args.interpolatedlinear:
        train_x = dataset[1].squeeze(1)
    elif args.interpolated:
        train_x = dataset[2].squeeze(1)
    else:
        train_x = dataset[0].squeeze(1)
        
    train_y = dataset[3]
    train_x = train_x[:,:,4:]   
    pad_mask_source = train_x[:,:,0]==0
    train_x[pad_mask_source] = 0.
        
    class_dict={}
    for y in train_y:
        if y not in class_dict:
            class_dict[y]=1
        else:
            class_dict[y]+=1
    print('Geolife:',dict(sorted(class_dict.items())))
    
    
    # filename_mtl = '/home/xieyuan/Transportation-mode/TS2Vec/datafiles/Huawei/traindata_4class_xy_traintest_interpolatedLinear_trip%d_new_001meters.pickle'%args.trip_time
    # filename_mtl = '/home/yichen/ts2vec/datafiles/MTL/traindata_4class_xy_traintest_interpolatedLinear_5s_trip%d_new_001meters_withdist_aligninterpolation_InsertAfterSeg_Both_10dim_1115.pickle'%args.trip_time
    filename_mtl = '/home/yichen/TS2Vec/datafiles/MTL/traindata_4class_xy_traintest_interpolatedNAN_5s_trip20_new_001meters_withdist_aligninterpolation_InsertAfterSeg_Both_11dim_doubletest_0226_sharedminmax_balanced.pickle'
    
    with open(filename_mtl, 'rb') as f:
        kfold_dataset, X_unlabeled_mtl = pickle.load(f)
    dataset_mtl = kfold_dataset
    
    if args.interpolatedlinear:
        train_x_mtl = dataset_mtl[1].squeeze(1)
        test_x = dataset_mtl[5].squeeze(1)
    elif args.interpolated:
        train_x_mtl = dataset_mtl[2].squeeze(1)
        test_x = dataset_mtl[5].squeeze(1)
    else:
        train_x_mtl = dataset_mtl[0].squeeze(1)
        test_x = dataset_mtl[4].squeeze(1)

    train_y_mtl = dataset_mtl[3]
    test_y = dataset_mtl[7]
    
    train_x_mtl = train_x_mtl[:,:,4:]
    test_x = test_x[:,:,4:]
    
    pad_mask_target_train = train_x_mtl[:,:,0]==0
    pad_mask_target_test = test_x[:,:,0]==0
    train_x_mtl[pad_mask_target_train] = 0.
    test_x[pad_mask_target_test] = 0.
    
    class_dict={}
    for y in train_y_mtl:
        if y not in class_dict:
            class_dict[y]=1
        else:
            class_dict[y]+=1
    print('MTL train:',dict(sorted(class_dict.items())))
    class_dict={}
    for y in test_y:
        if y not in class_dict:
            class_dict[y]=1
        else:
            class_dict[y]+=1
    print('MTL test:',dict(sorted(class_dict.items())))

    print('Reading Data: (train: geolife + MTL, test: MTL)')
    # logger.info('Total shape: '+str(train_data.shape))
    print('GeoLife shape: '+str(train_x.shape))
    print('MTL shape: '+str(train_x_mtl.shape))
    
    n_geolife = train_x.shape[0]
    n_mtl = train_x_mtl.shape[0]
    train_dataset_geolife = TensorDataset(
        torch.from_numpy(train_x).to(torch.float),
        torch.from_numpy(train_y),
        torch.from_numpy(np.array([0]*n_geolife)).float()
    )
    train_dataset_mtl = TensorDataset(
        torch.from_numpy(train_x_mtl).to(torch.float),
        torch.from_numpy(np.array([1]*n_mtl)).float(),
        torch.from_numpy(np.arange(n_mtl))
    )



    sampler = ImbalancedDatasetSampler(train_dataset_geolife, callback_get_label=get_label, num_samples=len(train_dataset_mtl))
    train_loader_source = DataLoader(train_dataset_geolife, batch_size=min(args.batch_size, len(train_dataset_geolife)), sampler=sampler, shuffle=False, drop_last=True)
    train_loader_target = DataLoader(train_dataset_mtl, batch_size=min(args.batch_size, len(train_dataset_mtl)), shuffle=True, drop_last=True)
    train_source_iter = ForeverDataIterator(train_loader_source)
    train_tgt_iter = ForeverDataIterator(train_loader_target)
    train_loader = (train_source_iter, train_tgt_iter)
    
    test_dataset = TensorDataset(
        torch.from_numpy(test_x).to(torch.float),
        torch.from_numpy(test_y),
    )
    test_loader = DataLoader(test_dataset, batch_size=min(args.batch_size, len(test_dataset)))

    return train_source_iter, train_tgt_iter, test_loader 



def get_label_single(single_dataset,idx,label_dict):
    label = single_dataset.labels[idx].item()
    return label

class create_single_dataset(torch.utils.data.Dataset):
    def __init__(self, imgs, labels, neighbors, distances, neighbor_labels=None, part='train', transform=None, dataset=''):
        super(create_single_dataset, self).__init__()
        self.imgs = imgs.astype(np.float32)
        self.labels = labels
        self.neighbors = neighbors
        self.distances = distances
        self.neighbor_labels = neighbor_labels
        self.dataset=dataset
        self.part=part
        self.transform = transform

    def __getitem__(self, index):
        img = self.imgs[index]

        neighbors = self.neighbors[index]
        distances = self.distances[index]

        if neighbors is None:
            neighbors = img[np.newaxis,...]
            distances = np.array([0.])

        if self.transform is not None:
            img = self.transform(img)
        img = torch.as_tensor(img)
        neighbors = torch.as_tensor(neighbors.astype(np.float32))
        distances = torch.as_tensor(distances.astype(np.float32))

        if self.part=='test':
            label = torch.as_tensor(self.labels[index])
            return img, label, neighbors, distances
        elif self.dataset=='src':
            label = torch.as_tensor(self.labels[index])
            if self.neighbor_labels[index] is None:
                neighbors_label = label
            else:
                neighbors_label = torch.as_tensor(self.neighbor_labels[index])
            return img, label, neighbors, neighbors_label, distances, torch.as_tensor(0)
        else:
            return img, neighbors, distances, torch.as_tensor(1), torch.as_tensor(index)
    

    def __len__(self):
        return len(self.imgs)

def get_neighbor_idx_v3(pairdist_all, indices_all, n_neighbors, all_seg, all_seg_labels, train_data, lat_min, lat_max, lon_min,lon_max):
    print(id(train_data))
    pad_mask = train_data[:,:,0:1]!=0
    train_data = train_data[:,:,:2]
    train_data[:,:,0] = train_data[:,:,0] * (lat_max-lat_min) + lat_min
    train_data[:,:,1] = train_data[:,:,1] * (lon_max-lon_min) + lon_min
    train_data = np.where(pad_mask, train_data, -np.ones_like(train_data)*100)

    neighbor_list=[]
    dist_list=[]
    labels_list=[]
    for idx,pairdist in enumerate(pairdist_all):
        
        if pairdist is None:
            neighbor_list.append(None)
            dist_list.append(None)
            labels_list.append(None)
            continue
        
        tmp_n_neighbors = min(n_neighbors,len(pairdist))
        if len(pairdist)>tmp_n_neighbors:
            neighbor_idx = np.argpartition(np.array(pairdist), kth=tmp_n_neighbors)[:tmp_n_neighbors]
        else:
            neighbor_idx = np.arange(tmp_n_neighbors)
        # dist = np.take(pairdist, neighbor_idx)
        
        # neighbor_idx = np.take(indices_all[idx], neighbor_idx)
        neighbor = all_seg[idx][neighbor_idx]
        labels = all_seg_labels[idx][neighbor_idx]
        dist = np.linalg.norm(neighbor[:,:,:2]-train_data[idx], axis=-1)
        
        neighbor_list.append(neighbor)
        dist_list.append(dist)
        labels_list.append(labels)

    return neighbor_list, labels_list, dist_list

def load_data_neighbor(args): 
    args.n_neighbors=8
    
    filename = '/home/yichen/TS2Vec/datafiles/Geolife/traindata_4class_xy_traintest_interpolatedNAN_5s_trip%d_new_001meters_withdist_aligninterpolation_InsertAfterSeg_Both_11dim_doubletest_0218.pickle'%args.trip_time
    print('Geolife loading: ',filename)

    with open(filename, 'rb') as f:
        kfold_dataset, X_unlabeled = pickle.load(f)
    dataset = kfold_dataset
    
    if args.interpolatedlinear:
        train_x = dataset[1].squeeze(1)
    elif args.interpolated:
        train_x = dataset[2].squeeze(1)
    else:
        train_x = dataset[0].squeeze(1)
    train_y = dataset[3]
    
    train_x = train_x[:,:,2:]   
    pad_mask_source = train_x[:,:,2]==0
    train_x[pad_mask_source] = 0.
        
    class_dict={}
    for y in train_y:
        if y not in class_dict:
            class_dict[y]=1
        else:
            class_dict[y]+=1
    print('Geolife:',dict(sorted(class_dict.items())))



    pairdist, indices, labels_allseg, candidates, candidates_inter = np.load('/home/yichen/TS2Vec/datafiles/0628_geolifetrain_20neighbor_inter3_BT1pt1000_nopad3pteucldist.npy', allow_pickle=True)
    print(id(train_x))
    neighbors, neighbor_labels, distances = get_neighbor_idx_v3(pairdist, indices, args.n_neighbors, candidates_inter, labels_allseg, train_x[:], 18.249901, 55.975593, -122.3315333, 126.998528)
    train_dataset_src = create_single_dataset(train_x, train_y, neighbors, distances, neighbor_labels, dataset='src')


    filename_mtl = '/home/yichen/TS2Vec/datafiles/MTL/traindata_4class_xy_traintest_interpolatedNAN_5s_trip20_new_001meters_withdist_aligninterpolation_InsertAfterSeg_Both_11dim_doubletest_0226_sharedminmax_balanced.pickle'

    with open(filename_mtl, 'rb') as f:
        kfold_dataset, X_unlabeled_mtl = pickle.load(f)
    dataset_mtl = kfold_dataset
    
    if args.interpolatedlinear:
        train_x_mtl = dataset_mtl[1].squeeze(1)
        test_x = dataset_mtl[5].squeeze(1)
    elif args.interpolated:
        train_x_mtl = dataset_mtl[2].squeeze(1)
        test_x = dataset_mtl[5].squeeze(1)
    else:
        train_x_mtl = dataset_mtl[0].squeeze(1)
        test_x = dataset_mtl[4].squeeze(1)

    train_y_mtl = dataset_mtl[3]
    test_y = dataset_mtl[7]

    train_x_mtl = train_x_mtl[:,:,2:]
    test_x = test_x[:,:,2:]
    
    pad_mask_target_train = train_x_mtl[:,:,2]==0
    pad_mask_target_test = test_x[:,:,2]==0




    train_x_mtl[pad_mask_target_train] = 0.
    test_x[pad_mask_target_test] = 0.
    if args.interpolated:
        test_x_interpolated[pad_mask_target_test_internan] = 0.
    
    class_dict={}
    for y in train_y_mtl:
        if y not in class_dict:
            class_dict[y]=1
        else:
            class_dict[y]+=1
    print('MTL train:',dict(sorted(class_dict.items())))
    class_dict={}
    for y in test_y:
        if y not in class_dict:
            class_dict[y]=1
        else:
            class_dict[y]+=1
    print('MTL test:',dict(sorted(class_dict.items())))

    
    pairdist, indices, labels_allseg, candidates, candidates_inter = np.load('/home/yichen/TS2Vec/datafiles/0628_mtltrain_20neighbor_inter3_BT1pt1000_nopad3pteucldist.npy', allow_pickle=True)
    neighbors, neighbor_labels, distances = get_neighbor_idx_v3(pairdist, indices, args.n_neighbors, candidates_inter, labels_allseg, train_x_mtl[:], 45.230416, 45.9997262293, -74.31479102, -72.81248199999999)
    train_dataset_tgt = create_single_dataset(train_x_mtl, None, neighbors, distances, neighbor_labels, dataset='tgt')

    
    pairdist, indices, labels_allseg, candidates, candidates_inter = np.load('/home/yichen/TS2Vec/datafiles/0628_mtltest_20neighbor_inter3_BT1pt1000_nopad3pteucldist.npy', allow_pickle=True)
    neighbors, neighbor_labels, distances = get_neighbor_idx_v3(pairdist, indices, args.n_neighbors, candidates_inter, labels_allseg, test_x[:], 45.230416, 45.9997262293, -74.31479102, -72.81248199999999)
    test_dataset = create_single_dataset(test_x, test_y, neighbors, distances, neighbor_labels, part='test', dataset='tgt')

    del pairdist, indices, labels_allseg, candidates, candidates_inter, neighbors, neighbor_labels, distances
    

    print('Reading Data: (train: geolife + MTL, test: MTL)')
    print('GeoLife shape: '+str(train_x.shape))
    print('MTL shape: '+str(train_x_mtl.shape))

    def collate_fn(batch):
        return tuple(zip(*batch))

    sampler = ImbalancedDatasetSampler(train_dataset_src, callback_get_label=get_label_single, num_samples=len(train_dataset_tgt))
    train_loader_source = DataLoader(train_dataset_src, batch_size=min(args.batch_size, len(train_dataset_src)), sampler=sampler, shuffle=False, drop_last=True, collate_fn=collate_fn)
    train_loader_target = DataLoader(train_dataset_tgt, batch_size=min(args.batch_size, len(train_dataset_tgt)), shuffle=True, drop_last=True, collate_fn=collate_fn)

    train_source_iter = ForeverDataIterator(train_loader_source)
    train_tgt_iter = ForeverDataIterator(train_loader_target)
    test_loader = DataLoader(test_dataset, batch_size=min(args.batch_size, len(test_dataset)), collate_fn=collate_fn)
    
    return train_source_iter, train_tgt_iter, test_loader




# THRES=30
# N_NEIGHBOR_LIMIT=10

def change_to_new_channel_v3(inputs, max_threshold, label=None): 
    total_input_new = np.zeros((max_threshold, 9))
    trip_length_interpolated = inputs.shape[0]
    if trip_length_interpolated < max_threshold:
        inputs = np.pad(inputs, ((0, max_threshold - trip_length_interpolated),(0,0)), 'constant')
    else:
        inputs = inputs[:max_threshold]
    
    total_input_new[:,0]=inputs[:, 7]
    total_input_new[:,1]=inputs[:, 8]
    total_input_new[:,2]=inputs[:, 1]
    total_input_new[:,3]=inputs[:, 0]
    total_input_new[:,4]=inputs[:, 3]
    total_input_new[:,5]=inputs[:, 4]
    total_input_new[:,6]=inputs[:, 5]
    total_input_new[:,7]=inputs[:, 6]
    total_input_new[:,8]=inputs[:, 9]
    
    return total_input_new

def change_to_new_channel_v4(inputs, max_threshold, low_anchor, high_anchor): 
    total_input_new = np.zeros((max_threshold, 9))
    trip_length_interpolated = inputs.shape[0]
    assert high_anchor-low_anchor==trip_length_interpolated
    if trip_length_interpolated < max_threshold:
        inputs = np.pad(inputs, ((low_anchor, max_threshold - trip_length_interpolated - low_anchor),(0,0)), 'constant')
    else:
        inputs = inputs[:max_threshold]
    
    total_input_new[:,0]=inputs[:, 7]
    total_input_new[:,1]=inputs[:, 8]
    total_input_new[:,2]=inputs[:, 1]
    total_input_new[:,3]=inputs[:, 0]
    total_input_new[:,4]=inputs[:, 3]
    total_input_new[:,5]=inputs[:, 4]
    total_input_new[:,6]=inputs[:, 5]
    total_input_new[:,7]=inputs[:, 6]
    total_input_new[:,8]=inputs[:, 9]
    
    return total_input_new

class create_single_dataset_idx(torch.utils.data.Dataset):
    def __init__(self, imgs, labels, neighbors_idx, total_trips, pos_trips, part='train', transform=None, dataset='', label_mask=None, nbr_limit=10):
        super(create_single_dataset_idx, self).__init__()
        
        self.imgs = imgs.astype(np.float32)
        self.labels = labels
        self.neighbors_idx = neighbors_idx
        self.total_trips = total_trips
        self.pos_trips = pos_trips
        # self.neighbor_labels = neighbor_labels
        self.label_mask = label_mask
        self.nbr_limit=nbr_limit
        
        self.dataset=dataset
        self.part=part
        self.transform = transform

    def __getitem__(self, index):
        img = self.imgs[index]
        
        # nbr_idxes = self.neighbors_idx[index]
        nbr_idxes = self.neighbors_idx[index]
        # print(len(nbr_idxes))
        if len(nbr_idxes)>self.nbr_limit:
            each_len = nbr_idxes[:,2]-nbr_idxes[:,1]
            selected_idx = np.argpartition(-each_len, self.nbr_limit)[:self.nbr_limit]
            nbr_idxes = np.take(nbr_idxes,selected_idx,axis=0)
        
        nbrs_list = []
        mask_list = []
        for nbr_idx in nbr_idxes:
            each_nbr,tmp_low,tmp_high,tmp_low_anchor,tmp_high_anchor,avg_pair_dist = nbr_idx
            each_nbr,tmp_low,tmp_high,tmp_low_anchor,tmp_high_anchor = int(each_nbr),int(tmp_low),int(tmp_high),int(tmp_low_anchor),int(tmp_high_anchor)
            trip_id = self.pos_trips[each_nbr][0]
            tmp_trip = self.total_trips[trip_id]
        
            # nbr_seg = change_to_new_channel_v3(tmp_trip[tmp_low:tmp_high,:], 650, label=None)
            nbr_seg = change_to_new_channel_v4(tmp_trip[tmp_low:tmp_high,:], 650, tmp_low_anchor, tmp_high_anchor)
            nbrs_list.append(nbr_seg)
            
            # mask = np.zeros(650)
            # mask[tmp_low_anchor:tmp_high_anchor]=1
            # mask_list.append(mask)
            
            # # label = trips[trip_id][1]
            # label = trip_labels_list[trip_id]
            # label_list.append(label)


        # neighbors = self.neighbors[index]
        # distances = self.distances[index]
        
        if len(nbr_idxes)==0:
            neighbors = img[np.newaxis,...]
            # mask_list = torch.ones([1,650])
            # # distances = np.array([0.])
        else:
            neighbors = np.stack(nbrs_list)
            # mask_list = np.stack(mask_list)
            # # label_list=np.stack(label_list)
            # mask_list = torch.as_tensor(mask_list.astype(np.int))

        if self.transform is not None:
            img = self.transform(img)
            neighbors = self.transform(neighbors)
        img = torch.as_tensor(img)
        neighbors = torch.as_tensor(neighbors.astype(np.float32))
        # distances = torch.as_tensor(distances.astype(np.float32))

        if self.part=='test':
            label = torch.as_tensor(self.labels[index])
            return img, label, neighbors
        elif self.dataset=='src':
            label = torch.as_tensor(self.labels[index])
            # if self.neighbor_labels[index] is None:
            #     neighbors_label = label#.unsqueeze(0)
            # else:
            #     neighbors_label = torch.as_tensor(self.neighbor_labels[index])
            return img, label, neighbors, torch.as_tensor(0)
        else:
            if self.labels is not None:
                label = torch.as_tensor(self.labels[index])
                label_mask = torch.as_tensor(self.label_mask[index])
            else:
                label=label_mask=None
            return img, label, label_mask, neighbors, torch.as_tensor(1), torch.as_tensor(index)
    
    def __len__(self):
        return len(self.imgs)



def change_to_new_channel_v5(inputs, max_threshold): 
    total_input_new = np.zeros((max_threshold, 9))
    total_input_new[:,0]=inputs[:, 7]
    total_input_new[:,1]=inputs[:, 8]
    total_input_new[:,2]=inputs[:, 1]
    total_input_new[:,3]=inputs[:, 0]
    total_input_new[:,4]=inputs[:, 3]
    total_input_new[:,5]=inputs[:, 4]
    total_input_new[:,6]=inputs[:, 5]
    total_input_new[:,7]=inputs[:, 6]
    total_input_new[:,8]=inputs[:, 9]
    return total_input_new

class create_single_dataset_idx_v3(torch.utils.data.Dataset):
    def __init__(self, imgs, labels, neighbors_idx, total_trips, pos_trips, part='train', transform=None, dataset='', label_mask=None, nbr_limit=10, neighbor_labels=None, args=None):
        super(create_single_dataset_idx_v3, self).__init__()
        
        self.imgs = imgs.astype(np.float32)
        self.labels = labels
        self.neighbors_idx = neighbors_idx
        self.total_trips = total_trips
        self.pos_trips = pos_trips
        self.neighbor_labels = neighbor_labels
        self.label_mask = label_mask
        self.nbr_limit=nbr_limit
        
        self.dataset=dataset
        self.part=part
        self.transform = transform
        
        self.nbr_label_mode = args.nbr_label_mode

    def __getitem__(self, index):
        img = self.imgs[index]
        
        nbr_idxes = self.neighbors_idx[index]
        if len(nbr_idxes)>self.nbr_limit:
            each_len = np.array([np.sum(np.array(nbr_idx)[:,2]-np.array(nbr_idx)[:,1]) for nbr_idx in nbr_idxes])
            # each_len = nbr_idxes[:,2]-nbr_idxes[:,1]
            selected_idx = np.argpartition(-each_len, self.nbr_limit)[:self.nbr_limit]
            nbr_idxes = np.take(nbr_idxes,selected_idx,axis=0)
        
        nbrs_list = []
        mask_list = []
        nbrs_list_label = []
        for nbr_idx_list in nbr_idxes:
            if len(nbr_idx_list)==1:
                nbr_idx = nbr_idx_list[0]
                each_nbr,tmp_low,tmp_high,tmp_low_anchor,tmp_high_anchor,avg_pair_dist = nbr_idx
                each_nbr,tmp_low,tmp_high,tmp_low_anchor,tmp_high_anchor = int(each_nbr),int(tmp_low),int(tmp_high),int(tmp_low_anchor),int(tmp_high_anchor)
                trip_id = self.pos_trips[each_nbr][0]
                tmp_trip = self.total_trips[trip_id]
                nbr_seg = change_to_new_channel_v4(tmp_trip[tmp_low:tmp_high,:], 650, tmp_low_anchor, tmp_high_anchor)
            else:
                nbr_seg = np.zeros([650,10])
                for nbr_idx in nbr_idx_list:
                    each_nbr,tmp_low,tmp_high,tmp_low_anchor,tmp_high_anchor,avg_pair_dist = nbr_idx
                    each_nbr,tmp_low,tmp_high,tmp_low_anchor,tmp_high_anchor = int(each_nbr),int(tmp_low),int(tmp_high),int(tmp_low_anchor),int(tmp_high_anchor)
                    trip_id = self.pos_trips[each_nbr][0]
                    tmp_trip = self.total_trips[trip_id]
                    nbr_seg[tmp_low_anchor:tmp_high_anchor]=tmp_trip[tmp_low:tmp_high,:]
                nbr_seg = change_to_new_channel_v5(nbr_seg, 650)
                
            # # nbr_seg = change_to_new_channel_v3(tmp_trip[tmp_low:tmp_high,:], 650, label=None)
            # nbr_seg = change_to_new_channel_v4(tmp_trip[tmp_low:tmp_high,:], 650, tmp_low_anchor, tmp_high_anchor)
            if self.neighbor_labels is not None:
                nbr_label = self.neighbor_labels[each_nbr]
                nbrs_list_label.append(nbr_label)
            nbrs_list.append(nbr_seg)
            


        if len(nbr_idxes)==0:
            neighbors = img[np.newaxis,...]
            # mask_list = torch.ones([1,650])
            # # distances = np.array([0.])
        else:
            neighbors = np.stack(nbrs_list)
            # mask_list = np.stack(mask_list)
            # # label_list=np.stack(label_list)
            # mask_list = torch.as_tensor(mask_list.astype(np.int))


        if self.transform is not None:
            img = self.transform(img)
            neighbors = self.transform(neighbors)
        
        
        # if len(nbrs_list_label)>0:
        nbrs_list_label = np.array(nbrs_list_label)
        nbrs_labels = np.array([np.sum(nbrs_list_label==0),np.sum(nbrs_list_label==1),np.sum(nbrs_list_label==2),np.sum(nbrs_list_label==3)])
        nbrs_labels = nbrs_labels / (np.sum(nbrs_labels)+1)
        nbrs_labels = nbrs_labels.astype(np.float32)
        if self.nbr_label_mode == 'combine_each_pt':
            nbrs_labels = np.tile(nbrs_labels[np.newaxis,::], [650,1])
            img = np.concatenate([img, nbrs_labels], axis=1)
            img[img[:,2]==0] = 0.
        else:
            nbrs_labels = torch.as_tensor(nbrs_labels)
        # else:
        #     nbrs_labels = torch.ones([4])/4
        
        
        img = torch.as_tensor(img) #650,9
        neighbors = torch.as_tensor(neighbors.astype(np.float32))
        # distances = torch.as_tensor(distances.astype(np.float32))
        

        if self.part=='test':
            label = torch.as_tensor(self.labels[index])
            # if self.nbr_label_mode == 'separate_input':
            return img, label, neighbors, nbrs_labels
            # else:
            #     return img, label, neighbors
        elif self.dataset=='src':
            label = torch.as_tensor(self.labels[index])
            # if self.neighbor_labels[index] is None:
            #     neighbors_label = label#.unsqueeze(0)
            # else:
            #     neighbors_label = torch.as_tensor(self.neighbor_labels[index])
            
            # if self.nbr_label_mode == 'separate_input':
            return img, label, neighbors, nbrs_labels, torch.as_tensor(0)
            # else:
            #     return img, label, neighbors, torch.as_tensor(0)
        else:
            if self.labels is not None:
                label = torch.as_tensor(self.labels[index])
                label_mask = torch.as_tensor(self.label_mask[index])
            else:
                label=label_mask=None
                
            # if self.nbr_label_mode == 'separate_input':
            return img, label, label_mask, neighbors, nbrs_labels, torch.as_tensor(1), torch.as_tensor(index)
            # else:
            #     return img, label, label_mask, neighbors, torch.as_tensor(1), torch.as_tensor(index)
    

    def __len__(self):
        return len(self.imgs)
    
    
class create_single_dataset_idx_v4(torch.utils.data.Dataset):
    def __init__(self, imgs, labels, neighbors_idx, total_trips, pos_trips=None, part='train', transform=None, dataset='', label_mask=None, nbr_limit=10, nbrs=None):
        super(create_single_dataset_idx_v4, self).__init__()
        
        self.imgs = imgs.astype(np.float32)
        self.labels = labels
        self.neighbors_idx = neighbors_idx
        self.total_trips = total_trips
        self.pos_trips = pos_trips
        # self.neighbor_labels = neighbor_labels
        self.label_mask = label_mask
        self.nbr_limit=nbr_limit
        
        self.dataset=dataset
        self.part=part
        self.transform = transform
        self.nbrs = nbrs

    def __getitem__(self, index):
        img = self.imgs[index]
        
        nbr_idxes = self.neighbors_idx[index]
        if len(nbr_idxes)>self.nbr_limit:
            each_len = nbr_idxes[:,2]-nbr_idxes[:,1]
            selected_idx = np.argpartition(-each_len, self.nbr_limit)[:self.nbr_limit]
            nbr_idxes = np.take(nbr_idxes,selected_idx,axis=0)
        
        nbrs_list = []
        for nbr_tuple in nbr_idxes:
            each_nbr,mask = nbr_tuple
            if self.nbrs is None:
                nbr_seg = self.imgs[each_nbr] * mask[::,np.newaxis]
            else:
                nbr_seg = self.nbrs[each_nbr] * mask[::,np.newaxis]
            nbrs_list.append(nbr_seg)

        if len(nbr_idxes)==0:
            neighbors = img[np.newaxis,...]
        else:
            neighbors = np.stack(nbrs_list)

        if self.transform is not None:
            img = self.transform(img)
            neighbors = self.transform(neighbors)
        img = torch.as_tensor(img)
        neighbors = torch.as_tensor(neighbors.astype(np.float32))

        if self.part=='test':
            label = torch.as_tensor(self.labels[index])
            return img, label, neighbors
        elif self.dataset=='src':
            label = torch.as_tensor(self.labels[index])
            return img, label, neighbors, torch.as_tensor(0)
        else:
            if self.labels is not None:
                label = torch.as_tensor(self.labels[index])
                label_mask = torch.as_tensor(self.label_mask[index])
            else:
                label=label_mask=None
            return img, label, label_mask, neighbors, torch.as_tensor(1), torch.as_tensor(index)
    

    def __len__(self):
        return len(self.imgs)
    
    
    
def load_data_neighbor_v2(args, pseudo_labels=None, pseudo_labels_mask=None): 
    raise NotImplementedError
    
    THRES = args.nbr_dist_thres
    # nbr_limit=args.nbr_limit
    # print(THRES,args.nbr_limit)
    
    with open('/home/yichen/TS2Vec/datafiles/Geolife/traindata_4class_xy_traintest_interpolatedNAN_5s_trip20_new_001meters_withdist_aligninterpolation_InsertAfterSeg_Both_11dim_doubletest_0809_withposandtrip_withinterpos.pickle', 'rb') as f:
        _,_,_,_,_,pos_trips,total_trips = pickle.load(f)
    
    filename = '/home/yichen/TS2Vec/datafiles/Geolife/traindata_4class_xy_traintest_interpolatedNAN_5s_trip%d_new_001meters_withdist_aligninterpolation_InsertAfterSeg_Both_11dim_doubletest_0218.pickle'%args.trip_time
    with open(filename, 'rb') as f:
        kfold_dataset, X_unlabeled = pickle.load(f)
    dataset = kfold_dataset
    
    if args.interpolatedlinear:
        train_x = dataset[1].squeeze(1)
    elif args.interpolated:
        train_x = dataset[2].squeeze(1)
    else:
        train_x = dataset[0].squeeze(1)
    train_y = dataset[3]
    train_x = train_x[:,:,2:]   
    pad_mask_source = train_x[:,:,2]==0
    train_x[pad_mask_source] = 0.
        
    class_dict={}
    for y in train_y:
        if y not in class_dict:
            class_dict[y]=1
        else:
            class_dict[y]+=1
    print('Geolife:',dict(sorted(class_dict.items())))
    
    # nbr_idx_tuple = np.load('/home/yichen/TS2Vec/datafiles/0810_geolifetrain_20neighbor_inter_find_neighbor_perpts_new_min10pts_idxonly_thres%dm.npy'%THRES, allow_pickle=True)
    nbr_idx_tuple = np.load('/home/yichen/TS2Vec/datafiles/1128_geolifetrain_100neighbor_inter_find_neighbor_perpts_min10pts_idxonly_thres%dm_filternearest.npy'%THRES, allow_pickle=True)
    train_dataset_src = create_single_dataset_idx(train_x, train_y, nbr_idx_tuple, total_trips, pos_trips, dataset='src', nbr_limit=args.nbr_limit)
    
    


    with open('/home/yichen/TS2Vec/datafiles/MTL/traindata_4class_xy_traintest_interpolatedNAN_5s_trip20_new_001meters_withdist_aligninterpolation_InsertAfterSeg_Both_11dim_doubletest_0809_withposandtrip_withinterpos.pickle', 'rb') as f:
        _,_,_,_,_,pos_trips,total_trips = pickle.load(f)

    # filename_mtl = '/home/yichen/TS2Vec/datafiles/MTL/traindata_4class_xy_traintest_interpolatedNAN_5s_trip20_new_001meters_withdist_aligninterpolation_InsertAfterSeg_Both_11dim_doubletest_0226_sharedminmax_balanced.pickle'
    filename_mtl = '/home/yichen/TS2Vec/datafiles/MTL/traindata_4class_xy_traintest_interpolatedLinear_5s_trip20_new_001meters_withdist_aligninterpolation_InsertAfterSeg_Both_11dim_0817_sharedminmax_balanced.pickle'
    print(filename_mtl)
    with open(filename_mtl, 'rb') as f:
        kfold_dataset, X_unlabeled_mtl = pickle.load(f)
    dataset_mtl = kfold_dataset
    
    if args.interpolatedlinear:
        train_x_mtl = dataset_mtl[1].squeeze(1)
        test_x = dataset_mtl[4].squeeze(1)
    elif args.interpolated:
        raise NotImplementedError
        train_x_mtl = dataset_mtl[2].squeeze(1)
        test_x = dataset_mtl[5].squeeze(1)
    else:
        train_x_mtl = dataset_mtl[0].squeeze(1)
        test_x = dataset_mtl[3].squeeze(1)
    train_y_mtl = dataset_mtl[2]
    test_y = dataset_mtl[5]
    
    train_x_mtl = train_x_mtl[:,:,2:]
    test_x = test_x[:,:,2:]
    pad_mask_target_train = train_x_mtl[:,:,2]==0
    pad_mask_target_test = test_x[:,:,2]==0
    train_x_mtl[pad_mask_target_train] = 0.
    test_x[pad_mask_target_test] = 0.
    if args.interpolated:
        test_x_interpolated[pad_mask_target_test_internan] = 0.
    
    class_dict={}
    for y in train_y_mtl:
        if y not in class_dict:
            class_dict[y]=1
        else:
            class_dict[y]+=1
    print('MTL train:',dict(sorted(class_dict.items())))
    class_dict={}
    for y in test_y:
        if y not in class_dict:
            class_dict[y]=1
        else:
            class_dict[y]+=1
    print('MTL test:',dict(sorted(class_dict.items())))
    
    # nbr_idx_tuple = np.load('/home/yichen/TS2Vec/datafiles/0810_mtltrain_20neighbor_inter_find_neighbor_perpts_new_min10pts_idxonly_thres%dm.npy'%THRES, allow_pickle=True)
    nbr_idx_tuple = np.load('/home/yichen/TS2Vec/datafiles/1128_mtltrain_100neighbor_inter_find_neighbor_perpts_min10pts_idxonly_thres%dm_filternearest.npy'%THRES, allow_pickle=True)
    if pseudo_labels is not None:
        pseudo_labels = np.array(pseudo_labels)
        pseudo_labels_mask = np.array(pseudo_labels_mask)
    train_dataset_tgt = create_single_dataset_idx(train_x_mtl, pseudo_labels, nbr_idx_tuple, total_trips, pos_trips, dataset='tgt', label_mask=pseudo_labels_mask, nbr_limit=args.nbr_limit)
    
    # nbr_idx_tuple = np.load('/home/yichen/TS2Vec/datafiles/0810_mtltest_20neighbor_inter_find_neighbor_perpts_new_min10pts_idxonly_thres%dm.npy'%THRES, allow_pickle=True)
    nbr_idx_tuple = np.load('/home/yichen/TS2Vec/datafiles/1128_mtltest_100neighbor_inter_find_neighbor_perpts_min10pts_idxonly_thres%dm_filternearest.npy'%THRES, allow_pickle=True)
    test_dataset = create_single_dataset_idx(test_x, test_y, nbr_idx_tuple, total_trips, pos_trips, part='test', dataset='tgt', nbr_limit=args.nbr_limit)
    

    print('Reading Data: (train: geolife + MTL, test: MTL)')
    print('GeoLife shape: '+str(train_x.shape))
    print('MTL shape: '+str(train_x_mtl.shape))
    
    def collate_fn(batch):
        return tuple(zip(*batch))

    sampler = ImbalancedDatasetSampler(train_dataset_src, callback_get_label=get_label_single, num_samples=len(train_dataset_tgt))
    train_loader_source = DataLoader(train_dataset_src, batch_size=min(args.batch_size, len(train_dataset_src)), sampler=sampler, shuffle=False, drop_last=True, collate_fn=collate_fn)
    train_loader_target = DataLoader(train_dataset_tgt, batch_size=min(args.batch_size, len(train_dataset_tgt)), shuffle=True, drop_last=True, collate_fn=collate_fn)

    train_source_iter = ForeverDataIterator(train_loader_source)
    train_tgt_iter = ForeverDataIterator(train_loader_target)
    test_loader = DataLoader(test_dataset, batch_size=min(args.batch_size, len(test_dataset)), collate_fn=collate_fn)
    
    return train_source_iter, train_tgt_iter, test_loader


def load_data_neighbor_v3(args, pseudo_labels=None, pseudo_labels_mask=None, shuffle=True, all_pseudo_labels_nbr=None): 
    
    THRES = args.nbr_dist_thres
    # nbr_limit=args.nbr_limit
    # print(THRES,args.nbr_limit)
    
    with open('/home/yichen/TS2Vec/datafiles/Geolife/traindata_4class_xy_traintest_interpolatedNAN_5s_trip20_new_001meters_withdist_aligninterpolation_InsertAfterSeg_Both_11dim_doubletest_0809_withposandtrip_withinterpos.pickle', 'rb') as f:
        _,_,_,_,_,pos_trips,total_trips = pickle.load(f)
    
    filename = '/home/yichen/TS2Vec/datafiles/Geolife/traindata_4class_xy_traintest_interpolatedNAN_5s_trip%d_new_001meters_withdist_aligninterpolation_InsertAfterSeg_Both_11dim_doubletest_0218.pickle'%args.trip_time
    with open(filename, 'rb') as f:
        kfold_dataset, X_unlabeled = pickle.load(f)
    dataset = kfold_dataset
    
    if args.interpolatedlinear:
        train_x = dataset[1].squeeze(1)
    elif args.interpolated:
        train_x = dataset[2].squeeze(1)
    else:
        train_x = dataset[0].squeeze(1)
    train_y = dataset[3]
    train_x = train_x[:,:,2:]   
    pad_mask_source = train_x[:,:,2]==0
    train_x[pad_mask_source] = 0.
        
    class_dict={}
    for y in train_y:
        if y not in class_dict:
            class_dict[y]=1
        else:
            class_dict[y]+=1
    print('Geolife:',dict(sorted(class_dict.items())))
    
    # # # nbr_idx_tuple = np.load('/home/yichen/TS2Vec/datafiles/0810_geolifetrain_20neighbor_inter_find_neighbor_perpts_new_min10pts_idxonly_thres%dm.npy'%THRES, allow_pickle=True)
    # # nbr_idx_tuple = np.load('/home/yichen/TS2Vec/datafiles/1128_geolifetrain_100neighbor_inter_find_neighbor_perpts_min10pts_idxonly_thres%dm_filternearest.npy'%THRES, allow_pickle=True)
    # # nbr_idx_tuple = np.load('/home/yichen/TS2Vec/datafiles/1210_geolifetrain_100neighbor_inter_find_neighbor_100pts_min10pts_idxonly_thres%dm_merge.npy'%THRES, allow_pickle=True)
    # nbr_idx_tuple = np.load('/home/yichen/TS2Vec/datafiles/1220_geolifetrain_100neighbor_inter_find_neighbor_perpts_min10pts_idxonly_thres%dm_merge.npy'%THRES, allow_pickle=True)
    if args.nbr_data_mode == 'mergemin5':
        nbr_idx_tuple = np.load('/home/yichen/TS2Vec/datafiles/0110_geolifetrain_100neighbor_inter_find_neighbor_perpts_min10pts_idxonly_thres%dm_mergemin5.npy'%THRES, allow_pickle=True)
        train_dataset_src = create_single_dataset_idx_v3(train_x, train_y, nbr_idx_tuple, total_trips, pos_trips, dataset='src', nbr_limit=args.nbr_limit, neighbor_labels=train_y, args=args)
    elif args.nbr_data_mode == 'mergenomin':
        nbr_idx_tuple = np.load('/home/yichen/TS2Vec/datafiles/0315_geolifetrain_100neighbor_inter_find_neighbor_perpts_min10pts_idxonly_thres%dm_mergenomin.npy'%THRES, allow_pickle=True)
        train_dataset_src = create_single_dataset_idx_v3(train_x, train_y, nbr_idx_tuple, total_trips, pos_trips, dataset='src', nbr_limit=args.nbr_limit, neighbor_labels=train_y, args=args)
    elif args.nbr_data_mode == 'mergetoori':
        nbr_idx_tuple = np.load('/home/yichen/TS2Vec/datafiles/0124_geolifetrain_100neighbor_inter_find_neighbor_perpts_idxonly_thres%dm_mergetoori_min50.npy'%THRES, allow_pickle=True)
        train_dataset_src = create_single_dataset_idx_v4(train_x, train_y, nbr_idx_tuple, total_trips, pos_trips, dataset='src', nbr_limit=args.nbr_limit, neighbor_labels=train_y, args=args)
    else:
        raise NotImplementedError
    
    


    with open('/home/yichen/TS2Vec/datafiles/MTL/traindata_4class_xy_traintest_interpolatedNAN_5s_trip20_new_001meters_withdist_aligninterpolation_InsertAfterSeg_Both_11dim_doubletest_0809_withposandtrip_withinterpos.pickle', 'rb') as f:
        _,_,_,_,_,pos_trips,total_trips = pickle.load(f)

    # filename_mtl = '/home/yichen/TS2Vec/datafiles/MTL/traindata_4class_xy_traintest_interpolatedNAN_5s_trip20_new_001meters_withdist_aligninterpolation_InsertAfterSeg_Both_11dim_doubletest_0226_sharedminmax_balanced.pickle'
    filename_mtl = '/home/yichen/TS2Vec/datafiles/MTL/traindata_4class_xy_traintest_interpolatedLinear_5s_trip20_new_001meters_withdist_aligninterpolation_InsertAfterSeg_Both_11dim_0817_sharedminmax_balanced.pickle'
    print(filename_mtl)
    with open(filename_mtl, 'rb') as f:
        kfold_dataset, X_unlabeled_mtl = pickle.load(f)
    dataset_mtl = kfold_dataset
    
    if args.interpolatedlinear:
        train_x_mtl = dataset_mtl[1].squeeze(1)
        test_x = dataset_mtl[4].squeeze(1)
    elif args.interpolated:
        raise NotImplementedError
        train_x_mtl = dataset_mtl[2].squeeze(1)
        test_x = dataset_mtl[5].squeeze(1)
    else:
        train_x_mtl = dataset_mtl[0].squeeze(1)
        test_x = dataset_mtl[3].squeeze(1)
    train_y_mtl = dataset_mtl[2]
    test_y = dataset_mtl[5]
    
    train_x_mtl = train_x_mtl[:,:,2:]
    test_x = test_x[:,:,2:]
    pad_mask_target_train = train_x_mtl[:,:,2]==0
    pad_mask_target_test = test_x[:,:,2]==0
    train_x_mtl[pad_mask_target_train] = 0.
    test_x[pad_mask_target_test] = 0.
    if args.interpolated:
        test_x_interpolated[pad_mask_target_test_internan] = 0.
    
    class_dict={}
    for y in train_y_mtl:
        if y not in class_dict:
            class_dict[y]=1
        else:
            class_dict[y]+=1
    print('MTL train:',dict(sorted(class_dict.items())))
    class_dict={}
    for y in test_y:
        if y not in class_dict:
            class_dict[y]=1
        else:
            class_dict[y]+=1
    print('MTL test:',dict(sorted(class_dict.items())))
    
    # # # nbr_idx_tuple = np.load('/home/yichen/TS2Vec/datafiles/0810_mtltrain_20neighbor_inter_find_neighbor_perpts_new_min10pts_idxonly_thres%dm.npy'%THRES, allow_pickle=True)
    # # nbr_idx_tuple = np.load('/home/yichen/TS2Vec/datafiles/1128_mtltrain_100neighbor_inter_find_neighbor_perpts_min10pts_idxonly_thres%dm_filternearest.npy'%THRES, allow_pickle=True)
    # # nbr_idx_tuple = np.load('/home/yichen/TS2Vec/datafiles/1210_mtltrain_100neighbor_inter_find_neighbor_100pts_min10pts_idxonly_thres%dm_merge.npy'%THRES, allow_pickle=True)
    # nbr_idx_tuple = np.load('/home/yichen/TS2Vec/datafiles/1220_mtltrain_100neighbor_inter_find_neighbor_perpts_min10pts_idxonly_thres%dm_merge.npy'%THRES, allow_pickle=True)
    if pseudo_labels is not None:
        pseudo_labels = np.array(pseudo_labels)
        pseudo_labels_mask = np.array(pseudo_labels_mask)
    if args.nbr_data_mode == 'mergemin5':
        nbr_idx_tuple = np.load('/home/yichen/TS2Vec/datafiles/0110_mtltrain_100neighbor_inter_find_neighbor_perpts_min10pts_idxonly_thres%dm_mergemin5.npy'%THRES, allow_pickle=True)
        train_dataset_tgt = create_single_dataset_idx_v3(train_x_mtl, pseudo_labels, nbr_idx_tuple, total_trips, pos_trips, dataset='tgt', label_mask=pseudo_labels_mask, nbr_limit=args.nbr_limit, neighbor_labels=pseudo_labels, args=args)
    elif args.nbr_data_mode == 'mergenomin':
        nbr_idx_tuple = np.load('/home/yichen/TS2Vec/datafiles/0315_mtltrain_100neighbor_inter_find_neighbor_perpts_min10pts_idxonly_thres%dm_mergenomin.npy'%THRES, allow_pickle=True)
        train_dataset_tgt = create_single_dataset_idx_v3(train_x_mtl, pseudo_labels, nbr_idx_tuple, total_trips, pos_trips, dataset='tgt', label_mask=pseudo_labels_mask, nbr_limit=args.nbr_limit, neighbor_labels=pseudo_labels, args=args)
    elif args.nbr_data_mode == 'mergetoori':
        nbr_idx_tuple = np.load('/home/yichen/TS2Vec/datafiles/0124_mtltrain_100neighbor_inter_find_neighbor_perpts_idxonly_thres%dm_mergetoori_min50.npy'%THRES, allow_pickle=True)
        train_dataset_tgt = create_single_dataset_idx_v4(train_x_mtl, pseudo_labels, nbr_idx_tuple, total_trips, pos_trips, dataset='tgt', label_mask=pseudo_labels_mask, nbr_limit=args.nbr_limit, neighbor_labels=pseudo_labels, args=args)
    else:
        raise NotImplementedError
    # train_dataset_tgt = create_single_dataset_idx_v3(train_x_mtl, pseudo_labels, nbr_idx_tuple, total_trips, pos_trips, dataset='tgt', label_mask=pseudo_labels_mask, nbr_limit=args.nbr_limit)
    
    
    # # # nbr_idx_tuple = np.load('/home/yichen/TS2Vec/datafiles/0810_mtltest_20neighbor_inter_find_neighbor_perpts_new_min10pts_idxonly_thres%dm.npy'%THRES, allow_pickle=True)
    # # nbr_idx_tuple = np.load('/home/yichen/TS2Vec/datafiles/1128_mtltest_100neighbor_inter_find_neighbor_perpts_min10pts_idxonly_thres%dm_filternearest.npy'%THRES, allow_pickle=True)
    # # nbr_idx_tuple = np.load('/home/yichen/TS2Vec/datafiles/1210_mtltest_100neighbor_inter_find_neighbor_100pts_min10pts_idxonly_thres%dm_merge.npy'%THRES, allow_pickle=True)
    # nbr_idx_tuple = np.load('/home/yichen/TS2Vec/datafiles/1220_mtltest_100neighbor_inter_find_neighbor_perpts_min10pts_idxonly_thres%dm_merge.npy'%THRES, allow_pickle=True)
    if args.nbr_data_mode == 'mergemin5':
        nbr_idx_tuple = np.load('/home/yichen/TS2Vec/datafiles/0110_mtltest_100neighbor_inter_find_neighbor_perpts_min10pts_idxonly_thres%dm_mergemin5.npy'%THRES, allow_pickle=True)
        test_dataset = create_single_dataset_idx_v3(test_x, test_y, nbr_idx_tuple, total_trips, pos_trips, part='test', dataset='tgt', nbr_limit=args.nbr_limit, neighbor_labels=pseudo_labels, args=args)
    elif args.nbr_data_mode == 'mergenomin':
        nbr_idx_tuple = np.load('/home/yichen/TS2Vec/datafiles/0315_mtltest_100neighbor_inter_find_neighbor_perpts_min10pts_idxonly_thres%dm_mergenomin.npy'%THRES, allow_pickle=True)
        test_dataset = create_single_dataset_idx_v3(test_x, test_y, nbr_idx_tuple, total_trips, pos_trips, part='test', dataset='tgt', nbr_limit=args.nbr_limit, neighbor_labels=pseudo_labels, args=args)
    elif args.nbr_data_mode == 'mergetoori':
        nbr_idx_tuple = np.load('/home/yichen/TS2Vec/datafiles/0124_mtltest_100neighbor_inter_find_neighbor_perpts_idxonly_thres%dm_mergetoori_min50.npy'%THRES, allow_pickle=True)
        test_dataset = create_single_dataset_idx_v4(test_x, test_y, nbr_idx_tuple, total_trips, pos_trips, part='test', dataset='tgt', nbr_limit=args.nbr_limit, nbrs=train_x_mtl, neighbor_labels=pseudo_labels, args=args)
    else:
        raise NotImplementedError
    # test_dataset = create_single_dataset_idx_v3(test_x, test_y, nbr_idx_tuple, total_trips, pos_trips, part='test', dataset='tgt', nbr_limit=args.nbr_limit)
    

    print('Reading Data: (train: geolife + MTL, test: MTL)')
    print('GeoLife shape: '+str(train_x.shape))
    print('MTL shape: '+str(train_x_mtl.shape))
    
    def collate_fn(batch):
        return tuple(zip(*batch))

    if shuffle:
        sampler = ImbalancedDatasetSampler(train_dataset_src, callback_get_label=get_label_single, num_samples=len(train_dataset_tgt))
    else:
        sampler=None
        
    train_loader_source = DataLoader(train_dataset_src, batch_size=min(args.batch_size, len(train_dataset_src)), sampler=sampler, shuffle=False, drop_last=True, collate_fn=collate_fn, num_workers=0)
    train_loader_target = DataLoader(train_dataset_tgt, batch_size=min(args.batch_size, len(train_dataset_tgt)), shuffle=shuffle, drop_last=True, collate_fn=collate_fn, num_workers=0)

    train_source_iter = ForeverDataIterator(train_loader_source)
    train_tgt_iter = ForeverDataIterator(train_loader_target)
    test_loader = DataLoader(test_dataset, batch_size=min(args.batch_size, len(test_dataset)), collate_fn=collate_fn, num_workers=0, shuffle=False)
    
    return train_source_iter, train_tgt_iter, test_loader, train_loader_source, train_loader_target




def validate(val_loader, model, args, device) -> float:
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    
    # per class acc
    label_dict = {"walk": 0, "bike": 1, "car": 2, "bus": 3}
    idx_dict={}
    for k,v in label_dict.items():
        idx_dict[v]=k
    nb_classes = 4
    confusion_matrix = torch.zeros(nb_classes, nb_classes)  

    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()
    if args.per_class_eval:
        confmat = ConfusionMatrix(len(args.class_names))
    else:
        confmat = None

    with torch.no_grad():
        end = time.time()
        for i, data in enumerate(val_loader):
            images, target = data[:2]
            images = images.to(device)
            target = target.to(device)

            # compute output
            output,_,_,_,_ = model(images)
            loss = F.cross_entropy(output, target)

            # measure accuracy and record loss
            acc1, = accuracy(output, target, topk=(1,))
            if confmat:
                confmat.update(target, output.argmax(1))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1.item(), images.size(0))

            # per calss acc
            _, preds = torch.max(output, 1)
            for t, p in zip(target.view(-1), preds.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)
        
        # per class acc
        per_class_acc = list((confusion_matrix.diag()/confusion_matrix.sum(1)).numpy())
        print('per class accuracy:')
        for idx,acc in enumerate(per_class_acc):
            print('\t '+str(idx_dict[idx])+': '+str(acc))

        print(' * Acc@1 {top1.avg:.3f}'.format(top1=top1))
        if confmat:
            print(confmat.format(args.class_names))

    return top1.avg




def empirical_risk_minimization(train_source_iter, model, optimizer, lr_scheduler, epoch, args, device):
    batch_time = AverageMeter('Time', ':3.1f')
    data_time = AverageMeter('Data', ':3.1f')
    losses = AverageMeter('Loss', ':3.2f')
    cls_accs = AverageMeter('Cls Acc', ':3.1f')

    progress = ProgressMeter(
        args.iters_per_epoch,
        [batch_time, data_time, losses, cls_accs],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i in range(args.iters_per_epoch):
        # x_s, labels_s = next(train_source_iter)[:2]
        x_s,labels_s,_ = next(train_source_iter)
        x_s = x_s.to(device)
        labels_s = labels_s.to(device)

        # measure data loading time
        data_time.update(time.time() - end)

        # compute output
        y_s,_,_,_ = model(x_s)

        cls_loss = F.cross_entropy(y_s, labels_s)
        loss = cls_loss

        cls_acc = accuracy(y_s, labels_s)[0]

        losses.update(loss.item(), x_s.size(0))
        cls_accs.update(cls_acc.item(), x_s.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # lr_scheduler.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)




class create_single_dataset_idx_singlepad(torch.utils.data.Dataset):
    def __init__(self, imgs, labels, neighbors_idx, total_trips, pos_trips, part='train', transform=None, dataset='', label_mask=None, nbr_limit=10):
        super(create_single_dataset_idx_singlepad, self).__init__()
        
        self.imgs = imgs.astype(np.float32)
        self.labels = labels
        self.neighbors_idx = neighbors_idx
        self.total_trips = total_trips
        self.pos_trips = pos_trips
        # self.neighbor_labels = neighbor_labels
        self.label_mask = label_mask
        self.nbr_limit=nbr_limit
        
        self.dataset=dataset
        self.part=part
        
        # if self.part=='test':
        #     self.imgs_all = np.concatenate([other_imgs.astype(np.float32),self.imgs],axis=0)
        # else:
        #     self.imgs_all = self.imgs
            
        # if part=='train':          
        #     if length:
        #         self.imgs=random.sample(self.imgs,length)
        #     else:
        #         random.shuffle(self.imgs)

        self.transform = transform

    def __getitem__(self, index):
        img = self.imgs[index]
        
        # nbr_idxes = self.neighbors_idx[index]
        nbr_idxes = self.neighbors_idx[index]
        if len(nbr_idxes)>self.nbr_limit:
            each_len = nbr_idxes[:,2]-nbr_idxes[:,1]
            selected_idx = np.argpartition(-each_len, self.nbr_limit)[:self.nbr_limit]
            nbr_idxes = np.take(nbr_idxes,selected_idx,axis=0)
        
        nbrs_list = []
        mask_list = []
        for nbr_idx in nbr_idxes:
            each_nbr,tmp_low,tmp_high,tmp_low_anchor,tmp_high_anchor = nbr_idx
            trip_id = self.pos_trips[each_nbr][0]
            tmp_trip = self.total_trips[trip_id]
        
            nbr_seg = change_to_new_channel_v3(tmp_trip[tmp_low:tmp_high,:], 650, label=None)
            # nbr_seg = change_to_new_channel_v4(tmp_trip[tmp_low:tmp_high,:], 650, tmp_low_anchor, tmp_high_anchor)
            nbrs_list.append(nbr_seg)
            
            # mask = np.zeros(650)
            # mask[tmp_low_anchor:tmp_high_anchor]=1
            # mask_list.append(mask)

        
        if len(nbr_idxes)==0:
            neighbors = img[np.newaxis,...]
            # mask_list = torch.ones([1,650])
        else:
            neighbors = np.stack(nbrs_list)
            # mask_list = np.stack(mask_list)

        if self.transform is not None:
            img = self.transform(img)
            neighbors = self.transform(neighbors)
        img = torch.as_tensor(img)
        neighbors = torch.as_tensor(neighbors.astype(np.float32))

        if self.part=='test':
            label = torch.as_tensor(self.labels[index])
            return img, label, neighbors
        elif self.dataset=='src':
            label = torch.as_tensor(self.labels[index])
            return img, label, neighbors, torch.as_tensor(0)
        else:
            label = torch.as_tensor(self.labels[index])
            label_mask = torch.as_tensor(self.label_mask[index])
            return img, label, label_mask, neighbors, torch.as_tensor(1), torch.as_tensor(index)
    

    def __len__(self):
        return len(self.imgs)
    

def load_data_neighbor_v2_singlepad(args, pseudo_labels=None, pseudo_labels_mask=None): 
    
    THRES = args.nbr_dist_thres
    # nbr_limit=args.nbr_limit
    # print(THRES,args.nbr_limit)
    
    with open('/home/yichen/TS2Vec/datafiles/Geolife/traindata_4class_xy_traintest_interpolatedNAN_5s_trip20_new_001meters_withdist_aligninterpolation_InsertAfterSeg_Both_11dim_doubletest_0809_withposandtrip_withinterpos.pickle', 'rb') as f:
        _,_,_,_,_,pos_trips,total_trips = pickle.load(f)
    
    filename = '/home/yichen/TS2Vec/datafiles/Geolife/traindata_4class_xy_traintest_interpolatedNAN_5s_trip%d_new_001meters_withdist_aligninterpolation_InsertAfterSeg_Both_11dim_doubletest_0218.pickle'%args.trip_time
    with open(filename, 'rb') as f:
        kfold_dataset, X_unlabeled = pickle.load(f)
    dataset = kfold_dataset
    
    if args.interpolatedlinear:
        train_x = dataset[1].squeeze(1)
    elif args.interpolated:
        train_x = dataset[2].squeeze(1)
    else:
        train_x = dataset[0].squeeze(1)
    train_y = dataset[3]
    train_x = train_x[:,:,2:]   
    pad_mask_source = train_x[:,:,2]==0
    train_x[pad_mask_source] = 0.
        
    class_dict={}
    for y in train_y:
        if y not in class_dict:
            class_dict[y]=1
        else:
            class_dict[y]+=1
    print('Geolife:',dict(sorted(class_dict.items())))
    
    # nbr_idx_tuple = np.load('/home/yichen/TS2Vec/datafiles/0810_geolifetrain_20neighbor_inter_find_neighbor_perpts_new_min10pts_idxonly_thres%dm.npy'%THRES, allow_pickle=True)
    nbr_idx_tuple = np.load('/home/yichen/TS2Vec/datafiles/1126_geolifetrain_100neighbor_inter_find_neighbor_3pts_min10pts_idxonly_thres%dm_filternearest.npy'%THRES, allow_pickle=True)
    train_dataset_src = create_single_dataset_idx_singlepad(train_x, train_y, nbr_idx_tuple, total_trips, pos_trips, dataset='src', nbr_limit=args.nbr_limit)
    
    


    with open('/home/yichen/TS2Vec/datafiles/MTL/traindata_4class_xy_traintest_interpolatedNAN_5s_trip20_new_001meters_withdist_aligninterpolation_InsertAfterSeg_Both_11dim_doubletest_0809_withposandtrip_withinterpos.pickle', 'rb') as f:
        _,_,_,_,_,pos_trips,total_trips = pickle.load(f)

    filename_mtl = '/home/yichen/TS2Vec/datafiles/MTL/traindata_4class_xy_traintest_interpolatedNAN_5s_trip20_new_001meters_withdist_aligninterpolation_InsertAfterSeg_Both_11dim_doubletest_0226_sharedminmax_balanced.pickle'
    with open(filename_mtl, 'rb') as f:
        kfold_dataset, X_unlabeled_mtl = pickle.load(f)
    dataset_mtl = kfold_dataset
    
    if args.interpolatedlinear:
        train_x_mtl = dataset_mtl[1].squeeze(1)
        test_x = dataset_mtl[5].squeeze(1)
    elif args.interpolated:
        train_x_mtl = dataset_mtl[2].squeeze(1)
        test_x = dataset_mtl[5].squeeze(1)
    else:
        train_x_mtl = dataset_mtl[0].squeeze(1)
        test_x = dataset_mtl[4].squeeze(1)
    train_y_mtl = dataset_mtl[3]
    test_y = dataset_mtl[7]
    train_x_mtl = train_x_mtl[:,:,2:]
    test_x = test_x[:,:,2:]
    pad_mask_target_train = train_x_mtl[:,:,2]==0
    pad_mask_target_test = test_x[:,:,2]==0
    train_x_mtl[pad_mask_target_train] = 0.
    test_x[pad_mask_target_test] = 0.
    if args.interpolated:
        test_x_interpolated[pad_mask_target_test_internan] = 0.
    
    class_dict={}
    for y in train_y_mtl:
        if y not in class_dict:
            class_dict[y]=1
        else:
            class_dict[y]+=1
    print('MTL train:',dict(sorted(class_dict.items())))
    class_dict={}
    for y in test_y:
        if y not in class_dict:
            class_dict[y]=1
        else:
            class_dict[y]+=1
    print('MTL test:',dict(sorted(class_dict.items())))
    
    # nbr_idx_tuple = np.load('/home/yichen/TS2Vec/datafiles/0810_mtltrain_20neighbor_inter_find_neighbor_perpts_new_min10pts_idxonly_thres%dm.npy'%THRES, allow_pickle=True)
    nbr_idx_tuple = np.load('/home/yichen/TS2Vec/datafiles/1126_mtltrain_100neighbor_inter_find_neighbor_3pts_min10pts_idxonly_thres%dm_filternearest.npy'%THRES, allow_pickle=True)
    if pseudo_labels is not None:
        pseudo_labels = np.array(pseudo_labels)
        pseudo_labels_mask = np.array(pseudo_labels_mask)
    train_dataset_tgt = create_single_dataset_idx_singlepad(train_x_mtl, pseudo_labels, nbr_idx_tuple, total_trips, pos_trips, dataset='tgt', label_mask=pseudo_labels_mask, nbr_limit=args.nbr_limit)
    
    # nbr_idx_tuple = np.load('/home/yichen/TS2Vec/datafiles/0810_mtltest_20neighbor_inter_find_neighbor_perpts_new_min10pts_idxonly_thres%dm.npy'%THRES, allow_pickle=True)
    nbr_idx_tuple = np.load('/home/yichen/TS2Vec/datafiles/1126_mtltest_100neighbor_inter_find_neighbor_3pts_min10pts_idxonly_thres%dm_filternearest.npy'%THRES, allow_pickle=True)
    test_dataset = create_single_dataset_idx_singlepad(test_x, test_y, nbr_idx_tuple, total_trips, pos_trips, part='test', dataset='tgt', nbr_limit=args.nbr_limit)
    

    print('Reading Data: (train: geolife + MTL, test: MTL)')
    print('GeoLife shape: '+str(train_x.shape))
    print('MTL shape: '+str(train_x_mtl.shape))
    
    def collate_fn(batch):
        return tuple(zip(*batch))

    sampler = ImbalancedDatasetSampler(train_dataset_src, callback_get_label=get_label_single, num_samples=len(train_dataset_tgt))
    train_loader_source = DataLoader(train_dataset_src, batch_size=min(args.batch_size, len(train_dataset_src)), sampler=sampler, shuffle=False, drop_last=True, collate_fn=collate_fn)
    train_loader_target = DataLoader(train_dataset_tgt, batch_size=min(args.batch_size, len(train_dataset_tgt)), shuffle=True, drop_last=True, collate_fn=collate_fn)

    train_source_iter = ForeverDataIterator(train_loader_source)
    train_tgt_iter = ForeverDataIterator(train_loader_target)
    test_loader = DataLoader(test_dataset, batch_size=min(args.batch_size, len(test_dataset)), collate_fn=collate_fn)
    
    return train_source_iter, train_tgt_iter, test_loader

