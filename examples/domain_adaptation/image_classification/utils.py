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
