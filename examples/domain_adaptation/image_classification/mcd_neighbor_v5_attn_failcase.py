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

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.optim import SGD,Adam
import torch.utils.data
from torch.utils.data import DataLoader
import torch.nn.functional as F

import utils
import tllib.vision.models as models

from tllib.alignment.mcd import ImageClassifierHead, entropy, classifier_discrepancy
from tllib.utils.data import ForeverDataIterator
from tllib.utils.metric import accuracy, ConfusionMatrix
from tllib.utils.meter import AverageMeter, ProgressMeter
from tllib.utils.logger import CompleteLogger
from tllib.utils.analysis import collect_feature, tsne, a_distance

import numpy as np
import pdb
import operator

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main(args: argparse.Namespace):
    logger = CompleteLogger(args.log, args.phase)
    print(args)

    if args.seed is not None:
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')
    cudnn.benchmark = True
    
    
    if args.nbr_label_mode == 'combine_each_pt':
        G = models.TSEncoder_new(input_dims=7+4).to(device)
    else:
        G = models.TSEncoder_new().to(device)

    if 'cat_samedim' in args.nbr_mode:
        raise NotImplementedError
        F1 = models.Classifier_clf_samedim(input_dim=64).to(device)
        F2 = models.Classifier_clf_samedim(input_dim=64).to(device)
        if args.mean_tea:
            F1_t = models.Classifier_clf_samedim(input_dim=64).to(device)
        else:
            F1_t = None
    elif 'cat' in args.nbr_mode:
        if args.nbr_label_mode == 'separate_input':
            input_dim=64*2+args.nbr_label_embed_dim
        else:
            input_dim=64*2
        F1 = models.Classifier_clf(input_dim=input_dim).to(device)
        F2 = models.Classifier_clf(input_dim=input_dim).to(device)
        if args.mean_tea:
            F1_t = models.Classifier_clf(input_dim=input_dim).to(device)
        else:
            F1_t = None
    elif 'add' in args.nbr_mode:
        F1 = models.Classifier_clf(input_dim=64).to(device)
        F2 = models.Classifier_clf(input_dim=64).to(device)
        if args.mean_tea:
            F1_t = models.Classifier_clf(input_dim=64).to(device)
        else:
            F1_t = None
    else:
        raise NotImplementedError
    attn_net = models.AttnNet().to(device)
    multihead_attn = nn.MultiheadAttention(64, num_heads=args.num_head, batch_first=True).to(device)
    nbr_label_encoder = models.LabelEncoder(input_dim=4, embed_dim=args.nbr_label_embed_dim).to(device)

    pseudo_labels=pseudo_labels_mask=None

    filename='0529_v1'
    ckpt_dir='/home/yichen/Transfer-Learning-Library/examples/domain_adaptation/image_classification/logs/0312_mcd_v5_freeze_01_entpropor0666_srcce1_ent01_tgtce01_qkvcat_0124newnbr20m_limit100_nbrgrad/checkpoints/best.pth'
    ckpt = torch.load(ckpt_dir, map_location='cuda:0')
    G.load_state_dict(ckpt['G'])#, strict=False)
    F1.load_state_dict(ckpt['F1'])#, strict=False)
    multihead_attn.load_state_dict(ckpt['multihead_attn'])#, strict=False)
    args.nbr_mode="qkv_cat"
    args.nbr_limit=100
    print(filename, args.nbr_mode)
    train_source_iter, train_target_iter, val_loader,_,_,_,_ = utils.load_data_neighbor_v3(args, pseudo_labels, pseudo_labels_mask)
    results = validate(val_loader, G, F1, F2, attn_net, args, F1_t, multihead_attn, nbr_label_encoder, filename=filename)
    print('---------------------')
    
    filename='0529_v2'
    ckpt_dir='/home/yichen/Transfer-Learning-Library/examples/domain_adaptation/image_classification/logs/0312_mcd_v5_freeze_01_entpropor0666_srcce1_ent01_tgtce01_perptcat_0124newnbr20m_limit10_nbrgrad/checkpoints/best.pth'
    ckpt = torch.load(ckpt_dir, map_location='cuda:0')
    G.load_state_dict(ckpt['G'])#, strict=False)
    F1.load_state_dict(ckpt['F1'])#, strict=False)
    args.nbr_mode="perpt_cat"
    args.nbr_limit=10
    print(filename, args.nbr_mode)
    train_source_iter, train_target_iter, val_loader,_,_,_,_ = utils.load_data_neighbor_v3(args, pseudo_labels, pseudo_labels_mask)
    results = validate(val_loader, G, F1, F2, attn_net, args, F1_t, multihead_attn, nbr_label_encoder, filename=filename)
    print('---------------------')
    


class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)
        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, context = None, kv_include_self = False):
        b, n, _, h = *x.shape, self.heads
        x = self.norm(x)
        context = default(context, x)

        if kv_include_self:
            context = torch.cat((x, context), dim = 1) # cross attention requires CLS token includes itself as key / value

        qkv = (self.to_q(x), *self.to_kv(context).chunk(2, dim = -1))
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


def softmax_mse_loss(input_logits, target_logits, masks=None):
    """Takes softmax on both sides and returns MSE loss
    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """
    assert input_logits.size() == target_logits.size()
    input_softmax = F.softmax(input_logits, dim=1)
    target_softmax = F.softmax(target_logits, dim=1)
    num_classes = input_logits.size()[1]
    # return F.mse_loss(input_softmax, target_softmax, size_average=False) / num_classes / target_logits.shape[0]
    if masks is not None:
        return torch.sum(torch.mean(F.mse_loss(input_softmax, target_softmax, reduce=False), dim=1) * (~masks)) / torch.sum(~masks)
    else:
        return torch.mean(F.mse_loss(input_softmax, target_softmax, reduce=False))


def train(train_src_iter: ForeverDataIterator, train_tgt_iter: ForeverDataIterator,
          G: nn.Module, F1: ImageClassifierHead, F2: ImageClassifierHead, attn_net: nn.Module,
          optimizer_g: SGD, optimizer_f: SGD, epoch: int, args: argparse.Namespace, F1_t, multihead_attn, nbr_label_encoder):
    raise NotImplemented
    batch_time = AverageMeter('Time', ':3.1f')
    data_time = AverageMeter('Data', ':3.1f')
    losses = AverageMeter('Loss', ':3.2f')
    losses_srcce = AverageMeter('SrcceLoss', ':3.2f')
    losses_ent = AverageMeter('EntLoss', ':3.2f')
    losses_tgtce = AverageMeter('TgtceLoss', ':3.2f')
    losses_meantea = AverageMeter('MTLoss', ':3.2f')
    losses_consis = AverageMeter('ConsisLoss', ':3.2f')
    # trans_losses = AverageMeter('Trans Loss', ':3.2f')
    cls_accs = AverageMeter('Cls Acc', ':3.1f')

    progress = ProgressMeter(
        args.iters_per_epoch,
        [batch_time, data_time, losses, losses_srcce, losses_ent, losses_tgtce, losses_meantea, losses_consis, cls_accs],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    G.train()
    F1.train()
    if args.mean_tea:
        F1_t.eval()
    if 'individual' in args.nbr_mode:
        F2.train()
    attn_net.train()
    multihead_attn.train()
    nbr_label_encoder.train()

    end = time.time()
    for i in range(args.iters_per_epoch):
        # if i>5:
        #     break
        
        # x_s, labels_s = next(train_source_iter)[:2]
        # x_t, = next(train_target_iter)[:1]
        
        # x_s = x_s.to(device)
        # x_t = x_t.to(device)
        # labels_s = labels_s.to(device)
        # x = torch.cat((x_s, x_t), dim=0)
        # assert x.requires_grad is False
        
        # # x_ori_src, labels_src, x_ori_src_neighbor, labels_neighbor, dist_src_neighbor, labels_domain_src = next(train_source_iter)
        # # x_ori_tgt, x_ori_tgt_neighbor, dist_tgt_neighbor, labels_domain_tgt, idx_tgt = next(train_target_iter)
        # x_ori_src, labels_src, x_ori_src_neighbor, masks_src_neighbor, labels_domain_src = next(train_src_iter)
        # x_ori_tgt, x_ori_tgt_neighbor, masks_tgt_neighbor, labels_domain_tgt, idx_tgt = next(train_tgt_iter)
        
        
        # if args.nbr_label_mode == 'separate_input':
        x_ori_src, labels_s, x_ori_src_neighbor, src_neighbor_labels, labels_domain_src = next(train_src_iter) 
        x_ori_tgt, labels_t, labels_t_mask, x_ori_tgt_neighbor, tgt_neighbor_labels, labels_domain_tgt, idx_tgt = next(train_tgt_iter)
        src_neighbor_labels, tgt_neighbor_labels = torch.stack(src_neighbor_labels).to(device), torch.stack(tgt_neighbor_labels).to(device)
        # else:
        #     x_ori_src, labels_s, x_ori_src_neighbor, labels_domain_src = next(train_src_iter) 
        #     x_ori_tgt, labels_t, labels_t_mask, x_ori_tgt_neighbor, labels_domain_tgt, idx_tgt = next(train_tgt_iter)

        x_ori_src, labels_s, labels_domain_src = torch.stack(x_ori_src), torch.stack(labels_s), torch.stack(labels_domain_src)
        x_ori_tgt, labels_t, labels_t_mask, idx_tgt, labels_domain_tgt = torch.stack(x_ori_tgt), torch.stack(labels_t), torch.stack(labels_t_mask), torch.stack(idx_tgt), torch.stack(labels_domain_tgt)
        x_ori_src, x_ori_tgt = x_ori_src[:,:,2:], x_ori_tgt[:,:,2:] # time, dist, v, a, jerk, bearing, is_real
        x_ori_src, x_ori_tgt, labels_s, labels_t, labels_t_mask, idx_tgt = x_ori_src.to(device), x_ori_tgt.to(device), labels_s.to(device), labels_t.to(device), labels_t_mask.to(device), idx_tgt.to(device)
        
        neighbor_idx_src = [neighbor.shape[0] for neighbor in x_ori_src_neighbor]
        neighbor_idx_src = np.insert(np.cumsum(neighbor_idx_src),0,0)
        x_ori_src_neighbor = torch.cat(x_ori_src_neighbor)
        
        neighbor_idx_tgt = [neighbor.shape[0] for neighbor in x_ori_tgt_neighbor]
        neighbor_idx_tgt = np.insert(np.cumsum(neighbor_idx_tgt),0,0)
        x_ori_tgt_neighbor = torch.cat(x_ori_tgt_neighbor)

        # measure data loading time
        data_time.update(time.time() - end)

        # Step A train all networks to minimize loss on source domain
        optimizer_g.zero_grad()
        optimizer_f.zero_grad()

        bs=x_ori_src.shape[0]
        x = torch.cat((x_ori_src, x_ori_tgt), dim=0)
        if 'perpt' in args.nbr_mode:
            _,_,g,_,_,_,_ = G(x)
        elif 'maskboth' in args.nbr_mode:
            _,g,_,_,_,_,_ = G(x, mask_for_ts2loss=True)
        else:
            _,g,_,_,_,_,_ = G(x)
        g_s,g_t = g[:bs],g[bs:]
        

        # with torch.no_grad():
        #     n_iter = x_ori_src_neighbor.shape[0]//bs
        #     tmp_list=[]
        #     for iter in range(n_iter):
        #         neighbor = x_ori_src_neighbor[iter*bs:(iter+1)*bs,:,2:].to(device)
        #         _,feat_src_avgpool_neighbor,_,_,_,_,_ = G(neighbor)
        #         tmp_list.append(feat_src_avgpool_neighbor.cpu())
        #     if x_ori_src_neighbor.shape[0]%bs!=0:
        #         neighbor = x_ori_src_neighbor[n_iter*bs:,:,2:].to(device)
        #         _,feat_src_avgpool_neighbor,_,_,_,_,_ = G(neighbor)
        #         tmp_list.append(feat_src_avgpool_neighbor.cpu())
        #     feat_src_avgpool_neighbor = torch.cat(tmp_list,dim=0)
                                   
        #     n_iter = x_ori_tgt_neighbor.shape[0]//bs
        #     tmp_list=[]
        #     for iter in range(n_iter):
        #         neighbor = x_ori_tgt_neighbor[iter*bs:(iter+1)*bs,:,2:].to(device)
        #         _,feat_tgt_avgpool_neighbor,_,_,_,_,_ = G(neighbor)
        #         tmp_list.append(feat_tgt_avgpool_neighbor.cpu())
        #     if x_ori_tgt_neighbor.shape[0]%bs!=0:
        #         neighbor = x_ori_tgt_neighbor[n_iter*bs:,:,2:].to(device)
        #         _,feat_tgt_avgpool_neighbor,_,_,_,_,_ = G(neighbor)
        #         tmp_list.append(feat_tgt_avgpool_neighbor.cpu())
        #     feat_tgt_avgpool_neighbor = torch.cat(tmp_list,dim=0)
        
        if "perpt" not in args.nbr_mode:
            if args.nbr_grad:
                n_iter = x_ori_src_neighbor.shape[0]//bs
                tmp_list=[]
                for iter in range(n_iter):
                    neighbor = x_ori_src_neighbor[iter*bs:(iter+1)*bs,:,2:].to(device)
                    if args.nbr_label_mode == 'combine_each_pt':
                        nbrs_labels = torch.ones([neighbor.shape[0],650,4]).to(device)/4
                        neighbor = torch.cat([neighbor, nbrs_labels], dim=-1)
                    _,feat_src_avgpool_neighbor,_,_,_,_,_ = G(neighbor)
                    tmp_list.append(feat_src_avgpool_neighbor.cpu())
                if x_ori_src_neighbor.shape[0]%bs!=0:
                    neighbor = x_ori_src_neighbor[n_iter*bs:,:,2:].to(device)
                    if args.nbr_label_mode == 'combine_each_pt':
                        nbrs_labels = torch.ones([neighbor.shape[0],650,4]).to(device)/4
                        neighbor = torch.cat([neighbor, nbrs_labels], dim=-1)
                    _,feat_src_avgpool_neighbor,_,_,_,_,_ = G(neighbor)
                    tmp_list.append(feat_src_avgpool_neighbor.cpu())
                feat_src_avgpool_neighbors = torch.cat(tmp_list,dim=0)
                        
                n_iter = x_ori_tgt_neighbor.shape[0]//bs
                tmp_list=[]
                for iter in range(n_iter):
                    neighbor = x_ori_tgt_neighbor[iter*bs:(iter+1)*bs,:,2:].to(device)
                    if args.nbr_label_mode == 'combine_each_pt':
                        nbrs_labels = torch.ones([neighbor.shape[0],650,4]).to(device)/4
                        neighbor = torch.cat([neighbor, nbrs_labels], dim=-1)
                    _,feat_tgt_avgpool_neighbor,_,_,_,_,_ = G(neighbor)
                    tmp_list.append(feat_tgt_avgpool_neighbor.cpu())
                if x_ori_tgt_neighbor.shape[0]%bs!=0:
                    neighbor = x_ori_tgt_neighbor[n_iter*bs:,:,2:].to(device)
                    if args.nbr_label_mode == 'combine_each_pt':
                        nbrs_labels = torch.ones([neighbor.shape[0],650,4]).to(device)/4
                        neighbor = torch.cat([neighbor, nbrs_labels], dim=-1)
                    _,feat_tgt_avgpool_neighbor,_,_,_,_,_ = G(neighbor)
                    tmp_list.append(feat_tgt_avgpool_neighbor.cpu())
                feat_tgt_avgpool_neighbors = torch.cat(tmp_list,dim=0)
                
            else:
                with torch.no_grad():
                    n_iter = x_ori_src_neighbor.shape[0]//bs
                    tmp_list=[]
                    for iter in range(n_iter):
                        neighbor = x_ori_src_neighbor[iter*bs:(iter+1)*bs,:,2:].to(device)
                        _,feat_src_avgpool_neighbor,_,_,_,_,_ = G(neighbor)
                        tmp_list.append(feat_src_avgpool_neighbor.cpu())
                    if x_ori_src_neighbor.shape[0]%bs!=0:
                        neighbor = x_ori_src_neighbor[n_iter*bs:,:,2:].to(device)
                        _,feat_src_avgpool_neighbor,_,_,_,_,_ = G(neighbor)
                        tmp_list.append(feat_src_avgpool_neighbor.cpu())
                    feat_src_avgpool_neighbors = torch.cat(tmp_list,dim=0)
                            
                    n_iter = x_ori_tgt_neighbor.shape[0]//bs
                    tmp_list=[]
                    for iter in range(n_iter):
                        neighbor = x_ori_tgt_neighbor[iter*bs:(iter+1)*bs,:,2:].to(device)
                        _,feat_tgt_avgpool_neighbor,_,_,_,_,_ = G(neighbor)
                        tmp_list.append(feat_tgt_avgpool_neighbor.cpu())
                    if x_ori_tgt_neighbor.shape[0]%bs!=0:
                        neighbor = x_ori_tgt_neighbor[n_iter*bs:,:,2:].to(device)
                        _,feat_tgt_avgpool_neighbor,_,_,_,_,_ = G(neighbor)
                        tmp_list.append(feat_tgt_avgpool_neighbor.cpu())
                    feat_tgt_avgpool_neighbors = torch.cat(tmp_list,dim=0)
        
        else:
            with torch.no_grad():
                assert args.nbr_data_mode=='mergemin5'
                mask_list=[]
                n_iter = x_ori_src_neighbor.shape[0]//bs
                mask_list=[]
                tmp_list=[]
                for iter in range(n_iter):
                    neighbor = x_ori_src_neighbor[iter*bs:(iter+1)*bs,:,2:].to(device)
                    _,_,feat_src_avgpool_neighbor,_,_,_,_ = G(neighbor) #32,650,64
                    mask = neighbor[:,:,0]!=0
                    mask_list.append(mask)
                    tmp_list.append(feat_src_avgpool_neighbor)
                if x_ori_src_neighbor.shape[0]%bs!=0:
                    neighbor = x_ori_src_neighbor[n_iter*bs:,:,2:].to(device)
                    _,_,feat_src_avgpool_neighbor,_,_,_,_ = G(neighbor)
                    mask = neighbor[:,:,0]!=0 #32,650
                    mask_list.append(mask)
                    tmp_list.append(feat_src_avgpool_neighbor)
                feat_src_avgpool_neighbors = torch.cat(tmp_list,dim=0)
                feat_src_avgpool_neighbors_mask = torch.cat(mask_list,dim=0)

                n_iter = x_ori_tgt_neighbor.shape[0]//bs
                mask_list=[]
                tmp_list=[]
                for iter in range(n_iter):
                    neighbor = x_ori_tgt_neighbor[iter*bs:(iter+1)*bs,:,2:].to(device)
                    _,_,feat_tgt_avgpool_neighbor,_,_,_,_ = G(neighbor) #32,650,64
                    mask = neighbor[:,:,0]!=0
                    mask_list.append(mask)
                    tmp_list.append(feat_tgt_avgpool_neighbor)
                if x_ori_tgt_neighbor.shape[0]%bs!=0:
                    neighbor = x_ori_tgt_neighbor[n_iter*bs:,:,2:].to(device)
                    _,_,feat_tgt_avgpool_neighbor,_,_,_,_ = G(neighbor)
                    mask = neighbor[:,:,0]!=0 #32,650
                    mask_list.append(mask)
                    tmp_list.append(feat_tgt_avgpool_neighbor)
                feat_tgt_avgpool_neighbors = torch.cat(tmp_list,dim=0)
                feat_tgt_avgpool_neighbors_mask = torch.cat(mask_list,dim=0)
        
        
        aux_feat=None
        if 'individual' in args.nbr_mode:
            g_s = g_s.unsqueeze(1)
            g_t = g_t.unsqueeze(1)
            
            # with torch.no_grad():
            feat_src_avgpool_neighbor_list=[]
            for neighbor_idx,_ in enumerate(neighbor_idx_src):
                if neighbor_idx==neighbor_idx_src.shape[0]-1:
                    break
                key = value = feat_src_avgpool_neighbor[neighbor_idx_src[neighbor_idx]:neighbor_idx_src[neighbor_idx+1]].to(device)
                query = g_s[neighbor_idx] # might change to dist?
                attn_output_neighbors_s, attn_output_weights_s = multihead_attn(query, key, value)
                feat_src_avgpool_neighbor_list.append(attn_output_neighbors_s)
            feat_src_avgpool_neighbors = torch.stack(feat_src_avgpool_neighbor_list).squeeze(1)
                
            feat_tgt_avgpool_neighbor_list=[]
            for neighbor_idx,_ in enumerate(neighbor_idx_tgt):
                if neighbor_idx==neighbor_idx_tgt.shape[0]-1:
                    break
                key = value = feat_tgt_avgpool_neighbor[neighbor_idx_tgt[neighbor_idx]:neighbor_idx_tgt[neighbor_idx+1]].to(device)
                query = g_t[neighbor_idx]
                attn_output_neighbors_t, attn_output_weights_t = multihead_attn(query, key, value)
                feat_tgt_avgpool_neighbor_list.append(attn_output_neighbors_t)
            feat_tgt_avgpool_neighbors = torch.stack(feat_tgt_avgpool_neighbor_list).squeeze(1)
            
            g_s = g_s.squeeze(1)
            g_t = g_t.squeeze(1)
            
            g_s1 = g_s
            g_t1 = g_t
            g_s2 = feat_src_avgpool_neighbors
            g_t2 = feat_tgt_avgpool_neighbors
            
            g1 = torch.cat((g_s1, g_t1), dim=0)
            g2 = torch.cat((g_s2, g_t2), dim=0)
            y_1 = F1(g1, aux_feat)
            y_2 = F2(g2, aux_feat)
            y_1, y_2 = F.softmax(y_1, dim=1), F.softmax(y_2, dim=1)
            y1_s, y1_t = y_1.chunk(2, dim=0)
            y2_s, y2_t = y_2.chunk(2, dim=0)
                
        elif args.nbr_mode == 'qkv_cat':
            g_s = g_s.unsqueeze(1)
            g_t = g_t.unsqueeze(1)
        
            feat_src_avgpool_neighbor_list=[]
            for neighbor_idx,_ in enumerate(neighbor_idx_src):
                if neighbor_idx==neighbor_idx_src.shape[0]-1:
                    break
                # tmp_neighbor = feat_src_avgpool_neighbor[neighbor_idx_src[neighbor_idx]:neighbor_idx_src[neighbor_idx+1]]
                # tmp_neighbor = torch.mean(tmp_neighbor, dim=0).to(device)
                
                key = value = feat_src_avgpool_neighbor[neighbor_idx_src[neighbor_idx]:neighbor_idx_src[neighbor_idx+1]].to(device)
                query = g_s[neighbor_idx] # might change to dist?
                attn_output_neighbors_s, attn_output_weights_s = multihead_attn(query, key, value)
                feat_src_avgpool_neighbor_list.append(attn_output_neighbors_s)
            feat_src_avgpool_neighbors = torch.stack(feat_src_avgpool_neighbor_list).squeeze(1)
                
            feat_tgt_avgpool_neighbor_list=[]
            for neighbor_idx,_ in enumerate(neighbor_idx_tgt):
                if neighbor_idx==neighbor_idx_tgt.shape[0]-1:
                    break
                key = value = feat_tgt_avgpool_neighbor[neighbor_idx_tgt[neighbor_idx]:neighbor_idx_tgt[neighbor_idx+1]].to(device)
                query = g_t[neighbor_idx]
                attn_output_neighbors_t, attn_output_weights_t = multihead_attn(query, key, value)
                
                # tmp_neighbor = torch.mean(tmp_neighbor, dim=0).to(device)
                feat_tgt_avgpool_neighbor_list.append(attn_output_neighbors_t)
            feat_tgt_avgpool_neighbors = torch.stack(feat_tgt_avgpool_neighbor_list).squeeze(1)
                    
            g_s = g_s.squeeze(1)
            g_t = g_t.squeeze(1)

            # query = g_s.unsqueeze(1)
            # key = value = feat_src_avgpool_neighbors.unsqueeze(1)
            # feat_src_avgpool_neighbors, attn_output_weights_s = multihead_attn(query, key, value)
            # query = g_t.unsqueeze(1)
            # key = value = feat_tgt_avgpool_neighbors.unsqueeze(1)
            # feat_tgt_avgpool_neighbors, attn_output_weights_t = multihead_attn(query, key, value)
            # # g = torch.cat((attn_output_s, attn_output_t), dim=0)
            
            if args.nbr_label_mode == 'separate_input':
                src_neighbor_labels_feat = nbr_label_encoder(src_neighbor_labels)
                feat_src_avgpool_neighbors = torch.cat([feat_src_avgpool_neighbors, src_neighbor_labels_feat],dim=1)
                tgt_neighbor_labels_feat = nbr_label_encoder(tgt_neighbor_labels)
                feat_tgt_avgpool_neighbors = torch.cat([feat_tgt_avgpool_neighbors, tgt_neighbor_labels_feat],dim=1)

            if 'cat' in args.nbr_mode:
                g_s = torch.cat([g_s,feat_src_avgpool_neighbors],dim=1)
                g_t = torch.cat([g_t,feat_tgt_avgpool_neighbors],dim=1)
            else:
                g_s = g_s + feat_src_avgpool_neighbors
                g_t = g_t + feat_tgt_avgpool_neighbors
            g = torch.cat((g_s, g_t), dim=0)
            y_1 = F1(g, aux_feat)
            y_1 = F.softmax(y_1, dim=1)
            y1_s, y1_t = y_1.chunk(2, dim=0)
    
        elif 'perpt' in args.nbr_mode:
            feat_src_avgpool_neighbor_list=[]
            for neighbor_idx,_ in enumerate(neighbor_idx_src):
                if neighbor_idx==neighbor_idx_tgt.shape[0]-1:
                    break
                tmp_neighbor = feat_src_avgpool_neighbors[neighbor_idx_src[neighbor_idx]:neighbor_idx_src[neighbor_idx+1]]
                tmp_mask = feat_src_avgpool_neighbors_mask[neighbor_idx_src[neighbor_idx]:neighbor_idx_src[neighbor_idx+1]]
                tmp_neighbor = torch.sum(tmp_neighbor, dim=0).to(device)
                mask = torch.sum(tmp_mask,dim=0).unsqueeze(1)
                tmp_neighbor = tmp_neighbor/(mask+1)
                feat_src_avgpool_neighbor_list.append(tmp_neighbor)
            feat_src_avgpool_neighbors = torch.stack(feat_src_avgpool_neighbor_list).squeeze(1)
            
            feat_tgt_avgpool_neighbor_list=[]
            for neighbor_idx,_ in enumerate(neighbor_idx_tgt):
                if neighbor_idx==neighbor_idx_tgt.shape[0]-1:
                    break
                tmp_neighbor = feat_tgt_avgpool_neighbors[neighbor_idx_tgt[neighbor_idx]:neighbor_idx_tgt[neighbor_idx+1]]
                tmp_mask = feat_tgt_avgpool_neighbors_mask[neighbor_idx_tgt[neighbor_idx]:neighbor_idx_tgt[neighbor_idx+1]]
                tmp_neighbor = torch.sum(tmp_neighbor, dim=0).to(device)
                mask = torch.sum(tmp_mask,dim=0).unsqueeze(1)
                tmp_neighbor = tmp_neighbor/(mask+1)
                feat_tgt_avgpool_neighbor_list.append(tmp_neighbor)
            feat_tgt_avgpool_neighbors = torch.stack(feat_tgt_avgpool_neighbor_list).squeeze(1)
            
            if 'cat' in args.nbr_mode:
                g_s = torch.cat([g_s,feat_src_avgpool_neighbors],dim=-1)
                g_t = torch.cat([g_t,feat_tgt_avgpool_neighbors],dim=-1)
            else:
                g_s = g_s + feat_src_avgpool_neighbors
                g_t = g_t + feat_tgt_avgpool_neighbors
            
            g_s = torch.mean(g_s, dim=1)
            g_t = torch.mean(g_t, dim=1)
            g = torch.cat((g_s, g_t), dim=0)
            y_1 = F1(g, aux_feat)
            y1 = F.softmax(y1, dim=1)
            y1_s, y1_t = y_1.chunk(2, dim=0)
        
        else:
            feat_src_avgpool_neighbor_list=[]
            for neighbor_idx,_ in enumerate(neighbor_idx_src):
                if neighbor_idx==neighbor_idx_src.shape[0]-1:
                    break
                tmp_neighbor = feat_src_avgpool_neighbor[neighbor_idx_src[neighbor_idx]:neighbor_idx_src[neighbor_idx+1]]
                tmp_neighbor = torch.mean(tmp_neighbor, dim=0).to(device)
                feat_src_avgpool_neighbor_list.append(tmp_neighbor)
            feat_src_avgpool_neighbors = torch.stack(feat_src_avgpool_neighbor_list)

            feat_tgt_avgpool_neighbor_list=[]
            for neighbor_idx,_ in enumerate(neighbor_idx_tgt):
                if neighbor_idx==neighbor_idx_tgt.shape[0]-1:
                    break
                tmp_neighbor = feat_tgt_avgpool_neighbor[neighbor_idx_tgt[neighbor_idx]:neighbor_idx_tgt[neighbor_idx+1]]
                tmp_neighbor = torch.mean(tmp_neighbor, dim=0).to(device)
                feat_tgt_avgpool_neighbor_list.append(tmp_neighbor)
            feat_tgt_avgpool_neighbors = torch.stack(feat_tgt_avgpool_neighbor_list) 
               
            if 'cat' in args.nbr_mode:
                g_s = torch.cat([g_s,feat_src_avgpool_neighbors],dim=1)
                g_t = torch.cat([g_t,feat_tgt_avgpool_neighbors],dim=1)
            else:
                g_s = g_s + feat_src_avgpool_neighbors
                g_t = g_t + feat_tgt_avgpool_neighbors
            g = torch.cat((g_s, g_t), dim=0)
        
        
        
        # # y_1 = F1(g, aux_feat)
        # # # y_2 = F2(g, aux_feat)
        # y1_s, y1_t = y_1.chunk(2, dim=0)
        # # y2_s, y2_t = y_2.chunk(2, dim=0)
        # # y1_t, y2_t = F.softmax(y1_t, dim=1), F.softmax(y2_t, dim=1)
        # y1_t = F.softmax(y1_t, dim=1)
        
        if args.mean_tea:
            y_tea = F1_t(g, aux_feat)
            y_s_tea, y_t_tea = y_tea.chunk(2, dim=0)
            y_t_tea = F.softmax(y_t_tea, dim=1)
            _,y_t_tea_hard = torch.max(y_t_tea, 1)

        if args.loss_mode=='tgtce':
            raise NotImplementedError
            loss = torch.sum(F.cross_entropy(y1_t, labels_t, reduce=False) * labels_t_mask)/torch.sum(labels_t_mask) 
        elif args.loss_mode=='srcce_ent_tgtce_tgtmeantea':
            raise NotImplementedError
            # rampup_ratio = sigmoid_rampup(self.current_epoch,self.rampup_length) 
            loss = F.cross_entropy(y1_s, labels_s) + entropy(y1_t) * args.trade_off_entropy + \
                    torch.sum(F.cross_entropy(y1_t, labels_t, reduce=False) * labels_t_mask) / torch.sum(labels_t_mask) * args.trade_off_pseudo + \
                        softmax_mse_loss(y1_t, y_t_tea) * args.trade_off_consis
        elif args.loss_mode=='srcce_ent_tgtce_tgtmeanteanomask':
            loss_srcce = F.cross_entropy(y1_s, labels_s)
            loss_ent = entropy(y1_t) * args.trade_off_entropy
            loss_tgtce = torch.sum(F.cross_entropy(y1_t, labels_t, reduce=False) * labels_t_mask) / torch.sum(labels_t_mask) * args.trade_off_pseudo
            loss_meantea = softmax_mse_loss(y1_t, y_t_tea, labels_t_mask) * args.trade_off_consis
            loss = loss_srcce + loss_ent  + loss_tgtce + loss_meantea
            loss_consistency=torch.tensor(0.)
        elif args.loss_mode=='srcce_ent_tgtce_tgtmeanteanomask_hardtea':
            loss_srcce = F.cross_entropy(y1_s, labels_s)
            loss_ent = entropy(y1_t) * args.trade_off_entropy
            loss_tgtce = torch.sum(F.cross_entropy(y1_t, labels_t, reduce=False) * labels_t_mask) / torch.sum(labels_t_mask) * args.trade_off_pseudo
            loss_meantea = torch.sum(F.cross_entropy(y1_t, y_t_tea_hard, reduce=False) * ~labels_t_mask) / torch.sum(~labels_t_mask) * args.trade_off_consis
            loss = loss_srcce + loss_ent  + loss_tgtce + loss_meantea 
            loss_consistency=torch.tensor(0.)
        elif args.loss_mode=='srcce_tgtce_tgtmeantea':
            raise NotImplementedError
            loss = F.cross_entropy(y1_s, labels_s) + \
                    torch.sum(F.cross_entropy(y1_t, labels_t, reduce=False) * labels_t_mask) / torch.sum(labels_t_mask) * args.trade_off_pseudo + \
                        softmax_mse_loss(y1_t, y_t_tea) * args.trade_off_consis

        # double src CE + tgt entropy + tgt pseudo CE + 2head consistency
        elif args.loss_mode=='v1':
            loss_srcce = 0.5 * (F.cross_entropy(y1_s, labels_s) + F.cross_entropy(y2_s, labels_s))
            loss_ent = 0.5 * args.trade_off_entropy * (entropy(y1_t) + entropy(y2_t))
            loss_tgtce = 0.5 * args.trade_off_pseudo * (torch.sum(F.cross_entropy(y1_t, labels_t, reduce=False) * labels_t_mask) / torch.sum(labels_t_mask) + torch.sum(F.cross_entropy(y2_t, labels_t, reduce=False) * labels_t_mask) / torch.sum(labels_t_mask))
            loss_consistency = softmax_mse_loss(y_1, y_2) * args.trade_off_consis
            loss = loss_srcce + loss_ent  + loss_tgtce + loss_consistency
        
        else:
            loss_srcce = F.cross_entropy(y1_s, labels_s)
            loss_ent = entropy(y1_t) * args.trade_off_entropy
            loss_tgtce = torch.sum(F.cross_entropy(y1_t, labels_t, reduce=False) * labels_t_mask) / torch.sum(labels_t_mask) * args.trade_off_pseudo
            loss = loss_srcce + loss_ent  + loss_tgtce 
            loss_consistency=torch.tensor(0.)
            
        loss.backward()
        optimizer_g.step()
        optimizer_f.step()
        
        losses.update(loss.item(), bs)
        losses_srcce.update(loss_srcce.item(), bs)
        losses_ent.update(loss_ent.item(), bs)
        losses_tgtce.update(loss_tgtce.item(), bs)
        losses_consis.update(loss_consistency.item(), bs)
        if args.mean_tea:
            losses_meantea.update(loss_meantea.item(), bs)
        batch_time.update(time.time() - end)
        end = time.time()
        if i % args.print_freq == 0:
            progress.display(i)


def validate(val_loader: DataLoader, G: nn.Module, F1: ImageClassifierHead, F2: ImageClassifierHead, attn_net:nn.Module, args: argparse.Namespace, F1_t, multihead_attn, nbr_label_encoder, filename) -> Tuple[float, float]:
    batch_time = AverageMeter('Time', ':6.3f')
    top1_1 = AverageMeter('Acc_1', ':6.2f')
    top1_2 = AverageMeter('Acc_2', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, top1_1, top1_2],
        prefix='Test: ')

    # switch to evaluate mode
    G.eval()
    F1.eval()
    if args.mean_tea:
        F1_t.eval()
    if 'individual' in args.nbr_mode:
        F2.eval()
    attn_net.eval()
    multihead_attn.eval()
    nbr_label_encoder.eval()


    if args.per_class_eval:
        confmat = ConfusionMatrix(len(args.class_names))
    else:
        confmat = None
    
    nb_classes = 4
    confusion_matrix1 = torch.zeros(nb_classes, nb_classes) 
    confusion_matrix2 = torch.zeros(nb_classes, nb_classes) 

    label_dict = {"walk": 0, "bike": 1, "car": 2, "train": 3}
    idx_dict={}
    for k,v in label_dict.items():
        idx_dict[v]=k

    wrong_list = {0: [[],[],[],[]], 1:  [[],[],[],[]], 2:  [[],[],[],[]], 3:  [[],[],[],[]]}
    with torch.no_grad():
        end = time.time()
        for i, data in enumerate(val_loader):
            # if i>5:
            #     break
            
            x, labels, neighbors, neighbor_labels,_ = data
            if neighbor_labels[0] is not None:
                neighbor_labels = torch.stack(neighbor_labels).to(device)
            x, labels = torch.stack(x), torch.stack(labels)
            x, labels = x.to(device), labels.to(device)
            bs = x.shape[0]
            x = x[:,:,2:]
            
            neighbor_idx_src = [neighbor.shape[0] for neighbor in neighbors]
            neighbor_idx_src = np.insert(np.cumsum(neighbor_idx_src),0,0)
            neighbors = torch.cat(neighbors)

            # compute output
            if 'perpt' in args.nbr_mode:
                _,_,g,_,_,_,_ = G(x)
            else:
                _,g,_,_,_,_,_ = G(x)
            
            if "perpt" not in args.nbr_mode:
                n_iter = neighbors.shape[0]//bs
                tmp_list=[]
                for iter in range(n_iter):
                    neighbor = neighbors[iter*bs:(iter+1)*bs,:,2:].to(device)
                    if args.nbr_label_mode == 'combine_each_pt':
                        nbrs_labels = torch.ones([neighbor.shape[0],650,4]).to(device)/4
                        neighbor = torch.cat([neighbor, nbrs_labels], dim=-1)
                    _,feat_neighbor,_,_,_,_,_ = G(neighbor)
                    tmp_list.append(feat_neighbor.cpu())
                if neighbors.shape[0]%bs!=0:
                    neighbor = neighbors[n_iter*bs:,:,2:].to(device)
                    if args.nbr_label_mode == 'combine_each_pt':
                        nbrs_labels = torch.ones([neighbor.shape[0],650,4]).to(device)/4
                        neighbor = torch.cat([neighbor, nbrs_labels], dim=-1)
                    _,feat_neighbor,_,_,_,_,_ = G(neighbor)
                    tmp_list.append(feat_neighbor.cpu())
                feat_neighbor = torch.cat(tmp_list,dim=0)
            else:
                n_iter = neighbors.shape[0]//bs
                mask_list=[]
                tmp_list=[]
                for iter in range(n_iter):
                    neighbor = neighbors[iter*bs:(iter+1)*bs,:,2:].to(device)
                    _,_,feat_neighbor,_,_,_,_ = G(neighbor) #32,650,64
                    mask = neighbor[:,:,0]!=0
                    mask_list.append(mask)
                    tmp_list.append(feat_neighbor)
                if neighbors.shape[0]%bs!=0:
                    neighbor = neighbors[n_iter*bs:,:,2:].to(device)
                    _,_,feat_neighbor,_,_,_,_ = G(neighbor)
                    mask = neighbor[:,:,0]!=0 #32,650
                    mask_list.append(mask)
                    tmp_list.append(feat_neighbor)
                feat_neighbors = torch.cat(tmp_list,dim=0)
                feat_neighbors_mask = torch.cat(mask_list,dim=0)
            
            
            
            aux_feat = None
            if 'individual' in args.nbr_mode:
                g = g.unsqueeze(1)
                feat_neighbor_list=[]
                for neighbor_idx,_ in enumerate(neighbor_idx_src):
                    if neighbor_idx==neighbor_idx_src.shape[0]-1:
                        break
                    key = value = feat_neighbor[neighbor_idx_src[neighbor_idx]:neighbor_idx_src[neighbor_idx+1]].to(device)
                    query = g[neighbor_idx]
                    attn_output_neighbors, attn_output_weights = multihead_attn(query, key, value)
                    feat_neighbor_list.append(attn_output_neighbors)
                feat_neighbors = torch.stack(feat_neighbor_list).squeeze(1)
                g = g.squeeze(1)

                g1 = g
                g2 = feat_neighbors

                y1 = F1(g1, aux_feat)
                y2 = F2(g2, aux_feat)
                acc2, = accuracy(y2, labels)
                top1_2.update(acc2.item(), bs)
                _, preds2 = torch.max(y2, 1)
                for t, p in zip(labels.view(-1), preds2.view(-1)):
                    confusion_matrix2[t.long(), p.long()] += 1
            
            elif 'qkv' in args.nbr_mode:
                g = g.unsqueeze(1)
                feat_neighbor_list=[]
                for neighbor_idx,_ in enumerate(neighbor_idx_src):
                    if neighbor_idx==neighbor_idx_src.shape[0]-1:
                        break
                    key = value = feat_neighbor[neighbor_idx_src[neighbor_idx]:neighbor_idx_src[neighbor_idx+1]].to(device)
                    query = g[neighbor_idx]
                    attn_output_neighbors, attn_output_weights = multihead_attn(query, key, value)
                    feat_neighbor_list.append(attn_output_neighbors)
                feat_neighbors = torch.stack(feat_neighbor_list).squeeze(1)
                g = g.squeeze(1)
                
                if args.nbr_label_mode == 'separate_input':
                    neighbor_labels_feat = nbr_label_encoder(neighbor_labels)
                    feat_neighbors = torch.cat([feat_neighbors, neighbor_labels_feat],dim=1)
                
                if 'cat' in args.nbr_mode:
                    g = torch.cat([g,feat_neighbors],dim=1)
                else:
                    g = g + feat_neighbors
                y1 = F1(g, aux_feat)
                
            elif 'perpt' in args.nbr_mode:
                feat_neighbor_list=[]
                for neighbor_idx,_ in enumerate(neighbor_idx_src):
                    if neighbor_idx==neighbor_idx_src.shape[0]-1:
                        break
                    tmp_neighbor = feat_neighbors[neighbor_idx_src[neighbor_idx]:neighbor_idx_src[neighbor_idx+1]]
                    tmp_mask = feat_neighbors_mask[neighbor_idx_src[neighbor_idx]:neighbor_idx_src[neighbor_idx+1]]
                    
                    tmp_neighbor = torch.sum(tmp_neighbor, dim=0).to(device)
                    mask = torch.sum(tmp_mask,dim=0).unsqueeze(1)
                    tmp_neighbor = tmp_neighbor/(mask+1)
                    feat_neighbor_list.append(tmp_neighbor)
                feat_neighbors = torch.stack(feat_neighbor_list).squeeze(1)
                
                if 'cat' in args.nbr_mode:
                    g = torch.cat([g,feat_neighbors],dim=-1)
                else:
                    g = g + feat_neighbors
                g = torch.mean(g, dim=1)
                y1 = F1(g, aux_feat)
        
            else:
                feat_neighbor_list=[]
                for neighbor_idx,_ in enumerate(neighbor_idx_src):
                    if neighbor_idx==neighbor_idx_src.shape[0]-1:
                        break
                    tmp_neighbor = feat_neighbor[neighbor_idx_src[neighbor_idx]:neighbor_idx_src[neighbor_idx+1]]
                    tmp_neighbor = torch.mean(tmp_neighbor, dim=0).to(device)
                    feat_neighbor_list.append(tmp_neighbor)
                feat_neighbors = torch.stack(feat_neighbor_list)
                
                if 'cat' in args.nbr_mode:
                    g = torch.cat([g,feat_neighbors],dim=1)
                else:
                    g = g + feat_neighbors
                y1 = F1(g, aux_feat)
                    

            # measure accuracy and record loss
            acc1, = accuracy(y1, labels)
            if confmat:
                confmat.update(labels, y1.argmax(1))
            top1_1.update(acc1.item(), bs)
            
            hard_y = torch.argmax(y1,dim=1)
            diff_ids = torch.nonzero(hard_y != labels).cpu()
            # diff_ids = torch.nonzero(hard_y == labels).cpu()
            for diff_id in diff_ids:
                tmp_label = labels[diff_id].item()
                tmp_pred = hard_y[diff_id].item()
                wrong_list[tmp_label][tmp_pred].append((diff_id+i*args.batch_size).item())
            
            _, preds1 = torch.max(y1, 1)
            for t, p in zip(labels.view(-1), preds1.view(-1)):
                confusion_matrix1[t.long(), p.long()] += 1

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)
                
        np.save("/home/yichen/Transfer-Learning-Library/examples/domain_adaptation/image_classification/logs/failcase0530/%s"%filename, np.array(wrong_list))
        # np.save("/home/yichen/Transfer-Learning-Library/examples/domain_adaptation/image_classification/logs/failcase_correct/%s"%filename, np.array(wrong_list))
        
        if top1_1.avg > top1_2.avg:
            confusion_matrix = confusion_matrix1
        else:
            confusion_matrix = confusion_matrix2
        print(str(confusion_matrix))
        per_class_acc = list((confusion_matrix.diag()/confusion_matrix.sum(1)).numpy())
        print('per class accuracy:')
        for idx,acc in enumerate(per_class_acc):
            print('\t '+str(idx_dict[idx])+': '+str(acc))

        print(' * Acc1 {top1_1.avg:.3f} Acc2 {top1_2.avg:.3f}'
              .format(top1_1=top1_1, top1_2=top1_2))
        if confmat:
            print(confmat.format(args.class_names))

    return top1_1.avg, top1_2.avg










if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MCD for Unsupervised Domain Adaptation')
    # dataset parameters
    # parser.add_argument('root', metavar='DIR',
    #                     help='root path of dataset')
    # parser.add_argument('-d', '--data', metavar='DATA', default='Office31', choices=utils.get_dataset_names(),
    #                     help='dataset: ' + ' | '.join(utils.get_dataset_names()) +
    #                          ' (default: Office31)')
    parser.add_argument('-s', '--source', help='source domain(s)', nargs='+')
    parser.add_argument('-t', '--target', help='target domain(s)', nargs='+')
    parser.add_argument('--train-resizing', type=str, default='default')
    parser.add_argument('--val-resizing', type=str, default='default')
    parser.add_argument('--resize-size', type=int, default=224,
                        help='the image size after resizing')
    parser.add_argument('--scale', type=float, nargs='+', default=[0.08, 1.0], metavar='PCT',
                        help='Random resize scale (default: 0.08 1.0)')
    parser.add_argument('--ratio', type=float, nargs='+', default=[3. / 4., 4. / 3.], metavar='RATIO',
                        help='Random resize aspect ratio (default: 0.75 1.33)')
    parser.add_argument('--no-hflip', action='store_true',
                        help='no random horizontal flipping during training')
    parser.add_argument('--norm-mean', type=float, nargs='+',
                        default=(0.485, 0.456, 0.406), help='normalization mean')
    parser.add_argument('--norm-std', type=float, nargs='+',
                        default=(0.229, 0.224, 0.225), help='normalization std')
    # model parameters
    # parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
    #                     # choices=utils.get_model_names(),
    #                     help='backbone architecture: ' +
    #                          ' | '.join(utils.get_model_names()) +
    #                          ' (default: resnet18)')
    parser.add_argument('--bottleneck-dim', default=1024, type=int)
    parser.add_argument('--no-pool', action='store_true',
                        help='no pool layer after the feature extractor.')
    parser.add_argument('--scratch', action='store_true', help='whether train from scratch.')
    parser.add_argument('--trade-off', default=1., type=float,
                        help='the trade-off hyper-parameter for transfer loss')
    parser.add_argument('--trade-off-entropy', default=0.01, type=float,
                        help='the trade-off hyper-parameter for entropy loss')
    parser.add_argument('--num-k', type=int, default=4, metavar='K',
                        help='how many steps to repeat the generator update')
    # training parameters
    parser.add_argument('-b', '--batch-size', default=32, type=int,
                        metavar='N',
                        help='mini-batch size (default: 32)')
    parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--wd', '--weight-decay', default=1e-3, type=float,
                    metavar='W', help='weight decay (default: 1e-3)',
                    dest='weight_decay')
    parser.add_argument('-j', '--workers', default=2, type=int, metavar='N',
                        help='number of data loading workers (default: 2)')
    parser.add_argument('--epochs', default=20, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-i', '--iters-per-epoch', default=1000, type=int,
                        help='Number of iterations per epoch')
    parser.add_argument('-p', '--print-freq', default=100, type=int,
                        metavar='N', help='print frequency (default: 100)')
    parser.add_argument('--seed', default=None, type=int,
                        help='seed for initializing training. ')
    parser.add_argument('--per-class-eval', action='store_true',
                        help='whether output per-class accuracy during evaluation')
    parser.add_argument("--log", type=str, default='mcd',
                        help="Where to save logs, checkpoints and debugging images.")
    parser.add_argument("--phase", type=str, default='train', choices=['train', 'test', 'analysis'],
                        help="When phase is 'test', only test the model."
                             "When phase is 'analysis', only analysis the model.")
    
    parser.add_argument('--use_unlabel', action="store_true", help='Whether to perform evaluation after training')
    parser.add_argument('--interpolated', action="store_true", help='Whether to perform evaluation after training')
    parser.add_argument('--interpolatedlinear', action="store_true", help='Whether to perform evaluation after training')
    parser.add_argument('--trip_time', type=int, default=20, help='')
    
    parser.add_argument("--cat_mode", type=str, default='cat', help="Where to save logs, checkpoints and debugging images.")
    parser.add_argument("--loss_mode", type=str, default='srcce_tgtce_tgtent', help="Where to save logs, checkpoints and debugging images.")
    parser.add_argument("--pseudo_mode", type=str, default='threshold', help="Where to save logs, checkpoints and debugging images.")
    parser.add_argument('--pseudo_thres', default=0.95, type=float, help='initial learning rate')
    parser.add_argument('--pseudo_ratio', default=0.666, type=float, help='initial learning rate')
    parser.add_argument('--nbr_dist_thres', default=30, type=int, help='initial learning rate')
    parser.add_argument('--nbr_limit', default=10, type=int, help='initial learning rate')
    parser.add_argument('--trade-off-pseudo', default=1., type=float)
    parser.add_argument('--trade-off-consis', default=1., type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--mean_tea', action="store_true", help='Whether to perform evaluation after training')
    parser.add_argument('--num_head', default=2, type=int, help='initial learning rate')
    parser.add_argument("--nbr_data_mode", type=str, default='mergemin5', help="Where to save logs, checkpoints and debugging images.")
    parser.add_argument("--nbr_mode", type=str, default='perpt_cat', help="Where to save logs, checkpoints and debugging images.")
    parser.add_argument('--nbr_grad', action="store_true", help='Whether to perform evaluation after training')
    parser.add_argument('--nbr_pseudo', action="store_true", help='Whether to perform evaluation after training')
    parser.add_argument("--nbr_label_mode", type=str, default='', help="Where to save logs, checkpoints and debugging images.")
    parser.add_argument('--nbr_label_embed_dim', default=16, type=int, help='initial learning rate')
    parser.add_argument('--pseudo_every_epoch', action="store_true", help='Whether to perform evaluation after training')

    
    parser.add_argument('--random_mask_nbr_ratio', default=1.0, type=float)
    parser.add_argument('--mask_early', action="store_true", help='Whether to perform evaluation after training')
    parser.add_argument('--mask_late', action="store_true", help='Whether to perform evaluation after training')
    parser.add_argument('--n_mask_late', default=5, type=int, help='initial learning rate')

    parser.add_argument('--bert_out_dim', default=64, type=int, help='initial learning rate')
    parser.add_argument('--G_no_frozen', action="store_true", help='Whether to perform evaluation after training')
    parser.add_argument('--token_len', default=10, type=int, help='initial learning rate')
    parser.add_argument('--token_max_len', default=60, type=int, help='initial learning rate')
    parser.add_argument("--steps_list", type=str, default='ABC', help="Where to save logs, checkpoints and debugging images.")
    parser.add_argument('--prompt_id', default=5, type=int, help='initial learning rate')
    parser.add_argument('--semi', action="store_true", help='Whether to perform evaluation after training')


    args = parser.parse_args()
    torch.set_num_threads(8)

    main(args)
