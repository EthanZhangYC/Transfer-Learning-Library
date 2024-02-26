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
        # random.seed(args.seed)
        # torch.manual_seed(args.seed)
        # cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    cudnn.benchmark = True

    # Data loading code
    # train_transform = utils.get_train_transform(args.train_resizing, scale=args.scale, ratio=args.ratio,
    #                                             random_horizontal_flip=not args.no_hflip,
    #                                             random_color_jitter=False, resize_size=args.resize_size,
    #                                             norm_mean=args.norm_mean, norm_std=args.norm_std)
    # val_transform = utils.get_val_transform(args.val_resizing, resize_size=args.resize_size,
    #                                         norm_mean=args.norm_mean, norm_std=args.norm_std)
    # print("train_transform: ", train_transform)
    # print("val_transform: ", val_transform)

    # train_source_dataset, train_target_dataset, val_dataset, test_dataset, num_classes, args.class_names = \
    #     utils.get_dataset(args.data, args.root, args.source, args.target, train_transform, val_transform)
    # train_source_loader = DataLoader(train_source_dataset, batch_size=args.batch_size,
    #                                  shuffle=True, num_workers=args.workers, drop_last=True)
    # train_target_loader = DataLoader(train_target_dataset, batch_size=args.batch_size,
    #                                  shuffle=True, num_workers=args.workers, drop_last=True)
    # val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    # test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    # num_classes = 4
    # train_source_dataset, train_target_dataset, val_dataset = utils.load_data(args)
    # train_source_loader = DataLoader(train_source_dataset, batch_size=args.batch_size,
    #                                  shuffle=True, num_workers=args.workers, drop_last=True)
    # train_target_loader = DataLoader(train_target_dataset, batch_size=args.batch_size,
    #                                  shuffle=True, num_workers=args.workers, drop_last=True)
    # val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    # train_source_iter = ForeverDataIterator(train_source_loader)
    # train_target_iter = ForeverDataIterator(train_target_loader)
    G = models.TSEncoder_new().to(device)
    classifier_features_dim=64
    num_classes = 4
    
    # # if args.ckpt_dir is not None:
    # # print('resuming model')
    # ckpt_dir='/home/yichen/TS2Vec/result/0402_pretrain/model_best.pth.tar'
    # ckpt = torch.load(ckpt_dir, map_location='cuda:0')
    # G.load_state_dict(ckpt['encoder_tgt_state_dict'], strict=False)


    # create model
    # print("=> using model '{}'".format(args.arch))
    # G = utils.get_model(args.arch, pretrain=not args.scratch).to(device)  # feature extractor
    # two image classifier heads
    pool_layer = nn.Identity() if args.no_pool else None
    # F1 = ImageClassifierHead(G.out_features, num_classes, args.bottleneck_dim, pool_layer).to(device)
    # F2 = ImageClassifierHead(G.out_features, num_classes, args.bottleneck_dim, pool_layer).to(device)
    F1_ori = models.Classifier_clf(input_dim=64).to(device)
    # F2_ori = models.Classifier_clf(input_dim=64).to(device)
    if args.cat_mode=='cat':
        F1 = models.Classifier_clf(input_dim=64*2).to(device)
        if args.mean_tea:
            F1_t = models.Classifier_clf(input_dim=64*2).to(device)
        else:
            F1_t = None
    elif args.cat_mode=='cat_samedim':
        F1 = models.Classifier_clf_samedim(input_dim=64).to(device)
        if args.mean_tea:
            F1_t = models.Classifier_clf_samedim(input_dim=64).to(device)
        else:
            F1_t = None
    elif args.cat_mode=='add':
        F1 = models.Classifier_clf(input_dim=64).to(device)
        if args.mean_tea:
            F1_t = models.Classifier_clf(input_dim=64).to(device)
        else:
            F1_t = None
    else:
        raise NotImplementedError
    # F2 = models.Classifier_clf(input_dim=64*2).to(device)
    # F1 = models.ViT(use_auxattn=True, double_attn=True).to(device)
    # F2 = models.ViT(use_auxattn=True, double_attn=True).to(device)
    attn_net = models.AttnNet().to(device)
    
    multihead_attn = nn.MultiheadAttention(64, num_heads=args.num_head, batch_first=True).to(device)

    
    
    ckpt_dir='/home/yichen/Transfer-Learning-Library/examples/domain_adaptation/image_classification/logs/mcd_1015/checkpoints/best.pth'
    ckpt = torch.load(ckpt_dir, map_location='cuda:0')
    G.load_state_dict(ckpt['G'], strict=False)
    F1_ori.load_state_dict(ckpt['F2'])#, strict=False)
    # F2_ori.load_state_dict(ckpt['F2'])#, strict=False)
    
    if args.cat_mode=='cat_samedim':
        F1.load_state_dict(ckpt['F2'], strict=False)
        for name,param in F1.named_parameters():
            if 'fc' in name:
                param.requires_grad = False 
        
    if args.mean_tea:
        for param_s, param_t in zip(F1.parameters(), F1_t.parameters()):
            param_t.data.copy_(param_s.data) 
            param_t.requires_grad=False
    
    for param in G.parameters():
        param.requires_grad = False 
    
    _,_,_,train_loader_target = utils.load_data(args)
    # pseudo_labels,pseudo_labels_mask = get_pseudo_labels(train_loader_target, G, F1_ori, args)
    pseudo_labels,pseudo_labels_mask = eval('get_pseudo_labels_by_'+args.pseudo_mode)(train_loader_target, G, F1_ori, args)
    del F1_ori#,F2_ori#,train_loader_target
    train_source_iter, train_target_iter, val_loader = utils.load_data_neighbor_v3(args, pseudo_labels, pseudo_labels_mask)

    

    # define optimizer
    # the learning rate is fixed according to origin paper
    # optimizer_g = SGD(G.parameters(), lr=args.lr, weight_decay=0.0005)
    # optimizer_f = SGD([
    #     {"params": F1.parameters()},
    #     {"params": F2.parameters()},
    # ], momentum=0.9, lr=args.lr, weight_decay=0.0005)
    optimizer_g = Adam(G.parameters(), args.lr, weight_decay=args.weight_decay, betas=(0.5, 0.99))
    optimizer_f = Adam([
        {"params": F1.parameters()},
        # {"params": F2.parameters()},
        {"params": multihead_attn.parameters()},
        {"params": attn_net.parameters()},
    ], args.lr, weight_decay=args.weight_decay, betas=(0.5, 0.99))

    # resume from the best checkpoint
    if args.phase != 'train':
        checkpoint = torch.load(logger.get_checkpoint_path('best'), map_location='cpu')
        G.load_state_dict(checkpoint['G'])
        F1.load_state_dict(checkpoint['F1'])
        F2.load_state_dict(checkpoint['F2'])

    # analysis the model
    if args.phase == 'analysis':
        # extract features from both domains
        feature_extractor = nn.Sequential(G, F1.pool_layer).to(device)
        source_feature = collect_feature(train_source_loader, feature_extractor, device)
        target_feature = collect_feature(train_target_loader, feature_extractor, device)
        # plot t-SNE
        tSNE_filename = osp.join(logger.visualize_directory, 'TSNE.pdf')
        tsne.visualize(source_feature, target_feature, tSNE_filename)
        print("Saving t-SNE to", tSNE_filename)
        # calculate A-distance, which is a measure for distribution discrepancy
        A_distance = a_distance.calculate(source_feature, target_feature, device)
        print("A-distance =", A_distance)
        return

    if args.phase == 'test':
        acc1 = validate(test_loader, G, F1, F2, args)
        print(acc1)
        return

    # start training
    best_acc1 = 0.
    best_results = None
    best_epoch = 0
    for epoch in range(args.epochs):
        # train for one epoch
        train(train_source_iter, train_target_iter, G, F1, attn_net, optimizer_g, optimizer_f, epoch, args, F1_t, multihead_attn)

        # evaluate on validation set
        results = validate(val_loader, G, F1, attn_net, args, F1_t, multihead_attn)

        # remember best acc@1 and save checkpoint
        torch.save({
            'G': G.state_dict(),
            'F1': F1.state_dict(),
            # 'F2': F2.state_dict(),
            'multihead_attn': multihead_attn.state_dict(),
            'attnnet':attn_net.state_dict()
        }, logger.get_checkpoint_path('latest'))
        if max(results) > best_acc1:
            shutil.copy(logger.get_checkpoint_path('latest'), logger.get_checkpoint_path('best'))
            best_acc1 = max(results)
            best_results = results
            best_epoch = epoch
        print("best_acc1 = {:3.1f} ({}), results = {}".format(best_acc1, best_epoch, best_results))

    # evaluate on test set
    checkpoint = torch.load(logger.get_checkpoint_path('best'), map_location='cpu')
    G.load_state_dict(checkpoint['G'])
    F1.load_state_dict(checkpoint['F1'])
    # F2.load_state_dict(checkpoint['F2'])
    # results = validate(test_loader, G, F1, F2, args)
    # print("test_acc1 = {:3.1f}".format(max(results)))

    logger.close()


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


def softmax_mse_loss(input_logits, target_logits, masks):
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
    return torch.sum(torch.mean(F.mse_loss(input_softmax, target_softmax, reduce=False), dim=1) * (~masks)) / torch.sum(~masks)


def train(train_src_iter: ForeverDataIterator, train_tgt_iter: ForeverDataIterator,
          G: nn.Module, F1: ImageClassifierHead, attn_net: nn.Module,
          optimizer_g: SGD, optimizer_f: SGD, epoch: int, args: argparse.Namespace, F1_t, multihead_attn):
    batch_time = AverageMeter('Time', ':3.1f')
    data_time = AverageMeter('Data', ':3.1f')
    losses = AverageMeter('Loss', ':3.2f')
    losses_srcce = AverageMeter('SrcceLoss', ':3.2f')
    losses_ent = AverageMeter('EntLoss', ':3.2f')
    losses_tgtce = AverageMeter('TgtceLoss', ':3.2f')
    losses_meantea = AverageMeter('MTLoss', ':3.2f')
    # trans_losses = AverageMeter('Trans Loss', ':3.2f')
    cls_accs = AverageMeter('Cls Acc', ':3.1f')

    progress = ProgressMeter(
        args.iters_per_epoch,
        [batch_time, data_time, losses, losses_srcce, losses_ent, losses_tgtce, losses_meantea, cls_accs],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    G.train()
    F1.train()
    if args.mean_tea:
        F1_t.eval()
    # F2.train()
    attn_net.train()
    multihead_attn.train()

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
        x_ori_src, labels_s, x_ori_src_neighbor, labels_domain_src = next(train_src_iter) 
        x_ori_tgt, labels_t, labels_t_mask, x_ori_tgt_neighbor, labels_domain_tgt, idx_tgt = next(train_tgt_iter)

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
        _,g,_,_,_,_,_ = G(x)
        g_s,g_t = g[:bs],g[bs:]
        

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
            feat_src_avgpool_neighbor = torch.cat(tmp_list,dim=0)
                                   
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
            feat_tgt_avgpool_neighbor = torch.cat(tmp_list,dim=0)
            
        
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
        # pdb.set_trace()
        # query = g_t.unsqueeze(1)
        # key = value = feat_tgt_avgpool_neighbors.unsqueeze(1)
        # feat_tgt_avgpool_neighbors, attn_output_weights_t = multihead_attn(query, key, value)
        # # g = torch.cat((attn_output_s, attn_output_t), dim=0)

        
        if 'cat' in args.cat_mode:
            g_s = torch.cat([g_s,feat_src_avgpool_neighbors],dim=1)
            g_t = torch.cat([g_t,feat_tgt_avgpool_neighbors],dim=1)
        else:
            g_s = g_s + feat_src_avgpool_neighbors
            g_t = g_t + feat_tgt_avgpool_neighbors
        g = torch.cat((g_s, g_t), dim=0)
        aux_feat=None

        y_1 = F1(g, aux_feat)
        # y_2 = F2(g, aux_feat)
        y1_s, y1_t = y_1.chunk(2, dim=0)
        # y2_s, y2_t = y_2.chunk(2, dim=0)
        # y1_t, y2_t = F.softmax(y1_t, dim=1), F.softmax(y2_t, dim=1)
        y1_t = F.softmax(y1_t, dim=1)
        
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
        elif args.loss_mode=='srcce_ent_tgtce_tgtmeanteanomask_hardtea':
            loss_srcce = F.cross_entropy(y1_s, labels_s)
            loss_ent = entropy(y1_t) * args.trade_off_entropy
            loss_tgtce = torch.sum(F.cross_entropy(y1_t, labels_t, reduce=False) * labels_t_mask) / torch.sum(labels_t_mask) * args.trade_off_pseudo
            loss_meantea = torch.sum(F.cross_entropy(y1_t, y_t_tea_hard, reduce=False) * ~labels_t_mask) / torch.sum(~labels_t_mask) * args.trade_off_consis
            loss = loss_srcce + loss_ent  + loss_tgtce + loss_meantea 
        elif args.loss_mode=='srcce_tgtce_tgtmeantea':
            raise NotImplementedError
            loss = F.cross_entropy(y1_s, labels_s) + \
                    torch.sum(F.cross_entropy(y1_t, labels_t, reduce=False) * labels_t_mask) / torch.sum(labels_t_mask) * args.trade_off_pseudo + \
                        softmax_mse_loss(y1_t, y_t_tea) * args.trade_off_consis
        else:
            loss_srcce = F.cross_entropy(y1_s, labels_s)
            loss_ent = entropy(y1_t) * args.trade_off_entropy
            loss_tgtce = torch.sum(F.cross_entropy(y1_t, labels_t, reduce=False) * labels_t_mask) / torch.sum(labels_t_mask) * args.trade_off_pseudo
            loss = loss_srcce + loss_ent  + loss_tgtce 
            
        loss.backward()
        # optimizer_g.step()
        optimizer_f.step()
        
        losses.update(loss.item(), bs)
        losses_srcce.update(loss_srcce.item(), bs)
        losses_ent.update(loss_ent.item(), bs)
        losses_tgtce.update(loss_tgtce.item(), bs)
        if args.mean_tea:
            losses_meantea.update(loss_meantea.item(), bs)
        batch_time.update(time.time() - end)
        end = time.time()
        if i % args.print_freq == 0:
            progress.display(i)


def validate(val_loader: DataLoader, G: nn.Module, F1: ImageClassifierHead, attn_net:nn.Module, args: argparse.Namespace, F1_t, multihead_attn) -> Tuple[float, float]:
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

    # F2.eval()
    attn_net.eval()
    multihead_attn.eval()

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

    with torch.no_grad():
        end = time.time()
        for i, data in enumerate(val_loader):
            # if i>5:
            #     break
            
            # images, target = data[:2]
            # images = images.to(device)
            # target = target.to(device)
            
            # x, labels, neighbors, dist_neighbors = data
            # x, labels, neighbors, masks = data
            x, labels, neighbors = data
            x, labels = torch.stack(x), torch.stack(labels)
            x, labels = x.to(device), labels.to(device)
            bs = x.shape[0]
            x = x[:,:,2:]
            
            neighbor_idx_src = [neighbor.shape[0] for neighbor in neighbors]
            neighbor_idx_src = np.insert(np.cumsum(neighbor_idx_src),0,0)
            neighbors = torch.cat(neighbors)

            # compute output
            # _,_,g,_ = G(x)
            _,g,_,_,_,_,_ = G(x)
            # aux_feat = torch.stack([images[:,:,0],images[:,:,-1]], axis=2)

            # # feat_neighbor_list=[]
            # # for neighbor,dist in zip(neighbors, dist_neighbors):
            # #     neighbor = neighbor[:,:,2:].to(device)
            # #     dist = dist.to(device)
            # #     # dist = F.softmax(torch.pow(dist, 1/args.dist_root))
            # #     _,_,feat_neighbor,_ = G(neighbor)
            # #     if neighbor.shape[0]==1:
            # #         feat_neighbor = feat_neighbor.squeeze(0)
            # #     else:
            # #         attn = attn_net(dist.unsqueeze(2).transpose(0, 1))
            # #         attn = F.softmax(attn[-1], dim=0)
            # #         feat_neighbor = torch.sum(attn * feat_neighbor, dim=0)
            # #     feat_neighbor_list.append(feat_neighbor)
            # # feat_neighbor = torch.stack(feat_neighbor_list)
            # feat_neighbor_list=[]
            # for neighbor,mask in zip(neighbors, masks):
            #     neighbor = neighbor[:,:,2:].to(device)
            #     mask = mask.unsqueeze(2).to(device)
            #     neighbor = neighbor * mask
                                            
            #     n_iter = neighbor.shape[0]//bs
            #     tmp_list=[]
            #     for iter in range(n_iter):
            #         _,feat_neighbor,_,_,_,_,_ = G(neighbor[iter*bs:(iter+1)*bs])
            #         tmp_list.append(feat_neighbor)
            #     if neighbor.shape[0]%bs!=0:
            #         _,feat_neighbor,_,_,_,_,_ = G(neighbor[n_iter*bs:])
            #         tmp_list.append(feat_neighbor)

            #     feat_neighbor = torch.cat(tmp_list,dim=0)
            #     feat_neighbor = torch.mean(feat_neighbor, dim=0)
            #     feat_neighbor_list.append(feat_neighbor)
            #     del tmp_list
            # feat_neighbor = torch.stack(feat_neighbor_list)
            n_iter = neighbors.shape[0]//bs
            tmp_list=[]
            for iter in range(n_iter):
                neighbor = neighbors[iter*bs:(iter+1)*bs,:,2:].to(device)
                _,feat_neighbor,_,_,_,_,_ = G(neighbor)
                tmp_list.append(feat_neighbor.cpu())
            if neighbors.shape[0]%bs!=0:
                neighbor = neighbors[n_iter*bs:,:,2:].to(device)
                _,feat_neighbor,_,_,_,_,_ = G(neighbor)
                tmp_list.append(feat_neighbor.cpu())
            feat_neighbor = torch.cat(tmp_list,dim=0)
            
            # feat_neighbor_list=[]
            # for neighbor_idx,_ in enumerate(neighbor_idx_src):
            #     if neighbor_idx==neighbor_idx_src.shape[0]-1:
            #         break
            #     tmp_neighbor = feat_neighbor[neighbor_idx_src[neighbor_idx]:neighbor_idx_src[neighbor_idx+1]]
            #     tmp_neighbor = torch.mean(tmp_neighbor, dim=0).to(device)
            #     feat_neighbor_list.append(tmp_neighbor)
            # feat_neighbors = torch.stack(feat_neighbor_list) 
            
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
        
            # query = g
            # key = value = feat_neighbors
            # feat_neighbors, attn_output_weights = multihead_attn(query, key, value)
            # # g = attn_output   

            if 'cat' in args.cat_mode:
                g = torch.cat([g,feat_neighbors],dim=1)
            else:
                g = g + feat_neighbors
            aux_feat = None

            
            y1 = F1(g, aux_feat)
            # y1, y2 = F1(g, aux_feat), F2(g, aux_feat)

            # measure accuracy and record loss
            acc1, = accuracy(y1, labels)
            # acc2, = accuracy(y2, labels)
            if confmat:
                confmat.update(labels, y1.argmax(1))
            top1_1.update(acc1.item(), bs)
            # top1_2.update(acc2.item(), bs)
            
            
            _, preds1 = torch.max(y1, 1)
            # _, preds2 = torch.max(y2, 1)
            for t, p in zip(labels.view(-1), preds1.view(-1)):
                confusion_matrix1[t.long(), p.long()] += 1
            # for t, p in zip(labels.view(-1), preds2.view(-1)):
            #     confusion_matrix2[t.long(), p.long()] += 1

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)
        
        # if top1_1.avg > top1_2.avg:
        confusion_matrix = confusion_matrix1
        # else:
        #     confusion_matrix = confusion_matrix2
        print(str(confusion_matrix))
        per_class_acc = list((confusion_matrix.diag()/confusion_matrix.sum(1)).numpy())
        print('per class accuracy:')
        for idx,acc in enumerate(per_class_acc):
            print('\t '+str(idx_dict[idx])+': '+str(acc))

        print(' * Acc1 {top1_1.avg:.3f} Acc2 {top1_2.avg:.3f}'
              .format(top1_1=top1_1, top1_2=top1_2))
        if confmat:
            print(confmat.format(args.class_names))

    return top1_1.avg, 0.#, top1_2.avg




def get_pseudo_labels_by_threshold(val_loader: DataLoader, G: nn.Module, F1: ImageClassifierHead, args: argparse.Namespace):
    G.eval()
    F1.eval()

    THRESHOLD=args.pseudo_thres
    y_list=[]
    mask_list=[]
    with torch.no_grad():
        for i, data in enumerate(val_loader):
            images = data[0]
            images = images.to(device)
            _,g,_,_,_,_,_ = G(images)
            y = F1(g, None)
            
            max_prob,_ = torch.max(F.softmax(y),dim=1)
            mask = (max_prob >= THRESHOLD).tolist()
            y = torch.argmax(y,dim=1).tolist()
            
            y_list+=y
            mask_list+=mask
            
    return y_list,mask_list

def get_pseudo_labels_by_proportion(val_loader: DataLoader, G: nn.Module, F1: ImageClassifierHead, args: argparse.Namespace):
    G.eval()
    F1.eval()

    # THRESHOLD=args.pseudo_thres
    y_list=[]
    per_class_dict={0:[],1:[],2:[],3:[]}
    cnt = 0
    with torch.no_grad():
        for i, data in enumerate(val_loader):
            images = data[0]
            images = images.to(device)
            _,g,_,_,_,_,_ = G(images)
            y = F1(g, None)
            
            max_prob,_ = torch.max(F.softmax(y),dim=1)
            # mask = (max_prob >= THRESHOLD).tolist()
            y = torch.argmax(y,dim=1).tolist()
            
            cnt += images.shape[0]
            y_list+=y
            # mask_list+=mask
            
            max_prob = max_prob.tolist()
            # pseudo_class = pseudo_class.tolist()
            for idx,prob in enumerate(max_prob):
                per_class_dict[y[idx]].append((i*args.batch_size+idx, prob))
            
            
    mask_list = torch.zeros(cnt)
    for c in per_class_dict:
        tmp_prob = torch.tensor(per_class_dict[c])
        n_total = tmp_prob.shape[0]
        probs,indices = torch.topk(tmp_prob[:,1], k=int(n_total*args.pseudo_ratio))
        indices_ori = tmp_prob[:,0][indices].long()
        mask_list[indices_ori] = 1
    mask_list = mask_list.bool().tolist()
            
    return y_list,mask_list

def get_pseudo_labels_by_confidence_and_proportion(val_loader: DataLoader, G: nn.Module, F1: ImageClassifierHead, args: argparse.Namespace):
    top1 = AverageMeter('Acc_1', ':6.2f')
    
    nb_classes = 4
    confusion_matrix = torch.zeros(nb_classes, nb_classes) 
    label_dict = {"walk": 0, "bike": 1, "car": 2, "train": 3}
    idx_dict={}
    for k,v in label_dict.items():
        idx_dict[v]=k
        
    G.eval()
    F1.eval()
    # attn_net.eval()

    THRESHOLD=args.pseudo_thres
    y_list=[]
    mask_list_conf=[]
    per_class_dict={0:[],1:[],2:[],3:[]}
    cnt = 0
    with torch.no_grad():
        for i, data in enumerate(val_loader):
            images = data[0]
            labels = data[1]
            images,labels = images.to(device), labels.to(device)
            _,g,_,_,_,_,_ = G(images)
            y = F1(g, None)
            
            acc1, = accuracy(y, labels)
            top1.update(acc1.item(), images.shape[0])
            _, preds = torch.max(y, 1)
            for t, p in zip(labels.view(-1), preds.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1
                
            max_prob,_ = torch.max(F.softmax(y),dim=1)
            mask = (max_prob >= THRESHOLD).tolist()
            y = torch.argmax(y,dim=1).tolist()
            
            cnt += images.shape[0]
            y_list += y
            mask_list_conf += mask

            max_prob = max_prob.tolist()
            for idx,prob in enumerate(max_prob):
                per_class_dict[y[idx]].append((i*args.batch_size+idx, prob))
            
            
    mask_list_propor = torch.zeros(cnt)
    for c in per_class_dict:
        tmp_prob = torch.tensor(per_class_dict[c])
        n_total = tmp_prob.shape[0]
        probs,indices = torch.topk(tmp_prob[:,1], k=int(n_total*args.pseudo_ratio))
        indices_ori = tmp_prob[:,0][indices].long()
        mask_list_propor[indices_ori] = 1
    mask_list_propor = mask_list_propor.bool().tolist()
    # mask_list = mask_list_conf or mask_list_propor
    mask_list = list(map(operator.and_, mask_list_conf, mask_list_propor))
    print(sum(mask_list),sum(mask_list_conf),sum(mask_list_propor)) 
    # 95:8489, 90:10939, 85:12819
    # 66:12541
    # (85,66): 11433 12819 12541
    # (90,66): 9661 10939 12541
    # (95,66): 7394 8489 12541

    print(str(confusion_matrix))
    per_class_acc = list((confusion_matrix.diag()/confusion_matrix.sum(1)).numpy())
    print('per class accuracy:')
    for idx,acc in enumerate(per_class_acc):
        print('\t '+str(idx_dict[idx])+': '+str(acc))
    print(' * Acc1 {top1.avg:.3f}'.format(top1=top1))
    
    return y_list, mask_list



def enable_dropout(model):
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()
                
def get_pseudo_labels_by_entropyproportion(val_loader: DataLoader, G: nn.Module, F1: ImageClassifierHead, args: argparse.Namespace):
    top1 = AverageMeter('Acc_1', ':6.2f')
    
    nb_classes = 4
    confusion_matrix = torch.zeros(nb_classes, nb_classes) 
    label_dict = {"walk": 0, "bike": 1, "car": 2, "train": 3}
    idx_dict={}
    for k,v in label_dict.items():
        idx_dict[v]=k
        
    G.eval()
    F1.eval()
    # F1.train()
    # attn_net.eval()
    
    cnt = 0
    # THRESHOLD=args.pseudo_thres
    y_list=[]
    # mask_list_conf=[]
    with torch.no_grad():
        for i, data in enumerate(val_loader):
            images = data[0]
            labels = data[1]
            images,labels = images.to(device), labels.to(device)
            _,g,_,_,_,_,_ = G(images)
            y = F1(g, None)
            y = F.softmax(y,dim=1)

            acc1, = accuracy(y, labels)
            top1.update(acc1.item(), images.shape[0])
            max_prob, preds = torch.max(y, 1)
            for t, p in zip(labels.view(-1), preds.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1
            
            # mask = (max_prob >= THRESHOLD).tolist()
            y = torch.argmax(y,dim=1).tolist()
            y_list += y
            cnt += images.shape[0]
            # mask_list_conf += mask
            
    print(str(confusion_matrix))
    per_class_acc = list((confusion_matrix.diag()/confusion_matrix.sum(1)).numpy())
    print('per class accuracy:')
    for idx,acc in enumerate(per_class_acc):
        print('\t '+str(idx_dict[idx])+': '+str(acc))
    print(' * Acc1 {top1.avg:.3f}'.format(top1=top1))
    
    
    
    # cnt = 0
    # enable_dropout(F1)
    # per_class_dict={0:[],1:[],2:[],3:[]}
    # with torch.no_grad():
    #     for i, data in enumerate(val_loader):
    #         images = data[0]
    #         labels = data[1]
    #         images,labels = images.to(device), labels.to(device)
    #         _,g,_,_,_,_,_ = G(images)
            
    #         all_y_list=[]
    #         for idx in range(5):
    #             y = F1(g, None)
    #             all_y_list.append(F.softmax(y,dim=1))
            
    #         cnt += images.shape[0]
            
    #         preds = torch.stack(all_y_list, dim=0)
    #         preds = torch.mean(preds, dim=0) 
    #         entropy = -1.0*torch.mean(preds*torch.log(preds + 1e-6), dim=1) #, keepdim=True) 
    #         entropy = entropy.tolist()
    #         for idx,ent in enumerate(entropy):
    #             per_class_dict[y_list[i*args.batch_size+idx]].append((i*args.batch_size+idx, ent))
    
    # # pdb.set_trace()
    # # np.save("/home/yichen/Transfer-Learning-Library/examples/domain_adaptation/image_classification/logs/entropy_per_class_dict.npy", per_class_dict)
    
    cnt = 18833
    per_class_dict = np.load("/home/yichen/Transfer-Learning-Library/examples/domain_adaptation/image_classification/logs/entropy_per_class_dict.npy", allow_pickle=True).item()
            
    mask_list_propor = torch.zeros(cnt)
    for c in per_class_dict:
        tmp_prob = torch.tensor(per_class_dict[c])
        n_total = tmp_prob.shape[0]
        probs,indices = torch.topk(-tmp_prob[:,1], k=int(n_total*args.pseudo_ratio))
        indices_ori = tmp_prob[:,0][indices].long()
        mask_list_propor[indices_ori] = 1
    mask_list_propor = mask_list_propor.bool().tolist()
    
    # mask_list = list(map(operator.and_, mask_list_conf, mask_list_propor))
    mask_list = mask_list_propor
    print(sum(mask_list))#,sum(mask_list_conf),sum(mask_list_propor)) 

    cnt=0
    F1.eval()
    top1 = AverageMeter('Acc_1', ':6.2f')
    confusion_matrix = torch.zeros(nb_classes, nb_classes) 
    with torch.no_grad():
        for i, data in enumerate(val_loader):
            images = data[0]
            labels = data[1]
            images,labels = images.to(device), labels.to(device)
            _,g,_,_,_,_,_ = G(images)
            y = F1(g, None)
            y = F.softmax(y,dim=1)

            select_idx = torch.nonzero(torch.tensor(mask_list[cnt:cnt+images.shape[0]])).squeeze(1)
            if len(select_idx)>0:
                y = y[select_idx]
                labels = labels[select_idx]
                
                acc1, = accuracy(y, labels)
                top1.update(acc1.item(), len(select_idx))
                _, preds = torch.max(y, 1)
                for t, p in zip(labels.view(-1), preds.view(-1)):
                    confusion_matrix[t.long(), p.long()] += 1
            
            cnt += images.shape[0]
            
    print('-------------------------------')
    print(str(confusion_matrix))
    per_class_acc = list((confusion_matrix.diag()/confusion_matrix.sum(1)).numpy())
    print('per class accuracy after filtering:')
    for idx,acc in enumerate(per_class_acc):
        print('\t '+str(idx_dict[idx])+': '+str(acc))
    print(' * Acc1 {top1.avg:.3f}'.format(top1=top1))
    print('-------------------------------')
    
    return y_list, mask_list

def get_pseudo_labels_by_confidence_and_entropyproportion(val_loader: DataLoader, G: nn.Module, F1: ImageClassifierHead, args: argparse.Namespace):
    top1 = AverageMeter('Acc_1', ':6.2f')
    
    nb_classes = 4
    confusion_matrix = torch.zeros(nb_classes, nb_classes) 
    label_dict = {"walk": 0, "bike": 1, "car": 2, "train": 3}
    idx_dict={}
    for k,v in label_dict.items():
        idx_dict[v]=k
        
    G.eval()
    F1.eval()
    # F1.train()
    # attn_net.eval()
    
    cnt = 0
    THRESHOLD=args.pseudo_thres
    y_list=[]
    mask_list_conf=[]
    with torch.no_grad():
        for i, data in enumerate(val_loader):
            images = data[0]
            labels = data[1]
            images,labels = images.to(device), labels.to(device)
            _,g,_,_,_,_,_ = G(images)
            y = F1(g, None)
            y = F.softmax(y,dim=1)

            acc1, = accuracy(y, labels)
            top1.update(acc1.item(), images.shape[0])
            max_prob, preds = torch.max(y, 1)
            for t, p in zip(labels.view(-1), preds.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1
            
            mask = (max_prob >= THRESHOLD).tolist()
            y = torch.argmax(y,dim=1).tolist()
            y_list += y
            mask_list_conf += mask
            cnt += images.shape[0]
                
    # print(str(confusion_matrix))
    # per_class_acc = list((confusion_matrix.diag()/confusion_matrix.sum(1)).numpy())
    # print('per class accuracy:')
    # for idx,acc in enumerate(per_class_acc):
    #     print('\t '+str(idx_dict[idx])+': '+str(acc))
    # print(' * Acc1 {top1.avg:.3f}'.format(top1=top1))
    
    
    
    # cnt = 0
    # enable_dropout(F1)
    # per_class_dict={0:[],1:[],2:[],3:[]}
    # with torch.no_grad():
    #     for i, data in enumerate(val_loader):
    #         images = data[0]
    #         labels = data[1]
    #         images,labels = images.to(device), labels.to(device)
    #         _,g,_,_,_,_,_ = G(images)
            
    #         all_y_list=[]
    #         for idx in range(5):
    #             y = F1(g, None)
    #             all_y_list.append(F.softmax(y,dim=1))
    #         # y=all_y_list[0]
            
    #         # acc1, = accuracy(y, labels)
    #         # top1.update(acc1.item(), images.shape[0])
    #         # _, preds = torch.max(y, 1)
    #         # for t, p in zip(labels.view(-1), preds.view(-1)):
    #         #     confusion_matrix[t.long(), p.long()] += 1
                
    #         # max_prob,_ = torch.max(y,dim=1)
    #         # mask = (max_prob >= THRESHOLD).tolist()
    #         # y = torch.argmax(y,dim=1).tolist()
            
    #         cnt += images.shape[0]
    #         # y_list += y
    #         # mask_list_conf += mask
            
    #         preds = torch.stack(all_y_list, dim=0)
    #         preds = torch.mean(preds, dim=0) 
    #         entropy = -1.0*torch.mean(preds*torch.log(preds + 1e-6), dim=1) #, keepdim=True) 
    #         entropy = entropy.tolist()
    #         for idx,ent in enumerate(entropy):
    #             per_class_dict[y_list[i*args.batch_size+idx]].append((i*args.batch_size+idx, ent))
      
            
    cnt = 18833
    per_class_dict = np.load("/home/yichen/Transfer-Learning-Library/examples/domain_adaptation/image_classification/logs/entropy_per_class_dict.npy", allow_pickle=True).item()
        
        
    mask_list_propor = torch.zeros(cnt)
    for c in per_class_dict:
        tmp_prob = torch.tensor(per_class_dict[c])
        n_total = tmp_prob.shape[0]
        probs,indices = torch.topk(-tmp_prob[:,1], k=int(n_total*args.pseudo_ratio))
        indices_ori = tmp_prob[:,0][indices].long()
        mask_list_propor[indices_ori] = 1
    mask_list_propor = mask_list_propor.bool().tolist()
    
    # mask_list = mask_list_conf or mask_list_propor
    mask_list = list(map(operator.and_, mask_list_conf, mask_list_propor))
    print(sum(mask_list),sum(mask_list_conf),sum(mask_list_propor)) 
    # (85,66): 11433 12819 12541
    # (90,66): 9661 10939 12541
    # (95,66): 7394 8489 12541

    # print(str(confusion_matrix))
    # per_class_acc = list((confusion_matrix.diag()/confusion_matrix.sum(1)).numpy())
    # print('per class accuracy:')
    # for idx,acc in enumerate(per_class_acc):
    #     print('\t '+str(idx_dict[idx])+': '+str(acc))
    # print(' * Acc1 {top1.avg:.3f}'.format(top1=top1))
    
    cnt=0
    F1.eval()
    top1 = AverageMeter('Acc_1', ':6.2f')
    confusion_matrix = torch.zeros(nb_classes, nb_classes) 
    with torch.no_grad():
        for i, data in enumerate(val_loader):
            images = data[0]
            labels = data[1]
            images,labels = images.to(device), labels.to(device)
            _,g,_,_,_,_,_ = G(images)
            y = F1(g, None)
            y = F.softmax(y,dim=1)

            select_idx = torch.nonzero(torch.tensor(mask_list[cnt:cnt+images.shape[0]])).squeeze(1)
            if len(select_idx)>0:
                y = y[select_idx]
                labels = labels[select_idx]
                
                acc1, = accuracy(y, labels)
                top1.update(acc1.item(), len(select_idx))
                _, preds = torch.max(y, 1)
                for t, p in zip(labels.view(-1), preds.view(-1)):
                    confusion_matrix[t.long(), p.long()] += 1
            
            cnt += images.shape[0]
            
    print('-------------------------------')
    print(str(confusion_matrix))
    per_class_acc = list((confusion_matrix.diag()/confusion_matrix.sum(1)).numpy())
    print('per class accuracy after filtering:')
    for idx,acc in enumerate(per_class_acc):
        print('\t '+str(idx_dict[idx])+': '+str(acc))
    print(' * Acc1 {top1.avg:.3f}'.format(top1=top1))
    print('-------------------------------')
    
    return y_list, mask_list





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


    args = parser.parse_args()
    torch.set_num_threads(8)

    main(args)
