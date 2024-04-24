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

from transformers.models.bert.modeling_bert import BertEncoder, BertPooler, BertEmbeddings
from transformers import AutoConfig,AutoModel
from transformers import EncoderDecoderModel, BertTokenizer

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


    
    G = models.TSEncoder_new().to(device)
    classifier_features_dim=64
    num_classes = 4
    
    ckpt_dir='/home/yichen/TS2Vec/result/0402_pretrain/model_best.pth.tar'
    # ckpt_dir='/home/yichen/Transfer-Learning-Library/examples/domain_adaptation/image_classification/logs/mcd_1015/checkpoints/best.pth'
    ckpt = torch.load(ckpt_dir, map_location='cuda:0')
    G.load_state_dict(ckpt['encoder_tgt_state_dict'], strict=False)
    # G.load_state_dict(ckpt['G'], strict=False)

    if 'cat' in args.nbr_mode:
        if args.nbr_label_mode == 'separate_input':
            input_dim=64*2+args.nbr_label_embed_dim
        else:
            input_dim=64+args.bert_out_dim
        F1 = models.Classifier_clf(input_dim=input_dim).to(device)
        F2 = models.Classifier_clf(input_dim=input_dim).to(device)
    elif 'add' in args.nbr_mode or 'bertonly' in args.nbr_mode:
        F1 = models.Classifier_clf(input_dim=64).to(device)
        F2 = models.Classifier_clf(input_dim=64).to(device)
    else:
        raise NotImplementedError
    attn_net = models.AttnNet().to(device)
    dim_converter = models.DimConverter(input_dim=768,out_dim=args.bert_out_dim).to(device)
    
    multihead_attn = nn.MultiheadAttention(64, num_heads=args.num_head, batch_first=True).to(device)
    nbr_label_encoder = models.LabelEncoder(input_dim=4, embed_dim=args.nbr_label_embed_dim).to(device)
    
    
    encoder_config = AutoConfig.from_pretrained("bert-base-uncased")#, force_download=True)
    enc_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")#, force_download=True)
    bert_model = AutoModel.from_pretrained("bert-base-uncased", config=encoder_config).to(device)#, force_download=True).to(device)    
    bert_learnable_tokens = torch.nn.Parameter(torch.rand([args.token_len,768]), requires_grad=True)
    for param in bert_model.parameters():
        param.requires_grad = False 
    
    train_source_iter, train_target_iter, val_loader,_,_ = utils.load_data_neighbor_v3(args, enc_tokenizer=enc_tokenizer)

    # define optimizer
    optimizer_g = Adam([
        {"params": G.parameters()},
        {"params": dim_converter.parameters()},
        {"params": bert_learnable_tokens}
    ], args.lr, weight_decay=args.weight_decay, betas=(0.5, 0.99))
    optimizer_f = Adam([
        {"params": F1.parameters()},
        {"params": F2.parameters()},
        {"params": multihead_attn.parameters()},
        {"params": attn_net.parameters()},
        {"params": nbr_label_encoder.parameters()}
    ], args.lr, weight_decay=args.weight_decay, betas=(0.5, 0.99))
    # optimizer_f = SGD([
    #     {"params": F1.parameters()},
    #     {"params": F2.parameters()},
    #     {"params": multihead_attn.parameters()},
    #     {"params": attn_net.parameters()},
    #     {"params": nbr_label_encoder.parameters()},
    #     {"params": dim_converter.parameters()},
    #     {"params": bert_learnable_tokens}
    # ], args.lr, weight_decay=args.weight_decay, momentum=0.9)
    # optimizer_others = Adam([
    #     {"params": multihead_attn.parameters()},
    #     {"params": attn_net.parameters()},
    #     {"params": nbr_label_encoder.parameters()},
    #     {"params": dim_converter.parameters()},
    #     {"params": bert_learnable_tokens}
    # ], args.lr, weight_decay=args.weight_decay, betas=(0.5, 0.99))
    bert_learnable_tokens = bert_learnable_tokens.to(device)


    # start training
    best_acc1 = 0.
    best_results = None
    best_epoch = 0
    for epoch in range(args.epochs):
        # train for one epoch
        train(train_source_iter, train_target_iter, G, F1, F2, attn_net, optimizer_g, optimizer_f, epoch, args, multihead_attn, \
            nbr_label_encoder, bert_model, dim_converter, bert_learnable_tokens, enc_tokenizer)

        # evaluate on validation set
        results = validate(val_loader, G, F1, F2, attn_net, args, multihead_attn, \
            nbr_label_encoder, bert_model, dim_converter, bert_learnable_tokens, enc_tokenizer)

        # remember best acc@1 and save checkpoint
        torch.save({
            'G': G.state_dict(),
            'F1': F1.state_dict(),
            'F2': F2.state_dict(),
            'multihead_attn': multihead_attn.state_dict(),
            'attnnet':attn_net.state_dict(),
            'nbr_label_encoder':nbr_label_encoder.state_dict(),
            'bert_encoder':bert_model.state_dict(),
            'dim_converter':dim_converter.state_dict(),
            'bert_learnable_tokens':bert_learnable_tokens
        }, logger.get_checkpoint_path('latest'))
        if max(results) > best_acc1:
            shutil.copy(logger.get_checkpoint_path('latest'), logger.get_checkpoint_path('best'))
            best_acc1 = max(results)
            best_results = results
            best_epoch = epoch
        print("best_acc1 = {:3.1f} ({}), results = {}".format(best_acc1, best_epoch, best_results))


    # # evaluate on test set
    # checkpoint = torch.load(logger.get_checkpoint_path('best'), map_location='cpu')
    # G.load_state_dict(checkpoint['G'])
    # F1.load_state_dict(checkpoint['F1'])
    # F2.load_state_dict(checkpoint['F2'])
    # # results = validate(test_loader, G, F1, F2, args)
    # # print("test_acc1 = {:3.1f}".format(max(results)))

    logger.close()


def train(train_src_iter: ForeverDataIterator, train_tgt_iter: ForeverDataIterator,
          G: nn.Module, F1: ImageClassifierHead, F2: ImageClassifierHead, attn_net: nn.Module,
          optimizer_g: SGD, optimizer_f: SGD, epoch: int, args: argparse.Namespace, multihead_attn, nbr_label_encoder, bert_model, dim_converter, bert_learnable_tokens, enc_tokenizer):
    batch_time = AverageMeter('Time', ':3.1f')
    data_time = AverageMeter('Data', ':3.1f')
    losses = AverageMeter('Loss', ':3.2f')
    trans_losses = AverageMeter('Trans Loss', ':3.2f')
    cls_accs = AverageMeter('Cls Acc', ':3.1f')

    progress = ProgressMeter(
        args.iters_per_epoch,
        [batch_time, data_time, losses, trans_losses, cls_accs],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    G.train()
    F1.train()
    F2.train()
    dim_converter.train()
    bert_model.eval()
    attn_net.train()
    multihead_attn.train()
    nbr_label_encoder.train()

    end = time.time()
    for i in range(args.iters_per_epoch):
        # if i>5:
        #     break

        x_ori_src, labels_s, x_ori_src_neighbor, src_neighbor_labels, labels_domain_src, bert_input_src  = next(train_src_iter) 
        x_ori_tgt, labels_t, labels_t_mask, x_ori_tgt_neighbor, tgt_neighbor_labels, labels_domain_tgt, idx_tgt, bert_input_tgt = next(train_tgt_iter)
        if tgt_neighbor_labels[0] is not None:
            src_neighbor_labels, tgt_neighbor_labels = torch.stack(src_neighbor_labels).to(device), torch.stack(tgt_neighbor_labels).to(device)

        x_ori_src, labels_s, labels_domain_src = torch.stack(x_ori_src), torch.stack(labels_s), torch.stack(labels_domain_src)
        x_ori_tgt, idx_tgt, labels_domain_tgt = torch.stack(x_ori_tgt), torch.stack(idx_tgt), torch.stack(labels_domain_tgt)
        x_ori_src, x_ori_tgt = x_ori_src[:,:,2:], x_ori_tgt[:,:,2:] # time, dist, v, a, jerk, bearing, is_real
        x_ori_src, x_ori_tgt, labels_s, idx_tgt = x_ori_src.to(device), x_ori_tgt.to(device), labels_s.to(device), idx_tgt.to(device)
        
        neighbor_idx_src = [neighbor.shape[0] for neighbor in x_ori_src_neighbor]
        neighbor_idx_src = np.insert(np.cumsum(neighbor_idx_src),0,0)
        x_ori_src_neighbor = torch.cat(x_ori_src_neighbor)
        
        neighbor_idx_tgt = [neighbor.shape[0] for neighbor in x_ori_tgt_neighbor]
        neighbor_idx_tgt = np.insert(np.cumsum(neighbor_idx_tgt),0,0)
        x_ori_tgt_neighbor = torch.cat(x_ori_tgt_neighbor)

        # measure data loading time
        data_time.update(time.time() - end)
        bs=x_ori_src.shape[0]
        
        # if "perpt" not in args.nbr_mode:
        #     with torch.no_grad():
        #         n_iter = x_ori_src_neighbor.shape[0]//bs
        #         tmp_list=[]
        #         for iter in range(n_iter):
        #             neighbor = x_ori_src_neighbor[iter*bs:(iter+1)*bs,:,2:].to(device)
        #             _,feat_src_avgpool_neighbor,_,_,_,_,_ = G(neighbor)
        #             tmp_list.append(feat_src_avgpool_neighbor.cpu())
        #         if x_ori_src_neighbor.shape[0]%bs!=0:
        #             neighbor = x_ori_src_neighbor[n_iter*bs:,:,2:].to(device)
        #             _,feat_src_avgpool_neighbor,_,_,_,_,_ = G(neighbor)
        #             tmp_list.append(feat_src_avgpool_neighbor.cpu())
        #         feat_src_avgpool_neighbors = torch.cat(tmp_list,dim=0)
                        
        #         n_iter = x_ori_tgt_neighbor.shape[0]//bs
        #         tmp_list=[]
        #         for iter in range(n_iter):
        #             neighbor = x_ori_tgt_neighbor[iter*bs:(iter+1)*bs,:,2:].to(device)
        #             _,feat_tgt_avgpool_neighbor,_,_,_,_,_ = G(neighbor)
        #             # _,_,feat_tgt_avgpool_neighbor,_,_,_,_ = G(neighbor)
        #             # feat_tgt_avgpool_neighbor = F.normalize(feat_tgt_avgpool_neighbor, dim=0)
        #             # mask = neighbor[:,:,0]!=0
        #             # feat_tgt_avgpool_neighbor = torch.sum(feat_tgt_avgpool_neighbor,dim=1) / (torch.sum(mask,dim=1,keepdim=True)+1)
        #             tmp_list.append(feat_tgt_avgpool_neighbor.cpu())
        #         if x_ori_tgt_neighbor.shape[0]%bs!=0:
        #             neighbor = x_ori_tgt_neighbor[n_iter*bs:,:,2:].to(device)
        #             _,feat_tgt_avgpool_neighbor,_,_,_,_,_ = G(neighbor)
        #             # _,_,feat_tgt_avgpool_neighbor,_,_,_,_ = G(neighbor)
        #             # feat_tgt_avgpool_neighbor = F.normalize(feat_tgt_avgpool_neighbor, dim=0)
        #             # mask = neighbor[:,:,0]!=0
        #             # feat_tgt_avgpool_neighbor = torch.sum(feat_tgt_avgpool_neighbor,dim=1) / (torch.sum(mask,dim=1,keepdim=True)+1)
        #             tmp_list.append(feat_tgt_avgpool_neighbor.cpu())
        #         feat_tgt_avgpool_neighbors = torch.cat(tmp_list,dim=0)
        # else:
        #     assert args.nbr_data_mode=='mergemin5'
        #     mask_list=[]
        #     n_iter = x_ori_src_neighbor.shape[0]//bs
        #     mask_list=[]
        #     tmp_list=[]
        #     for iter in range(n_iter):
        #         neighbor = x_ori_src_neighbor[iter*bs:(iter+1)*bs,:,2:].to(device)
        #         _,_,feat_src_avgpool_neighbor,_,_,_,_ = G(neighbor) #32,650,64
        #         mask = neighbor[:,:,0]!=0
        #         mask_list.append(mask)
        #         tmp_list.append(feat_src_avgpool_neighbor)
        #     if x_ori_src_neighbor.shape[0]%bs!=0:
        #         neighbor = x_ori_src_neighbor[n_iter*bs:,:,2:].to(device)
        #         _,_,feat_src_avgpool_neighbor,_,_,_,_ = G(neighbor)
        #         mask = neighbor[:,:,0]!=0 #32,650
        #         mask_list.append(mask)
        #         tmp_list.append(feat_src_avgpool_neighbor)
        #     # feat_src_avgpool_neighbors = tmp_list/mask_cnt #650,64
        #     feat_src_avgpool_neighbors = torch.cat(tmp_list,dim=0)
        #     feat_src_avgpool_neighbors_mask = torch.cat(mask_list,dim=0)

        #     n_iter = x_ori_tgt_neighbor.shape[0]//bs
        #     mask_list=[]
        #     tmp_list=[]
        #     for iter in range(n_iter):
        #         neighbor = x_ori_tgt_neighbor[iter*bs:(iter+1)*bs,:,2:].to(device)
        #         _,_,feat_tgt_avgpool_neighbor,_,_,_,_ = G(neighbor) #32,650,64
        #         # feat_tgt_avgpool_neighbor = F.normalize(feat_tgt_avgpool_neighbor, dim=0) 
        #         mask = neighbor[:,:,0]!=0
        #         mask_list.append(mask)
        #         tmp_list.append(feat_tgt_avgpool_neighbor)
        #     if x_ori_tgt_neighbor.shape[0]%bs!=0:
        #         neighbor = x_ori_tgt_neighbor[n_iter*bs:,:,2:].to(device)
        #         _,_,feat_tgt_avgpool_neighbor,_,_,_,_ = G(neighbor)
        #         mask = neighbor[:,:,0]!=0 #32,650
        #         mask_list.append(mask)
        #         tmp_list.append(feat_tgt_avgpool_neighbor)
        #     # feat_tgt_avgpool_neighbors = tmp_list/mask_cnt #650,64
        #     feat_tgt_avgpool_neighbors = torch.cat(tmp_list,dim=0)
        #     feat_tgt_avgpool_neighbors_mask = torch.cat(mask_list,dim=0)

               
        # if 'qkv' or 'perpt' in args.nbr_mode:
        #     1
        # else:
        #     feat_src_avgpool_neighbor_list=[]
        #     for neighbor_idx,_ in enumerate(neighbor_idx_src):
        #         if neighbor_idx==neighbor_idx_src.shape[0]-1:
        #             break
        #         tmp_neighbor = feat_src_avgpool_neighbor[neighbor_idx_src[neighbor_idx]:neighbor_idx_src[neighbor_idx+1]]
        #         tmp_neighbor = torch.mean(tmp_neighbor, dim=0).to(device)
        #         feat_src_avgpool_neighbor_list.append(tmp_neighbor)
        #     feat_src_avgpool_neighbors = torch.stack(feat_src_avgpool_neighbor_list)

        #     feat_tgt_avgpool_neighbor_list=[]
        #     for neighbor_idx,_ in enumerate(neighbor_idx_tgt):
        #         if neighbor_idx==neighbor_idx_tgt.shape[0]-1:
        #             break
        #         tmp_neighbor = feat_tgt_avgpool_neighbor[neighbor_idx_tgt[neighbor_idx]:neighbor_idx_tgt[neighbor_idx+1]]
        #         tmp_neighbor = torch.mean(tmp_neighbor, dim=0).to(device)
        #         feat_tgt_avgpool_neighbor_list.append(tmp_neighbor)
        #     feat_tgt_avgpool_neighbors = torch.stack(feat_tgt_avgpool_neighbor_list)
        
 
        optimizer_g.zero_grad()
        optimizer_f.zero_grad()


        aux_feat=None
        if 'bertonly' in args.nbr_mode:
            bert_input_src, bert_input_tgt = torch.stack(bert_input_src).to(device), torch.stack(bert_input_tgt).to(device)
            bert_input = torch.cat([bert_input_src, bert_input_tgt],dim=0) #128,60,768
            bert_embedding = bert_model.embeddings(bert_input)
            if 'learnable' in args.nbr_mode:
                bert_embedding = torch.cat([bert_learnable_tokens.unsqueeze(0).tile(args.batch_size*2,1,1) ,bert_embedding],dim=1) #128,10,768 vs #128,60,768
            if 'cross_sim' in args.nbr_mode:
                raise NotImplementedError
                bert_input_class = ['A trajectory of person','A trajectory of bike','A trajectory of car','A trajectory of public transport']
            bert_feature = bert_model.encoder(bert_embedding) #128.70.768
            bert_feature = bert_model.pooler(bert_feature[0]) #128,768
            bert_feature = dim_converter(bert_feature)

            y_1 = F1(bert_feature , aux_feat)
            y_2 = F2(bert_feature , aux_feat)
            
        else:
            x = torch.cat((x_ori_src, x_ori_tgt), dim=0)
            if 'perpt' in args.nbr_mode:
                _,_,g,_,_,_,_ = G(x, args, mask_early=args.mask_early, mask_late=args.mask_late)
            else:
                _,g,_,_,_,_,_ = G(x, args, mask_early=args.mask_early, mask_late=args.mask_late)
            if args.mask_late:
                g,g_list = g
            g_s,g_t = g[:bs],g[bs:]

            if 'bert' in args.nbr_mode:
                bert_input_src, bert_input_tgt = torch.stack(bert_input_src).to(device), torch.stack(bert_input_tgt).to(device)
                bert_input = torch.cat([bert_input_src, bert_input_tgt],dim=0) #128,60,768
                bert_embedding = bert_model.embeddings(bert_input)
                if 'learnable' in args.nbr_mode:
                    bert_embedding = torch.cat([bert_learnable_tokens.unsqueeze(0).tile(args.batch_size*2,1,1) ,bert_embedding],dim=1) #128,10,768 vs #128,60,768
                
                if 'crosssim' in args.nbr_mode:
                    raise NotImplementedError
                    with torch.no_grad():
                        bert_word_class = ['A trajectory of person','A trajectory of bike','A trajectory of car','A trajectory of public transport']
                        bert_input_class = []
                        for sentence in bert_word_class:
                            bert_input_class.append(torch.as_tensor(enc_tokenizer(sentence, max_length = args.token_max_len, truncation = True, padding = "max_length")['input_ids']))
                        bert_input_class = torch.stack(bert_input_class).to(device)
                        bert_embedding_class = bert_model.embeddings(bert_input_class)
                    bert_embedding = torch.cat([bert_embedding, bert_embedding_class],dim=0) #128,60,768 vs 4,60,768
                    
                bert_feature = bert_model.encoder(bert_embedding) #128.70.768
                bert_feature = bert_model.pooler(bert_feature[0]) #128,768
                bert_feature = dim_converter(bert_feature)
                
                if 'crosssim' in args.nbr_mode:
                    bert_feature_class = bert_feature[-4:]
                    bert_feature = bert_feature[:-4]
                bert_feature_s,bert_feature_t = bert_feature[:bs],bert_feature[bs:]
                
                if 'cat' in args.nbr_mode:
                    g_s = torch.cat([g_s,bert_feature_s],dim=1)
                    g_t = torch.cat([g_t,bert_feature_t],dim=1)
                else:
                    g_s = g_s + bert_feature_s
                    g_t = g_t + bert_feature_t
                g = torch.cat((g_s, g_t), dim=0)
                
                if 'crosssim' in args.nbr_mode:
                    pdb.set_trace()
                    y_1 = torch.matmul(g,bert_feature_class.T) # 128,64 x 64,4 -> 128,4
                    y_1=y_2=None
                else:
                    y_1 = F1(g, aux_feat)
                    y_2 = F2(g, aux_feat)
                
            else:
                
                if args.nbr_mode=='qkv_cat':
                    g_s = g_s.unsqueeze(1)
                    g_t = g_t.unsqueeze(1)
                    
                    feat_src_avgpool_neighbor_list=[]
                    for neighbor_idx,_ in enumerate(neighbor_idx_src):
                        if neighbor_idx==neighbor_idx_src.shape[0]-1:
                            break
                        key = value = feat_src_avgpool_neighbors[neighbor_idx_src[neighbor_idx]:neighbor_idx_src[neighbor_idx+1]].to(device)
                        query = g_s[neighbor_idx] # might change to dist?
                        attn_output_neighbors_s, attn_output_weights_s = multihead_attn(query, key, value)
                        feat_src_avgpool_neighbor_list.append(attn_output_neighbors_s)
                    feat_src_avgpool_neighbors = torch.stack(feat_src_avgpool_neighbor_list).squeeze(1)
                        
                    feat_tgt_avgpool_neighbor_list=[]
                    for neighbor_idx,_ in enumerate(neighbor_idx_tgt):
                        if neighbor_idx==neighbor_idx_tgt.shape[0]-1:
                            break
                        key = value = feat_tgt_avgpool_neighbors[neighbor_idx_tgt[neighbor_idx]:neighbor_idx_tgt[neighbor_idx+1]].to(device)
                        query = g_t[neighbor_idx]
                        attn_output_neighbors_t, attn_output_weights_t = multihead_attn(query, key, value)
                        feat_tgt_avgpool_neighbor_list.append(attn_output_neighbors_t)
                    feat_tgt_avgpool_neighbors = torch.stack(feat_tgt_avgpool_neighbor_list).squeeze(1)
                    
                    g_s = g_s.squeeze(1)
                    g_t = g_t.squeeze(1)
                    
                    g_s = torch.cat([g_s,feat_src_avgpool_neighbors],dim=1)
                    g_t = torch.cat([g_t,feat_tgt_avgpool_neighbors],dim=1)
                    
                    g = torch.cat((g_s, g_t), dim=0)
                    y_1 = F1(g, aux_feat)
                    y_2 = F2(g, aux_feat)
                
                elif args.nbr_mode=='qkv_individual':
                    g_s = g_s.unsqueeze(1)
                    g_t = g_t.unsqueeze(1)
                    
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
                    y_2 = F2(g, aux_feat)
                
                else:
                    raise NotImplementedError
                    if 'cat' in args.cat_mode:
                        g_s = torch.cat([g_s,feat_src_avgpool_neighbors],dim=1)
                        g_t = torch.cat([g_t,feat_tgt_avgpool_neighbors],dim=1)
                    else:
                        g_s = g_s + feat_src_avgpool_neighbors
                        g_t = g_t + feat_tgt_avgpool_neighbors
                        
                    g = torch.cat((g_s, g_t), dim=0)
                    
                    y_1 = F1(g, aux_feat)
                    y_2 = F2(g, aux_feat)
                
                
        y1_s, y1_t = y_1.chunk(2, dim=0)
        y2_s, y2_t = y_2.chunk(2, dim=0)
        y1_t, y2_t = F.softmax(y1_t, dim=1), F.softmax(y2_t, dim=1)

        if 'sourceonly' in args.steps_list:
            loss = F.cross_entropy(y1_s, labels_s) + F.cross_entropy(y2_s, labels_s) 
            losses.update(loss.item(), bs)
        else:
            loss = F.cross_entropy(y1_s, labels_s) + F.cross_entropy(y2_s, labels_s) + \
                 (entropy(y1_t) + entropy(y2_t)) * args.trade_off_entropy

        loss.backward()
        optimizer_g.step()
        optimizer_f.step()
        



        # Step B train classifier to maximize discrepancy
        if 'B' in args.steps_list:
            # optimizer_g.zero_grad()
            optimizer_f.zero_grad()
            
            # if "perpt" in args.nbr_mode:
            #     with torch.no_grad():
            #         mask_list=[]
            #         n_iter = x_ori_src_neighbor.shape[0]//bs
            #         mask_list=[]
            #         tmp_list=[]
            #         for iter in range(n_iter):
            #             neighbor = x_ori_src_neighbor[iter*bs:(iter+1)*bs,:,2:].to(device)
            #             _,_,feat_src_avgpool_neighbor,_,_,_,_ = G(neighbor) #32,650,64
            #             mask = neighbor[:,:,0]!=0
            #             mask_list.append(mask)
            #             tmp_list.append(feat_src_avgpool_neighbor)
            #         if x_ori_src_neighbor.shape[0]%bs!=0:
            #             neighbor = x_ori_src_neighbor[n_iter*bs:,:,2:].to(device)
            #             _,_,feat_src_avgpool_neighbor,_,_,_,_ = G(neighbor)
            #             mask = neighbor[:,:,0]!=0 #32,650
            #             mask_list.append(mask)
            #             tmp_list.append(feat_src_avgpool_neighbor)
            #         feat_src_avgpool_neighbors = torch.cat(tmp_list,dim=0)
            #         feat_src_avgpool_neighbors_mask = torch.cat(mask_list,dim=0)

            #         n_iter = x_ori_tgt_neighbor.shape[0]//bs
            #         mask_list=[]
            #         tmp_list=[]
            #         for iter in range(n_iter):
            #             neighbor = x_ori_tgt_neighbor[iter*bs:(iter+1)*bs,:,2:].to(device)
            #             _,_,feat_tgt_avgpool_neighbor,_,_,_,_ = G(neighbor) #32,650,64
            #             mask = neighbor[:,:,0]!=0
            #             mask_list.append(mask)
            #             tmp_list.append(feat_tgt_avgpool_neighbor)
            #         if x_ori_tgt_neighbor.shape[0]%bs!=0:
            #             neighbor = x_ori_tgt_neighbor[n_iter*bs:,:,2:].to(device)
            #             _,_,feat_tgt_avgpool_neighbor,_,_,_,_ = G(neighbor)
            #             mask = neighbor[:,:,0]!=0 #32,650
            #             mask_list.append(mask)
            #             tmp_list.append(feat_tgt_avgpool_neighbor)
            #         feat_tgt_avgpool_neighbors = torch.cat(tmp_list,dim=0)
            #         feat_tgt_avgpool_neighbors_mask = torch.cat(mask_list,dim=0)

            aux_feat=None
            if 'bertonly' in args.nbr_mode:
                bert_input = torch.cat([bert_input_src, bert_input_tgt],dim=0) #128,60,768
                bert_embedding = bert_model.embeddings(bert_input)
                if 'learnable' in args.nbr_mode:
                    bert_embedding = torch.cat([bert_learnable_tokens.unsqueeze(0).tile(args.batch_size*2,1,1) ,bert_embedding],dim=1) #128,10,768 vs #128,60,768
                bert_feature = bert_model.encoder(bert_embedding) #128.70.768
                bert_feature = bert_model.pooler(bert_feature[0]) #128,768
                bert_feature = dim_converter(bert_feature)

                y_1 = F1(bert_feature , aux_feat)
                y_2 = F2(bert_feature , aux_feat)
                
            else:
                x = torch.cat((x_ori_src, x_ori_tgt), dim=0)
                if 'perpt' in args.nbr_mode:
                    _,_,g,_,_,_,_ = G(x, args, mask_early=args.mask_early, mask_late=args.mask_late)
                else:
                    _,g,_,_,_,_,_ = G(x, args, mask_early=args.mask_early, mask_late=args.mask_late)
                if args.mask_late:
                    g,g_list = g
                g_s,g_t = g[:bs],g[bs:]
            
                if 'bert' in args.nbr_mode:
                    bert_input = torch.cat([bert_input_src, bert_input_tgt],dim=0) #128,60,768
                    bert_embedding = bert_model.embeddings(bert_input)
                    if 'learnable' in args.nbr_mode:
                        bert_embedding = torch.cat([bert_learnable_tokens.unsqueeze(0).tile(args.batch_size*2,1,1) ,bert_embedding],dim=1) #128,10,768 vs #128,60,768
                    bert_feature = bert_model.encoder(bert_embedding) #128.70.768
                    bert_feature = bert_model.pooler(bert_feature[0]) #128,768
                    bert_feature = dim_converter(bert_feature)
                    bert_feature_s,bert_feature_t = bert_feature[:bs],bert_feature[bs:]
                    
                    if 'cat' in args.nbr_mode:
                        g_s = torch.cat([g_s,bert_feature_s],dim=1)
                        g_t = torch.cat([g_t,bert_feature_t],dim=1)
                    else:
                        g_s = g_s + bert_feature_s
                        g_t = g_t + bert_feature_t
                    g = torch.cat((g_s, g_t), dim=0)
                    y_1 = F1(g, aux_feat)
                    y_2 = F2(g, aux_feat)
                
                else:
                    
                    if args.nbr_mode=='qkv_cat':
                        g_s = g_s.unsqueeze(1)
                        g_t = g_t.unsqueeze(1)
                        
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
                        
                        g_s = torch.cat([g_s,feat_src_avgpool_neighbors],dim=1)
                        g_t = torch.cat([g_t,feat_tgt_avgpool_neighbors],dim=1)
                        
                        g = torch.cat((g_s, g_t), dim=0)
                        y_1 = F1(g, aux_feat)
                        y_2 = F2(g, aux_feat)
                        
                    elif args.nbr_mode=='qkv_individual':
                        g_s = g_s.unsqueeze(1)
                        g_t = g_t.unsqueeze(1)
                        
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
                    
                    elif args.nbr_mode=='qkv_individual_vnbr':
                        g_s = g_s.unsqueeze(1)
                        g_t = g_t.unsqueeze(1)
                        
                        feat_src_avgpool_neighbor_list=[]
                        for neighbor_idx,_ in enumerate(neighbor_idx_src):
                            if neighbor_idx==neighbor_idx_src.shape[0]-1:
                                break
                            key = feat_src_avgpool_neighbor[neighbor_idx_src[neighbor_idx]:neighbor_idx_src[neighbor_idx+1]].to(device)
                            query = value = g_s[neighbor_idx] # might change to dist?
                            attn_output_neighbors_s, attn_output_weights_s = multihead_attn(query, key, value)
                            feat_src_avgpool_neighbor_list.append(attn_output_neighbors_s)
                        feat_src_avgpool_neighbors = torch.stack(feat_src_avgpool_neighbor_list).squeeze(1)
                            
                        feat_tgt_avgpool_neighbor_list=[]
                        for neighbor_idx,_ in enumerate(neighbor_idx_tgt):
                            if neighbor_idx==neighbor_idx_tgt.shape[0]-1:
                                break
                            key = feat_tgt_avgpool_neighbor[neighbor_idx_tgt[neighbor_idx]:neighbor_idx_tgt[neighbor_idx+1]].to(device)
                            query = value = g_t[neighbor_idx]
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
                        y_2 = F2(g, aux_feat)
                    
                    else:
                        if 'cat' in args.cat_mode:
                            g_s = torch.cat([g_s,feat_src_avgpool_neighbors],dim=1)
                            g_t = torch.cat([g_t,feat_tgt_avgpool_neighbors],dim=1)
                        else:
                            g_s = g_s + feat_src_avgpool_neighbors
                            g_t = g_t + feat_tgt_avgpool_neighbors
                            
                        g = torch.cat((g_s, g_t), dim=0)
                        y_1 = F1(g, aux_feat)
                        y_2 = F2(g, aux_feat)
                        
                        
                        
            y1_s, y1_t = y_1.chunk(2, dim=0)
            y2_s, y2_t = y_2.chunk(2, dim=0)
            y1_t, y2_t = F.softmax(y1_t, dim=1), F.softmax(y2_t, dim=1)
            loss = F.cross_entropy(y1_s, labels_s) + F.cross_entropy(y2_s, labels_s) + (entropy(y1_t) + entropy(y2_t)) * args.trade_off_entropy \
                - classifier_discrepancy(y1_t, y2_t) * args.trade_off
            loss.backward()
            optimizer_f.step()



        # Step C train genrator to minimize discrepancy
        if 'C' in args.steps_list:
            for k in range(args.num_k):
                optimizer_g.zero_grad()

                aux_feat=None
                if 'bertonly' in args.nbr_mode:
                    bert_input = torch.cat([bert_input_src, bert_input_tgt],dim=0) #128,60,768
                    bert_embedding = bert_model.embeddings(bert_input)
                    if 'learnable' in args.nbr_mode:
                        bert_embedding = torch.cat([bert_learnable_tokens.unsqueeze(0).tile(args.batch_size*2,1,1) ,bert_embedding],dim=1) #128,10,768 vs #128,60,768
                    bert_feature = bert_model.encoder(bert_embedding) #128.70.768
                    bert_feature = bert_model.pooler(bert_feature[0]) #128,768
                    bert_feature = dim_converter(bert_feature)

                    y_1 = F1(bert_feature , aux_feat)
                    y_2 = F2(bert_feature , aux_feat)
                
                else:
                    x = torch.cat((x_ori_src, x_ori_tgt), dim=0)
                    if 'perpt' in args.nbr_mode:
                        _,_,g,_,_,_,_ = G(x, args, mask_early=args.mask_early, mask_late=args.mask_late)
                    else:
                        _,g,_,_,_,_,_ = G(x, args, mask_early=args.mask_early, mask_late=args.mask_late)
                    if args.mask_late:
                        g,g_list = g
                    g_s,g_t = g[:bs],g[bs:]
                    
                    if 'bert' in args.nbr_mode:
                        bert_input = torch.cat([bert_input_src, bert_input_tgt],dim=0) #128,60,768
                        bert_embedding = bert_model.embeddings(bert_input)
                        if 'learnable' in args.nbr_mode:
                            bert_embedding = torch.cat([bert_learnable_tokens.unsqueeze(0).tile(args.batch_size*2,1,1) ,bert_embedding],dim=1) #128,10,768 vs #128,60,768
                        bert_feature = bert_model.encoder(bert_embedding) #128.70.768
                        bert_feature = bert_model.pooler(bert_feature[0]) #128,768
                        bert_feature = dim_converter(bert_feature)
                        bert_feature_s,bert_feature_t = bert_feature[:bs],bert_feature[bs:]
                        
                        if 'cat' in args.nbr_mode:
                            g_s = torch.cat([g_s,bert_feature_s],dim=1)
                            g_t = torch.cat([g_t,bert_feature_t],dim=1)
                        else:
                            g_s = g_s + bert_feature_s
                            g_t = g_t + bert_feature_t
                        g = torch.cat((g_s, g_t), dim=0)
                        y_1 = F1(g, aux_feat)
                        y_2 = F2(g, aux_feat)
                    
                    else:
                        
                        if args.nbr_mode=="c_feat_nbr":
                            g1 = torch.cat((g_s, g_t), dim=0)
                            g2 = torch.cat((feat_src_avgpool_neighbors, feat_tgt_avgpool_neighbors), dim=0)
                            y_1 = F1(g1, aux_feat)
                            y_2 = F2(g2, aux_feat)
                        
                        elif args.nbr_mode=="c_featfeat_nbrnbr":
                            g_s1 = torch.cat([feat_src_avgpool_neighbors,feat_src_avgpool_neighbors],dim=1)
                            g_t1 = torch.cat([feat_tgt_avgpool_neighbors,feat_tgt_avgpool_neighbors],dim=1)
                            g_s = torch.cat([g_s,g_s],dim=1)
                            g_t = torch.cat([g_t,g_t],dim=1)
                            
                            g1 = torch.cat((g_s, g_t), dim=0)
                            g2 = torch.cat((g_s1, g_t1), dim=0)
                            y_1 = F1(g1, aux_feat)
                            y_2 = F2(g2, aux_feat)
                        
                        elif args.nbr_mode=='qkv_cat':
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
                            
                            g_s = torch.cat([g_s,feat_src_avgpool_neighbors],dim=1)
                            g_t = torch.cat([g_t,feat_tgt_avgpool_neighbors],dim=1)
                            
                            g = torch.cat((g_s, g_t), dim=0)
                            y_1 = F1(g, aux_feat)
                            y_2 = F2(g, aux_feat)
                        
                        elif args.nbr_mode=='qkv_individual':
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
                            
                        elif args.nbr_mode=='qkv_individual_perpt':
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
                            y_2 = F2(g, aux_feat)
                        
                        else:
                            if 'cat' in args.cat_mode:
                                g_s = torch.cat([g_s,feat_src_avgpool_neighbors],dim=1)
                                g_t = torch.cat([g_t,feat_tgt_avgpool_neighbors],dim=1)
                            else:
                                g_s = g_s + feat_src_avgpool_neighbors
                                g_t = g_t + feat_tgt_avgpool_neighbors
                            g = torch.cat((g_s, g_t), dim=0)
                            y_1 = F1(g, aux_feat)
                            y_2 = F2(g, aux_feat)
                        
                        
                y1_s, y1_t = y_1.chunk(2, dim=0)
                y2_s, y2_t = y_2.chunk(2, dim=0)
                y1_t, y2_t = F.softmax(y1_t, dim=1), F.softmax(y2_t, dim=1)
                mcd_loss = classifier_discrepancy(y1_t, y2_t) * args.trade_off
                mcd_loss.backward()
                optimizer_g.step()

            cls_acc = accuracy(y1_s, labels_s)[0]

            losses.update(loss.item(), bs)
            cls_accs.update(cls_acc.item(), bs)
            trans_losses.update(mcd_loss.item(), bs)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)


def validate(val_loader: DataLoader, G: nn.Module, F1: ImageClassifierHead, F2: ImageClassifierHead, attn_net:nn.Module, args: argparse.Namespace, multihead_attn, nbr_label_encoder, bert_model, dim_converter, bert_learnable_tokens, enc_tokenizer) -> Tuple[float, float]:
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
    F2.eval()
    bert_model.eval()
    dim_converter.eval()
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

    with torch.no_grad():
        end = time.time()
        for i, data in enumerate(val_loader):
            # if i>5:
            #     break
            x, labels, neighbors, neighbor_labels, bert_input = data
            if neighbor_labels[0] is not None:
                neighbor_labels = torch.stack(neighbor_labels).to(device)
            x, labels = torch.stack(x), torch.stack(labels)
            x, labels = x.to(device), labels.to(device)
            bs = x.shape[0]
            x = x[:,:,2:]
            
            neighbor_idx_src = [neighbor.shape[0] for neighbor in neighbors]
            neighbor_idx_src = np.insert(np.cumsum(neighbor_idx_src),0,0)
            neighbors = torch.cat(neighbors)


            # if "perpt" not in args.nbr_mode:
            #     n_iter = neighbors.shape[0]//bs
            #     tmp_list=[]
            #     for iter in range(n_iter):
            #         neighbor = neighbors[iter*bs:(iter+1)*bs,:,2:].to(device)
            #         _,feat_neighbor,_,_,_,_,_ = G(neighbor)
            #         # _,_,feat_neighbor,_,_,_,_ = G(neighbor)
            #         # feat_neighbor = F.normalize(feat_neighbor, dim=0)
            #         # mask = neighbor[:,:,0]!=0
            #         # feat_neighbor = torch.sum(feat_neighbor,dim=1) / (torch.sum(mask,dim=1,keepdim=True)+1)
            #         tmp_list.append(feat_neighbor.cpu())
            #     if neighbors.shape[0]%bs!=0:
            #         neighbor = neighbors[n_iter*bs:,:,2:].to(device)
            #         _,feat_neighbor,_,_,_,_,_ = G(neighbor)
            #         # _,_,feat_neighbor,_,_,_,_ = G(neighbor)
            #         # feat_neighbor = F.normalize(feat_neighbor, dim=0)
            #         # mask = neighbor[:,:,0]!=0
            #         # feat_neighbor = torch.sum(feat_neighbor,dim=1) / (torch.sum(mask,dim=1,keepdim=True)+1)
            #         tmp_list.append(feat_neighbor.cpu())
            #     feat_neighbor = torch.cat(tmp_list,dim=0)
            # else:
            #     n_iter = neighbors.shape[0]//bs
            #     mask_list=[]
            #     tmp_list=[]
            #     for iter in range(n_iter):
            #         neighbor = neighbors[iter*bs:(iter+1)*bs,:,2:].to(device)
            #         _,_,feat_neighbor,_,_,_,_ = G(neighbor) #32,650,64
            #         mask = neighbor[:,:,0]!=0
            #         mask_list.append(mask)
            #         tmp_list.append(feat_neighbor)
            #     if neighbors.shape[0]%bs!=0:
            #         neighbor = neighbors[n_iter*bs:,:,2:].to(device)
            #         _,_,feat_neighbor,_,_,_,_ = G(neighbor)
            #         mask = neighbor[:,:,0]!=0 #32,650
            #         mask_list.append(mask)
            #         tmp_list.append(feat_neighbor)
            #     feat_neighbors = torch.cat(tmp_list,dim=0)
            #     feat_neighbors_mask = torch.cat(mask_list,dim=0)
            

            aux_feat = None
            if 'bertonly' in args.nbr_mode:
                bert_input = torch.stack(bert_input).to(device)
                bert_embedding = bert_model.embeddings(bert_input)
                if 'learnable' in args.nbr_mode:
                    bert_embedding = torch.cat([bert_learnable_tokens.unsqueeze(0).tile(args.batch_size*2,1,1) ,bert_embedding],dim=1) #128,10,768 vs #128,60,768
                bert_feature = bert_model.encoder(bert_embedding) #128.70.768
                bert_feature = bert_model.pooler(bert_feature[0]) #128,768
                bert_feature = dim_converter(bert_feature)

                y1 = F1(bert_feature, aux_feat)
                y2 = F2(bert_feature, aux_feat)
            
            else:
                # compute output
                if "perpt" in args.nbr_mode:
                    _,_,g,_,_,_,_ = G(x)
                else:
                    _,g,_,_,_,_,_ = G(x)
            
                
                if 'bert' in args.nbr_mode:
                    bert_input = torch.stack(bert_input).to(device)
                    # bert_feature = bert_encoder(input_ids=bert_input)
                    # bert_feature = bert_feature[1]
                    
                    bert_embedding = bert_model.embeddings(bert_input)
                    if 'learnable' in args.nbr_mode:
                        bert_embedding = torch.cat([bert_learnable_tokens.unsqueeze(0).tile(bs,1,1) ,bert_embedding],dim=1) #128,10,768 vs #128,60,768
                    bert_feature = bert_model.encoder(bert_embedding) #128.70.768
                    bert_feature = bert_model.pooler(bert_feature[0]) #128,768
                    bert_feature = dim_converter(bert_feature)

                    if 'cat' in args.nbr_mode:
                        g = torch.cat([g,bert_feature],dim=1)
                    else:
                        g = g + bert_feature
                    y1 = F1(g, aux_feat)
                    y2 = F2(g, aux_feat)
                
                else:
                    
                    if args.nbr_mode=='qkv_cat':
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
                        g = torch.cat([g,feat_neighbors],dim=1)
                        
                        y1 = F1(g, aux_feat)
                        y2 = F2(g, aux_feat)
                    
                    elif args.nbr_mode=='qkv_individual':
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
                        y2 = F2(g, aux_feat)
                    
                    else:
                        feat_neighbor_list=[]
                        for neighbor_idx,_ in enumerate(neighbor_idx_src):
                            if neighbor_idx==neighbor_idx_src.shape[0]-1:
                                break
                            tmp_neighbor = feat_neighbor[neighbor_idx_src[neighbor_idx]:neighbor_idx_src[neighbor_idx+1]]
                            tmp_neighbor = torch.mean(tmp_neighbor, dim=0).to(device)
                            feat_neighbor_list.append(tmp_neighbor)
                        feat_neighbors = torch.stack(feat_neighbor_list) 
                        
                        if 'cat' in args.cat_mode:
                            g = torch.cat([g,feat_neighbors],dim=1)
                        else:
                            g = g + feat_neighbors
                            
                        y1 = F1(g, aux_feat)
                        y2 = F2(g, aux_feat)


            # measure accuracy and record loss
            acc1, = accuracy(y1, labels)
            acc2, = accuracy(y2, labels)
            if confmat:
                confmat.update(labels, y1.argmax(1))
            top1_1.update(acc1.item(), bs)
            top1_2.update(acc2.item(), bs)
            
            
            _, preds1 = torch.max(y1, 1)
            _, preds2 = torch.max(y2, 1)
            for t, p in zip(labels.view(-1), preds1.view(-1)):
                confusion_matrix1[t.long(), p.long()] += 1
            for t, p in zip(labels.view(-1), preds2.view(-1)):
                confusion_matrix2[t.long(), p.long()] += 1

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)
        
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
    parser.add_argument('--nbr_dist_thres', default=10, type=int, help='initial learning rate')
    parser.add_argument('--nbr_limit', default=100000, type=int, help='initial learning rate')
    parser.add_argument("--nbr_mode", type=str, default='', help="Where to save logs, checkpoints and debugging images.")
    parser.add_argument('--num_head', default=2, type=int, help='initial learning rate')
    parser.add_argument("--nbr_data_mode", type=str, default='mergemin5', help="Where to save logs, checkpoints and debugging images.")
    parser.add_argument("--nbr_label_mode", type=str, default='', help="Where to save logs, checkpoints and debugging images.")
    parser.add_argument('--nbr_label_embed_dim', default=16, type=int, help='initial learning rate')
    
    parser.add_argument('--random_mask_nbr_ratio', default=1.0, type=float)
    parser.add_argument('--mask_early', action="store_true", help='Whether to perform evaluation after training')
    parser.add_argument('--mask_late', action="store_true", help='Whether to perform evaluation after training')
    parser.add_argument('--n_mask_late', default=5, type=int, help='initial learning rate')
    parser.add_argument("--loss_mode", type=str, default='v1', help="Where to save logs, checkpoints and debugging images.")

    parser.add_argument('--bert_out_dim', default=64, type=int, help='initial learning rate')
    parser.add_argument('--G_no_frozen', action="store_true", help='Whether to perform evaluation after training')
    parser.add_argument('--token_len', default=10, type=int, help='initial learning rate')
    parser.add_argument('--token_max_len', default=60, type=int, help='initial learning rate')
    parser.add_argument("--steps_list", type=str, default='ABC', help="Where to save logs, checkpoints and debugging images.")
    parser.add_argument('--prompt_id', default=5, type=int, help='initial learning rate')




    args = parser.parse_args()
    
    # import resource
    # rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    # resource.setrlimit(resource.RLIMIT_NOFILE, (2048, rlimit[1]))
    
    # torch.multiprocessing.set_sharing_strategy('file_system')

    torch.set_num_threads(8)
    main(args)
