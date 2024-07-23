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

    # train_source_iter, train_target_iter, val_loader = utils.load_data_neighbor(args)
    train_source_iter, train_target_iter, val_loader,_,_,source_loader, train_target_iter_lbd = utils.load_data_neighbor_v3(args, None, None, enc_tokenizer=None)
    G = models.TSEncoder().to(device)
    classifier_features_dim=64
    num_classes = 4
    
    # if args.ckpt_dir is not None:
    # print('resuming model')
    ckpt_dir='/home/yichen/TS2Vec/result/0402_pretrain/model_best.pth.tar'
    ckpt = torch.load(ckpt_dir, map_location='cuda:0')
    G.load_state_dict(ckpt['encoder_tgt_state_dict'], strict=False)

    # create model
    # print("=> using model '{}'".format(args.arch))
    # G = utils.get_model(args.arch, pretrain=not args.scratch).to(device)  # feature extractor
    # two image classifier heads
    pool_layer = nn.Identity() if args.no_pool else None
    # F1 = ImageClassifierHead(G.out_features, num_classes, args.bottleneck_dim, pool_layer).to(device)
    # F2 = ImageClassifierHead(G.out_features, num_classes, args.bottleneck_dim, pool_layer).to(device)
    F1 = models.Classifier_clf(input_dim=64).to(device)
    F2 = models.Classifier_clf(input_dim=64).to(device)
    # F1 = models.ViT(use_auxattn=True, double_attn=True).to(device)
    # F2 = models.ViT(use_auxattn=True, double_attn=True).to(device)
    
    attn_net = models.AttnNet().to(device)

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
        {"params": F2.parameters()},
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
        train(train_source_iter, train_target_iter, G, F1, F2, attn_net, optimizer_g, optimizer_f, epoch, args,  train_target_iter_lbd)

        # evaluate on validation set
        results = validate(val_loader, G, F1, F2, attn_net, args)

        # remember best acc@1 and save checkpoint
        torch.save({
            'G': G.state_dict(),
            'F1': F1.state_dict(),
            'F2': F2.state_dict(),
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
    F2.load_state_dict(checkpoint['F2'])
    # results = validate(test_loader, G, F1, F2, args)
    # print("test_acc1 = {:3.1f}".format(max(results)))

    logger.close()


def train(train_source_iter: ForeverDataIterator, train_target_iter: ForeverDataIterator,
          G: nn.Module, F1: ImageClassifierHead, F2: ImageClassifierHead, attn_net: nn.Module,
          optimizer_g: SGD, optimizer_f: SGD, epoch: int, args: argparse.Namespace, train_tgt_iter_lbd):
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
    attn_net.train()

    end = time.time()
    for i in range(args.iters_per_epoch):
        # if i>5:
        #     break
        
        
        # x_ori_src, labels_src, x_ori_src_neighbor, labels_neighbor, dist_src_neighbor, labels_domain_src = next(train_source_iter)
        # x_ori_tgt, _,_,x_ori_tgt_neighbor, dist_tgt_neighbor, labels_domain_tgt, idx_tgt,_ = next(train_target_iter)
        x_ori_src, labels_s, x_ori_src_neighbor, src_neighbor_labels, labels_domain_src, bert_input_src  = next(train_source_iter) 
        x_ori_tgt, labels_t, labels_t_mask, x_ori_tgt_neighbor, tgt_neighbor_labels, labels_domain_tgt, idx_tgt, bert_input_tgt = next(train_target_iter)

        x_ori_src, labels_s, labels_domain_src = torch.stack(x_ori_src), torch.stack(labels_s), torch.stack(labels_domain_src)
        x_ori_tgt, idx_tgt, labels_domain_tgt = torch.stack(x_ori_tgt), torch.stack(idx_tgt), torch.stack(labels_domain_tgt)
        x_ori_src, x_ori_tgt = x_ori_src[:,:,2:], x_ori_tgt[:,:,2:] # time, dist, v, a, jerk, bearing, is_real
        x_ori_src, x_ori_tgt, labels_s, idx_tgt = x_ori_src.to(device), x_ori_tgt.to(device), labels_s.to(device), idx_tgt.to(device)
        
        if args.semi:
            x_ori_tgt_lbd, labels_t_lbd, x_ori_tgt_neighbor_lbd, tgt_neighbor_labels_lbd, labels_domain_tgt_lbd, bert_input_tgt_lbd = next(train_tgt_iter_lbd)
            x_ori_tgt_lbd, labels_t_lbd = torch.stack(x_ori_tgt_lbd), torch.stack(labels_t_lbd)
            x_ori_tgt_lbd = x_ori_tgt_lbd[:,:,2:] # time, dist, v, a, jerk, bearing, is_real
            x_ori_tgt_lbd, labels_t_lbd = x_ori_tgt_lbd.to(device), labels_t_lbd.to(device)

        # measure data loading time
        data_time.update(time.time() - end)

        # Step A train all networks to minimize loss on source domain
        optimizer_g.zero_grad()
        optimizer_f.zero_grad()

        bs=x_ori_src.shape[0]
       
        if args.semi:
            x = torch.cat((x_ori_src, x_ori_tgt, x_ori_tgt_lbd), dim=0)
        else:
            x = torch.cat((x_ori_src, x_ori_tgt), dim=0)
        _,g,_,_ = G(x)
        g_s,g_t = g[:bs],g[bs:]
        
        # feat_src_avgpool_neighbor_list=[]
        # for neighbor,dist in zip(x_ori_src_neighbor, dist_src_neighbor):
        #     neighbor = neighbor[:,:,2:].to(device)
        #     dist = dist.to(device)
        #     # attn = F.softmax(torch.pow(dist, 1/args.dist_root)).unsqueeze(1)
        #     _,_,feat_src_avgpool_neighbor,_ = G(neighbor)
        #     if neighbor.shape[0]==1:
        #         feat_src_avgpool_neighbor = feat_src_avgpool_neighbor.squeeze(0)
        #     else:
        #         attn = attn_net(dist.unsqueeze(2).transpose(0, 1))
        #         attn = F.softmax(attn[-1], dim=0)
        #         feat_src_avgpool_neighbor = torch.sum(attn * feat_src_avgpool_neighbor, dim=0)
        #     feat_src_avgpool_neighbor_list.append(feat_src_avgpool_neighbor)
        # feat_src_avgpool_neighbor = torch.stack(feat_src_avgpool_neighbor_list)
        
        # feat_tgt_avgpool_neighbor_list=[]
        # for neighbor,dist in zip(x_ori_tgt_neighbor, dist_tgt_neighbor):
        #     neighbor = neighbor[:,:,2:].to(device)
        #     dist = dist.to(device)
        #     # attn = F.softmax(torch.pow(dist, 1/args.dist_root)).unsqueeze(1)
        #     _,_,feat_tgt_avgpool_neighbor,_ = G(neighbor)
        #     if neighbor.shape[0]==1:
        #         feat_tgt_avgpool_neighbor = feat_tgt_avgpool_neighbor.squeeze(0)
        #     else:
        #         attn = attn_net(dist.unsqueeze(2).transpose(0, 1))
        #         attn = F.softmax(attn[-1], dim=0)
        #         feat_tgt_avgpool_neighbor = torch.sum(attn * feat_tgt_avgpool_neighbor, dim=0)
        #     feat_tgt_avgpool_neighbor_list.append(feat_tgt_avgpool_neighbor)
        # feat_tgt_avgpool_neighbor = torch.stack(feat_tgt_avgpool_neighbor_list)
        
        # g_s = torch.cat([g_s,feat_src_avgpool_neighbor],dim=1)
        # g_t = torch.cat([g_t,feat_tgt_avgpool_neighbor],dim=1)
        # g = torch.cat((g_s, g_t), dim=0)
        aux_feat=None

        y_1 = F1(g, aux_feat)
        y_2 = F2(g, aux_feat)
        # y1_s, y1_t = y_1.chunk(2, dim=0)
        # y2_s, y2_t = y_2.chunk(2, dim=0)
        y1_s,y1_t,y1_t_lbd = y_1[:bs],y_1[bs:2*bs],y_1[2*bs:]
        y2_s,y2_t,y2_t_lbd = y_2[:bs],y_2[bs:2*bs],y_2[2*bs:]
        y1_t, y2_t = F.softmax(y1_t, dim=1), F.softmax(y2_t, dim=1)

        # loss = F.cross_entropy(y1_s, labels_s) + F.cross_entropy(y2_s, labels_s) + F.cross_entropy(y1_t_lbd, labels_t_lbd) + F.cross_entropy(y2_t_lbd, labels_t_lbd) + \
        loss = F.cross_entropy(torch.cat([y1_s,y1_t_lbd],dim=0), torch.cat([labels_s,labels_t_lbd],dim=0)) + F.cross_entropy(torch.cat([y2_s,y2_t_lbd],dim=0), torch.cat([labels_s,labels_t_lbd],dim=0)) + \
            (entropy(y1_t) + entropy(y2_t)) * args.trade_off_entropy
        loss.backward()
        optimizer_g.step()
        optimizer_f.step()

        # Step B train classifier to maximize discrepancy
        optimizer_g.zero_grad()
        optimizer_f.zero_grad()

        _,g,_,_ = G(x)
        # _,_,_,g = G(x)
        g_s,g_t = g[:bs],g[bs:]
        
        # feat_src_avgpool_neighbor_list=[]
        # for neighbor,dist in zip(x_ori_src_neighbor, dist_src_neighbor):
        #     neighbor = neighbor[:,:,2:].to(device)
        #     dist = dist.to(device)
        #     # attn = F.softmax(torch.pow(dist, 1/args.dist_root)).unsqueeze(1)
        #     _,_,feat_src_avgpool_neighbor,_ = G(neighbor)
        #     if neighbor.shape[0]==1:
        #         feat_src_avgpool_neighbor = feat_src_avgpool_neighbor.squeeze(0)
        #     else:
        #         attn = attn_net(dist.unsqueeze(2).transpose(0, 1))
        #         attn = F.softmax(attn[-1], dim=0)
        #         feat_src_avgpool_neighbor = torch.sum(attn * feat_src_avgpool_neighbor, dim=0)
        #     feat_src_avgpool_neighbor_list.append(feat_src_avgpool_neighbor)
        # feat_src_avgpool_neighbor = torch.stack(feat_src_avgpool_neighbor_list)
        
        # feat_tgt_avgpool_neighbor_list=[]
        # for neighbor,dist in zip(x_ori_tgt_neighbor, dist_tgt_neighbor):
        #     neighbor = neighbor[:,:,2:].to(device)
        #     dist = dist.to(device)
        #     # attn = F.softmax(torch.pow(dist, 1/args.dist_root)).unsqueeze(1)
        #     _,_,feat_tgt_avgpool_neighbor,_ = G(neighbor)
        #     if neighbor.shape[0]==1:
        #         feat_tgt_avgpool_neighbor = feat_tgt_avgpool_neighbor.squeeze(0)
        #     else:
        #         attn = attn_net(dist.unsqueeze(2).transpose(0, 1))
        #         attn = F.softmax(attn[-1], dim=0)
        #         feat_tgt_avgpool_neighbor = torch.sum(attn * feat_tgt_avgpool_neighbor, dim=0)
        #     feat_tgt_avgpool_neighbor_list.append(feat_tgt_avgpool_neighbor)
        # feat_tgt_avgpool_neighbor = torch.stack(feat_tgt_avgpool_neighbor_list)
        
        # g_s = torch.cat([g_s,feat_src_avgpool_neighbor],dim=1)
        # g_t = torch.cat([g_t,feat_tgt_avgpool_neighbor],dim=1)
        # g = torch.cat((g_s, g_t), dim=0)
        aux_feat=None
        
        y_1 = F1(g, aux_feat)
        y_2 = F2(g, aux_feat)
        # y1_s, y1_t = y_1.chunk(2, dim=0)
        # y2_s, y2_t = y_2.chunk(2, dim=0)
        y1_s,y1_t,y1_t_lbd = y_1[:bs],y_1[bs:2*bs],y_1[2*bs:]
        y2_s,y2_t,y2_t_lbd = y_2[:bs],y_2[bs:2*bs],y_2[2*bs:]

        y1_t, y2_t = F.softmax(y1_t, dim=1), F.softmax(y2_t, dim=1)
        loss = F.cross_entropy(y1_s, labels_s) + F.cross_entropy(y2_s, labels_s) + F.cross_entropy(y1_t_lbd, labels_t_lbd) + F.cross_entropy(y2_t_lbd, labels_t_lbd) + \
               (entropy(y1_t) + entropy(y2_t)) * args.trade_off_entropy - \
               classifier_discrepancy(y1_t, y2_t) * args.trade_off
        loss.backward()
        optimizer_f.step()

        # Step C train genrator to minimize discrepancy
        for k in range(args.num_k):
            optimizer_g.zero_grad()
            _,g,_,_ = G(x)
            # _,_,_,g = G(x)
            g_s,g_t = g[:bs],g[bs:]
        
            # feat_src_avgpool_neighbor_list=[]
            # for neighbor,dist in zip(x_ori_src_neighbor, dist_src_neighbor):
            #     neighbor = neighbor[:,:,2:].to(device)
            #     dist = dist.to(device)
            #     # attn = F.softmax(torch.pow(dist, 1/args.dist_root)).unsqueeze(1)
            #     _,_,feat_src_avgpool_neighbor,_ = G(neighbor)
            #     if neighbor.shape[0]==1:
            #         feat_src_avgpool_neighbor = feat_src_avgpool_neighbor.squeeze(0)
            #     else:
            #         attn = attn_net(dist.unsqueeze(2).transpose(0, 1))
            #         attn = F.softmax(attn[-1], dim=0)
            #         feat_src_avgpool_neighbor = torch.sum(attn * feat_src_avgpool_neighbor, dim=0)
            #     feat_src_avgpool_neighbor_list.append(feat_src_avgpool_neighbor)
            # feat_src_avgpool_neighbor = torch.stack(feat_src_avgpool_neighbor_list)
            
            # feat_tgt_avgpool_neighbor_list=[]
            # for neighbor,dist in zip(x_ori_tgt_neighbor, dist_tgt_neighbor):
            #     neighbor = neighbor[:,:,2:].to(device)
            #     dist = dist.to(device)
            #     # attn = F.softmax(torch.pow(dist, 1/args.dist_root)).unsqueeze(1)
            #     _,_,feat_tgt_avgpool_neighbor,_ = G(neighbor)
            #     if neighbor.shape[0]==1:
            #         feat_tgt_avgpool_neighbor = feat_tgt_avgpool_neighbor.squeeze(0)
            #     else:
            #         attn = attn_net(dist.unsqueeze(2).transpose(0, 1))
            #         attn = F.softmax(attn[-1], dim=0)
            #         feat_tgt_avgpool_neighbor = torch.sum(attn * feat_tgt_avgpool_neighbor, dim=0)
            #     feat_tgt_avgpool_neighbor_list.append(feat_tgt_avgpool_neighbor)
            # feat_tgt_avgpool_neighbor = torch.stack(feat_tgt_avgpool_neighbor_list)
            
            # g_s = torch.cat([g_s,feat_src_avgpool_neighbor],dim=1)
            # g_t = torch.cat([g_t,feat_tgt_avgpool_neighbor],dim=1)
            # g = torch.cat((g_s, g_t), dim=0)
            aux_feat=None
            
            y_1 = F1(g, aux_feat)
            y_2 = F2(g, aux_feat)
            # y1_s, y1_t = y_1.chunk(2, dim=0)
            # y2_s, y2_t = y_2.chunk(2, dim=0)
            y1_s,y1_t,y1_t_lbd = y_1[:bs],y_1[bs:2*bs],y_1[2*bs:]
            y2_s,y2_t,y2_t_lbd = y_2[:bs],y_2[bs:2*bs],y_2[2*bs:]
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


def validate(val_loader: DataLoader, G: nn.Module, F1: ImageClassifierHead,
             F2: ImageClassifierHead, attn_net:nn.Module, args: argparse.Namespace) -> Tuple[float, float]:
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
    attn_net.eval()

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
            
            # x, labels, neighbors, dist_neighbors = data
            x, labels, neighbors, neighbor_labels, bert_input = data

            x, labels = torch.stack(x), torch.stack(labels)
            x, labels = x.to(device), labels.to(device)
            bs = x.shape[0]
            x = x[:,:,2:]

            # compute output
            _,g,_,_ = G(x)
            # # _,_,_,g = G(images)
            # # aux_feat = torch.stack([images[:,:,0],images[:,:,-1]], axis=2)

            # feat_neighbor_list=[]
            # for neighbor,dist in zip(neighbors, dist_neighbors):
            #     neighbor = neighbor[:,:,2:].to(device)
            #     dist = dist.to(device)
            #     # dist = F.softmax(torch.pow(dist, 1/args.dist_root))
            #     _,_,feat_neighbor,_ = G(neighbor)
            #     if neighbor.shape[0]==1:
            #         feat_neighbor = feat_neighbor.squeeze(0)
            #     else:
            #         attn = attn_net(dist.unsqueeze(2).transpose(0, 1))
            #         attn = F.softmax(attn[-1], dim=0)
            #         feat_neighbor = torch.sum(attn * feat_neighbor, dim=0)
            #     feat_neighbor_list.append(feat_neighbor)
            # feat_neighbor = torch.stack(feat_neighbor_list)
            
            # g = torch.cat([g,feat_neighbor],dim=1)
            aux_feat = None

            
            y1, y2 = F1(g, aux_feat), F2(g, aux_feat)

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
    parser.add_argument('--token_len', default=0, type=int, help='initial learning rate')
    parser.add_argument('--token_max_len', default=60, type=int, help='initial learning rate')
    parser.add_argument('--prompt_id', default=5, type=int, help='initial learning rate')

    parser.add_argument('--proto_momentum', default=0.9, type=float)
    parser.add_argument("--update_strategy", type=str, default='iter', help="Where to save logs, checkpoints and debugging images.")
    parser.add_argument('--self_train', action="store_true", help='Whether to perform evaluation after training')
    parser.add_argument('--semi', action="store_true", help='Whether to perform evaluation after training')
    parser.add_argument('--n_shots', default=5, type=int, help='initial learning rate')

    args = parser.parse_args()
    main(args)
