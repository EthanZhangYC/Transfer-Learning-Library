"""
@author: Baixu Chen
@contact: cbx_99_hasta@outlook.com
"""
import random
import time
import warnings
import argparse
import shutil
import os.path as osp

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.optim import SGD, Adam
from torch.utils.data import DataLoader
import torch.nn.functional as F

# import encoder
import utils
import tllib.vision.models as models

from tllib.normalization.afn import AdaptiveFeatureNorm, ImageClassifier
from tllib.modules.entropy import entropy
from tllib.utils.data import ForeverDataIterator
from tllib.utils.metric import accuracy
from tllib.utils.meter import AverageMeter, ProgressMeter
from tllib.utils.logger import CompleteLogger
from tllib.utils.analysis import collect_feature, tsne, a_distance
import pdb

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

    # create model
    # print("=> using model '{}'".format(args.arch))
    # backbone = utils.get_model(args.arch, pretrain=not args.scratch)
    # pool_layer = nn.Identity() if args.no_pool else None
    # classifier = ImageClassifier(backbone, num_classes, args.num_blocks,
    #                              bottleneck_dim=args.bottleneck_dim, dropout_p=args.dropout_p,
    #                              pool_layer=pool_layer, finetune=not args.scratch).to(device)
    train_src_iter, train_tgt_iter, val_loader = utils.load_data(args)
    classifier = models.TSEncoder().to(device)
    adaptive_feature_norm = AdaptiveFeatureNorm(args.delta).to(device)

    # define optimizer
    # the learning rate is fixed according to origin paper
    # optimizer = SGD(classifier.parameters(), args.lr, momentum=0.9, weight_decay=args.weight_decay)
    optimizer = Adam(classifier.parameters(), args.lr, weight_decay=args.weight_decay, betas=(0.5, 0.99))

    # resume from the best checkpoint
    if args.phase != 'train':
        raise NotImplemented
        checkpoint = torch.load(logger.get_checkpoint_path('best'), map_location='cpu')
        classifier.load_state_dict(checkpoint)

    # analysis the model
    if args.phase == 'analysis':
        raise NotImplemented
        # extract features from both domains
        feature_extractor = nn.Sequential(classifier.backbone, classifier.pool_layer, classifier.bottleneck).to(device)
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
        raise NotImplemented
        acc1 = utils.validate(test_loader, classifier, args, device)
        print(acc1)
        return

    # start training
    best_acc1 = 0.
    best_epoch = 0
    for epoch in range(args.epochs):
        # train for one epoch
        train(train_src_iter, train_tgt_iter, classifier, adaptive_feature_norm, optimizer, epoch, args)

        # evaluate on validation set
        acc1 = utils.validate(val_loader, classifier, args, device)

        # remember best acc@1 and save checkpoint
        torch.save(classifier.state_dict(), logger.get_checkpoint_path('latest'))
        if acc1 > best_acc1:
            shutil.copy(logger.get_checkpoint_path('latest'), logger.get_checkpoint_path('best'))
            best_epoch = epoch
        best_acc1 = max(acc1, best_acc1)

        # print("best_acc1 = {:3.1f}".format(best_acc1))
        print("best_acc1 = {:3.1f}({:d})".format(best_acc1, best_epoch))

    # evaluate on test set
    # classifier.load_state_dict(torch.load(logger.get_checkpoint_path('best')))
    # acc1 = utils.validate(test_loader, classifier, args, device)
    # print("test_acc1 = {:3.1f}".format(acc1))

    logger.close()


# def train(train_source_iter: ForeverDataIterator, train_target_iter: ForeverDataIterator, model: ImageClassifier,
#           adaptive_feature_norm: AdaptiveFeatureNorm, optimizer: SGD, epoch: int, args: argparse.Namespace):
def train(train_source_iter, train_target_iter, model,
          adaptive_feature_norm, optimizer, epoch, args):
    batch_time = AverageMeter('Time', ':3.1f')
    data_time = AverageMeter('Data', ':3.1f')
    cls_losses = AverageMeter('Cls Loss', ':3.2f')
    norm_losses = AverageMeter('Norm Loss', ':3.2f')
    src_feature_norm = AverageMeter('Source Feature Norm', ':3.2f')
    tgt_feature_norm = AverageMeter('Target Feature Norm', ':3.2f')
    cls_accs = AverageMeter('Cls Acc', ':3.1f')

    progress = ProgressMeter(
        args.iters_per_epoch,
        [cls_losses, norm_losses, src_feature_norm, tgt_feature_norm, cls_accs],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i in range(args.iters_per_epoch):
        end = time.time()
        x_s,labels_s,_ = next(train_source_iter)
        x_t,_,_ = next(train_target_iter)

        x_s = x_s.to(device)
        x_t = x_t.to(device)
        labels_s = labels_s.to(device)

        # measure data loading time
        data_time.update(time.time() - end)

        # compute output
        y_s, f_s, _, _ = model(x_s, True)
        y_t, f_t, _, _ = model(x_t, True)

        # classification loss
        cls_loss = F.cross_entropy(y_s, labels_s)
        # norm loss
        norm_loss = adaptive_feature_norm(f_s) + adaptive_feature_norm(f_t)
        loss = cls_loss + norm_loss * args.trade_off_norm

        # using entropy minimization
        if args.trade_off_entropy:
            y_t = F.softmax(y_t, dim=1)
            entropy_loss = entropy(y_t, reduction='mean')
            loss += entropy_loss * args.trade_off_entropy

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # update statistics
        cls_acc = accuracy(y_s, labels_s)[0]

        cls_losses.update(cls_loss.item(), x_s.size(0))
        norm_losses.update(norm_loss.item(), x_s.size(0))
        src_feature_norm.update(f_s.norm(p=2, dim=1).mean().item(), x_s.size(0))
        tgt_feature_norm.update(f_t.norm(p=2, dim=1).mean().item(), x_s.size(0))
        cls_accs.update(cls_acc.item(), x_s.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='AFN for Unsupervised Domain Adaptation')

    parser.add_argument('-s', '--source', help='source domain(s)', nargs='+')
    parser.add_argument('-t', '--target', help='target domain(s)', nargs='+')

    # model parameters

    parser.add_argument('--scratch', action='store_true', help='whether train from scratch.')
    parser.add_argument('-n', '--num-blocks', default=1, type=int, help='Number of basic blocks for classifier')
    parser.add_argument('--bottleneck-dim', default=1000, type=int, help='Dimension of bottleneck')
    parser.add_argument('--dropout-p', default=0.5, type=float,
                        help='Dropout probability')
    # training parameters
    parser.add_argument('-b', '--batch-size', default=32, type=int,
                        metavar='N',
                        help='mini-batch size (default: 32)')
    parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=5e-4, type=float,
                        metavar='W', help='weight decay (default: 5e-4)',
                        dest='weight_decay')
    
    parser.add_argument('--trade-off-norm', default=0.05, type=float,
                        help='the trade-off hyper-parameter for norm loss')
    parser.add_argument('--trade-off-entropy', default=None, type=float,
                        help='the trade-off hyper-parameter for entropy loss')
    parser.add_argument('-r', '--delta', default=1, type=float, help='Increment for L2 norm')
    parser.add_argument('-j', '--workers', default=2, type=int, metavar='N',
                        help='number of data loading workers (default: 2)')
    parser.add_argument('--epochs', default=20, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-i', '--iters-per-epoch', default=500, type=int,
                        help='Number of iterations per epoch')
    parser.add_argument('-p', '--print-freq', default=100, type=int,
                        metavar='N', help='print frequency (default: 100)')
    parser.add_argument('--seed', default=None, type=int,
                        help='seed for initializing training. ')
    parser.add_argument('--per-class-eval', action='store_true',
                        help='whether output per-class accuracy during evaluation')
    parser.add_argument("--log", type=str, default='afn',
                        help="Where to save logs, checkpoints and debugging images.")
    parser.add_argument("--phase", type=str, default='train', choices=['train', 'test', 'analysis'],
                        help="When phase is 'test', only test the model."
                             "When phase is 'analysis', only analysis the model.")
    
    parser.add_argument('--use_unlabel', action="store_true", help='Whether to perform evaluation after training')
    parser.add_argument('--interpolated', action="store_true", help='Whether to perform evaluation after training')
    parser.add_argument('--interpolatedlinear', action="store_true", help='Whether to perform evaluation after training')
    parser.add_argument('--trip_time', type=int, default=20, help='')

    args = parser.parse_args()
    main(args)
