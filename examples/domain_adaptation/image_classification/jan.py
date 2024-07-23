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

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
import torch.nn.functional as F

import utils
import tllib.vision.models as models

from tllib.alignment.jan import JointMultipleKernelMaximumMeanDiscrepancy, ImageClassifier, Theta
from tllib.modules.kernels import GaussianKernel
from tllib.utils.data import ForeverDataIterator
from tllib.utils.metric import accuracy
from tllib.utils.meter import AverageMeter, ProgressMeter
from tllib.utils.logger import CompleteLogger
from tllib.utils.analysis import collect_feature, tsne, a_distance

from tllib.alignment.mcd import ImageClassifierHead, entropy, classifier_discrepancy
from tllib.utils.data import ForeverDataIterator
from tllib.utils.metric import accuracy, ConfusionMatrix
from tllib.utils.logger import CompleteLogger


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

    # # create model
    # print("=> using model '{}'".format(args.arch))
    # backbone = utils.get_model(args.arch, pretrain=not args.scratch)
    # pool_layer = nn.Identity() if args.no_pool else None
    # classifier = ImageClassifier(backbone, num_classes, bottleneck_dim=args.bottleneck_dim,
    #                              pool_layer=pool_layer, finetune=not args.scratch).to(device)
    train_source_iter, train_target_iter, val_loader,_ = utils.load_data(args)
    classifier = models.TSEncoder().to(device)
    classifier_features_dim=64
    num_classes=4
    F1 = models.Classifier_clf().to(device)

    # define loss function
    if args.adversarial:
        thetas = [Theta(dim).to(device) for dim in (classifier_features_dim, num_classes)]
    else:
        thetas = None
    jmmd_loss = JointMultipleKernelMaximumMeanDiscrepancy(
        kernels=(
            [GaussianKernel(alpha=2 ** k) for k in range(-3, 2)],
            (GaussianKernel(sigma=0.92, track_running_stats=False),)
        ),
        linear=args.linear, thetas=thetas
    ).to(device)

    # parameters = classifier.get_parameters()
    # if thetas is not None:
    #     parameters += [{"params": theta.parameters(), 'lr': 0.1} for theta in thetas]

    # define optimizer
    # optimizer = torch.optim.Adam(classifier.get_parameters(), lr=args.lr, weight_decay=args.wd)
    # optimizer = Adam(parameters, args.lr, momentum=args.momentum, weight_decay=args.wd, nesterov=True)
    # lr_scheduler = LambdaLR(optimizer, lambda x: args.lr * (1. + args.lr_gamma * float(x)) ** (-args.lr_decay))
    if args.adversarial:
        optimizer = Adam([{'params':classifier.parameters()},{'params':F1.parameters()}]+[{"params": theta.parameters(), 'lr': 0.1} for theta in thetas],
                     args.lr, weight_decay=args.wd, betas=(0.5, 0.99))
    else:
        optimizer = Adam([{'params':classifier.parameters()},{'params':F1.parameters()}],args.lr, weight_decay=args.wd, betas=(0.5, 0.99))
    lr_scheduler=None

    # # resume from the best checkpoint
    # if args.phase != 'train':
    #     checkpoint = torch.load(logger.get_checkpoint_path('best'), map_location='cpu')
    #     classifier.load_state_dict(checkpoint)

    # analysis the model
    if args.phase == 'analysis':
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
        ckpt_dir='/home/yichen/Transfer-Learning-Library/examples/domain_adaptation/image_classification/logs/0508_jan_adv_01/checkpoints/best.pth'
        ckpt = torch.load(ckpt_dir, map_location='cuda:0')
        classifier.load_state_dict(ckpt['G'])#, strict=False)
        F1.load_state_dict(ckpt['F1'])#, strict=False)
        acc1 = validate(val_loader, classifier, F1, args, device)
        print(acc1)
        return

    # start training
    best_acc1 = 0.
    best_epoch = 0
    for epoch in range(args.epochs):
        # train for one epoch
        train(train_source_iter, train_target_iter, classifier, F1, jmmd_loss, optimizer,
              epoch, args)

        # evaluate on validation set
        results = validate(val_loader, classifier, F1, args, device)

        # # remember best acc@1 and save checkpoint
        # torch.save(classifier.state_dict(), logger.get_checkpoint_path('latest'))
        # if acc1 > best_acc1:
        #     shutil.copy(logger.get_checkpoint_path('latest'), logger.get_checkpoint_path('best'))
        #     best_epoch = epoch
        # best_acc1 = max(acc1, best_acc1)
        
        torch.save({
            'G': classifier.state_dict(),
            'F1': F1.state_dict()
        }, logger.get_checkpoint_path('latest'))
        if results > best_acc1:
            print('copying best ckpt')
            shutil.copy(logger.get_checkpoint_path('latest'), logger.get_checkpoint_path('best'))
            best_acc1 = results
            best_epoch = epoch

    # print("best_acc1 = {:3.1f}".format(best_acc1))
    print("best_acc1 = {:3.1f}({:d})".format(best_acc1, best_epoch))

    # evaluate on test set
    # classifier.load_state_dict(torch.load(logger.get_checkpoint_path('best')))
    # acc1 = utils.validate(test_loader, classifier, args, device)
    # print("test_acc1 = {:3.1f}".format(acc1))

    logger.close()


def train(train_source_iter: ForeverDataIterator, train_target_iter: ForeverDataIterator, model: ImageClassifier, F1,
          jmmd_loss: JointMultipleKernelMaximumMeanDiscrepancy, optimizer: Adam,
          epoch: int, args: argparse.Namespace):
    batch_time = AverageMeter('Time', ':4.2f')
    data_time = AverageMeter('Data', ':3.1f')
    losses = AverageMeter('Loss', ':3.2f')
    trans_losses = AverageMeter('Trans Loss', ':5.4f')
    cls_accs = AverageMeter('Cls Acc', ':3.1f')

    progress = ProgressMeter(
        args.iters_per_epoch,
        [batch_time, data_time, losses, trans_losses, cls_accs],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()
    F1.train()
    jmmd_loss.train()

    end = time.time()
    for i in range(args.iters_per_epoch):
        # x_s, labels_s = next(train_source_iter)[:2]
        # x_t, = next(train_target_iter)[:1]
        x_s, labels_s = next(train_source_iter)[:2]
        x_t, = next(train_target_iter)[:1]


        x_s = x_s.to(device)
        x_t = x_t.to(device)
        labels_s = labels_s.to(device)

        # measure data loading time
        data_time.update(time.time() - end)

        # compute output
        x = torch.cat((x_s, x_t), dim=0)
        _,g,_,_ = model(x)
        y = F1(g, None)
        y = F.softmax(y, dim=1)
        y_s, y_t = y.chunk(2, dim=0)
        f_s, f_t = g.chunk(2, dim=0)

        cls_loss = F.cross_entropy(y_s, labels_s)
        transfer_loss = jmmd_loss(
            (f_s, F.softmax(y_s, dim=1)),
            (f_t, F.softmax(y_t, dim=1))
        )
        loss = cls_loss + transfer_loss * args.trade_off

        cls_acc = accuracy(y_s, labels_s)[0]

        losses.update(loss.item(), x_s.size(0))
        cls_accs.update(cls_acc.item(), x_s.size(0))
        trans_losses.update(transfer_loss.item(), x_s.size(0))

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



def validate(val_loader, model, F1, args, device) -> float:
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
    F1.eval()
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
            _,f,_,_ = model(images)
            output = F1(f, None)
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='JAN for Unsupervised Domain Adaptation')
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
    parser.add_argument('--bottleneck-dim', default=256, type=int,
                        help='Dimension of bottleneck')
    parser.add_argument('--no-pool', action='store_true',
                        help='no pool layer after the feature extractor.')
    parser.add_argument('--scratch', action='store_true', help='whether train from scratch.')
    parser.add_argument('--linear', default=False, action='store_true',
                        help='whether use the linear version')
    parser.add_argument('--adversarial', default=False, action='store_true',
                        help='whether use adversarial theta')
    parser.add_argument('--trade-off', default=1., type=float,
                        help='the trade-off hyper-parameter for transfer loss')
    # training parameters
    parser.add_argument('-b', '--batch-size', default=32, type=int,
                        metavar='N',
                        help='mini-batch size (default: 32)')
    parser.add_argument('--lr', '--learning-rate', default=0.003, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--lr-gamma', default=0.0003, type=float, help='parameter for lr scheduler')
    parser.add_argument('--lr-decay', default=0.75, type=float, help='parameter for lr scheduler')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=0.0005, type=float,
                        metavar='W', help='weight decay (default: 5e-4)')
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
    parser.add_argument("--log", type=str, default='jan',
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
