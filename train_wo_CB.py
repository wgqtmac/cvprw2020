# train.py
#!/usr/bin/env	python3

""" train network using pytorch

author baiyu
"""

import os
import sys
import argparse
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from torch.utils.data import DataLoader
#from dataset import *
from torch.autograd import Variable

from tensorboardX import SummaryWriter

from dataset.ImgLoader import ImgLoader
from class_balanced_loss import CB_loss
from imbalanced_dataset_sampler import ImbalancedDatasetSampler
from collections import Counter
from conf import settings
from utils import get_network, Logger, WarmUpLR
import roc

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def train(epoch):

    net.train()
    for batch_index, (images, labels) in enumerate(train_loader):
        if epoch <= args.warm:
            warmup_scheduler.step()

        cnt = Counter(np.array(labels))
        samples_per_cls = []
        samples_per_cls.append(cnt[0])
        samples_per_cls.append(cnt[1])

        images = Variable(images)
        labels = Variable(labels)

        labels = labels.cuda()
        images = images.cuda()

        optimizer.zero_grad()
        outputs = net(images)
        loss = loss_function(outputs, labels)
        # loss = CB_loss(labels, outputs, samples_per_cls, 2, 'softmax', 0.9999, 2.0)
        loss.backward()
        optimizer.step()

        n_iter = (epoch - 1) * len(train_loader) + batch_index + 1

        last_layer = list(net.children())[-1]
        # for name, para in last_layer.named_parameters():
        #     if 'weight' in name:
        #         writer.add_scalar('LastLayerGradients/grad_norm2_weights', para.grad.norm(), n_iter)
        #     if 'bias' in name:
        #         writer.add_scalar('LastLayerGradients/grad_norm2_bias', para.grad.norm(), n_iter)

        print('Training Epoch: {epoch} [{trained_samples}/{total_samples}]\tLoss: {:0.4f}\tLR: {:0.6f}'.format(
            loss.item(),
            optimizer.param_groups[0]['lr'],
            epoch=epoch,
            trained_samples=batch_index * args.tb + len(images),
            total_samples=len(train_loader.dataset)
        ))

        #update training loss for each iteration
        writer.add_scalar('Train/loss', loss.item(), n_iter)

    for name, param in net.named_parameters():
        layer, attr = os.path.splitext(name)
        attr = attr[1:]
        writer.add_histogram("{}/{}".format(layer, attr), param, epoch)

def eval_training(epoch):
    net.eval()

    test_loss = 0.0 # cost function error
    correct = 0.0

    result_list = []
    label_list = []
    TP = 0.
    TN = 0.
    FP = 0.
    FN = 0.


    for (images, labels) in test_loader:
        images = Variable(images)
        labels = Variable(labels)

        images = images.cuda()
        labels = labels.cuda()

        outputs = net(images)
        loss = loss_function(outputs, labels)
        # loss = CB_loss(labels, outputs, samples_per_cls, 2, 'softmax', 0.9999, 2.0)
        test_loss += loss.item()
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum()

        for i in range(len(preds)):
            if labels[i] == 1 and preds[i] == 1:
                TP += 1
            elif labels[i] == 0 and preds[i] == 0:
                TN += 1
            elif labels[i] == 1 and preds[i] == 0:
                FN += 1
            elif labels[i] == 0 and preds[i] == 1:
                FP += 1



        outputs = torch.softmax(outputs, dim=-1)
        preds_prob = outputs.to('cpu').detach().numpy()
        labels = labels.to('cpu').detach().numpy()
        for i_batch in range(preds.shape[0]):
            result_list.append(preds_prob[i_batch, 1])
            label_list.append(labels[i_batch])

    TP_rate = float(TP / (TP + FN))
    TN_rate = float(TN / (TN + FP))

    HTER = 1 - (TP_rate + TN_rate) / 2
    metric = roc.cal_metric(label_list, result_list, False)

    # print('Test set: Average loss: {:.4f}, Accuracy: {:.4f}, Auc: {:.4f}, HTER: {:.4f}'.format(
    #     test_loss / len(test_loader.dataset),
    #     correct.float() / len(test_loader.dataset),
    #     metric[2], HTER
    # ))
    log.write('Test set: Average loss: {:.4f}, Accuracy: {:.4f}, Auc: {:.4f}, HTER: {:.4f}'.format(
        test_loss / len(test_loader.dataset),
        correct.float() / len(test_loader.dataset),
        metric[2], HTER
    ))
    print()

    #add informations to tensorboard
    writer.add_scalar('Test/Average loss', test_loss / len(test_loader.dataset), epoch)
    writer.add_scalar('Test/Accuracy', correct.float() / len(test_loader.dataset), epoch)

    return correct.float() / len(test_loader.dataset)

def eval_turn(model, dataloader, epoch):

    model.train(False)

    TP = 0.
    TN = 0.
    FP = 0.
    FN = 0.

    val_corrects = 0
    item_count = len(dataloader.dataset)
    with torch.no_grad():
        for cnt, data in enumerate(dataloader, 0):

            img, label = data
            batch_size = img.size(0)


            preds = model(img.cuda())

            preds_ = preds.data.max(1)[1]
            batch_correct = preds_.eq(label.cuda().data).cpu().sum()

            val_corrects += batch_correct

            for i in range(len(preds_)):
                if label[i] == 1 and preds_[i] == 1:
                    TP += 1
                elif label[i] == 0 and preds_[i] == 0:
                    TN += 1
                elif label[i] == 1 and preds_[i] == 0:
                    FN += 1
                elif label[i] == 0 and preds_[i] == 1:
                    FP += 1

        TP_rate = float(TP / (TP + FN))
        TN_rate = float(TN / (TN + FP))

        HTER = 1 - (TP_rate + TN_rate) / 2

        # print('total eval item {:d}'.format(item_count))
        val_acc = float(float(val_corrects) / (item_count))

        # print('acc: %.4f, total item: %d, correct item: %d, TP rate: %.4f, TN rate: %.4f, HTER : %.4f' % (val_acc, item_count, val_corrects, TP_rate, TN_rate, HTER))
        print('epoch: %d acc: %.4f, total item: %d, correct item: %d, TP rate: %.4f, TN rate: %.4f, HTER : %.4f \n' % (epoch, val_acc, item_count, val_corrects, TP_rate, TN_rate, HTER))

    return val_acc


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-net', type=str, default='resnet18', help='net type')
    parser.add_argument('-gpu', type=bool, default=True, help='use gpu or not')
    parser.add_argument('-w', type=int, default=2, help='number of workers for dataloader')
    parser.add_argument('-tb', type=int, default=64, help='batch size for train dataloader')
    parser.add_argument('-vb', type=int, default=32, help='batch size for val dataloader')
    parser.add_argument('-s', type=bool, default=True, help='whether shuffle the dataset')
    parser.add_argument('-warm', type=int, default=1, help='warm up training phase')
    parser.add_argument('-lr', type=float, default=0.01, help='initial learning rate')
    parser.add_argument('-train_list', type=str, default='4train_list.txt', help='initial learning rate')
    parser.add_argument('-test_list', type=str, default='2test_list.txt', help='initial learning rate')
    parser.add_argument('-root_folder', type=str, default='/home/gqwang/Spoof_Croped', help='initial learning rate')
    args = parser.parse_args()

    net = get_network(args, use_gpu=args.gpu)

    out_dir = './logs'
    log = Logger()
    log.open(os.path.join(out_dir, args.net + '4to5_wo_CB.txt'), mode='a')
        
    #data preprocessing:
    # cifar100_training_loader = get_training_dataloader(
    #     settings.CIFAR100_TRAIN_MEAN,
    #     settings.CIFAR100_TRAIN_STD,
    #     num_workers=args.w,
    #     batch_size=args.b,
    #     shuffle=args.s
    # )
    #
    # cifar100_test_loader = get_test_dataloader(
    #     settings.CIFAR100_TRAIN_MEAN,
    #     settings.CIFAR100_TRAIN_STD,
    #     num_workers=args.w,
    #     batch_size=args.b,
    #     shuffle=args.s
    # )

    train_dataset = ImgLoader(args.root_folder, os.path.join(args.root_folder, args.train_list),
                              transforms.Compose([
                                  transforms.Resize(248),
                                  # transforms.RandomAffine(10),
                                  transforms.CenterCrop(248),
                                  transforms.RandomHorizontalFlip(),
                                  transforms.RandomRotation(15),
                                  transforms.ToTensor()
                                  # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                  # transforms.RandomRotation(15),
                                  # transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),

                              ]))

    weights = [3 if label == 1 else 1 for data, label in train_dataset.items]
    from torch.utils.data.sampler import WeightedRandomSampler

    sampler = WeightedRandomSampler(weights,
                                    num_samples=len(train_dataset.items),
                                    replacement=True) 
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=args.tb,
                                               num_workers=2,
                                               # shuffle=True,
                                               sampler=ImbalancedDatasetSampler(train_dataset),
                                               pin_memory=True)

    test_dataset = ImgLoader(args.root_folder, os.path.join(args.root_folder, args.test_list),
                             transforms.Compose([
                                 transforms.Resize(248),
                                 # transforms.RandomAffine(10),
                                 transforms.CenterCrop(248),
                                 transforms.RandomRotation(15),
                                 transforms.ToTensor()
                                 # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

                             ]))
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=args.vb,
                                              num_workers=2,
                                              pin_memory=True)

    loss_function = nn.CrossEntropyLoss()
    # loss_function = CB_loss()
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    train_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=settings.MILESTONES, gamma=0.2) #learning rate decay
    iter_per_epoch = len(train_loader)
    warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch * args.warm)
    checkpoint_path = os.path.join(settings.CHECKPOINT_PATH, args.net, settings.TIME_NOW)

    # use tensorboard
    if not os.path.exists(settings.LOG_DIR):
        os.mkdir(settings.LOG_DIR)
    writer = SummaryWriter(log_dir=os.path.join(
            settings.LOG_DIR, args.net, settings.TIME_NOW))
    input_tensor = torch.Tensor(12, 3, 248, 248).cuda()
    writer.add_graph(net, Variable(input_tensor, requires_grad=True))

    #create checkpoint folder to save model
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    checkpoint_path = os.path.join(checkpoint_path, '{net}-{epoch}-{type}.pth')

    best_acc = 0.0
    for epoch in range(1, settings.EPOCH):
        if epoch > args.warm:
            train_scheduler.step(epoch)

        train(epoch)
        acc = eval_training(epoch)

        #start to save best performance model after learning rate decay to 0.01 
        if epoch > settings.MILESTONES[1] and best_acc < acc:
            torch.save(net.state_dict(), checkpoint_path.format(net=args.net, epoch=epoch, type='best'))
            best_acc = acc
            continue

        if not epoch % settings.SAVE_EPOCH:
            torch.save(net.state_dict(), checkpoint_path.format(net=args.net, epoch=epoch, type='regular'))

    writer.close()
