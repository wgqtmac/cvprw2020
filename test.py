#test.py
#!/usr/bin/env python3

""" test neuron network performace
print top1 and top5 err on test dataset
of a model

author baiyu
"""

import argparse
#from dataset import *

#from skimage import io
from matplotlib import pyplot as plt

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable
from dataset.ImgLoader import ImgLoader
import os
import roc

from conf import settings
from utils import get_network, get_test_dataloader

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-net', type=str, default='resnet18', help='net type')
    parser.add_argument('-weights', type=str, default='./checkpoint/resnet18/2019-12-30T20:04:53.202543/resnet18-20-regular.pth', help='the weights file you want to test')
    parser.add_argument('-gpu', type=bool, default=True, help='use gpu or not')
    parser.add_argument('-w', type=int, default=2, help='number of workers for dataloader')
    parser.add_argument('-b', type=int, default=16, help='batch size for dataloader')
    parser.add_argument('-s', type=bool, default=True, help='whether shuffle the dataset')
    parser.add_argument('-test_list', type=str, default='4test_list.txt', help='initial learning rate')
    parser.add_argument('-root_folder', type=str, default='/home/gqwang/Spoof_Croped', help='initial learning rate')
    args = parser.parse_args()

    net = get_network(args)

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
                                              batch_size=args.b,
                                              num_workers=2,
                                              pin_memory=True)

    net.load_state_dict(torch.load(args.weights), args.gpu)
    print(net)
    net.eval()

    correct = 0.0
    total = 0

    result_list = []
    label_list = []
    TP = 0.
    TN = 0.
    FP = 0.
    FN = 0.

    for n_iter, (image, label) in enumerate(test_loader):
        print("iteration: {}\ttotal {} iterations".format(n_iter + 1, len(test_loader)))
        image = Variable(image).cuda()
        labels = Variable(label).cuda()
        outputs = net(image)
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
    metric = roc.cal_metric(label_list, result_list, True)

    print('Test set: Accuracy: {:.4f}, Auc: {:.4f}, HTER: {:.4f}'.format(
        correct.float() / len(test_loader.dataset), metric[0], HTER
    ))
    print()
        # _, pred = output.topk(1, 1, largest=True, sorted=True)

        # label = label.view(label.size(0), -1).expand_as(pred)
        # correct = pred.eq(label).float()
        #
        # # #compute top 5
        # # correct_5 += correct[:, :5].sum()
        #
        # #compute top1
        # correct_1 += correct[:, :1].sum()


    # print()
    # print("Top 1 err: ", 1 - correct_1 / len(test_loader.dataset))
    # # print("Top 5 err: ", 1 - correct_5 / len(test_loader.dataset))
    # print("Parameter numbers: {}".format(sum(p.numel() for p in net.parameters())))