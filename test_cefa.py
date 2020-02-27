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
# from dataset.ImgLoader import ImgLoader
from dataset.TestImgLoader import ImgLoader
import os
import roc
import numpy as np

from conf import settings
from utils import get_network, get_test_dataloader

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-net', type=str, default='resnet18', help='net type')
    parser.add_argument('-weights', type=str, default='./checkpoint/resnet18/2020-02-26T23:06:01.904184/resnet18-42-best.pth', help='the weights file you want to test')
    parser.add_argument('-gpu', type=bool, default=True, help='use gpu or not')
    parser.add_argument('-w', type=int, default=2, help='number of workers for dataloader')
    parser.add_argument('-b', type=int, default=1, help='batch size for dataloader')
    parser.add_argument('-s', type=bool, default=True, help='whether shuffle the dataset')
    parser.add_argument('-test_list', type=str, default='4@all_test_res.txt', help='initial learning rate')
    parser.add_argument('-root_folder', type=str, default='/home/gqwang/Spoof_Croped/CASIA_CeFA', help='initial learning rate')
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

                             ]), stage='Test')
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

    fp = open("./result@all.txt", 'a+')

    for n_iter, (image, image_name) in enumerate(test_loader):
        print("iteration: {}\ttotal {} iterations".format(n_iter + 1, len(test_loader)))

        image = Variable(image).cuda()
        outputs = net(image)

        probability = torch.nn.functional.softmax(outputs, dim=1)[:, 1].detach().tolist()
        probability_value = np.array(probability)

        newline = image_name[0] + ' ' + "{:.8f}".format(probability_value[0]) + "\n"
        fp.write(newline)
        print(newline)

    fp.close()

    print()
