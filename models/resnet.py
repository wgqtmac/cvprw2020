# """resnet in pytorch
#
#
#
# [1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun.
#
#     Deep Residual Learning for Image Recognition
#     https://arxiv.org/abs/1512.03385v1
# """
#
# import torch
# import torch.nn as nn
#
# class BasicBlock(nn.Module):
#     """Basic Block for resnet 18 and resnet 34
#
#     """
#
#     #BasicBlock and BottleNeck block
#     #have different output size
#     #we use class attribute expansion
#     #to distinct
#     expansion = 1
#
#     def __init__(self, in_channels, out_channels, stride=1):
#         super().__init__()
#
#         #residual function
#         self.residual_function = nn.Sequential(
#             nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(out_channels, out_channels * BasicBlock.expansion, kernel_size=3, padding=1, bias=False),
#             nn.BatchNorm2d(out_channels * BasicBlock.expansion)
#         )
#
#         #shortcut
#         self.shortcut = nn.Sequential()
#
#         #the shortcut output dimension is not the same with residual function
#         #use 1*1 convolution to match the dimension
#         if stride != 1 or in_channels != BasicBlock.expansion * out_channels:
#             self.shortcut = nn.Sequential(
#                 nn.Conv2d(in_channels, out_channels * BasicBlock.expansion, kernel_size=1, stride=stride, bias=False),
#                 nn.BatchNorm2d(out_channels * BasicBlock.expansion)
#             )
#
#     def forward(self, x):
#         return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))
#
# class BottleNeck(nn.Module):
#     """Residual block for resnet over 50 layers
#
#     """
#     expansion = 4
#     def __init__(self, in_channels, out_channels, stride=1):
#         super().__init__()
#         self.residual_function = nn.Sequential(
#             nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(out_channels, out_channels, stride=stride, kernel_size=3, padding=1, bias=False),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(out_channels, out_channels * BottleNeck.expansion, kernel_size=1, bias=False),
#             nn.BatchNorm2d(out_channels * BottleNeck.expansion),
#         )
#
#         self.shortcut = nn.Sequential()
#
#         if stride != 1 or in_channels != out_channels * BottleNeck.expansion:
#             self.shortcut = nn.Sequential(
#                 nn.Conv2d(in_channels, out_channels * BottleNeck.expansion, stride=stride, kernel_size=1, bias=False),
#                 nn.BatchNorm2d(out_channels * BottleNeck.expansion)
#             )
#
#     def forward(self, x):
#         return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))
#
# class ResNet(nn.Module):
#
#     def __init__(self, block, num_block, num_classes=2):
#         super().__init__()
#
#         self.in_channels = 64
#
#         self.conv1 = nn.Sequential(
#             nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False),
#             nn.BatchNorm2d(64),
#             nn.ReLU(inplace=True))
#         #we use a different inputsize than the original paper
#         #so conv2_x's stride is 1
#         self.conv2_x = self._make_layer(block, 64, num_block[0], 1)
#         self.conv3_x = self._make_layer(block, 128, num_block[1], 2)
#         self.conv4_x = self._make_layer(block, 256, num_block[2], 2)
#         self.conv5_x = self._make_layer(block, 512, num_block[3], 2)
#         self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
#         self.fc = nn.Linear(512 * block.expansion, num_classes)
#
#     def _make_layer(self, block, out_channels, num_blocks, stride):
#         """make resnet layers(by layer i didnt mean this 'layer' was the
#         same as a neuron netowork layer, ex. conv layer), one layer may
#         contain more than one residual block
#
#         Args:
#             block: block type, basic block or bottle neck block
#             out_channels: output depth channel number of this layer
#             num_blocks: how many blocks per layer
#             stride: the stride of the first block of this layer
#
#         Return:
#             return a resnet layer
#         """
#
#         # we have num_block blocks per layer, the first block
#         # could be 1 or 2, other blocks would always be 1
#         strides = [stride] + [1] * (num_blocks - 1)
#         layers = []
#         for stride in strides:
#             layers.append(block(self.in_channels, out_channels, stride))
#             self.in_channels = out_channels * block.expansion
#
#         return nn.Sequential(*layers)
#
#     def forward(self, x):
#         output = self.conv1(x)
#         output = self.conv2_x(output)
#         output = self.conv3_x(output)
#         output = self.conv4_x(output)
#         output = self.conv5_x(output)
#         output = self.avg_pool(output)
#         output = output.view(output.size(0), -1)
#         output = self.fc(output)
#
#         return output
#
# def resnet18():
#     """ return a ResNet 18 object
#     """
#     return ResNet(BasicBlock, [2, 2, 2, 2])
#
# def resnet34():
#     """ return a ResNet 34 object
#     """
#     return ResNet(BasicBlock, [3, 4, 6, 3])
#
# def resnet50():
#     """ return a ResNet 50 object
#     """
#     return ResNet(BottleNeck, [3, 4, 6, 3])
#
# def resnet101():
#     """ return a ResNet 101 object
#     """
#     return ResNet(BottleNeck, [3, 4, 23, 3])
#
# def resnet152():
#     """ return a ResNet 152 object
#     """
#     return ResNet(BottleNeck, [3, 8, 36, 3])
#
#
#


import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def _make_deconv(inplanes, outplanes, kernel_size, stride, padding, bias=True):
    return nn.Sequential(
            nn.BatchNorm2d(inplanes),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(inplanes, outplanes, kernel_size=kernel_size,
                               stride=stride, padding=padding, bias=bias) )


def _make_conv(inplanes, outplanes, kernel_size=1, stride=1, padding=0, bias=True):
    return nn.Sequential(
            nn.BatchNorm2d(inplanes),
            nn.ReLU(inplace=True),
            nn.Conv2d(inplanes, outplanes, kernel_size=kernel_size,
                      stride=stride, padding=padding, bias=bias) )


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000):
        super(ResNet, self).__init__()

        self.restored = False
        self.inplanes = 64
        num_feats = 256
        self.bias = False
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, 2)
        # self.fc = nn.Sequential(nn.Linear(512 * block.expansion, 256),
        #               nn.ReLU(),
        #               nn.Linear(256, 128)
        #               )
        # self.fc = nn.Sequential(nn.Linear(512 * block.expansion, 256)
        #                         )




        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        x = self.avgpool(x4)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x




def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    return model

def resnet34(pretrained=False, **kwargs):
    """ return a ResNet 34 object
    """
    return ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)

def resnet50(pretrained=False, **kwargs):
    """ return a ResNet 50 object
    """
    return ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)

def resnet101(pretrained=False, **kwargs):
    """ return a ResNet 101 object
    """
    return ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)

def resnet152(pretrained=False, **kwargs):
    """ return a ResNet 152 object
    """
    return ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)


if __name__ == '__main__':
    print(torch.__version__)
    x = torch.autograd.Variable(torch.Tensor(2, 3, 256, 256))
    # model = resnet80(num_classes=41857)
    model = resnet34()
    print(model)
    out = model(x)
    print(out.shape)