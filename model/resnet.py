# Modified ResNet with Conditional Batch Norm (CBN) layers instead of the batch norm layers
# Features from block 4 are used for the VQA task

import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo

from model.cbn import CBN

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)
'''
This modules returns both the conv feature map and the lstm question embedding (unchanges)
since subsequent CBN layers in nn.Sequential will require both inputs
'''
class Conv2d(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size=1, stride=1, bias=True):
        super(Conv2d, self).__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, bias=bias)

    def forward(self, x, lstm_emb=None):
        out = self.conv(x)
        return out, lstm_emb


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, lstm_size, emb_size, stride=1, downsample=None, cbn=True):
        super(BasicBlock, self).__init__()
        self.cbn = cbn
        self.conv1 = conv3x3(inplanes, planes, stride)
        if self.cbn:
            self.bn1 = CBN(lstm_size, emb_size, planes)
        else:
            self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        if self.cbn:
            self.bn2 = CBN(lstm_size, emb_size, planes)
        else:
            self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x, lstm_emb=None):
        residual = x

        out = self.conv1(x)
        if self.cbn:
            out, _ = self.bn1(out, lstm_emb)
        else:
            out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        if self.cbn:
            out, _ = self.bn2(out, lstm_emb)
        else:
            out = self.bn2(out)

        if self.downsample is not None:
            if self.cbn:
                residual, _ = self.downsample(x, lstm_emb)
            else:
                residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out, lstm_emb


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, lstm_size, emb_size, stride=1, downsample=None, cbn=True):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)

        self.cbn = cbn
        if self.cbn:
            self.bn1 = CBN(lstm_size, emb_size, planes)
            self.bn2 = CBN(lstm_size, emb_size, planes)
            self.bn3 = CBN(lstm_size, emb_size, planes * 4)
        else:
            self.bn1 = nn.BatchNorm2d(planes)
            self.bn2 = nn.BatchNorm2d(planes)
            self.bn3 = nn.BatchNorm2d(planes * 4)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x, lstm_emb=None):
        residual = x

        out = self.conv1(x)
        if self.cbn:
            out, _ = self.bn1(out, lstm_emb)
        else:
            out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        if self.cbn:
            out, _ = self.bn2(out, lstm_emb)
        else:
            out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        if self.cbn:
            out, _ = self.bn3(out, lstm_emb)
        else:
            out = self.bn3(out)

        if self.downsample is not None:
            if self.cbn:
                residual, _ = self.downsample(x, lstm_emb)
            else:
                residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        # if we use condtioned batch norm,
        # we will have lstm != emb -> return out, lstm_emb for modified Sequential
        # Otherwise, simply return out
        if lstm_emb is None:
            return out
        else:
            return out, lstm_emb

class ResNet(nn.Module):

    def __init__(self, block, layers, lstm_size, emb_size, num_classes=1000, cbn=True):
        self.inplanes = 64
        self.lstm_size = lstm_size
        self.emb_size = emb_size
        self.cbn = cbn

        # if use conditioned batch norm, we need to modify Sequential to accept two inputs
        if self.cbn:
            from model.sequential_modified import Sequential
            self.seq = Sequential
        else:
            from torch.nn import Sequential
            self.seq = Sequential
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False).cuda()
        
        if self.cbn:
            self.bn1 = CBN(self.lstm_size, self.emb_size, 64)
        else:
            self.bn1 = nn.BatchNorm2d(64)

        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            if self.cbn:
                conv = Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False)
                bn = CBN(self.lstm_size, self.emb_size, planes * block.expansion)
            else:
                conv = nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False)
                bn = nn.BatchNorm2d(planes * block.expansion)
            downsample = self.seq(conv, bn)

        layers = []
        layers.append(block(self.inplanes, planes, self.lstm_size, self.emb_size, stride, downsample, self.cbn))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, self.lstm_size, self.emb_size, cbn=self.cbn))

        return self.seq(*layers)

    def forward(self, x, lstm_emb):
        x = self.conv1(x)
        if self.cbn:
            x, _ = self.bn1(x, lstm_emb)
        else:
            x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        if self.cbn:
            x, _ = self.layer1(x, lstm_emb)
            x, _ = self.layer2(x, lstm_emb)
            x, _ = self.layer3(x, lstm_emb)
            x, _ = self.layer4(x, lstm_emb)
        else:
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)

        # not required currently
        # x = self.avgpool(x)
        # x = x.view(x.size(0), -1)
        # x = self.fc(x)

        return x

def resnet18(lstm_size, emb_size, pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], lstm_size, emb_size, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']), strict=False)
    return model


def resnet34(lstm_size, emb_size, pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], lstm_size, emb_size,**kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34']), strict=False)
    return model


def resnet50(lstm_size, emb_size, pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], lstm_size, emb_size, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']), strict=False)
    return model


def resnet101(lstm_size, emb_size, pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], lstm_size, emb_size, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101']), strict=False)
    return model


def resnet152(lstm_size, emb_size, pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], lstm_size, emb_size, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152']), strict=False)
    return model