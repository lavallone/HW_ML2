import torch.nn as nn
from torchvision.models import ResNet

########################################################################
## helper function ##
def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)
########################################################################

class SEBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None,
                 *, reduction=16):
        super(SEBasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, 1)
        self.bn2 = nn.BatchNorm2d(planes)
        self.se = SELayer(planes, reduction)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class SEBottleneck(nn.Module):

    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None,
                 *, reduction=16):
        super(SEBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.se = SELayer(planes * 4, reduction)
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
        out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class SEResNet(nn.Module):
    def __init__(self):
        super(SEResNet, self).__init__()

    @staticmethod
    def get_classifiers():
        return ['se_rn18', 'se_rn34', 'se_rn50', 'se_rn101','se_rn152']
            
    @classmethod # cls sta a indicare la classe stessa, in questo caso la SE_RESNET
    def build_classifier(cls, arch: str, num_classes: int, input_channels: int):
        _, type=arch.split("se_rn")[1]
        if type=='18':
            cls_instance = ResNet(SEBasicBlock, [2, 2, 2, 2], num_classes=num_classes)
            cls_instance.avg_pool = nn.AdaptiveAvgPool2d(1)
        elif type=='34':
            cls_instance = ResNet(SEBasicBlock, [3, 4, 6, 3], num_classes=num_classes)
            cls_instance.avg_pool = nn.AdaptiveAvgPool2d(1)
        elif type=='50':
            cls_instance = ResNet(SEBottleneck, [3, 4, 6, 3], num_classes=num_classes)
            #cls_instance.avg_pool = nn.AdaptiveAvgPool2d(1)
        elif type=='101':
            cls_instance = ResNet(SEBottleneck, [3, 4, 23, 3], num_classes=num_classes)
            cls_instance.avg_pool = nn.AdaptiveAvgPool2d(1)
        elif type=='152':
            cls_instance = ResNet(SEBottleneck, [3, 8, 36, 3], num_classes=num_classes)
            cls_instance.avg_pool = nn.AdaptiveAvgPool2d(1)
        return cls_instance
        