import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.vision.models import resnet50

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2D(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias_attr=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2D(in_planes, out_planes, kernel_size=1, stride=stride, bias_attr=False)

class ResNetSeries(nn.Layer):
    def __init__(self, pretrained):
        super(ResNetSeries, self).__init__()

        if pretrained == 'supervised':
            print(f'Loading supervised pretrained parameters!')
            model = resnet50(pretrained=True)
        elif pretrained == 'mocov2':
            print(f'Loading unsupervised {pretrained} pretrained parameters!')
            model = resnet50(pretrained=False)
            checkpoint = paddle.load('moco_r50_v2-e3b0c442.pth')
            model.set_state_dict(checkpoint['state_dict'])
        elif pretrained == 'detco':
            print(f'Loading unsupervised {pretrained} pretrained parameters!')
            model = resnet50(pretrained=False)
            checkpoint = paddle.load('detco_200ep.pth')
            model.set_state_dict(checkpoint['state_dict'])
        else:
            raise NotImplementedError

        self.conv1 = model.conv1
        self.bn1 = model.bn1
        self.relu = nn.ReLU()
        self.maxpool = model.maxpool
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4

    def forward(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x1 = self.layer3(x)
        x2 = self.layer4(x1)

        return paddle.concat([x2, x1], axis=1)

class Disentangler(nn.Layer):
    def __init__(self, cin):
        super(Disentangler, self).__init__()

        self.activation_head = nn.Conv2D(cin, 1, kernel_size=3, padding=1, bias_attr=False)
        self.bn_head = nn.BatchNorm2D(1)

    def forward(self, x, inference=False):
        N, C, H, W = x.shape
        if inference:
            ccam = self.bn_head(self.activation_head(x))
        else:
            ccam = F.sigmoid(self.bn_head(self.activation_head(x)))

        ccam_ = ccam.reshape([N, 1, H * W])                          # [N, 1, H*W]
        x = x.reshape([N, C, H * W]).transpose([0, 2, 1]).contiguous()   # [N, H*W, C]
        fg_feats = paddle.matmul(ccam_, x) / (H * W)                # [N, 1, C]
        bg_feats = paddle.matmul(1 - ccam_, x) / (H * W)            # [N, 1, C]

        return fg_feats.reshape([x.shape[0], -1]), bg_feats.reshape([x.shape[0], -1]), ccam


class Network(nn.Layer):
    def __init__(self, pretrained='mocov2', cin=None):
        super(Network, self).__init__()

        self.backbone = ResNetSeries(pretrained=pretrained)
        self.ac_head = Disentangler(cin)
        self.from_scratch_layers = [self.ac_head]

    def forward(self, x, inference=False):

        feats = self.backbone(x)
        fg_feats, bg_feats, ccam = self.ac_head(feats, inference=inference)

        return fg_feats, bg_feats, ccam

    def get_parameter_groups(self):
        groups = ([], [], [], [])
        print('======================================================')
        for m in self.sublayers():
            if isinstance(m, nn.Conv2D) or isinstance(m, nn.BatchNorm2D):

                if m.weight is not None and m.weight.trainable:
                    if m in self.from_scratch_layers:
                        groups[2].append(m.weight)
                    else:
                        groups[0].append(m.weight)

                if m.bias is not None and m.bias.trainable:
                    if m in self.from_scratch_layers:
                        groups[3].append(m.bias)
                    else:
                        groups[1].append(m.bias)
        return groups


def get_model(pretrained, cin=2048+1024):
    return Network(pretrained=pretrained, cin=cin)
