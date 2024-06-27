import math
import paddle
import paddle.nn as nn

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2D(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias_attr=False)


class BasicBlock(nn.Layer):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2D(planes)
        self.relu = nn.ReLU()
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2D(planes)
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


class Bottleneck(nn.Layer):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2D(inplanes, planes, kernel_size=1, bias_attr=False)
        self.bn1 = nn.BatchNorm2D(planes)
        self.conv2 = nn.Conv2D(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias_attr=False)
        self.bn2 = nn.BatchNorm2D(planes)
        self.conv3 = nn.Conv2D(planes, planes * self.expansion, kernel_size=1, bias_attr=False)
        self.bn3 = nn.BatchNorm2D(planes * self.expansion)
        self.relu = nn.ReLU()
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


class ResNet(nn.Layer):
    def __init__(self, block, layers, stride=None):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2D(3, 64, kernel_size=7, stride=2, padding=3,
                               bias_attr=False)
        self.bn1 = nn.BatchNorm2D(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2D(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], stride=stride[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=stride[1])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=stride[2])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=stride[3])
        self.avgpool = nn.AdaptiveAvgPool2D((1, 1))
        self.fc = nn.Linear(512 * block.expansion, 1000)

        for m in self.sublayers():
            if isinstance(m, nn.Conv2D):
                n = m.weight.shape[1] * m.weight.shape[2] * m.weight.shape[3]
                m.weight.set_value(paddle.normal(0, math.sqrt(2. / n), m.weight.shape))
            elif isinstance(m, nn.BatchNorm2D):
                m.weight.set_value(paddle.ones(m.weight.shape))
                m.bias.set_value(paddle.zeros(m.bias.shape))

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2D(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias_attr=False),
                nn.BatchNorm2D(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = paddle.flatten(x, start_axis=1)
        embedded = x
        x = self.fc(x)

        return embedded, x

    def get_parameter_groups(self, print_fn=print):
        groups = ([], [], [], [])

        for name, value in self.named_parameters():
            # pretrained weights
            if 'fc' not in name:
                if 'weight' in name:
                    print_fn(f'pretrained weights : {name}')
                    groups[0].append(value)
                else:
                    print_fn(f'pretrained bias : {name}')
                    groups[1].append(value)

            # scratched weights
            else:
                if 'weight' in name:
                    if print_fn is not None:
                        print_fn(f'scratched weights : {name}')
                    groups[2].append(value)
                else:
                    if print_fn is not None:
                        print_fn(f'scratched bias : {name}')
                    groups[3].append(value)
        return groups


def resnet18(pretrained=False, stride=None, num_classes=1000, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if stride is None:
        stride = [1, 2, 2, 1]
    model = ResNet(BasicBlock, [2, 2, 2, 2], stride=stride, **kwargs)
    if pretrained:
        state_dict = paddle.utils.download.get_weights_path_from_url(model_urls['resnet18'])
        model.set_state_dict(paddle.load(state_dict))
    model.fc = nn.Linear(512 * BasicBlock.expansion, num_classes)
    return model


def resnet34(pretrained=False, stride=None, num_classes=1000, **kwargs):
    """Constructs a ResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if stride is None:
        stride = [1, 2, 2, 1]
    model = ResNet(BasicBlock, [3, 4, 6, 3], stride=stride, **kwargs)
    if pretrained:
        state_dict = paddle.utils.download.get_weights_path_from_url(model_urls['resnet34'])
        model.set_state_dict(paddle.load(state_dict))
    model.fc = nn.Linear(512 * BasicBlock.expansion, num_classes)
    return model


def resnet50(pretrained=False, stride=None, num_classes=1000, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if stride is None:
        stride = [1, 2, 2, 1]
    model = ResNet(Bottleneck, [3, 4, 6, 3], stride=stride, **kwargs)
    if pretrained:
        state_dict = paddle.utils.download.get_weights_path_from_url(model_urls['resnet50'])
        model.set_state_dict(paddle.load(state_dict))
    model.fc = nn.Linear(512 * Bottleneck.expansion, num_classes)
    return model


def resnet101(pretrained=False, stride=None, num_classes=1000, **kwargs):
    """Constructs a ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if stride is None:
        stride = [1, 2, 2, 1]
    model = ResNet(Bottleneck, [3, 4, 23, 3], stride=stride, **kwargs)
    if pretrained:
        state_dict = paddle.utils.download.get_weights_path_from_url(model_urls['resnet101'])
        model.set_state_dict(paddle.load(state_dict))
    model.fc = nn.Linear(512 * Bottleneck.expansion, num_classes)
    return model


def resnet152(pretrained=False, stride=None, num_classes=1000, **kwargs):
    """Constructs a ResNet-152 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if stride is None:
        stride = [1, 2, 2, 1]
    model = ResNet(Bottleneck, [3, 8, 36, 3], stride=stride, **kwargs)
    if pretrained:
        state_dict = paddle.utils.download.get_weights_path_from_url(model_urls['resnet152'])
        model.set_state_dict(paddle.load(state_dict))
    model.fc = nn.Linear(512 * Bottleneck.expansion, num_classes)
    return model


if __name__ == '__main__':
    model = ResNet(Bottleneck, [3, 4, 6, 3])
    print('-----')
    model.get_parameter_groups()
