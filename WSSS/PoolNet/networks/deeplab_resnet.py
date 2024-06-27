import paddle
import paddle.nn as nn
import paddle.nn.functional as F

affine_par = True

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2D(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias_attr=False)

class BasicBlock(nn.Layer):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2D(planes, affine=affine_par)
        self.relu = nn.ReLU()
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2D(planes, affine=affine_par)
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

    def __init__(self, inplanes, planes, stride=1, dilation_=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2D(inplanes, planes, kernel_size=1, stride=stride, bias_attr=False)
        self.bn1 = nn.BatchNorm2D(planes, affine=affine_par)
        self.conv2 = nn.Conv2D(planes, planes, kernel_size=3, stride=1, padding=dilation_, dilation=dilation_, bias_attr=False)
        self.bn2 = nn.BatchNorm2D(planes, affine=affine_par)
        self.conv3 = nn.Conv2D(planes, planes * 4, kernel_size=1, bias_attr=False)
        self.bn3 = nn.BatchNorm2D(planes * 4, affine=affine_par)
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
    def __init__(self, block, layers):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2D(3, 64, kernel_size=7, stride=2, padding=3, bias_attr=False)
        self.bn1 = nn.BatchNorm2D(64, affine=affine_par)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2D(kernel_size=3, stride=2, padding=1, ceil_mode=True)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilation_=2)

        for m in self.sublayers():
            if isinstance(m, nn.Conv2D):
                n = m.weight.shape[0] * m.weight.shape[1] * m.weight.shape[2] * m.weight.shape[3]
                m.weight.set_value(paddle.randn(m.weight.shape) * (math.sqrt(2. / n)))
            elif isinstance(m, nn.BatchNorm2D):
                m.weight.set_value(paddle.ones(m.weight.shape))
                m.bias.set_value(paddle.zeros(m.bias.shape))

    def _make_layer(self, block, planes, blocks, stride=1, dilation_=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion or dilation_ == 2 or dilation_ == 4:
            downsample = nn.Sequential(
                nn.Conv2D(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias_attr=False),
                nn.BatchNorm2D(planes * block.expansion, affine=affine_par),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, dilation_=dilation_, downsample=downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation_=dilation_))

        return nn.Sequential(*layers)

    def forward(self, x):
        tmp_x = []
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        tmp_x.append(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        tmp_x.append(x)
        x = self.layer2(x)
        tmp_x.append(x)
        x = self.layer3(x)
        tmp_x.append(x)
        x = self.layer4(x)
        tmp_x.append(x)

        return tmp_x

class ResNet_locate(nn.Layer):
    def __init__(self, block, layers):
        super(ResNet_locate, self).__init__()
        self.resnet = ResNet(block, layers)
        self.in_planes = 512
        self.out_planes = [512, 256, 256, 128]

        self.ppms_pre = nn.Conv2D(2048, self.in_planes, 1, 1, bias_attr=False)
        ppms, infos = [], []
        for ii in [1, 3, 5]:
            ppms.append(nn.Sequential(nn.AdaptiveAvgPool2D(ii), nn.Conv2D(self.in_planes, self.in_planes, 1, 1, bias_attr=False), nn.ReLU()))
        self.ppms = nn.LayerList(ppms)

        self.ppm_cat = nn.Sequential(nn.Conv2D(self.in_planes * 4, self.in_planes, 3, 1, 1, bias_attr=False), nn.ReLU())
        for ii in self.out_planes:
            infos.append(nn.Sequential(nn.Conv2D(self.in_planes, ii, 3, 1, 1, bias_attr=False), nn.ReLU()))
        self.infos = nn.LayerList(infos)

        for m in self.sublayers():
            if isinstance(m, nn.Conv2D):
                n = m.weight.shape[0] * m.weight.shape[1] * m.weight.shape[2] * m.weight.shape[3]
                m.weight.set_value(paddle.randn(m.weight.shape) * (math.sqrt(2. / n)))
            elif isinstance(m, nn.BatchNorm2D):
                m.weight.set_value(paddle.ones(m.weight.shape))
                m.bias.set_value(paddle.zeros(m.bias.shape))

    def load_pretrained_model(self, model):
        state_dict = paddle.load(model)
        self.set_state_dict(state_dict, strict=False)

    def forward(self, x):
        x_size = x.shape[2:]
        xs = self.resnet(x)

        xs_1 = self.ppms_pre(xs[-1])
        xls = [xs_1]
        for k in range(len(self.ppms)):
            xls.append(F.interpolate(self.ppms[k](xs_1), xs_1.shape[2:], mode='bilinear'))
        xls = self.ppm_cat(paddle.concat(xls, axis=1))

        infos = []
        for k in range(len(self.infos)):
            infos.append(self.infos[k](F.interpolate(xls, xs[len(self.infos) - 1 - k].shape[2:], mode='bilinear')))

        return xs, infos

def resnet50_locate():
    model = ResNet_locate(Bottleneck, [3, 4, 6, 3])
    return model
