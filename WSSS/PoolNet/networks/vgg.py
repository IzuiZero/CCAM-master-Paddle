import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import math

# VGG definition
def vgg(cfg, i, batch_norm=False):
    layers = []
    in_channels = i
    stage = 1
    for v in cfg:
        if v == 'M':
            stage += 1
            if stage == 6:
                layers += [nn.MaxPool2D(kernel_size=3, stride=1, padding=1)]
            else:
                layers += [nn.MaxPool2D(kernel_size=3, stride=2, padding=1)]
        else:
            if stage == 6:
                conv2d = nn.Conv2D(in_channels, v, kernel_size=3, padding=1)
            else:
                conv2d = nn.Conv2D(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2D(v), nn.ReLU()]
            else:
                layers += [conv2d, nn.ReLU()]
            in_channels = v
    return layers

class VGG16(nn.Layer):
    def __init__(self):
        super(VGG16, self).__init__()
        self.cfg = {'tun': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'], 'tun_ex': [512, 512, 512]}
        self.extract = [8, 15, 22, 29] # [3, 8, 15, 22, 29]
        self.base = nn.LayerList(vgg(self.cfg['tun'], 3))
        for m in self.sublayers():
            if isinstance(m, nn.Conv2D):
                n = m.weight.shape[0] * m.weight.shape[1] * m.weight.shape[2] * m.weight.shape[3]
                m.weight.set_value(paddle.randn(m.weight.shape) * 0.01)
            elif isinstance(m, nn.BatchNorm2D):
                m.weight.set_value(paddle.ones(m.weight.shape))
                m.bias.set_value(paddle.zeros(m.bias.shape))

    def load_pretrained_model(self, model):
        self.base.set_state_dict(model, strict=False)

    def forward(self, x):
        tmp_x = []
        for k, layer in enumerate(self.base):
            x = layer(x)
            if k in self.extract:
                tmp_x.append(x)
        return tmp_x

class VGG16_locate(nn.Layer):
    def __init__(self):
        super(VGG16_locate, self).__init__()
        self.vgg16 = VGG16()
        self.in_planes = 512
        self.out_planes = [512, 256, 128]

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
                m.weight.set_value(paddle.randn(m.weight.shape) * 0.01)
            elif isinstance(m, nn.BatchNorm2D):
                m.weight.set_value(paddle.ones(m.weight.shape))
                m.bias.set_value(paddle.zeros(m.bias.shape))

    def load_pretrained_model(self, model):
        self.vgg16.load_pretrained_model(model)

    def forward(self, x):
        x_size = x.shape[2:]
        xs = self.vgg16(x)

        xls = [xs[-1]]
        for k in range(len(self.ppms)):
            xls.append(F.interpolate(self.ppms[k](xs[-1]), xs[-1].shape[2:], mode='bilinear', align_corners=True))
        xls = self.ppm_cat(paddle.concat(xls, axis=1))
        infos = []
        for k in range(len(self.infos)):
            infos.append(self.infos[k](F.interpolate(xls, xs[len(self.infos) - 1 - k].shape[2:], mode='bilinear', align_corners=True)))

        return xs, infos
