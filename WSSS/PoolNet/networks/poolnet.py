import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import math

from .deeplab_resnet import resnet50_locate
from .vgg import vgg16_locate


config_vgg = {'convert': [[128,256,512,512,512],[64,128,256,512,512]], 'deep_pool': [[512, 512, 256, 128], [512, 256, 128, 128], [True, True, True, False], [True, True, True, False]], 'score': 128}  # no convert layer, no conv6

config_resnet = {'convert': [[64,256,512,1024,2048],[128,256,256,512,512]], 'deep_pool': [[512, 512, 256, 256, 128], [512, 256, 256, 128, 128], [False, True, True, True, False], [True, True, True, True, False]], 'score': 128}

class ConvertLayer(nn.Layer):
    def __init__(self, list_k):
        super(ConvertLayer, self).__init__()
        up = []
        for i in range(len(list_k[0])):
            up.append(nn.Sequential(nn.Conv2D(list_k[0][i], list_k[1][i], 1, 1, bias_attr=False), nn.ReLU()))
        self.convert0 = nn.LayerList(up)

    def forward(self, list_x):
        resl = []
        for i in range(len(list_x)):
            resl.append(self.convert0[i](list_x[i]))
        return resl

class DeepPoolLayer(nn.Layer):
    def __init__(self, k, k_out, need_x2, need_fuse):
        super(DeepPoolLayer, self).__init__()
        self.pools_sizes = [2,4,8]
        self.need_x2 = need_x2
        self.need_fuse = need_fuse
        pools, convs = [],[]
        for i in self.pools_sizes:
            pools.append(nn.AvgPool2D(kernel_size=i, stride=i))
            convs.append(nn.Conv2D(k, k, 3, 1, 1, bias_attr=False))
        self.pools = nn.LayerList(pools)
        self.convs = nn.LayerList(convs)
        self.relu = nn.ReLU()
        self.conv_sum = nn.Conv2D(k, k_out, 3, 1, 1, bias_attr=False)
        if self.need_fuse:
            self.conv_sum_c = nn.Conv2D(k_out, k_out, 3, 1, 1, bias_attr=False)

    def forward(self, x, x2=None, x3=None):
        x_size = x.shape
        resl = x
        for i in range(len(self.pools_sizes)):
            y = self.convs[i](self.pools[i](x))
            resl = paddle.add(resl, F.interpolate(y, x_size[2:], mode='bilinear'))
        resl = self.relu(resl)
        if self.need_x2:
            resl = F.interpolate(resl, x2.shape[2:], mode='bilinear')
        resl = self.conv_sum(resl)
        if self.need_fuse:
            resl = self.conv_sum_c(paddle.add(paddle.add(resl, x2), x3))
        return resl

class ScoreLayer(nn.Layer):
    def __init__(self, k):
        super(ScoreLayer, self).__init__()
        self.score = nn.Conv2D(k ,1, 1, 1)

    def forward(self, x, x_size=None):
        x = self.score(x)
        if x_size is not None:
            x = F.interpolate(x, x_size[2:], mode='bilinear')
        return x

def extra_layer(base_model_cfg, vgg):
    if base_model_cfg == 'vgg':
        config = config_vgg
    elif base_model_cfg == 'resnet':
        config = config_resnet
    convert_layers, deep_pool_layers, score_layers = [], [], []
    convert_layers = ConvertLayer(config['convert'])

    for i in range(len(config['deep_pool'][0])):
        deep_pool_layers += [DeepPoolLayer(config['deep_pool'][0][i], config['deep_pool'][1][i], config['deep_pool'][2][i], config['deep_pool'][3][i])]

    score_layers = ScoreLayer(config['score'])

    return vgg, convert_layers, deep_pool_layers, score_layers


class PoolNet(nn.Layer):
    def __init__(self, base_model_cfg, base, convert_layers, deep_pool_layers, score_layers):
        super(PoolNet, self).__init__()
        self.base_model_cfg = base_model_cfg
        self.base = base
        self.deep_pool = nn.LayerList(deep_pool_layers)
        self.score = score_layers
        if self.base_model_cfg == 'resnet':
            self.convert = convert_layers

    def forward(self, x):
        x_size = x.shape
        conv2merge, infos = self.base(x)
        if self.base_model_cfg == 'resnet':
            conv2merge = self.convert(conv2merge)
        conv2merge = conv2merge[::-1]

        edge_merge = []
        merge = self.deep_pool[0](conv2merge[0], conv2merge[1], infos[0])
        for k in range(1, len(conv2merge)-1):
            merge = self.deep_pool[k](merge, conv2merge[k+1], infos[k])

        merge = self.deep_pool[-1](merge)
        merge = self.score(merge, x_size)
        return merge

def build_model(base_model_cfg='vgg'):
    if base_model_cfg == 'vgg':
        return PoolNet(base_model_cfg, *extra_layer(base_model_cfg, vgg16_locate()))
    elif base_model_cfg == 'resnet':
        return PoolNet(base_model_cfg, *extra_layer(base_model_cfg, resnet50_locate()))

def weights_init(m):
    if isinstance(m, nn.Conv2D):
        m.weight.set_value(paddle.randn(m.weight.shape) * 0.01)
        if m.bias is not None:
            m.bias.set_value(paddle.zeros(m.bias.shape))
