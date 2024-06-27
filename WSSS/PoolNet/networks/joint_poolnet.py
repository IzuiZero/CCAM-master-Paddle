import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from .deeplab_resnet import resnet50_locate
from .vgg import vgg16_locate

config_vgg = {'convert': [[128,256,512,512,512],[64,128,256,512,512]], 'deep_pool': [[512, 512, 256, 128], [512, 256, 128, 128], [True, True, True, False], [True, True, True, False]], 'score': 256, 'edgeinfoc':[48,128], 'block': [[512, [16]], [256, [16]], [128, [16]]], 'fuse': [[16, 16, 16], True]}  # no convert layer, no conv6

config_resnet = {'convert': [[64,256,512,1024,2048],[128,256,256,512,512]], 'deep_pool': [[512, 512, 256, 256, 128], [512, 256, 256, 128, 128], [False, True, True, True, False], [True, True, True, True, False]], 'score': 256, 'edgeinfoc':[64,128], 'block': [[512, [16]], [256, [16]], [256, [16]], [128, [16]]], 'fuse': [[16, 16, 16, 16], True]}

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
            resl = paddle.add(resl, F.interpolate(y, x_size[2:], mode='bilinear', align_corners=True))
        resl = self.relu(resl)
        if self.need_x2:
            resl = F.interpolate(resl, x2.shape[2:], mode='bilinear', align_corners=True)
        resl = self.conv_sum(resl)
        if self.need_fuse:
            resl = self.conv_sum_c(paddle.add(paddle.add(resl, x2), x3))
        return resl

class BlockLayer(nn.Layer):
    def __init__(self, k_in, k_out_list):
        super(BlockLayer, self).__init__()
        up_in1, up_mid1, up_in2, up_mid2, up_out = [], [], [], [], []

        for k in k_out_list:
            up_in1.append(nn.Conv2D(k_in, k_in//4, 1, 1, bias_attr=False))
            up_mid1.append(nn.Sequential(nn.Conv2D(k_in//4, k_in//4, 3, 1, 1, bias_attr=False), nn.Conv2D(k_in//4, k_in, 1, 1, bias_attr=False)))
            up_in2.append(nn.Conv2D(k_in, k_in//4, 1, 1, bias_attr=False))
            up_mid2.append(nn.Sequential(nn.Conv2D(k_in//4, k_in//4, 3, 1, 1, bias_attr=False), nn.Conv2D(k_in//4, k_in, 1, 1, bias_attr=False)))
            up_out.append(nn.Conv2D(k_in, k, 1, 1, bias_attr=False))

        self.block_in1 = nn.LayerList(up_in1)
        self.block_in2 = nn.LayerList(up_in2)
        self.block_mid1 = nn.LayerList(up_mid1)
        self.block_mid2 = nn.LayerList(up_mid2)
        self.block_out = nn.LayerList(up_out)
        self.relu = nn.ReLU()

    def forward(self, x, mode=0):
        x_tmp = self.relu(x + self.block_mid1[mode](self.block_in1[mode](x)))
        # x_tmp = self.block_mid2[mode](self.block_in2[mode](self.relu(x + x_tmp)))
        x_tmp = self.relu(x_tmp + self.block_mid2[mode](self.block_in2[mode](x_tmp)))
        x_tmp = self.block_out[mode](x_tmp)

        return x_tmp

class EdgeInfoLayerC(nn.Layer):
    def __init__(self, k_in, k_out):
        super(EdgeInfoLayerC, self).__init__()
        self.trans = nn.Sequential(nn.Conv2D(k_in, k_in, 3, 1, 1, bias_attr=False), nn.ReLU(),
                                   nn.Conv2D(k_in, k_out, 3, 1, 1, bias_attr=False), nn.ReLU(),
                                   nn.Conv2D(k_out, k_out, 3, 1, 1, bias_attr=False), nn.ReLU(),
                                   nn.Conv2D(k_out, k_out, 3, 1, 1, bias_attr=False), nn.ReLU())

    def forward(self, x, x_size):
        tmp_x = []
        for i_x in x:
            tmp_x.append(F.interpolate(i_x, x_size[2:], mode='bilinear', align_corners=True))
        x = self.trans(paddle.concat(tmp_x, axis=1))
        return x

class FuseLayer1(nn.Layer):
    def __init__(self, list_k, deep_sup):
        super(FuseLayer1, self).__init__()
        up = []
        for i in range(len(list_k)):
            up.append(nn.Conv2D(list_k[i], 1, 1, 1))
        self.trans = nn.LayerList(up)
        self.fuse = nn.Conv2D(len(list_k), 1, 1, 1)
        self.deep_sup = deep_sup

    def forward(self, list_x, x_size):
        up_x = []
        for i, i_x in enumerate(list_x):
            up_x.append(F.interpolate(self.trans[i](i_x), x_size[2:], mode='bilinear', align_corners=True))
        out_fuse = self.fuse(paddle.concat(up_x, axis=1))
        if self.deep_sup:
            out_all = []
            for up_i in up_x:
                out_all.append(up_i)
            return [out_fuse, out_all]
        else:
            return [out_fuse]

class ScoreLayer(nn.Layer):
    def __init__(self, k):
        super(ScoreLayer, self).__init__()
        self.score = nn.Conv2D(k ,1, 3, 1, 1)

    def forward(self, x, x_size=None):
        x = self.score(x)
        if x_size is not None:
            x = F.interpolate(x, x_size[2:], mode='bilinear', align_corners=True)
        return x

def extra_layer(base_model_cfg, base):
    if base_model_cfg == 'vgg':
        config = config_vgg
    elif base_model_cfg == 'resnet':
        config = config_resnet
    convert_layers, deep_pool_layers, block_layers, fuse_layers, edgeinfo_layers, score_layers = [], [], [], [], [], []
    convert_layers = ConvertLayer(config['convert'])

    for k in config['block']:
        block_layers += [BlockLayer(k[0], k[1])]

    for i in range(len(config['deep_pool'][0])):
        deep_pool_layers += [DeepPoolLayer(config['deep_pool'][0][i], config['deep_pool'][1][i], config['deep_pool'][2][i], config['deep_pool'][3][i])]

    fuse_layers = FuseLayer1(config['fuse'][0], config['fuse'][1])
    edgeinfo_layers = EdgeInfoLayerC(config['edgeinfoc'][0], config['edgeinfoc'][1])
    score_layers = ScoreLayer(config['score'])

    return base, convert_layers, deep_pool_layers, block_layers, fuse_layers, edgeinfo_layers, score_layers


class PoolNet(nn.Layer):
    def __init__(self, base_model_cfg, base, convert_layers, deep_pool_layers, block_layers, fuse_layers, edgeinfo_layers, score_layers):
        super(PoolNet, self).__init__()
        self.base_model_cfg = base_model_cfg
        self.base = base
        self.block = nn.LayerList(block_layers)
        self.deep_pool = nn.LayerList(deep_pool_layers)
        self.fuse = fuse_layers
        self.edgeinfo = edgeinfo_layers
        self.score = score_layers
        if self.base_model_cfg == 'resnet':
            self.convert = convert_layers

    def forward(self, x, mode):
        x_size = x.shape
        conv2merge, infos = self.base(x)
        if self.base_model_cfg == 'resnet':
            conv2merge = self.convert(conv2merge)
        conv2merge = conv2merge[::-1]

        edge_merge = []
        merge = self.deep_pool[0](conv2merge[0], conv2merge[1], infos[0])
        edge_merge.append(merge)
        for k in range(1, len(conv2merge)-1):
            merge = self.deep_pool[k](merge, conv2merge[k+1], infos[k])
            edge_merge.append(merge)

        if mode == 0:
            edge_merge = [self.block[i](kk) for i, kk in enumerate(edge_merge)]
            merge = self.fuse(edge_merge, x_size)
        elif mode == 1:
            merge = self.deep_pool[-1](merge)
            edge_merge = [self.block[i](kk).detach() for i, kk in enumerate(edge_merge)]
            edge_merge = self.edgeinfo(edge_merge, merge.shape)
            merge = self.score(paddle.concat([merge, edge_merge], axis=1), x_size)
        return merge

def build_model(base_model_cfg='vgg'):
    if base_model_cfg == 'vgg':
        return PoolNet(base_model_cfg, *extra_layer(base_model_cfg, vgg16_locate()))
    elif base_model_cfg == 'resnet':
        return PoolNet(base_model_cfg, *extra_layer(base_model_cfg, resnet50_locate()))

def weights_init(m):
    if isinstance(m, nn.Conv2D):
        m.weight.data.normal_(0, 0.01)
        if m.bias is not None:
            m.bias.data.zero_()
