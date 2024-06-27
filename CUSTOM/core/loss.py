import paddle
import paddle.nn as nn
import paddle.nn.functional as F

def cos_simi(embedded_fg, embedded_bg):
    embedded_fg = F.normalize(embedded_fg, axis=1)
    embedded_bg = F.normalize(embedded_bg, axis=1)
    sim = paddle.matmul(embedded_fg, embedded_bg.transpose(perm=[1, 0]))
    
    return paddle.clip(sim, min=0.0005, max=0.9995)

def cos_distance(embedded_fg, embedded_bg):
    embedded_fg = F.normalize(embedded_fg, axis=1)
    embedded_bg = F.normalize(embedded_bg, axis=1)
    sim = paddle.matmul(embedded_fg, embedded_bg.transpose(perm=[1, 0]))
    
    return 1 - sim

def l2_distance(embedded_fg, embedded_bg):
    N, C = embedded_fg.shape
    
    # embedded_fg = F.normalize(embedded_fg, dim=1)
    # embedded_bg = F.normalize(embedded_bg, dim=1)
    
    embedded_fg = paddle.unsqueeze(embedded_fg, axis=1).expand([N, N, C])
    embedded_bg = paddle.unsqueeze(embedded_bg, axis=0).expand([N, N, C])
    
    return paddle.pow(embedded_fg - embedded_bg, 2).sum(axis=2) / C

class SimMinLoss(nn.Layer):
    def __init__(self, metric='cos', reduction='mean'):
        super(SimMinLoss, self).__init__()
        self.metric = metric
        self.reduction = reduction

    def forward(self, embedded_bg, embedded_fg):
        if self.metric == 'l2':
            raise NotImplementedError
        elif self.metric == 'cos':
            sim = cos_simi(embedded_bg, embedded_fg)
            loss = -paddle.log(1 - sim)
        else:
            raise NotImplementedError

        if self.reduction == 'mean':
            return paddle.mean(loss)
        elif self.reduction == 'sum':
            return paddle.sum(loss)

class SimMaxLoss(nn.Layer):
    def __init__(self, metric='cos', alpha=0.25, reduction='mean'):
        super(SimMaxLoss, self).__init__()
        self.metric = metric
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, embedded_bg):
        if self.metric == 'l2':
            raise NotImplementedError
        elif self.metric == 'cos':
            sim = cos_simi(embedded_bg, embedded_bg)
            loss = -paddle.log(sim)
            loss = paddle.maximum(loss, paddle.zeros_like(loss))
            _, indices = paddle.sort(sim, descending=True, axis=1)
            _, rank = paddle.sort(indices, axis=1)
            rank = rank - 1
            rank_weights = paddle.exp(-rank.astype('float32') * self.alpha)
            loss = loss * rank_weights
        else:
            raise NotImplementedError

        if self.reduction == 'mean':
            return paddle.mean(loss)
        elif self.reduction == 'sum':
            return paddle.sum(loss)

if __name__ == '__main__':
    paddle.set_device('gpu')
    
    fg_embedding = paddle.randn((4, 12))
    bg_embedding = paddle.randn((4, 12))

    sim_min_loss = SimMinLoss(metric='cos')
    loss_min = sim_min_loss(bg_embedding, fg_embedding)
    print(loss_min.numpy())

    sim_max_loss = SimMaxLoss(metric='cos')
    loss_max = sim_max_loss(bg_embedding)
    print(loss_max.numpy())

    examplar = paddle.to_tensor([[1, 2, 3, 4], [2, 3, 1, 4], [4, 2, 1, 3]])

    _, indices = paddle.sort(examplar, descending=True, axis=1)
    print(indices.numpy())
    _, rank = paddle.sort(indices, axis=1)
    print(rank.numpy())
    rank_weights = paddle.exp(-rank.astype('float32') * 0.25)
    print(rank_weights.numpy())
