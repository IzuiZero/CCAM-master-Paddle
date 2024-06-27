import cv2
import math
import random
import numpy as np
import paddle
import paddle.nn.functional as F
from paddle.optimizer.lr import LambdaDecay

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    paddle.seed(seed)

def rotation(x, k):
    return paddle.rot90(x, k, (1, 2))

def interleave(x, size):
    s = list(x.shape)
    return x.reshape([-1, size] + s[1:]).transpose([1, 0] + list(range(2, len(s) + 1))).reshape([-1] + s[1:])

def de_interleave(x, size):
    s = list(x.shape)
    return x.reshape([size, -1] + s[1:]).transpose([1, 0] + list(range(2, len(s) + 1))).reshape([-1] + s[1:])

def resize_for_tensors(tensors, size, mode='bilinear', align_corners=False):
    return F.interpolate(tensors, size=size, mode=mode, align_corners=align_corners)

def L1_Loss(A_tensors, B_tensors):
    return paddle.abs(A_tensors - B_tensors)

def L2_Loss(A_tensors, B_tensors):
    return paddle.pow(A_tensors - B_tensors, 2)

# ratio = 0.2, top=20%
def Online_Hard_Example_Mining(values, ratio=0.2):
    b, c, h, w = values.shape
    return paddle.topk(values.reshape([b, -1]), k=int(c * h * w * ratio), axis=-1)[0]

def shannon_entropy_loss(logits, activation=torch.sigmoid, epsilon=1e-5):
    v = activation(logits)
    return -paddle.sum(v * paddle.log(v + epsilon), axis=1).mean()

def make_cam(x, epsilon=1e-5):
    # relu(x) = max(x, 0)
    x = F.relu(x)
    
    b, c, h, w = x.shape

    flat_x = x.reshape([b, c, (h * w)])
    max_value = flat_x.max(axis=-1, keepdim=True)[0].reshape([b, c, 1, 1])
    
    return F.relu(x - epsilon) / (max_value + epsilon)

def one_hot_embedding(label, classes):
    """Embedding labels to one-hot form.

    Args:
      labels: (int) class labels.
      num_classes: (int) number of classes.

    Returns:
      (tensor) encoded labels, sized [N, #classes].
    """
    
    vector = np.zeros((classes), dtype=np.float32)
    if len(label) > 0:
        vector[label] = 1.
    return vector

def calculate_parameters(model):
    return sum(param.numel() for param in model.parameters()) / 1000000.0

def get_learning_rate_from_optimizer(optimizer):
    return optimizer.get_lr()

def get_numpy_from_tensor(tensor):
    return tensor.numpy()

def load_model(model, model_path, parallel=False):
    if parallel:
        model.set_state_dict(paddle.load(model_path))
    else:
        model.set_state_dict(paddle.load(model_path))

def save_model(model, model_path, parallel=False):
    if parallel:
        paddle.save(model.state_dict(), model_path)
    else:
        paddle.save(model.state_dict(), model_path)

def transfer_model(pretrained_model, model):
    pretrained_dict = pretrained_model.state_dict()
    model_dict = model.state_dict()
    
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    
    model_dict.update(pretrained_dict)
    model.set_state_dict(model_dict)

def get_learning_rate(optimizer):
    lr = []
    for param_group in optimizer.param_groups:
        lr += [param_group['lr']]
    return lr

def get_cosine_schedule_with_warmup(optimizer,
                                    warmup_iteration,
                                    max_iteration,
                                    cycles=7./16.
                                    ):
    def _lr_lambda(current_iteration):
        if current_iteration < warmup_iteration:
            return float(current_iteration) / float(max(1, warmup_iteration))

        no_progress = float(current_iteration - warmup_iteration) / float(max(1, max_iteration - warmup_iteration))
        return max(0., math.cos(math.pi * cycles * no_progress))
    
    return LambdaDecay(learning_rate=optimizer.get_lr(),
                       lr_lambda=_lr_lambda)
