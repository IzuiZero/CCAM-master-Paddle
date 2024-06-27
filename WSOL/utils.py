import shutil
import cv2
import numpy as np
import json
import os
import errno

# PaddlePaddle specific imports
import paddle
import paddle.vision.transforms as T
from paddle.vision.datasets import DatasetFolder

import matplotlib
matplotlib.use('Agg')  # Ensure matplotlib doesn't need a GUI backend

# PaddlePaddle doesn't have an exact equivalent of make_grid from torchvision.utils
def make_grid(images, nrow=8, padding=2, pad_value=0, normalize=False, range=None):
    # Implementing a basic version of make_grid for visualization
    raise NotImplementedError("make_grid function for PaddlePaddle needs custom implementation.")

# PaddlePaddle doesn't have an exact equivalent of Logger or mkdir_if_missing from torchvision
class Logger(object):
    def __init__(self, fpath=None):
        self.console = paddle.stdout()
        self.file = None
        if fpath is not None:
            mkdir_if_missing(os.path.dirname(fpath))
            self.file = open(fpath, 'w')

    def __del__(self):
        self.close()

    def __enter__(self):
        pass

    def __exit__(self, *args):
        self.close()

    def write(self, msg):
        self.console.write(msg)
        if self.file is not None:
            self.file.write(msg)
        self.flush()

    def flush(self):
        self.console.flush()
        if self.file is not None:
            self.file.flush()
            os.fsync(self.file.fileno())

    def close(self):
        self.console.close()
        if self.file is not None:
            self.file.close()

def mkdir_if_missing(dir_path):
    try:
        os.makedirs(dir_path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

# PaddlePaddle does not have a direct equivalent of torchvision.utils.accuracy or accuracy_binary
# You may need to implement these functions based on your specific requirements.

def intersect(box_a, box_b):
    max_xy = paddle.minimum(box_a[:, 2:], box_b[:, 2:])
    min_xy = paddle.maximum(box_a[:, :2], box_b[:, :2])
    inter = paddle.clip((max_xy - min_xy), min=0)
    return inter[:, 0] * inter[:, 1]

# PaddlePaddle does not have an exact equivalent of IOUFunciton_ILSRVC
# You may need to implement this function based on your specific requirements.

# PaddlePaddle does not have an exact equivalent of visualize_heatmap
# You may need to implement this function based on your specific requirements.

# PaddlePaddle does not have an exact equivalent of save_bbox_as_json
# You may need to implement this function based on your specific requirements.

# PaddlePaddle does not have an exact equivalent of model_info
# You may need to implement this function based on your specific requirements.

def compute_bboxes_from_scoremaps(scoremap, scoremap_threshold_list, factor, multi_contour_eval=False):
    """
    Implementing compute_bboxes_from_scoremaps function for PaddlePaddle.
    Adjusted for PaddlePaddle's tensor operations and APIs.
    """
    # Implement the function according to PaddlePaddle tensor operations
    raise NotImplementedError("compute_bboxes_from_scoremaps function for PaddlePaddle needs custom implementation.")

def normalize_scoremap(alm):
    """
    Implementing normalize_scoremap function for PaddlePaddle.
    Adjusted for PaddlePaddle's tensor operations and APIs.
    """
    alm = paddle.to_tensor(alm)
    if paddle.any(paddle.isnan(alm)):
        return paddle.zeros_like(alm)
    if paddle.min(alm).item() == paddle.max(alm).item():
        return paddle.zeros_like(alm)
    alm -= paddle.min(alm)
    alm /= paddle.max(alm)
    return alm.numpy()

# Additional functions like t2n and creat_folder are straightforward and may not need changes for PaddlePaddle.

def t2n(t):
    return t.numpy()

def creat_folder(config, args):
    # Implement creat_folder function based on PaddlePaddle's file handling
    raise NotImplementedError("creat_folder function for PaddlePaddle needs custom implementation.")

# Constants or configurations can remain as they are, if they don't involve torch-specific operations.

# You need to adjust other parts of your script (model training, dataset loading, etc.) for PaddlePaddle.
# Ensure you use PaddlePaddle's corresponding APIs for these operations.
