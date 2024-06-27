import math
import numbers
import random
import warnings
import copy
from PIL import Image
from paddle.vision.transforms import functional as F

from .func import *


class RandomHorizontalFlipBBox(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, bbox):
        if random.random() < self.p:
            flipbox = copy.deepcopy(bbox)
            flipbox[0] = 1 - bbox[2]
            flipbox[2] = 1 - bbox[0]
            return F.hflip(img), flipbox

        return img, bbox


class RandomResizedBBoxCrop(object):
    def __init__(self, size, scale=(0.2, 1.0), ratio=(3. / 4., 4. / 3.), interpolation=Image.BILINEAR):
        if isinstance(size, tuple):
            self.size = size
        else:
            self.size = (size, size)
        if (scale[0] > scale[1]) or (ratio[0] > ratio[1]):
            warnings.warn("range should be of kind (min, max)")

        self.interpolation = interpolation
        self.scale = scale
        self.ratio = ratio

    @staticmethod
    def get_params(img, bbox, scale, ratio):
        area = img.size[0] * img.size[1]

        for attempt in range(30):
            target_area = random.uniform(*scale) * area
            log_ratio = (math.log(ratio[0]), math.log(ratio[1]))
            aspect_ratio = math.exp(random.uniform(*log_ratio))

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if w <= img.size[0] and h <= img.size[1]:
                i = random.randint(0, img.size[1] - h)
                j = random.randint(0, img.size[0] - w)
                intersec = compute_intersec(i, j, h, w, bbox)

                if intersec[2] - intersec[0] > 0 and intersec[3] - intersec[1] > 0:
                    intersec = normalize_intersec(i, j, h, w, intersec)
                    return i, j, h, w, intersec

        in_ratio = img.size[0] / img.size[1]
        if in_ratio < min(ratio):
            w = img.size[0]
            h = int(round(w / min(ratio)))
        elif in_ratio > max(ratio):
            h = img.size[1]
            w = int(round(h * max(ratio)))
        else:
            w = img.size[0]
            h = img.size[1]

        i = (img.size[1] - h) // 2
        j = (img.size[0] - w) // 2

        intersec = compute_intersec(i, j, h, w, bbox)
        intersec = normalize_intersec(i, j, h, w, intersec)
        return i, j, h, w, intersec

    def __call__(self, img, bbox):
        i, j, h, w, crop_bbox = self.get_params(img, bbox, self.scale, self.ratio)
        return F.resized_crop(img, i, j, h, w, self.size, self.interpolation), crop_bbox


class RandomBBoxCrop(object):
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    @staticmethod
    def get_params(img, bbox, output_size):
        w, h = img.size
        th, tw = output_size
        if w == tw and h == th:
            return 0, 0, h, w

        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
        intersec = compute_intersec(i, j, h, w, bbox)
        intersec = normalize_intersec(i, j, h, w, intersec)
        return i, j, th, tw, intersec

    def __call__(self, img, bbox):
        i, j, h, w, crop_bbox = self.get_params(img, bbox, self.size)
        return F.crop(img, i, j, h, w), crop_bbox

    def __repr__(self):
        return self.__class__.__name__ + '(size={0}, padding={1})'.format(self.size, self.padding)


class ResizedBBoxCrop(object):
    def __init__(self, size, interpolation=Image.BILINEAR):
        self.size = size
        self.interpolation = interpolation

    @staticmethod
    def get_params(img, bbox, size):
        if isinstance(size, int):
            w, h = img.size
            if (w <= h and w == size) or (h <= w and h == size):
                img = copy.deepcopy(img)
                ow, oh = w, h
            if w < h:
                ow = size
                oh = int(size * h / w)
            else:
                oh = size
                ow = int(size * w / h)
        else:
            ow, oh = size[::-1]
            w, h = img.size

        intersec = copy.deepcopy(bbox)
        ratew = ow / w
        rateh = oh / h
        intersec[0] = bbox[0] * ratew
        intersec[2] = bbox[2] * ratew
        intersec[1] = bbox[1] * rateh
        intersec[3] = bbox[3] * rateh

        return (oh, ow), intersec

    def __call__(self, img, bbox):
        size, crop_bbox = self.get_params(img, bbox, self.size)
        return F.resize(img, self.size, self.interpolation), crop_bbox


class CenterBBoxCrop(object):
    def __init__(self, size, interpolation=Image.BILINEAR):
        self.size = size
        self.interpolation = interpolation

    @staticmethod
    def get_params(img, bbox, size):
        if isinstance(size, numbers.Number):
            output_size = (int(size), int(size))

        w, h = img.size
        th, tw = output_size

        i = int(round((h - th) / 2.))
        j = int(round((w - tw) / 2.))

        intersec = compute_intersec(i, j, th, tw, bbox)
        intersec = normalize_intersec(i, j, th, tw, intersec)

        return i, j, th, tw, intersec

    def __call__(self, img, bbox):
        i, j, th, tw, crop_bbox = self.get_params(img, bbox, self.size)
        return F.center_crop(img, self.size), crop_bbox
