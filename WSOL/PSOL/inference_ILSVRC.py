import argparse
import os
import numpy as np
from PIL import Image
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.vision import transforms
from models.models import choose_locmodel, choose_clsmodel  # Assuming these are paddle models
from utils.IoU import *
from utils.augment import *
from utils.vis import *

parser = argparse.ArgumentParser(description='Parameters for CCAM evaluation')
parser.add_argument('--loc_model', metavar='locarg', type=str, default='resnet50', dest='locmodel')
parser.add_argument('--cls_model', metavar='clsarg', type=str, default='efficientnetb7', dest='clsmodel')
parser.add_argument('--ckpt', type=str, required=True)
parser.add_argument('--input_size', default=256, dest='input_size')
parser.add_argument('--crop_size', default=224, dest='crop_size')
parser.add_argument('--cls_input_size', default=685)
parser.add_argument('--cls_crop_size', default=600)
parser.add_argument('--ten-crop', help='tencrop', action='store_true', dest='tencrop')
parser.add_argument('--gpu', help='which gpu to use', default='4', dest='gpu')
parser.add_argument('data', metavar='DIR', help='path to imagenet dataset')

args = parser.parse_args()
paddle.set_device(args.gpu)
TEN_CROP = args.tencrop

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
transform = transforms.Compose([
    transforms.Resize((args.input_size, args.input_size)),
    transforms.CenterCrop(args.crop_size),
    transforms.ToTensor(),
    normalize
])
cls_transform = transforms.Compose([
    transforms.Resize((args.cls_input_size, args.cls_input_size)),
    transforms.CenterCrop(args.cls_crop_size),
    transforms.ToTensor(),
    normalize
])
ten_crop_aug = transforms.Compose([
    transforms.Resize((args.input_size, args.input_size)),
    transforms.TenCrop(args.crop_size),
    transforms.Lambda(lambda crops: paddle.stack([transforms.ToTensor()(crop) for crop in crops])),
    transforms.Lambda(lambda crops: paddle.stack([normalize(crop) for crop in crops])),
])

locname = args.locmodel
model = choose_locmodel(locname, True, ckpt_path=args.ckpt)
model.eval()

clsname = args.clsmodel
cls_model = choose_clsmodel(clsname)
cls_model.eval()

root = args.data
val_imagedir = os.path.join(root, 'val')
anno_root = os.path.join(root, 'val_boxes')
val_annodir = os.path.join(anno_root, 'val')

classes = os.listdir(val_imagedir)
classes.sort()
temp_softmax = nn.Softmax()

class_to_idx = {classes[i]: i for i in range(len(classes))}

result = {}

accs = []
accs_top5 = []
loc_accs = []
cls_accs = []
loc_maxboxaccv2 = []
final_cls = []
final_loc = []
final_clsloc = []
final_clsloctop5 = []
final_ind = []

for k in range(1000):
    cls = classes[k]

    IoUSet = []
    IoUSetTop5 = []
    LocSet = []
    ClsSet = []

    files = os.listdir(os.path.join(val_imagedir, cls))
    files.sort()

    for (i, name) in enumerate(files):
        now_index = int(name.split('_')[-1].split('.')[0])
        final_ind.append(now_index - 1)
        xmlfile = os.path.join(val_annodir, name.split('.')[0] + '.xml')
        gt_boxes = get_cls_gt_boxes(xmlfile, cls)
        if len(gt_boxes) == 0:
            continue

        raw_img = Image.open(os.path.join(val_imagedir, cls, name)).convert('RGB')
        w, h = raw_img.size

        with paddle.no_grad():
            img = transform(raw_img)
            img = paddle.unsqueeze(img, 0)
            img = img.to(args.gpu)
            reg_outputs = model(img)

            bbox = to_data(reg_outputs)
            bbox = paddle.squeeze(bbox)
            bbox = bbox.numpy()

            if TEN_CROP:
                img = ten_crop_aug(raw_img)
                img = img.to(args.gpu)
                vgg16_out = cls_model(img)
                vgg16_out = temp_softmax(vgg16_out)
                vgg16_out = paddle.mean(vgg16_out, axis=0, keepdim=True)
                vgg16_out = paddle.topk(vgg16_out, 5, axis=1)[1]
            else:
                img = cls_transform(raw_img)
                img = paddle.unsqueeze(img, 0)
                img = img.to(args.gpu)
                vgg16_out = cls_model(img)
                vgg16_out = paddle.topk(vgg16_out, 5, axis=1)[1]

            vgg16_out = to_data(vgg16_out)
            vgg16_out = paddle.squeeze(vgg16_out)
            vgg16_out = vgg16_out.numpy()
            out = vgg16_out

        ClsSet.append(out[0] == class_to_idx[cls])

        for j in range(len(gt_boxes)):
            temp_list = list(gt_boxes[j])
            raw_img_i, gt_bbox_i = ResizedBBoxCrop((256, 256))(raw_img, temp_list)
            raw_img_i, gt_bbox_i = CenterBBoxCrop((224))(raw_img_i, gt_bbox_i)
            w, h = raw_img_i.size

            gt_bbox_i[0] = gt_bbox_i[0] * w
            gt_bbox_i[2] = gt_bbox_i[2] * w
            gt_bbox_i[1] = gt_bbox_i[1] * h
            gt_bbox_i[3] = gt_bbox_i[3] * h

            gt_boxes[j] = gt_bbox_i

        w, h = raw_img_i.size

        bbox[0] = bbox[0] * w
        bbox[2] = bbox[2] * w + bbox[0]
        bbox[1] = bbox[1] * h
        bbox[3] = bbox[3] * h + bbox[1]

        max_iou = -1
        for gt_bbox in gt_boxes:
            iou = IoU(bbox, gt_bbox)
            if iou > max_iou:
                max_iou = iou

        LocSet.append(max_iou)
        temp_loc_iou = max_iou
        if out[0] != class_to_idx[cls]:
            max_iou = 0

        result[os.path.join(cls, name)] = max_iou
        IoUSet.append(max_iou)

        max_iou = 0
        for i in range(5):
            if out[i] == class_to_idx[cls]:
                max_iou = temp_loc_iou
        IoUSetTop5.append(max_iou)

    cls_loc_acc = np.sum(np.array(IoUSet) > 0.5) / len(IoUSet)
    final_clsloc.extend(IoUSet)
    cls_loc_acc_top5 = np.sum(np.array(IoUSetTop5) > 0.5) / len(IoUSetTop5)
    final_clsloctop5.extend(IoUSetTop5)
    loc_acc = np.sum(np.array(LocSet) > 0.5) / len(LocSet)
    final_loc.extend(LocSet)
    cls_acc = np.sum(np.array(ClsSet)) / len(ClsSet)
    final_cls.extend(ClsSet)

    with open('inference_CorLoc.txt', 'a+') as corloc_f:
        corloc_f.write('{} {}\n'.format(cls, loc_acc))

    accs.append(cls_loc_acc)
    accs_top5.append(cls_loc_acc_top5)
    loc_accs.append(loc_acc)
    cls_accs.append(cls_acc)

    maxboxaccv2 = (np.sum(np.array(LocSet) > 0.3) / len(LocSet) +
                   np.sum(np.array(LocSet) > 0.5) / len(LocSet) +
                   np.sum(np.array(LocSet) > 0.7) / len(LocSet)) / 3

    loc_maxboxaccv2.append(maxboxaccv2)

    if (k + 1) % 100 == 0:
        print(k)

print(accs)
print('Cls-Loc acc {}'.format(np.mean(accs)))
print('Cls-Loc acc Top 5 {}'.format(np.mean(accs_top5)))

print('GT Loc acc {}'.format(np.mean(loc_accs)))
print('Max Box acc {}'.format(np.mean(loc_maxboxaccv2)))
print('{} cls acc {}'.format(clsname, np.mean(cls_accs)))

with open('Corloc_result.txt', 'w') as f:
    for k in sorted(result.keys()):
        f.write('{} {}\n'.format(k, str(result[k])))
