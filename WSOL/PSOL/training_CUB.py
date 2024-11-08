import os
import time
import warnings

import paddle
import paddle.nn as nn
from paddle.optimizer import SGD
from paddle.optimizer.lr import CosineAnnealingDecay
from paddle.regularizer import L2Decay
from paddle.metric import Accuracy
from paddle.vision import transforms
from paddle.io import DataLoader

from loader.cub_loader import CUBDataset
from models.models import *
from utils.IoU import *

warnings.filterwarnings("ignore", "(Possibly )?corrupt EXIF data", UserWarning)
import argparse

def compute_reg_acc(preds, targets, theta=0.5):
    IoU = compute_IoU(preds, targets)
    corr = (IoU >= theta).astype('float32').sum()
    return float(corr) / float(preds.shape[0])

def compute_cls_acc(preds, targets):
    pred = np.argmax(preds, axis=1)
    num_correct = np.sum(pred == targets.numpy())
    return float(num_correct) / float(preds.shape[0])

def compute_acc(reg_preds, reg_targets, cls_preds, cls_targets, theta=0.5):
    IoU = compute_IoU(reg_preds, reg_targets)
    reg_corr = (IoU >= theta)

    pred = np.argmax(cls_preds, axis=1)
    cls_corr = (pred == cls_targets.numpy())

    corr = np.logical_and(reg_corr, cls_corr).astype('float32').sum()

    return float(corr) / float(reg_preds.shape[0])

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.cnt = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt

train_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Training

parser = argparse.ArgumentParser(description='Parameters for CCAM evaluation')
parser.add_argument('--loc_model', metavar='locarg', type=str, default='resnet50', dest='locmodel')
parser.add_argument('--input_size', default=256, dest='input_size')
parser.add_argument('--crop_size', default=224, dest='crop_size')
parser.add_argument('--epochs', default=100, dest='epochs')
parser.add_argument('--gpu', help='which gpu to use', default='4,5,6,7', dest='gpu')
parser.add_argument('--pseudo_bboxes_path', help='generated ddt path', required=True, dest="pseudo_bboxes_path")
parser.add_argument('--save_path', help='model save path', default='CUB', dest='save_path')
parser.add_argument('--batch_size', default=64, dest='batch_size')
parser.add_argument('data', metavar='DIR', help='path to imagenet dataset')

args = parser.parse_args()
batch_size = int(args.batch_size)
lr = 0.002
momentum = 0.9
weight_decay = 1e-4
print_freq = 500
root = args.data
savepath = args.save_path
os.environ['MKL_NUM_THREADS'] = '20'

MyTrainData = CUBDataset(root=root, pseudo_bboxes_path=args.pseudo_bboxes_path, train=True,
                         input_size=args.input_size, crop_size=args.crop_size,
                         transform=train_transform)
MyTestData = CUBDataset(root=root, pseudo_bboxes_path=args.pseudo_bboxes_path, train=False,
                        input_size=args.input_size, crop_size=args.crop_size,
                        transform=test_transform)

train_loader = DataLoader(dataset=MyTrainData,
                          batch_size=batch_size,
                          shuffle=True, num_workers=20, drop_last=True)
test_loader = DataLoader(dataset=MyTestData, batch_size=batch_size,
                         num_workers=8, drop_last=False)
dataloaders = {'train': train_loader, 'test': test_loader}

# construct model
model = choose_locmodel(args.locmodel)
model = paddle.DataParallel(model)
reg_criterion = nn.MSELoss()
# reg_criterion = nn.SmoothL1Loss()

if 'densenet' in args.locmodel:
    dense1_params = list(map(id, model.module.classifier.parameters()))
    rest_params = filter(lambda x: id(x) not in dense1_params, model.parameters())
    param_list = [{'params': model.module.classifier.parameters(), 'lr': 2 * lr},
                  {'params': rest_params, 'lr': 1 * lr}]
else:
    dense1_params = list(map(id, model.module.fc.parameters()))
    rest_params = filter(lambda x: id(x) not in dense1_params, model.parameters())
    param_list = [{'params': model.module.fc.parameters(), 'lr': 2 * lr},
                  {'params': rest_params, 'lr': 1 * lr}]

optimizer = SGD(parameters=param_list, learning_rate=lr, momentum=momentum, weight_decay=L2Decay(weight_decay))
scheduler = CosineAnnealingDecay(learning_rate=optimizer.get_lr(), T_max=len(train_loader) * int(args.epochs))
paddle.set_device('gpu')
best_model_state = model.state_dict()
best_epoch = -1
best_acc = 0.0

epoch_loss = {'train': [], 'test': []}
epoch_acc = {'train': [], 'test': []}
epochs = int(args.epochs)
lambda_reg = 0
for epoch in range(epochs):
    lambda_reg = 50
    for phase in ('train',):
        reg_accs = AverageMeter()
        accs = AverageMeter()
        reg_losses = AverageMeter()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        if phase == 'train':
            model.train()
        else:
            model.eval()

        end = time.time()
        cnt = 0
        for ims, labels, boxes in dataloaders[phase]:
            data_time.update(time.time() - end)
            inputs = ims
            boxes = boxes
            labels = labels

            optimizer.clear_grad()

            # forward
            if phase == 'train':
                if 'inception' in args.locmodel:
                    reg_outputs1, reg_outputs2 = model(inputs)
                    reg_loss1 = reg_criterion(reg_outputs1, boxes)
                    reg_loss2 = reg_criterion(reg_outputs2, boxes)
                    reg_loss = 1 * reg_loss1 + 0.3 * reg_loss2
                    reg_outputs = reg_outputs1
                else:
                    reg_outputs = model(inputs)
                    reg_loss = reg_criterion(reg_outputs, boxes)
            else:
                with paddle.no_grad():
                    reg_outputs = model(inputs)
                    reg_loss = reg_criterion(reg_outputs, boxes)
            loss = lambda_reg * reg_loss
            reg_acc = compute_reg_acc(reg_outputs.numpy(), boxes.numpy())

            nsample = inputs.shape[0]
            reg_accs.update(reg_acc, nsample)
            reg_losses.update(reg_loss.numpy()[0], nsample)
            if phase == 'train':
                loss.backward()
                optimizer.step()
                # fixed learning
                scheduler.step()
            batch_time.update(time.time() - end)
            end = time.time()
            if cnt % print_freq == 0:
                print(
                    '[{}]\tEpoch: {}/{}\t lr: {:.4f} \t Iter: {}/{} Time {:.3f} ({:.3f})\t Data {:.3f} ({:.3f})\tLoc Loss: {:.4f}\tLoc Acc: {:.2%}\t'.format(
                        phase, epoch + 1, epochs, scheduler.get_lr(), cnt, len(dataloaders[phase]),
                        batch_time.val, batch_time.avg,
                        data_time.val, data_time.avg, lambda_reg * reg_losses.avg, reg_accs.avg))
            cnt += 1
        if phase == 'test' and reg_accs.avg > best_acc:
            best_acc = reg_accs.avg
            best_epoch = epoch
            best_model_state = model.state_dict()

        elapsed_time = time.time() - end
        print(
            '[{}]\tEpoch: {}/{}\tLoc Loss: {:.4f}\tLoc Acc: {:.2%}\tTime: {:.3f}'.format(
                phase, epoch + 1, epochs, lambda_reg * reg_losses.avg, reg_accs.avg, elapsed_time))
        epoch_loss[phase].append(reg_losses.avg)
        epoch_acc[phase].append(reg_accs.avg)

    if not os.path.exists(savepath):
        os.makedirs(savepath)
    paddle.save(model.state_dict(), os.path.join(savepath,
                                                'checkpoint_localization_imagenet_' + args.locmodel + "_" + str(
                                                    epoch + 1) + '.pdparams'))
    paddle.save(best_model_state, os.path.join(savepath,
                                              'best_cls_localization_imagenet_' + args.locmodel + "_" + str(
                                                  epoch + 1) + '.pdparams'))
