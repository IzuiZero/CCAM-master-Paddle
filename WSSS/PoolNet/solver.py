import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.optimizer import Adam
from paddle.io import DataLoader
import numpy as np
import os
import cv2
import time
from torchvision.utils import make_grid

# Importing the PoolNet and weights_init functions from the networks module (assuming they are defined there)
from networks.poolnet import build_model, weights_init

class Solver(object):
    def __init__(self, train_loader, test_loader, config):
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.config = config
        self.iter_size = config.iter_size
        self.show_every = config.show_every
        self.lr_decay_epoch = [15,]
        self.build_model()
        if config.mode == 'test':
            print('Loading pre-trained model from %s...' % self.config.model)
            if self.config.cuda:
                self.net.set_state_dict(paddle.load(self.config.model))
            else:
                self.net.set_state_dict(paddle.load(self.config.model, map_location='cpu'))
            self.net.eval()

    # print the network information and parameter numbers
    def print_network(self, model, name):
        num_params = sum(p.numel() for p in model.parameters())
        print(name)
        print(model)
        print("The number of parameters: {}".format(num_params))

    # build the network
    def build_model(self):
        self.net = build_model(self.config.arch)
        if self.config.cuda:
            self.net = self.net.cuda()
        # self.net.train()
        self.net.eval()  # use_global_stats = True
        self.net.apply(weights_init)

        if self.config.load == '':
            # Assuming `load_pretrained_model` is a method in the base model of PoolNet
            self.net.base.load_pretrained_model(paddle.load(self.config.pretrained_model))
        else:
            self.net.set_state_dict(paddle.load(self.config.load))

        self.lr = self.config.lr
        self.wd = self.config.wd

        self.optimizer = Adam(parameters=filter(lambda p: p.requires_grad, self.net.parameters()), learning_rate=self.lr, weight_decay=self.wd)

    def test(self):
        mode_name = 'sal_fuse'
        time_s = time.time()
        img_num = len(self.test_loader)
        for i, data_batch in enumerate(self.test_loader()):
            images, image_id = data_batch['sal_image'], data_batch['image_id'][0]
            with paddle.no_grad():
                images = paddle.to_tensor(images)
                b, c, h, w = images.shape
                if h <= 100 or w <= 100:
                    images = F.interpolate(images, scale_factor=2, mode='bilinear')
                if self.config.cuda:
                    images = images.cuda()
                preds = self.net(images)
                if h <= 100 or w <= 100:
                    preds = F.interpolate(preds, size=(h, w), mode='bilinear')

                pred = np.squeeze(paddle.sigmoid(preds).cpu().numpy())
                multi_fuse = 255 * pred
                if os.path.isfile(os.path.join(self.config.sal_folder, image_id + '.png')):
                    continue
                cv2.imwrite(os.path.join(self.config.sal_folder, image_id + '.png'), multi_fuse)
        time_e = time.time()
        print('Speed: %f FPS' % (img_num/(time_e-time_s)))
        print('Test Done!')

    # training phase
    def train(self):
        mode_name = 'sal_fuse'
        iter_num = len(self.train_loader) // self.config.batch_size
        aveGrad = 0
        for epoch in range(self.config.epoch):
            r_sal_loss= 0
            self.net.clear_gradients()
            for i, data_batch in enumerate(self.train_loader()):
                sal_image, sal_label, image_id = data_batch['sal_image'], data_batch['sal_label'], data_batch['image_id'][0]
                # print(sal_image.shape, sal_label.shape)
                if (sal_image.shape[2] != sal_label.shape[2]) or (sal_image.shape[3] != sal_label.shape[3]):
                    print('IMAGE ERROR, PASSING```')
                    continue
                b, c, h, w = sal_image.shape
                if h <= 100 or w <= 100:
                    continue
                sal_image, sal_label= paddle.to_tensor(sal_image), paddle.to_tensor(sal_label)
                if self.config.cuda:
                    sal_image, sal_label = sal_image.cuda(), sal_label.cuda()

                sal_pred = self.net(sal_image)
                sal_loss_fuse = F.binary_cross_entropy_with_logits(sal_pred, sal_label, reduction='sum')
                sal_loss = sal_loss_fuse / (self.iter_size * self.config.batch_size)
                r_sal_loss += sal_loss.numpy()

                sal_loss.backward()

                aveGrad += 1

                # accumulate gradients as done in DSS
                if aveGrad % self.iter_size == 0:
                    self.optimizer.step()
                    self.optimizer.clear_gradients()
                    aveGrad = 0

                if i % (self.show_every // self.config.batch_size) == 0:
                    if i == 0:
                        x_showEvery = 1
                    print('epoch: [%2d/%2d], iter: [%5d/%5d]  ||  Sal : %10.4f' % (
                        epoch, self.config.epoch, i, iter_num, r_sal_loss/x_showEvery))
                    print('Learning rate: ' + str(self.lr))
                    r_sal_loss= 0
                if i % 200 == 0:
                    pred = np.squeeze(paddle.sigmoid(sal_pred).cpu().numpy())
                    multi_fuse = 255 * pred
                    cv2.imwrite(os.path.join(self.config.test_fold, image_id + '_sal' + '.png'), multi_fuse)

                    grid = make_grid(sal_image, nrow=1, padding=0, pad_value=0,
                         normalize=True, range=None)
                    # Add 0.5 after unnormalizing to [0, 255] to round to nearest integer
                    image = grid.mul_(255).add_(0.5).clip(0, 255).numpy().astype('uint8')
                    cv2.imwrite(os.path.join(self.config.test_fold, image_id + '.png'), image)

            if (epoch + 1) % self.config.epoch_save == 0:
                paddle.save(self.net.state_dict(), '%s/models/epoch_%d.pth' % (self.config.save_folder, epoch + 1))

            if epoch in self.lr_decay_epoch:
                self.lr = self.lr * 0.1
                self.optimizer = Adam(parameters=filter(lambda p: p.requires_grad, self.net.parameters()), learning_rate=self.lr, weight_decay=self.wd)

        paddle.save(self.net.state_dict(), '%s/models/final.pth' % self.config.save_folder)

def bce2d(input, target, reduction=None):
    assert input.shape == target.shape
    pos = paddle.equal(target, 1.0).astype('float32')
    neg = paddle.equal(target, 0.0).astype('float32')

    num_pos = paddle.sum(pos)
    num_neg = paddle.sum(neg)
    num_total = num_pos + num_neg

    alpha = num_neg / num_total
    beta = 1.1 * num_pos / num_total
    # target pixel = 1 -> weight beta
    # target pixel = 0 -> weight 1-beta
    weights = alpha * pos + beta * neg

    return F.binary_cross_entropy_with_logits(input, target, weight=weights, reduction=reduction)
