import argparse
import time
import os
import sys
import random
import numpy as np
import paddle
import paddle.nn as nn
import paddle.distributed as dist
from paddle.io import DataLoader
from paddle.vision import transforms
from paddle.optimizer.lr import CosineAnnealingDecay
from paddle.metric import Accuracy
from models.model import get_model  # Assuming you have implemented this in 'models/model.py'
from dataset.ilsvrc import ILSVRC2012  # Assuming you have implemented this in 'dataset/ilsvrc.py'
from utils import creat_folder, Logger, visualize_heatmap, compute_bboxes_from_scoremaps, save_bbox_as_json

# Benchmark before running
paddle.set_device('gpu')
paddle.disable_static()
paddle.seed(1)
np.random.seed(1)
random.seed(1)

def parse_arg():
    parser = argparse.ArgumentParser(description="train CCAM on ILSRVC dataset")
    parser.add_argument('--cfg', type=str, default='config/CCAM_ILSVRC.yaml',
                        help='experiment configuration filename')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--port', type=int, default=2345)
    parser.add_argument('--epoch', type=int, default=5)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--arch', type=str, default='resnet')
    parser.add_argument('--ckpt', type=str, default='')
    parser.add_argument('--alpha', type=float, default=0.05)
    parser.add_argument('--experiment', type=str, required=True, help='record different experiments')
    parser.add_argument('--pretrained', type=str, required=True,
                        help='adopt different pretrained parameters, [supervised, mocov2, detco]')
    parser.add_argument('--evaluate', type=bool, default=False, help='evaluation mode')

    args = parser.parse_args()

    with open(args.cfg, 'r') as f:
        config = yaml.load(f)
        config = edict(config)
    config.EXPERIMENT = args.experiment
    config.EVALUATE = args.evaluate
    config.PORT = args.port
    config.LR = args.lr
    config.ALPHA = args.alpha
    config.EPOCHS = args.epoch
    config.SEED = args.seed
    config.PRETRAINED = args.pretrained

    return config, args


def main():
    config, args = parse_arg()
    config.BATCH_SIZE = args.batch_size

    if config.SEED != -1:
        paddle.seed(config.SEED)
        np.random.seed(config.SEED)
        random.seed(config.SEED)

    # Create folder for logging
    creat_folder(config, args)

    # Log
    sys.stdout = Logger('{}/{}_log.txt'.format(config.LOG_DIR, config.EXPERIMENT))

    config.nprocs = paddle.distributed.ParallelEnv().world_size
    print(config.nprocs, 'processes!')

    paddle.distributed.spawn(main_worker, args=(config.nprocs, config, args), nprocs=config.nprocs)


def main_worker(local_rank, nprocs, config, args):
    dist.init_parallel_env()

    # Create model
    print("=> Creating model...")
    print("alphas is :", config.ALPHA)
    model = get_model(config.PRETRAINED)
    param_groups = model.get_parameter_groups()

    # Move model to GPU
    paddle.set_device(f'gpu:{local_rank}')
    model = paddle.DataParallel(model)

    # Define criterion
    criterion = [SimMaxLoss(metric='cos', alpha=config.ALPHA), SimMinLoss(metric='cos'),
                 SimMaxLoss(metric='cos', alpha=config.ALPHA)]

    # Data augmentation and transformation
    train_transforms = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    test_transforms = transforms.Compose([
        transforms.Resize((480, 480)),
        transforms.CenterCrop((448, 448)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Wrap datasets
    train_data = ILSVRC2012(root=config.ROOT, input_size=256, crop_size=224, train=True, transform=train_transforms)
    test_data = ILSVRC2012(root=config.ROOT, input_size=480, crop_size=448, train=False, transform=test_transforms)

    if local_rank == 0:
        print(f'Loaded {len(train_data)} train images!')
        print(f'Loaded {len(test_data)} test images!')

    # Use DistributedSampler
    train_sampler = paddle.io.DistributedBatchSampler(train_data, batch_size=config.BATCH_SIZE, shuffle=True)
    test_sampler = paddle.io.DistributedBatchSampler(test_data, batch_size=config.BATCH_SIZE, shuffle=False)

    # DataLoaders
    train_loader = DataLoader(train_data, batch_sampler=train_sampler, num_workers=config.WORKERS)
    test_loader = DataLoader(test_data, batch_sampler=test_sampler, num_workers=config.WORKERS)

    # Define optimizer and scheduler
    max_step = len(train_data) // config.BATCH_SIZE * config.EPOCHS
    optimizer = paddle.optimizer.PolyOptimizer([
        {'params': param_groups[0], 'lr': config.LR, 'weight_decay': config.WEIGHT_DECAY},
        {'params': param_groups[1], 'lr': 2 * config.LR, 'weight_decay': 0},
        {'params': param_groups[2], 'lr': 10 * config.LR, 'weight_decay': config.WEIGHT_DECAY},
        {'params': param_groups[3], 'lr': 20 * config.LR, 'weight_decay': 0}
    ], learning_rate=config.LR, weight_decay=config.WEIGHT_DECAY, total_steps=max_step)

    # Optionally resume from checkpoint
    start_epoch = 0

    # Training loop
    for epoch in range(start_epoch, config.EPOCHS):
        train_sampler.set_epoch(epoch)
        test_sampler.set_epoch(epoch)

        # Train
        train(config, train_loader, model, criterion, optimizer, epoch, local_rank, nprocs)

        # Test
        best_CorLoc, best_threshold = test(config, test_loader, model, criterion, epoch, local_rank, nprocs)

        if local_rank == 0:
            paddle.save(
                {"state_dict": model.state_dict(),
                 "epoch": epoch + 1,
                 "CorLoc": best_CorLoc,
                 "Threshold": best_threshold,
                 "Flag": flag,
                 }, f'{config.DEBUG}/checkpoints/{config.EXPERIMENT}/current_epoch_{epoch + 1}.pdparams')

    print('Training finished...')

    if local_rank == 0:
        print('Use the last checkpoint for evaluation...')

    # Evaluate
    evaluate(config, test_loader, model, criterion, best_threshold, flag, local_rank, nprocs)

    if local_rank == 0:
        print('Extracting class-agnostic bboxes using best threshold...')
        print('--------------------------------------------------------')

    # Extract class-agnostic bboxes
    train_transforms = transforms.Compose([
        transforms.Resize((480, 480)),
        transforms.CenterCrop((448, 448)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_data = ILSVRC2012(root=config.ROOT, input_size=480, crop_size=448, train=True, transform=train_transforms)

    if local_rank == 0:
        print(f'Loaded {len(train_data)} train images!')

    train_sampler = paddle.io.DistributedBatchSampler(train_data, batch_size=config.BATCH_SIZE, shuffle=False)
    train_loader = DataLoader(train_data, batch_sampler=train_sampler, num_workers=config.WORKERS)

    extract(config, train_loader, model, best_threshold, flag, local_rank)

    if local_rank == 0:
        paddle.save(
            {"state_dict": model.state_dict(),
             "CorLoc": best_CorLoc,
             "Threshold": best_threshold,
             "Flag": flag,
             }, f'{config.DEBUG}/checkpoints/{config.EXPERIMENT}/last_epoch.pdparams')


def train(config, train_loader, model, criterion, optimizer, epoch, local_rank, nprocs):
    # Set up AverageMeters
    batch_time = paddle.metric.AverageMeter()
    losses = paddle.metric.AverageMeter()
    losses_bg_bg = paddle.metric.AverageMeter()
    losses_bg_fg = paddle.metric.AverageMeter()
    losses_fg_fg = paddle.metric.AverageMeter()

    model.train()

    end = time.time()

    for i, (input, target, cls_name, img_name) in enumerate(train_loader):
        input = input.cuda()

        optimizer.clear_grad()
        fg_feats, bg_feats, ccam = model(input)

        loss1 = criterion[0](bg_feats)
        loss2 = criterion[1](fg_feats)
        loss3 = criterion[2](ccam)

        loss = loss1 + loss2 + loss3

        loss.backward()
        optimizer.step()

        # Measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # Print to console
        if i % config.PRINT_FREQ == 0:
            print(f'Epoch: [{epoch + 1}][{i + 1}/{len(train_loader)}]  '
                  f'Loss {loss.avg:.4f}  Loss (bg-bg): {losses_bg_bg.avg:.4f}  '
                  f'Loss (bg-fg): {losses_bg_fg.avg:.4f}  Loss (fg-fg): {losses_fg_fg.avg:.4f}  '
                  f'Time {batch_time.avg:.4f} sec  ', end='\r')


def test(config, test_loader, model, criterion, epoch, local_rank, nprocs):
    batch_time = paddle.metric.AverageMeter()
    losses = paddle.metric.AverageMeter()
    losses_bg_bg = paddle.metric.AverageMeter()
    losses_bg_fg = paddle.metric.AverageMeter()
    losses_fg_fg = paddle.metric.AverageMeter()

    model.eval()

    end = time.time()
    with paddle.no_grad():
        for i, (input, target, cls_name, img_name) in enumerate(test_loader):
            input = input.cuda()

            # Compute output
            fg_feats, bg_feats, ccam = model(input)

            loss1 = criterion[0](bg_feats)
            loss2 = criterion[1](fg_feats)
            loss3 = criterion[2](ccam)

            loss = loss1 + loss2 + loss3

            # Measure accuracy and record loss
            losses.update(loss)
            losses_bg_bg.update(loss1)
            losses_bg_fg.update(loss2)
            losses_fg_fg.update(loss3)

            # Measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # Print to console
            if i % config.PRINT_FREQ == 0:
                print(f'Test: [{epoch + 1}][{i + 1}/{len(test_loader)}]  '
                      f'Loss {losses.avg:.4f}  Loss (bg-bg): {losses_bg_bg.avg:.4f}  '
                      f'Loss (bg-fg): {losses_bg_fg.avg:.4f}  Loss (fg-fg): {losses_fg_fg.avg:.4f}  '
                      f'Time {batch_time.avg:.4f} sec  ', end='\r')

        print(f'Test: [{epoch + 1}/{config.EPOCHS}]  '
              f'Loss {losses.avg:.4f}  Loss (bg-bg): {losses_bg_bg.avg:.4f}  '
              f'Loss (bg-fg): {losses_bg_fg.avg:.4f}  Loss (fg-fg): {losses_fg_fg.avg:.4f}  '
              f'Time {batch_time.avg:.4f} sec  ')

    return 0.0, 0.0


def evaluate(config, test_loader, model, criterion, threshold, flag, local_rank, nprocs):
    model.eval()

    CorLocs = []
    for i, (input, target, cls_name, img_name) in enumerate(test_loader):
        input = input.cuda()

        with paddle.no_grad():
            # Compute output
            fg_feats, bg_feats, ccam = model(input)

            # Evaluate on CorLoc
            if flag == 1:
                pred_cboxes = compute_bboxes_from_scoremaps(ccam, threshold)
            else:
                pred_cboxes = compute_bboxes_from_scoremaps(fg_feats, threshold)

            clocs = CorLoc(pred_cboxes, cls_name, img_name)

            CorLocs.append(clocs)

    CorLocs = np.array(CorLocs)
    corloc = np.mean(CorLocs)

    return corloc


def extract(config, train_loader, model, threshold, flag, local_rank):
    model.eval()

    all_pred_cboxes = []
    for i, (input, target, cls_name, img_name) in enumerate(train_loader):
        input = input.cuda()

        with paddle.no_grad():
            # Compute output
            fg_feats, bg_feats, ccam = model(input)

            # Extract bounding boxes
            if flag == 1:
                pred_cboxes = compute_bboxes_from_scoremaps(ccam, threshold)
            else:
                pred_cboxes = compute_bboxes_from_scoremaps(fg_feats, threshold)

            all_pred_cboxes.extend(pred_cboxes)

    save_bbox_as_json(all_pred_cboxes, config, local_rank)


if __name__ == '__main__':
    main()
