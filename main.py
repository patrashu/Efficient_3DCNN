import os
import json
import torch
from torch import nn
from torch import optim
from torch.optim import lr_scheduler
from tensorboardX import SummaryWriter

from opts import parse_opts
from lib.models.select_model import generate_model
from lib.transforms.spatial_transforms import *
from lib.transforms.temporal_transforms import *
from lib.transforms.target_transforms import ClassLabel, VideoID
from lib.datasets.dataset import get_training_set, get_validation_set, get_test_set
from lib.utils.utils import *
from train import train_epoch, val_epoch
import inference


if __name__ == '__main__':
    ## set opt
    opt = parse_opts()
    opt.arch = '{}'.format(opt.model)
    opt.mean = get_mean(opt.norm_value, dataset=opt.mean_dataset)
    opt.std = get_std(opt.norm_value)
    opt.store_name = '_'.join([opt.dataset, opt.model, str(opt.width_mult) + 'x',
                               'downsample_' + str(opt.sample_duration)])
    print(opt)
    
    with open(os.path.join(opt.result_path, 'opts.json'), 'w') as opt_file:
        json.dump(vars(opt), opt_file)

    torch.manual_seed(opt.manual_seed)

    model, parameters = generate_model(opt)
    criterion = nn.CrossEntropyLoss()
    if not opt.no_cuda:
        criterion = criterion.cuda()    

    norm_method = Normalize(opt.mean, opt.std)
    ## train
    writer = SummaryWriter(log_dir='runs/lr_scheduling')

    if not opt.no_train:
        assert opt.train_crop in ['random', 'corner', 'center']
        if opt.train_crop == 'random':
            crop_method = MultiScaleRandomCrop(1, opt.sample_size)
        elif opt.train_crop == 'corner':
            crop_method = MultiScaleCornerCrop(1, opt.sample_size)
        elif opt.train_crop == 'center':
            crop_method = MultiScaleCornerCrop(
                1, opt.sample_size, crop_positions=['c'])
        
        spatial_transform = Compose([
            Resize(opt.resize_w, opt.resize_h),
            crop_method,
            SaltImage(),
            ToTensor(opt.norm_value),
            norm_method
        ])
        temporal_transform = TemporalRandomCrop(opt.sample_duration, opt.downsample)
        target_transform = ClassLabel()
        training_data = get_training_set(opt, spatial_transform, temporal_transform, target_transform)

        train_loader = torch.utils.data.DataLoader(
            training_data,
            batch_size=opt.batch_size,
            shuffle=True,
            num_workers=opt.n_threads,
            pin_memory=True,
            )

        
        # visualize dataloader with transform applied
        # import sys
        # import numpy as np
        # from PIL import Image
        # import matplotlib.pyplot as plt

        # i = 0
        # if not os.path.exists('./result_images'):
        #     os.makedirs('./result_images')
        # for _ in range(1):
        #     image, label = next(iter(train_loader))
        #     print(image.size())
        
        #     for img in image:
        #         img = img.permute(1, 0, 2, 3)
        #         print(img.size())

        #         for im in img:
        #             print(im.size())
        #             im = im.permute(1, 2, 0)
        #             np_img = im.cpu().detach().numpy()
        #             np_img = np_img * 255

        #             cv2.imwrite(f'./result_images/image{i}.jpg', np_img)
        #             i += 1

        # sys.exit(0)

        train_logger = Logger(
            os.path.join(opt.result_path, 'train.log'),
            ['epoch', 'loss', 'prec1', 'prec5', 'lr'])
        train_batch_logger = Logger(
            os.path.join(opt.result_path, 'train_batch.log'),
            ['epoch', 'batch', 'iter', 'loss', 'prec1', 'prec5', 'lr'])

        optimizer = optim.SGD(
            parameters,
            lr=opt.learning_rate,
            momentum=opt.momentum,
            dampening=0,
            weight_decay=opt.weight_decay,
            nesterov=True)
        scheduler = lr_scheduler.ReduceLROnPlateau(
            optimizer, 'min', patience=opt.lr_patience)

    # val
    if not opt.no_val:
        spatial_transform = Compose([
            Resize(opt.resize_w, opt.resize_h),
            CenterCrop(opt.sample_size),
            SaltImage(),
            ToTensor(opt.norm_value),
            norm_method
        ])

        temporal_transform = TemporalCenterCrop(opt.sample_duration, opt.downsample)
        target_transform = ClassLabel()
        validation_data = get_validation_set(
            opt, spatial_transform, temporal_transform, target_transform)

        val_loader = torch.utils.data.DataLoader(
            validation_data,
            batch_size=4,
            shuffle=False,
            num_workers=opt.n_threads,
            pin_memory=True)

        val_logger = Logger(
            os.path.join(opt.result_path, 'val.log'), ['epoch', 'loss', 'prec1', 'prec5'])
    
    # resume training
    best_prec1 = 0
    if opt.resume_path:
        print('loading checkpoint {}'.format(opt.resume_path))
        checkpoint = torch.load(opt.resume_path)
        assert opt.arch == checkpoint['arch']
        best_prec1 = checkpoint['best_prec1']
        opt.begin_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])

    # run train/val epoch
    print('run')
    for i in range(opt.begin_epoch, opt.n_epochs + 1):
        if not opt.no_train:
            # adjust_learning_rate(optimizer, i, opt)
            train_epoch(i, train_loader, model, criterion, optimizer, opt,
                        train_logger, train_batch_logger, writer)
            state = {
                'epoch': i,
                'arch': opt.arch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_prec1': best_prec1
                }
            if (i+1) % 5 == 0:
                save_checkpoint(state, i, False, opt)
            
        if not opt.no_val:
            validation_loss, prec1 = val_epoch(i, val_loader, model, criterion, opt,
                                        val_logger, writer)
            is_best = prec1 > best_prec1
            best_prec1 = max(prec1, best_prec1)
            state = {
                'epoch': i,
                'arch': opt.arch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_prec1': best_prec1
                }
            # save_checkpoint(state, i, is_best, opt)
    writer.close()

    # test
    if opt.test:
        spatial_transform = Compose([
            Resize(opt.resize_w, opt.resize_h),
            CenterCrop(opt.sample_size),
            # SaltImage(),
            ToTensor(opt.norm_value), 
            norm_method
        ])
        # temporal_transform = LoopPadding(opt.sample_duration, opt.downsample)
        temporal_transform = TemporalRandomCrop(opt.sample_duration, opt.downsample)
        target_transform = VideoID()

        test_data = get_test_set(
            opt, spatial_transform, temporal_transform, target_transform
        )
        
        test_loader = torch.utils.data.DataLoader(
            test_data,
            batch_size=8,
            shuffle=False,
            num_workers=opt.n_threads,
            pin_memory=True
        )
        
        checkpoint = torch.load(opt.pretrain_path)
        model.load_state_dict(checkpoint['state_dict'])
        inference.test(test_loader, model, opt, test_data.class_names)




