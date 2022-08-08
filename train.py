import torch
from torch.autograd import Variable
import time
import os
import sys

from lib.utils.utils import *


CLASS_NAMES = [
    'travelling', 
    'lifting brick', 
    'lifting rebar', 
    'measuring rebar', 
    'tying rebar', 
    'hammering', 
    'drilling', 
    'idle'
]

# CLASS_COUNT = [4, 4, 4, 5, 4, 3, 3, 5]
# flag = True

# txt_file_name = 'results/save_epoch.txt'
# cur_txt_file = txt_file_name


def train_epoch(epoch, data_loader, model, criterion, optimizer, opt,
                epoch_logger, batch_logger, writer):
    print('train at epoch {}'.format(epoch))

    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    end_time = time.time()
    for i, (inputs, targets) in enumerate(data_loader):
        data_time.update(time.time() - end_time)

        if not opt.no_cuda:
            targets = targets.cuda()

        inputs = Variable(inputs)
        targets = Variable(targets)
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        losses.update(loss.data, inputs.size(0))
        prec1, prec5 = calculate_accuracy(outputs.data, targets.data, topk=(1,5))
        top1.update(prec1, inputs.size(0))
        top5.update(prec5, inputs.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end_time)
        end_time = time.time()

        batch_logger.log({
            'epoch': epoch,
            'batch': i + 1,
            'iter': (epoch - 1) * len(data_loader) + (i + 1),
            'loss': losses.val.item(),
            'prec1': top1.val.item(),
            'prec5': top5.val.item(),
            'lr': optimizer.param_groups[0]['lr']
        })

        print('Epoch: [{0}][{1}/{2}]\t lr: {lr:.5f}\t'
                'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                'Prec@1 {top1.val:.5f} ({top1.avg:.5f})\t'
                'Prec@5 {top5.val:.5f} ({top5.avg:.5f})'.format(
                    epoch,
                    i,
                    len(data_loader),
                    batch_time=batch_time,
                    data_time=data_time,
                    loss=losses,
                    top1=top1,
                    top5=top5,
                    lr=optimizer.param_groups[0]['lr']))

    epoch_logger.log({
        'epoch': epoch,
        'loss': losses.avg.item(),
        'prec1': top1.avg.item(),
        'prec5': top5.avg.item(),
        'lr': optimizer.param_groups[0]['lr']
    })

    writer.add_scalar('training loss', losses.avg.item(), epoch)
    writer.add_scalar('training_acc', top1.avg.item()/100, epoch)



def val_epoch(epoch, data_loader, model, criterion, opt, logger, writer):
    correct_pred = {classname: 0 for classname in CLASS_NAMES}
    total_pred = {classname: 0 for classname in CLASS_NAMES} 
    # global cur_txt_file, flag

    print('validation at epoch {}'.format(epoch))

    model.eval()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    end_time = time.time()

    for i, (inputs, targets) in enumerate(data_loader):
        data_time.update(time.time() - end_time)

        if not opt.no_cuda:
            targets = targets.cuda()
        with torch.no_grad():
            inputs = Variable(inputs)
            targets = Variable(targets)

        outputs = model(inputs)
        loss = criterion(outputs, targets)

        _, predictions = torch.max(outputs, 1)
        for label, prediction in zip(targets, predictions):
            if label == prediction:
                correct_pred[CLASS_NAMES[label]] += 1
            total_pred[CLASS_NAMES[label]] += 1

        prec1, prec5 = calculate_accuracy(outputs.data, targets.data, topk=(1,5))

        top1.update(prec1, inputs.size(0))
        top5.update(prec5, inputs.size(0))

        losses.update(loss.data, inputs.size(0))

        batch_time.update(time.time() - end_time)
        end_time = time.time()

        print('Epoch: [{0}][{1}/{2}]\t'
              'Time {batch_time.val:.5f} ({batch_time.avg:.5f})\t'
              'Data {data_time.val:.5f} ({data_time.avg:.5f})\t'
              'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
              'Prec@1 {top1.val:.5f} ({top1.avg:.5f})\t'
              'Prec@5 {top5.val:.5f} ({top5.avg:.5f})'.format(
                  epoch,
                  i + 1,
                  len(data_loader),
                  batch_time=batch_time,
                  data_time=data_time,
                  loss=losses,
                  top1=top1,
                  top5=top5))

        logger.log({'epoch': epoch,
                'loss': losses.avg.item(),
                'prec1': top1.avg.item(),
                'prec5': top5.avg.item()})

    for classname, correct_count in correct_pred.items():
        # acc = correct_count / total_pred[classname]
        acc = correct_count / total_pred[classname] * 100
        print(f'Accuracy for class: {classname:5s} is {acc:.1f} %')
                
    writer.add_scalar('validation loss', losses.avg.item(), epoch)
    writer.add_scalar('validation acc', top1.avg.item()/100, epoch)

    return losses.avg.item(), top1.avg.item()

    # # make txt file if not exist txt file
    # if not os.path.exists(cur_txt_file):
    #     f = open(cur_txt_file, 'w')
    #     f.write('travelling lifting brick lifting rebar measuring rebar tying rebar hammering drilling idle\n')
    #     flag = False
    # else:
    #     # check resuming
    #     cf = open(cur_txt_file, 'r')
    #     txt_len = len(cf.readlines())
    #     if txt_len-1 >= epoch-100:
    #         print('weight for duplicate class accuracy')
    #         with open(cur_txt_file, 'r') as f:
    #             tmp = f.readlines()
    #     else:
    #         flag = False
    #     cf.close()

    #     # if resume with checkpoint, run this code
    #     if flag:
    #         with open(cur_txt_file, 'w') as f:
    #             for j, line in enumerate(tmp):
    #                 if j <= epoch-1:
    #                     f.write(line)
    #                 if j == epoch:
    #                     a = line.split(' ')

    #         # cur_txt_file = 'results/resume.txt'
    #         j = 0
    #         for k in total_pred:
    #             total_pred[k] = CLASS_COUNT[j] * (100)
    #             j += 1
            
    #         j = 1
    #         for k  in correct_pred:
    #             correct_pred[k] = total_pred[k] * float(a[j]) // 100
    #             j += 1

    #         f.close()
    #     f = open(cur_txt_file, 'a')

    #     f.write(f'{epoch}')
    #     print(correct_pred)
    #     print(total_pred)

        # for classname, correct_count in correct_pred.items():
        #     accuracy = 100 * float(correct_count) / total_pred[classname]
        #     print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')
        #     f.write(f' {accuracy:.1f}')

        # f.write('\n')
        # f.close()

    
