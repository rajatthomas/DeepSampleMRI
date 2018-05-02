from torch.autograd import Variable
import time
import numpy as np
import torch

from utils import AverageMeter, calculate_accuracy
import glob as glob
import os


def load_subject_data(dir_name, opt):


    DX = dir_name.split('_')[1]
    dir_name = os.path.join(opt.root_path, dir_name)
    all_patches = glob.glob(dir_name+'/*.npy')

    batch_size = len(all_patches)

    if DX == 'ASD':
        targets = np.ones(batch_size)

    if DX == 'CON':
        targets = np.zeros(batch_size)

    size_x, size_y, size_z = opt.image_size
    inputs = np.zeros((batch_size, 1, size_x, size_y, size_z))

    for i, p in enumerate(all_patches):
        inputs[i, 0, :, :, :] = np.load(p)

    return torch.from_numpy(inputs).type(torch.FloatTensor), torch.from_numpy(targets).type(torch.LongTensor)


def test_epoch(test_subjects_list, model, criterion, opt, logger):

    model.eval()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    accuracies = AverageMeter()

    end_time = time.time()

    for f in test_subjects_list:

        inputs, targets = load_subject_data(f, opt)

        data_time.update(time.time() - end_time)

        if not opt.no_cuda:
            targets = targets.cuda(async=True)
        inputs = Variable(inputs, volatile=True)
        targets = Variable(targets, volatile=True)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        acc = calculate_accuracy(outputs, targets)

        losses.update(loss.data[0], inputs.size(0))
        accuracies.update(acc, inputs.size(0))

        _, max_loc = torch.max(outputs, 1)
        sample_correct = float(sum(max_loc == targets)/len(targets))

        batch_time.update(time.time() - end_time)
        end_time = time.time()

        print('Test data:\t'
              'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
              'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
              'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
              'Acc {acc.val:.2f} ({acc.avg:.3f})\t'
              'Vote {sample_correct:.3f}'.format(
                  len(test_subjects_list),
                  batch_time=batch_time,
                  data_time=data_time,
                  loss=losses,
                  acc=accuracies,
                  sample_correct=sample_correct))

    logger.log({'loss': losses.avg, 'acc': accuracies.avg, 'vote': sample_correct})

    return losses.avg
