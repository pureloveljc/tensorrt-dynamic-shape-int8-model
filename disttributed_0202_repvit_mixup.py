import argparse
import os
import random
import shutil
import time
import warnings
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from timm.data import Mixup
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from torchvision import transforms
from PIL import ImageOps


from timm.models import create_model
import model
model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))
filenmes = "/home/all/ljc/code_commit/repvit_cls_train/white_1114/"
best_filenames = os.path.join(filenmes, 'model_best_1114_repvit_4.pth')
checkpoint_filenames = os.path.join(filenmes, 'model_1114_repvit_4.pth.tar')
# pre_path = '/home/all/ljc/code_commit/EfficientNet-PyTorch/icon_0922/model_best_0922_69.pth'
pre_path = None


FINAL_CLASSES = ['????', '????', '????', '??????']
print('FINAL_CLASSES_len:' + str(len(FINAL_CLASSES)))

lr_pre = 0.0001
batch_size = 160
num_class = len(FINAL_CLASSES)
default_data = '/home/all/ljc/white_screen_cls_all/all//'
parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--data',
                    metavar='DIR',
                    default=default_data,
                    help='path to dataset')
parser.add_argument('-a',
                    '--arch',
                    metavar='ARCH',
                    default='resnet34',
                    choices=model_names,
                    help='model architecture: ' + ' | '.join(model_names) +
                         ' (default: resnet18)')
parser.add_argument('-j',
                    '--workers',
                    default=8,
                    type=int,
                    metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs',
                    default=150,
                    type=int,
                    metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch',
                    default=0,
                    type=int,
                    metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b',
                    '--batch-size',
                    default=batch_size,
                    type=int,
                    metavar='N',
                    help='mini-batch size (default: 3200), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr',
                    '--learning-rate',
                    default=lr_pre,
                    type=float,
                    metavar='LR',
                    help='initial learning rate',
                    dest='lr')
parser.add_argument('--momentum',
                    default=0.9,
                    type=float,
                    metavar='M',
                    help='momentum')
parser.add_argument('--local_rank',
                    default=-1,
                    type=int,
                    help='node rank for distributed training')
parser.add_argument('--wd',
                    '--weight-decay',
                    default=1e-4,
                    type=float,
                    metavar='W',
                    help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p',
                    '--print-freq',
                    default=10,
                    type=int,
                    metavar='N',
                    help='print frequency (default: 10)')
parser.add_argument('-e',
                    '--evaluate',
                    dest='evaluate',
                    action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained',
                    dest='pretrained',
                    action='store_true',
                    help='use pre-trained model')
parser.add_argument('--pretrained_path',
                    default=pre_path,
                    help='use pre-trained model')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')

parser.add_argument('--seed',
                    default=42,
                    type=int,
                    help='seed for initializing training. ')

parser.add_argument('--local-rank', default=-1, type=int, help='Local rank for distributed training')


def reduce_mean(tensor, nprocs):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= nprocs
    return rt


def main():
    args = parser.parse_args()
    args.nprocs = torch.cuda.device_count()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    main_worker(args.local_rank, args.nprocs, args)


def main_worker(local_rank, nprocs, args):
    best_acc1 = .0

    dist.init_process_group(backend='nccl')


    model = create_model(
        'repvit_m2_3',
        num_classes=num_class,
        pretrained=False,
    )
    print(model)


    torch.cuda.set_device(local_rank)
    model.cuda(local_rank)
    # When using a single GPU per process and per
    # DistributedDataParallel, we need to divide the batch size
    # ourselves based on the total number of GPUs we have
    args.batch_size = int(args.batch_size / nprocs)
    model = torch.nn.parallel.DistributedDataParallel(model,
                                                      device_ids=[local_rank])

    # # torch load  finetune
    # checkpoint = torch.load(args.pretrained_path)
    # # print(checkpoint)
    # model.load_state_dict(checkpoint['state_dict'])

    # criterion = nn.CrossEntropyLoss().cuda(local_rank)
    criterion = SoftTargetCrossEntropy().cuda(local_rank)


    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=args.lr,
                                 weight_decay=args.weight_decay)

    if args.pretrained_path:
        checkpoint = torch.load(args.pretrained_path, map_location=lambda storage, loc: storage.cuda(local_rank))
        # If you saved the entire model (not recommended)
        # model = torch.load(args.pretrained_path)

        # If you saved only the state dict (recommended)
        # For strict=False, It will ignore non-matching keys
        model.load_state_dict(checkpoint['state_dict'])

    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            if args.gpu is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                best_acc1 = best_acc1.to(args.gpu)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # Data loading code
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'valid')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    from torchvision.transforms import functional as F
    from PIL import ImageOps
    import cv2
    from PIL import Image
    import numpy as np
    from torchvision.transforms import Lambda
    import io
    def convert_to_bgr(image):
        if image.mode != "RGB":
            image = image.convert("RGB")
        image = ImageOps.exif_transpose(image)  # ????????EXIF????????????????????????
        image = Image.fromarray(cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR))  # ????????????BGR????????????PIL.Image
        return image

    class RandomBlur(object):
        def __init__(self, kernel_size, p=0.5):
            self.kernel_size = kernel_size
            self.p = p

        def __call__(self, img):
            if random.random() < self.p:
                img = np.array(img)
                img = cv2.blur(img, (self.kernel_size, self.kernel_size))
                img = F.to_pil_image(img)
            return img

    class RandomImageEncode:
        def __call__(self, image):
            formats = ['JPEG', 'PNG']
            chosen_format = random.choice(formats)

            buffer = io.BytesIO()
            image.save(buffer, format=chosen_format)
            buffer.seek(0)
            return Image.open(buffer)

    random_image_encode = Lambda(RandomImageEncode())

    import random
    from PIL import ImageOps

    class ExpandEdgeTransform:
        def __init__(self, min_expand, max_expand, probability=0.4):
            self.min_expand = min_expand
            self.max_expand = max_expand
            self.probability = probability

        def __call__(self, image):
            if random.random() < self.probability:
                expand_pixel = random.randint(self.min_expand, self.max_expand)
                edge_color = image.getpixel((0, 0))
                return ImageOps.expand(image, border=expand_pixel, fill=edge_color)
            else:
                return image


    edge_expand_transform = ExpandEdgeTransform(10, 30)

    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomRotation(5),
            # transforms.Resize(80),
            edge_expand_transform,
            transforms.Resize((224, 224)),  # ????????????????????????????
            # transforms.RandomHorizontalFlip(p=0.5),#???????????????????????? ????????????????????????????????
            # transforms.RandomVerticalFlip(p=0.5),#????????????????????????
            # transforms.RandomCrop(130, padding=2),
            # transforms.RandomCrop((222, 222), padding=2),  # randomly jitter 1-2 pixels
            transforms.ColorJitter(brightness=0.2, contrast=0.3, saturation=0.3, hue=0.3),
            RandomBlur(3, p=0.5),


            #  transforms.ColorJitter(brightness=0.1),
            random_image_encode,
            convert_to_bgr,

            transforms.ToTensor(),
            #  transforms.Normalize([0, 0, 0], [1, 1, 1]),#????????????????????????????????????,
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),

        ]))

    # original_classes = train_dataset.classes
    # print(original_classes)

    class_to_idx = {cls: idx for idx, cls in enumerate(FINAL_CLASSES)}

    mixup_fn = Mixup(mixup_alpha=0.4, cutmix_alpha=0, prob=1.0, switch_prob=0.5, mode='batch', label_smoothing=0.1,
                     num_classes=num_class)
    train_dataset.class_to_idx = class_to_idx
    # train_dataset.targets = [class_to_idx[class_name] for class_name, _ in train_dataset.samples]

    new_samples = [(path, class_to_idx[os.path.dirname(path).split('/')[-1]]) for path, _ in train_dataset.samples]
    train_dataset.samples = new_samples
    train_dataset.targets = [item[1] for item in new_samples]

    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset)
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=args.batch_size,
                                               num_workers=8,
                                               pin_memory=True,
                                               sampler=train_sampler)

    val_dataset = datasets.ImageFolder(
        valdir,
        transforms.Compose([
            transforms.Resize((224, 224)),
            convert_to_bgr,
            transforms.ToTensor(),
            # transforms.Normalize([0, 0, 0], [1, 1, 1]),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]))

    class_to_idx_val = {cls: idx for idx, cls in enumerate(FINAL_CLASSES)}
    val_dataset.class_to_idx = class_to_idx_val
    new_samples_val = [(path, class_to_idx_val[os.path.dirname(path).split('/')[-1]]) for path, _ in
                       val_dataset.samples]
    val_dataset.samples = new_samples_val
    val_dataset.targets = [item[1] for item in new_samples_val]

    # print(train_dataset.targets)
    # print(val_dataset.targets)
    val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=args.batch_size,
                                             num_workers=4,
                                             pin_memory=True,
                                             sampler=val_sampler)
    print("Class to index mapping for training:", train_dataset.class_to_idx)
    print("Class to index mapping for validation:", val_dataset.class_to_idx)

    if args.evaluate:
        validate(val_loader, model, criterion, local_rank, args)
        return

    for epoch in range(args.start_epoch, args.epochs):
        train_sampler.set_epoch(epoch)
        val_sampler.set_epoch(epoch)

        adjust_learning_rate(optimizer, epoch, args)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, local_rank,
              args, mixup_fn)

        # evaluate on validation set
        acc1 = validate(val_loader, model, criterion, local_rank, args)

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        if args.local_rank == 0:
            save_checkpoint(
                {
                    'epoch': epoch + 1,
                    'arch': args.arch,
                    'state_dict': model.state_dict(),
                    'best_acc1': best_acc1,
                    'optimizer': optimizer.state_dict(),

                }, is_best, best_filenames, checkpoint_filenames)


def train(train_loader, model, criterion, optimizer, epoch, local_rank, args, mixup_fn):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(len(train_loader),
                             [batch_time, data_time, losses, top1, top5],
                             prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        if len(images) % 2 != 0:
            print('len(images) % 2 != 0')
            continue
        if mixup_fn is not None:
            images, target = mixup_fn(images, target)


        images = images.cuda(local_rank, non_blocking=True)
        target = target.cuda(local_rank, non_blocking=True)

        # compute output
        output = model(images)
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 4))

        torch.distributed.barrier()

        reduced_loss = reduce_mean(loss, args.nprocs)
        reduced_acc1 = reduce_mean(acc1, args.nprocs)
        reduced_acc5 = reduce_mean(acc5, args.nprocs)

        losses.update(reduced_loss.item(), images.size(0))
        top1.update(reduced_acc1.item(), images.size(0))
        top5.update(reduced_acc5.item(), images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)


def validate(val_loader, model, criterion, local_rank, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(len(val_loader), [batch_time, losses, top1, top5],
                             prefix='Test: ')
    criterion = torch.nn.CrossEntropyLoss().cuda(local_rank)

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            images = images.cuda(local_rank, non_blocking=True)
            target = target.cuda(local_rank, non_blocking=True)

            # compute output
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 4))

            torch.distributed.barrier()

            reduced_loss = reduce_mean(loss, args.nprocs)
            reduced_acc1 = reduce_mean(acc1, args.nprocs)
            reduced_acc5 = reduce_mean(acc5, args.nprocs)

            losses.update(reduced_loss.item(), images.size(0))
            top1.update(reduced_acc1.item(), images.size(0))
            top5.update(reduced_acc5.item(), images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

        # TODO: this should also be done with the ProgressMeter
        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'.format(top1=top1,
                                                                    top5=top5))

    return top1.avg


def save_checkpoint(state, is_best, best_filenmes, ckp_filenames):
    torch.save(state, ckp_filenames)
    if is_best:
        shutil.copyfile(ckp_filenames, best_filenmes)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
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

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


# def accuracy(output, target, topk=(1,)):
#     """Computes the accuracy over the k top predictions for the specified values of k"""
#     with torch.no_grad():
#         maxk = max(topk)
#         batch_size = target.size(0)
#
#         _, pred = output.topk(maxk, 1, True, True)
#         pred = pred.t()
#         correct = pred.eq(target.view(1, -1).expand_as(pred))
#
#         res = []
#         for k in topk:
#             correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
#             res.append(correct_k.mul_(100.0 / batch_size))
#         return res
def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()


        if target.dim() == 1:
            correct = pred.eq(target.view(1, -1).expand_as(pred))
        else:
            target_max, target_max_indices = target.max(dim=1)
            correct = pred.eq(target_max_indices.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

if __name__ == '__main__':
    main()
# CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 /home/all/ljc/code_commit/repvit_cls_train/disttributed_0202_repvit_1031_white_cls_mixup.py
