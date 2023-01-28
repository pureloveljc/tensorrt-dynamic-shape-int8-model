import argparse
import datetime
import numpy as np
import time
import json
import os
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn

from timm.models.layers import trunc_normal_
from timm.data.mixup import Mixup
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.utils import ModelEma
from optim_factory import create_optimizer, LayerDecayValueAssigner

from datasets import build_dataset
from engine_finetune import train_one_epoch, evaluate

import utils
from utils import NativeScalerWithGradNormCount as NativeScaler
from utils import str2bool, remap_checkpoint_keys
import models.convnextv2 as convnextv2
import torchvision.transforms as transforms
import cv2
from PIL import Image
import torch.nn.functional as F
from torch import nn

def get_args_parser():
    parser = argparse.ArgumentParser('FCMAE fine-tuning', add_help=False)
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Per GPU batch size')
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--update_freq', default=1, type=int,
                        help='gradient accumulation steps')

    # Model parameters
    parser.add_argument('--model', default='convnextv2_tiny', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--input_size', default=224, type=int,
                        help='image input size')
    parser.add_argument('--drop_path', type=float, default=0., metavar='PCT',
                        help='Drop path rate (default: 0.1)')
    parser.add_argument('--layer_decay_type', type=str, choices=['single', 'group'], default='single',
                        help="""Layer decay strategies. The single strategy assigns a distinct decaying value for each layer,
                        whereas the group strategy assigns the same decaying value for three consecutive layers""")

    # EMA related parameters
    parser.add_argument('--model_ema', type=str2bool, default=False)
    parser.add_argument('--model_ema_decay', type=float, default=0.9999, help='')
    parser.add_argument('--model_ema_force_cpu', type=str2bool, default=False, help='')
    parser.add_argument('--model_ema_eval', type=str2bool, default=False, help='Using ema to eval during training.')

    # Optimization parameters
    parser.add_argument('--clip_grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=5e-4, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--layer_decay', type=float, default=1.0)
    parser.add_argument('--min_lr', type=float, default=1e-6, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-6)')
    parser.add_argument('--warmup_epochs', type=int, default=20, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')

    parser.add_argument('--warmup_steps', type=int, default=-1, metavar='N',
                        help='num of steps to warmup LR, will overload warmup_epochs if set > 0')
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--opt_eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt_betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight_decay_end', type=float, default=None, help="""Final value of the
        weight decay. We use a cosine schedule for WD and using a larger decay by
        the end of training improves performance for ViTs.""")

    # Augmentation parameters
    parser.add_argument('--color_jitter', type=float, default=None, metavar='PCT',
                        help='Color jitter factor (enabled only when not using Auto/RandAug)')
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". " + "(default: rand-m9-mstd0.5-inc1)')
    parser.add_argument('--smoothing', type=float, default=0.1,
                        help='Label smoothing (default: 0.1)')

    parser.add_argument('--train_interpolation', type=str, default='bicubic',
                        help='Training interpolation (random, bilinear, bicubic default: "bicubic")')

    # * Random Erase params
    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                        help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')
    parser.add_argument('--resplit', type=str2bool, default=False,
                        help='Do not random erase first (clean) augmentation split')

    # * Mixup params
    parser.add_argument('--mixup', type=float, default=0.,
                        help='mixup alpha, mixup enabled if > 0.')
    parser.add_argument('--cutmix', type=float, default=0.,
                        help='cutmix alpha, cutmix enabled if > 0.')
    parser.add_argument('--cutmix_minmax', type=float, nargs='+', default=None,
                        help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    parser.add_argument('--mixup_prob', type=float, default=1.0,
                        help='Probability of performing mixup or cutmix when either/both is enabled')
    parser.add_argument('--mixup_switch_prob', type=float, default=0.5,
                        help='Probability of switching to cutmix when both mixup and cutmix enabled')
    parser.add_argument('--mixup_mode', type=str, default='batch',
                        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')

    # * Finetuning params
    parser.add_argument('--finetune', default='',
                        help='finetune from checkpoint')
    parser.add_argument('--head_init_scale', default=0.001, type=float,
                        help='classifier head initial scale, typically adjusted in fine-tuning')
    parser.add_argument('--model_key', default='model|module', type=str,
                        help='which key to load from saved state dict, usually model or model_ema')
    parser.add_argument('--model_prefix', default='', type=str)

    # Dataset parameters
    parser.add_argument('--data_path', default='/datasets01/imagenet_full_size/061417/', type=str,
                        help='dataset path')
    parser.add_argument('--nb_classes', default=41, type=int,
                        help='number of the classification types')
    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default=None,
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')

    parser.add_argument('--eval_data_path', default=None, type=str,
                        help='dataset path for evaluation')
    parser.add_argument('--imagenet_default_mean_and_std', type=str2bool, default=True)
    parser.add_argument('--data_set', default='IMNET', choices=['CIFAR', 'IMNET', 'image_folder'],
                        type=str, help='ImageNet dataset path')
    parser.add_argument('--auto_resume', type=str2bool, default=True)
    parser.add_argument('--save_ckpt', type=str2bool, default=True)
    parser.add_argument('--save_ckpt_freq', default=1, type=int)
    parser.add_argument('--save_ckpt_num', default=3, type=int)

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', type=str2bool, default=False,
                        help='Perform evaluation only')
    parser.add_argument('--dist_eval', type=str2bool, default=True,
                        help='Enabling distributed evaluation')
    parser.add_argument('--disable_eval', type=str2bool, default=False,
                        help='Disabling evaluation during training')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', type=str2bool, default=True,
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')

    # Evaluation parameters
    parser.add_argument('--crop_pct', type=float, default=None)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', type=str2bool, default=False)
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')

    parser.add_argument('--use_amp', type=str2bool, default=False,
                        help="Use apex AMP (Automatic Mixed Precision) or not")
    return parser

# Resize(size=256, interpolation=bicubic)
# CenterCrop(size=(224, 224))
# ToTensor()
# Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
transform_test = transforms.Compose([
# (args.input_size, args.input_size),
#                             interpolation=transforms.InterpolationMode.BICUBIC)
    transforms.Resize(size=(256, 256), interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(size=(224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
# transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
# transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

# DEFAULT_CROP_PCT = 0.875
# IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
# IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)
# IMAGENET_INCEPTION_MEAN = (0.5, 0.5, 0.5)
# IMAGENET_INCEPTION_STD = (0.5, 0.5, 0.5)
# IMAGENET_DPN_MEAN = (124 / 255, 117 / 255, 104 / 255)
# IMAGENET_DPN_STD = tuple([1 / (.0167 * 255)] * 3)

parser = argparse.ArgumentParser('FCMAE fine-tuning', parents=[get_args_parser()])
args = parser.parse_args()



# _, config = parse_option()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = convnextv2.__dict__[args.model](
        num_classes=args.nb_classes,
        drop_path_rate=args.drop_path,
        head_init_scale=args.head_init_scale,
    )


class Mycls(nn.Module):
    def __init__(self, ori_net):
        super(Mycls, self).__init__()
        self.ori_net = ori_net

    def forward(self, x):
        x = self.ori_net(x)
        x = F.softmax(x, dim=1)
        scores, index = x.topk(k=1, dim=1)
        index = index.float()
        print(scores)
        print(index)
        outputs = torch.cat([index, scores], dim=1)

        return outputs
# print(model_ft)
# GPU????
model_ft = Mycls(model)
model_ft = model_ft.to(device)


checkpoint = torch.load('/home/all/ljc/code/ConvNeXt-V2/save_0128/checkpoint-best.pth', map_location='cpu')


checkpoint = checkpoint['model']
new_dict_state = {}
for key, value in checkpoint.items():
    # print(key)
    if "module." in key:
        key = key[7:]
    # print(key)
    key = "ori_net."+key

    new_dict_state[key] = value
model_dict = model_ft.state_dict()
model_dict.update(new_dict_state)

model_ft.load_state_dict(model_dict)

model_ft.eval().cuda()
torch.cuda.empty_cache()

# model_ft = model_ft.to(device)
# model_ft.load_state_dict(checkpoint['model'])
# model_ft.eval()


x = torch.ones((1, 3, 224, 224)).cuda()
with torch.no_grad():
    print('getting onnx...')

    y_onnx = torch.onnx._export(model_ft,
                     x,
                     "./convnext.onnx",
                     export_params=True,#????????????????
                     opset_version=11,
                     input_names=["input"], #????????????????????????????????key????
                     output_names=['output'],
                     dynamic_axes={'input':{0:'batchsize'},
                                                      'output':{0:'batchsize'}
                                   },
                                verbose=True,
                                do_constant_folding=True
                     )
with torch.no_grad():
    y = model_ft(x)
    print('~~~~~~~~~')
    print("result", torch.max(torch.abs(y - y_onnx)))
