import numpy as np
import pandas as pd
import albumentations
import argparse
import collections
import cv2
import datetime
import gc
import glob
import logging
import math
import operator
import os 
import pickle
import pkg_resources
import random
import re
import scipy.stats as stats
import seaborn as sns
import shutil
import sys
import time
import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torchvision import models, transforms
from contextlib import contextmanager
from collections import OrderedDict
from sklearn import metrics
from sklearn import model_selection
from sklearn.model_selection import KFold, GroupKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import mean_squared_log_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from torch.nn import CrossEntropyLoss, MSELoss
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils import model_zoo
from torch.utils.data import (Dataset,DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
import tensorflow as tf
import PIL
from PIL import Image

from tqdm import tqdm, tqdm_notebook, trange
import warnings
warnings.filterwarnings('ignore')
# from apex import amp

from torch.utils.data.distributed import DistributedSampler

try:
    from torch.utils.tensorboard import SummaryWriter
except:
    from tensorboardX import SummaryWriter

import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

sys.path.append("/usr/src/app/kaggle/Kaggle_Bengali2019")

# from src.augmentations import GridMask
from src.machine_learning_util import seed_everything, prepare_labels, DownSampler, timer, \
                                      to_pickle, unpickle
from src.iterative_stratification import MultilabelStratifiedKFold
from src.image_util import resize_to_square_PIL, pad_PIL, threshold_image, \
                           bbox, crop_resize, Resize, \
                           image_to_tensor, train_one_epoch, validate, macro_recall
from src.scheduler import GradualWarmupScheduler
from src.layers import ResidualBlock
from src.image_bengali import rand_bbox, cutmix, mixup, cutmix_criterion, mixup_criterion
from src.trainer_bengali import train_one_epoch_mixup_cutmix, train_one_epoch_mixup_cutmix_for_single_output, validate_for_single_output, \
                                train_one_epoch_cutmix_for_single_output, train_one_epoch_mixup_for_single_output, train_one_epoch_for_single_output

from src.resnet_model import resnext50_32x4d

SEED = 1129
seed_everything(SEED)


LOGGER = logging.getLogger()
FORMATTER = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")


def setup_logger(out_file=None, stderr=True, stderr_level=logging.INFO, file_level=logging.DEBUG):
    LOGGER.handlers = []
    LOGGER.setLevel(min(stderr_level, file_level))

    if stderr:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(FORMATTER)
        handler.setLevel(stderr_level)
        LOGGER.addHandler(handler)

    if out_file is not None:
        handler = logging.FileHandler(out_file)
        handler.setFormatter(FORMATTER)
        handler.setLevel(file_level)
        LOGGER.addHandler(handler)

    LOGGER.info("logger set up")
    return LOGGER


EXP_ID = "exp43_over9000_gem_mixup_75epoch_4e-4"
LOGGER_PATH = f"logs/log_{EXP_ID}.txt"
setup_logger(out_file=LOGGER_PATH)
LOGGER.info("seed={}".format(SEED))


SIZE = 128
HEIGHT=137
WIDTH=236
OUT_DIR = 'models'

import numpy as np
import albumentations
from albumentations.core.transforms_interface import DualTransform


class GridMask(DualTransform):

    def __init__(self, num_grid=3, fill_value=0, rotate=0, mode=0, always_apply=False, p=0.5):
        super(GridMask, self).__init__(always_apply, p)
        if isinstance(num_grid, int):
            num_grid = (num_grid, num_grid)
        if isinstance(rotate, int):
            rotate = (-rotate, rotate)
        self.num_grid = num_grid
        self.fill_value = fill_value
        self.rotate = rotate
        self.mode = mode
        self.masks = None
        self.rand_h_max = []
        self.rand_w_max = []
        
    def init_masks(self, height, width):
        if self.masks is None:
            self.masks = []
            n_masks = self.num_grid[1] - self.num_grid[0] + 1
            for n, n_g in enumerate(range(self.num_grid[0], self.num_grid[1] + 1, 1)):
                grid_h = height / n_g
                grid_w = width / n_g
                this_mask = np.ones((int((n_g + 1) * grid_h), int((n_g + 1) * grid_w))).astype(np.uint8)
                for i in range(n_g + 1):
                    for j in range(n_g + 1):
                        this_mask[
                             int(i * grid_h) : int(i * grid_h + grid_h / 2),
                             int(j * grid_w) : int(j * grid_w + grid_w / 2)
                        ] = self.fill_value
                        if self.mode == 2:
                            this_mask[
                                 int(i * grid_h + grid_h / 2) : int(i * grid_h + grid_h),
                                 int(j * grid_w + grid_w / 2) : int(j * grid_w + grid_w)
                            ] = self.fill_value

                if self.mode == 1:
                    this_mask = 1 - this_mask

                self.masks.append(this_mask)
                self.rand_h_max.append(grid_h)
                self.rand_w_max.append(grid_w)

    def apply(self, image, mask, rand_h, rand_w, angle, **params):
        h, w = image.shape[:2]
        mask = albumentations.augmentations.functional.rotate(mask, angle) if self.rotate[1] > 0 else mask
        mask = mask[:,:,np.newaxis] if image.ndim == 3 else mask
        image *= mask[rand_h:rand_h+h, rand_w:rand_w+w].astype(image.dtype)
        return image

    def get_params_dependent_on_targets(self, params):
        img = params['image']
        height, width = img.shape[:2]
        self.init_masks(height, width)

        mid = np.random.randint(len(self.masks))
        mask = self.masks[mid]
        rand_h = np.random.randint(self.rand_h_max[mid])
        rand_w = np.random.randint(self.rand_w_max[mid])
        angle = np.random.randint(self.rotate[0], self.rotate[1]) if self.rotate[1] > 0 else 0

        return {'mask': mask, 'rand_h': rand_h, 'rand_w': rand_w, 'angle': angle}
    
    @property
    def targets_as_params(self):
        return ['image']

    def get_transform_init_args_names(self):
        return ('num_grid', 'fill_value', 'rotate', 'mode')


## Over9000 Optimizer . Inspired by Iafoss . Over and Out !
##https://github.com/mgrankin/over9000/blob/master/ralamb.py
import torch, math
from torch.optim.optimizer import Optimizer

# RAdam + LARS
class Ralamb(Optimizer):

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        self.buffer = [[None, None, None] for ind in range(10)]
        super(Ralamb, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(Ralamb, self).__setstate__(state)

    def step(self, closure=None):

        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data.float()
                if grad.is_sparse:
                    raise RuntimeError('Ralamb does not support sparse gradients')

                p_data_fp32 = p.data.float()

                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p_data_fp32)
                    state['exp_avg_sq'] = torch.zeros_like(p_data_fp32)
                else:
                    state['exp_avg'] = state['exp_avg'].type_as(p_data_fp32)
                    state['exp_avg_sq'] = state['exp_avg_sq'].type_as(p_data_fp32)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                # Decay the first and second moment running average coefficient
                # m_t
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                # v_t
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)

                state['step'] += 1
                buffered = self.buffer[int(state['step'] % 10)]

                if state['step'] == buffered[0]:
                    N_sma, radam_step_size = buffered[1], buffered[2]
                else:
                    buffered[0] = state['step']
                    beta2_t = beta2 ** state['step']
                    N_sma_max = 2 / (1 - beta2) - 1
                    N_sma = N_sma_max - 2 * state['step'] * beta2_t / (1 - beta2_t)
                    buffered[1] = N_sma

                    # more conservative since it's an approximated value
                    if N_sma >= 5:
                        radam_step_size = math.sqrt((1 - beta2_t) * (N_sma - 4) / (N_sma_max - 4) * (N_sma - 2) / N_sma * N_sma_max / (N_sma_max - 2)) / (1 - beta1 ** state['step'])
                    else:
                        radam_step_size = 1.0 / (1 - beta1 ** state['step'])
                    buffered[2] = radam_step_size

                if group['weight_decay'] != 0:
                    p_data_fp32.add_(-group['weight_decay'] * group['lr'], p_data_fp32)

                # more conservative since it's an approximated value
                radam_step = p_data_fp32.clone()
                if N_sma >= 5:
                    denom = exp_avg_sq.sqrt().add_(group['eps'])
                    radam_step.addcdiv_(-radam_step_size * group['lr'], exp_avg, denom)
                else:
                    radam_step.add_(-radam_step_size * group['lr'], exp_avg)

                radam_norm = radam_step.pow(2).sum().sqrt()
                weight_norm = p.data.pow(2).sum().sqrt().clamp(0, 10)
                if weight_norm == 0 or radam_norm == 0:
                    trust_ratio = 1
                else:
                    trust_ratio = weight_norm / radam_norm

                state['weight_norm'] = weight_norm
                state['adam_norm'] = radam_norm
                state['trust_ratio'] = trust_ratio

                if N_sma >= 5:
                    p_data_fp32.addcdiv_(-radam_step_size * group['lr'] * trust_ratio, exp_avg, denom)
                else:
                    p_data_fp32.add_(-radam_step_size * group['lr'] * trust_ratio, exp_avg)

                p.data.copy_(p_data_fp32)

        return loss

# Lookahead implementation from https://github.com/rwightman/pytorch-image-models/blob/master/timm/optim/lookahead.py

""" Lookahead Optimizer Wrapper.
Implementation modified from: https://github.com/alphadl/lookahead.pytorch
Paper: `Lookahead Optimizer: k steps forward, 1 step back` - https://arxiv.org/abs/1907.08610
"""
import torch
from torch.optim.optimizer import Optimizer
from collections import defaultdict

class Lookahead(Optimizer):
    def __init__(self, base_optimizer, alpha=0.5, k=6):
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f'Invalid slow update rate: {alpha}')
        if not 1 <= k:
            raise ValueError(f'Invalid lookahead steps: {k}')
        defaults = dict(lookahead_alpha=alpha, lookahead_k=k, lookahead_step=0)
        self.base_optimizer = base_optimizer
        self.param_groups = self.base_optimizer.param_groups
        self.defaults = base_optimizer.defaults
        self.defaults.update(defaults)
        self.state = defaultdict(dict)
        # manually add our defaults to the param groups
        for name, default in defaults.items():
            for group in self.param_groups:
                group.setdefault(name, default)

    def update_slow(self, group):
        for fast_p in group["params"]:
            if fast_p.grad is None:
                continue
            param_state = self.state[fast_p]
            if 'slow_buffer' not in param_state:
                param_state['slow_buffer'] = torch.empty_like(fast_p.data)
                param_state['slow_buffer'].copy_(fast_p.data)
            slow = param_state['slow_buffer']
            slow.add_(group['lookahead_alpha'], fast_p.data - slow)
            fast_p.data.copy_(slow)

    def sync_lookahead(self):
        for group in self.param_groups:
            self.update_slow(group)

    def step(self, closure=None):
        # print(self.k)
        #assert id(self.param_groups) == id(self.base_optimizer.param_groups)
        loss = self.base_optimizer.step(closure)
        for group in self.param_groups:
            group['lookahead_step'] += 1
            if group['lookahead_step'] % group['lookahead_k'] == 0:
                self.update_slow(group)
        return loss

    def state_dict(self):
        fast_state_dict = self.base_optimizer.state_dict()
        slow_state = {
            (id(k) if isinstance(k, torch.Tensor) else k): v
            for k, v in self.state.items()
        }
        fast_state = fast_state_dict['state']
        param_groups = fast_state_dict['param_groups']
        return {
            'state': fast_state,
            'slow_state': slow_state,
            'param_groups': param_groups,
        }

    def load_state_dict(self, state_dict):
        fast_state_dict = {
            'state': state_dict['state'],
            'param_groups': state_dict['param_groups'],
        }
        self.base_optimizer.load_state_dict(fast_state_dict)

        # We want to restore the slow state, but share param_groups reference
        # with base_optimizer. This is a bit redundant but least code
        slow_state_new = False
        if 'slow_state' not in state_dict:
            print('Loading state_dict from optimizer without Lookahead applied.')
            state_dict['slow_state'] = defaultdict(dict)
            slow_state_new = True
        slow_state_dict = {
            'state': state_dict['slow_state'],
            'param_groups': state_dict['param_groups'],  # this is pointless but saves code
        }
        super(Lookahead, self).load_state_dict(slow_state_dict)
        self.param_groups = self.base_optimizer.param_groups  # make both ref same container
        if slow_state_new:
            # reapply defaults to catch missing lookahead specific ones
            for name, default in self.defaults.items():
                for group in self.param_groups:
                    group.setdefault(name, default)

def LookaheadAdam(params, alpha=0.5, k=6, *args, **kwargs):
     adam = Adam(params, *args, **kwargs)
     return Lookahead(adam, alpha, k)


# RAdam + LARS + LookAHead

# Lookahead implementation from https://github.com/lonePatient/lookahead_pytorch/blob/master/optimizer.py
# RAdam + LARS implementation from https://gist.github.com/redknightlois/c4023d393eb8f92bb44b2ab582d7ec20

def Over9000(params, alpha=0.5, k=6, *args, **kwargs):
     ralamb = Ralamb(params, *args, **kwargs)
     return Lookahead(ralamb, alpha, k)

RangerLars = Over9000 


# https://albumentations.readthedocs.io/en/latest/api/augmentations.html
data_transforms = albumentations.Compose([
    albumentations.Resize(128, 128, p=1),
    # albumentations.Cutout(p=0.3),
    ])

data_transforms_test = albumentations.Compose([
    albumentations.Resize(128, 128, p=1),
    ])


class BengaliAIDataset(torch.utils.data.Dataset):
    
    def __init__(self, df, y=None, transform=None):
        self.df = df.iloc[:, 1:].values

        self.y = y
        self.transform = transform
        self.size = 128

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):

        input_dic = {}
        image = self.df[idx, :].reshape(128, 128).astype(np.uint8)
        image = np.array(image)
        image = (image.astype(np.uint8) - 0.0692) / 0.2051

        if self.transform is not None:
            image = self.transform(image=image)['image']
        else:
            pass
        
        image = image_to_tensor(image, normalize=None)
        input_dic["image"] = image

        if self.y is not None:
            label1 = self.y.vowel_diacritic.values[idx]
            label2 = self.y.grapheme_root.values[idx]
            label3 = self.y.consonant_diacritic.values[idx]
            return input_dic, label1, label2, label3
        else:
            return input_dic


class Swish(nn.Module):

    def __init__(self, inplace=False):
        super().__init__()

        self.inplace = True

    def forward(self, x):
        if self.inplace:
            x.mul_(F.sigmoid(x))
            return x
        else:
            return x * F.sigmoid(x)


# replace relu to swish
def convert_model_ReLU2Swish(module): 
    mod = module
    if isinstance(module, torch.nn.ReLU):
        mod = Swish(inplace=True)
    for name, child in module.named_children():
        mod.add_module(name, convert_model_ReLU2Swish(child))
    return mod


batch_size_list = [14, 42, 46, 64, 72, 128]


with timer('load csv data'):
    fold_id = 0
    epochs = 75
    batch_size = batch_size_list[2]
    
    train = pd.read_csv('input/train.csv')

    mis_label_index = [49823, 2819, 20689]
   
    y = train[["grapheme_root", "vowel_diacritic", "consonant_diacritic"]]

    num_folds = 10
    kf = MultilabelStratifiedKFold(n_splits = num_folds, random_state = SEED)
    splits = list(kf.split(X=train, y=y))
    train_idx = splits[fold_id][0]
    val_idx = splits[fold_id][1]

    print(len(train_idx), len(val_idx))

    for lbl in mis_label_index:
        if lbl in train_idx:
            train_idx.remove(lbl)
        if lbl in val_idx:
            val_idx.remove(lbl)   

    print(len(train_idx), len(val_idx))

    gc.collect()


with timer('load feather data'):
    train_path = [
        'input/resize_cropped_128_train_image_data_0.feather',
        'input/resize_cropped_128_train_image_data_1.feather',
        'input/resize_cropped_128_train_image_data_2.feather',
        'input/resize_cropped_128_train_image_data_3.feather'
    ]

    data0 = pd.read_feather(train_path[0])
    data1 = pd.read_feather(train_path[1])
    data2 = pd.read_feather(train_path[2])
    data3 = pd.read_feather(train_path[3])

    data = pd.concat([data0, data1, data2, data3])
    print(data.shape)
    del data0, data1, data2, data3
    gc.collect()


with timer('prepare validation data'):

    y_train = y.iloc[train_idx]
    train_dataset = BengaliAIDataset(data.iloc[train_idx], y=y_train, transform=data_transforms)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size*4, shuffle=True, num_workers=0, pin_memory=True)
  
    y_val = y.iloc[val_idx]

    val_dataset = BengaliAIDataset(data.iloc[val_idx], y=y_val, transform=data_transforms_test)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size*2, shuffle=False, num_workers=0, pin_memory=True)

    del train_dataset, val_dataset
    gc.collect()


from torch.nn.parameter import Parameter
def gem(x, p=3, eps=1e-6):
    return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1./p)
class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super(GeM,self).__init__()
        self.p = Parameter(torch.ones(1)*p)
        self.eps = eps
    def forward(self, x):
        return gem(x, p=self.p, eps=self.eps)       
    def __repr__(self):
        return self.__class__.__name__ + '(' + 'p=' + '{:.4f}'.format(self.p.data.tolist()[0]) + ', ' + 'eps=' + str(self.eps) + ')'


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.fc(x)
        return x * y


class CNNHead(nn.Module):
    def __init__(self):
        super(CNNHead, self).__init__()

        self.fc = nn.Sequential(
                          nn.Linear(in_features=512*4, out_features=512, bias=True),
                          nn.BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                          nn.Dropout(p=0.2),
                          nn.ReLU(inplace=True),
                          )

        self.se_layer = SELayer(512*4)

        # vowel_diacritic
        self.fc1 = nn.Linear(512,11)
        # grapheme_root
        self.fc2 = nn.Linear(512*4,168)
        # consonant_diacritic
        self.fc3 = nn.Linear(512,7)

    def forward(self, x):

        x_ = self.fc(x)

        x1 = self.fc1(x_)
        x2 = self.fc2(self.se_layer(x))
        x3 = self.fc3(x_)

        out = torch.cat([x1, x2, x3], 1)
        return out


with timer('create model'):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    model = torchvision.models.resnext50_32x4d(pretrained=True)
    # model = convert_model_ReLU2Swish(model)
    # w = model.conv1.weight
    model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    model.avg_pool = GeM()
    model.fc = CNNHead()
    # model.conv1.weight = nn.Parameter(torch.mean(w, dim=1, keepdim=True))
    model = model.to(device)
    # model.load_state_dict(torch.load("models/exp19_custom_fc_mixup_cutmix_45epoch_4e-4_fold0.pth"))
    # LOGGER.info("exp19 model loaded")

    criterion = nn.CrossEntropyLoss(reduction='mean').to(device)
    # optimizer = optim.Adam(model.parameters(), lr=1e-4)
    optimizer =Over9000(model.parameters(), lr=2e-3, weight_decay=1e-3) ## New once
    scheduler = GradualWarmupScheduler(optimizer, multiplier=1.1, total_epoch=5,
                                       after_scheduler=None)



with timer('training loop'):
    best_score = -999
    best_epoch = 0
    for epoch in range(1, epochs + 1):

        LOGGER.info("Starting {} epoch...".format(epoch))
        # tr_loss = train_one_epoch_mixup_cutmix_for_single_output(model, train_loader, criterion, optimizer, device)
        tr_loss =train_one_epoch_cutmix_for_single_output(model, train_loader, criterion, optimizer, device)
        LOGGER.info('Mean train loss: {}'.format(round(tr_loss, 5)))

        val_pred, y_true, val_loss = validate_for_single_output(model, val_loader, criterion, device)
        score = macro_recall(y_true, val_pred)
        LOGGER.info('Mean valid loss: {} score: {}'.format(round(val_loss, 5), round(score, 5)))
        if score > best_score:
            best_score = score
            best_epoch = epoch
            torch.save(model.state_dict(), os.path.join(OUT_DIR, '{}_fold{}.pth'.format(EXP_ID, fold_id)))
            np.save(os.path.join(OUT_DIR, "{}_fold{}.npy".format(EXP_ID, fold_id)), val_pred)
            LOGGER.info("save model at score={} on epoch={}".format(best_score, best_epoch))
        scheduler.step()

    LOGGER.info("best score={} on epoch={}".format(best_score, best_epoch))

