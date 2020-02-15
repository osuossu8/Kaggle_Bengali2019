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

sys.path.append("/usr/src/app/kaggle/bengaliai-cv19")

from src.machine_learning_util import seed_everything, prepare_labels, DownSampler, timer, \
                                      to_pickle, unpickle
from src.image_util import resize_to_square_PIL, pad_PIL, threshold_image, \
                           bbox, crop_resize, Resize, \
                           image_to_tensor, train_one_epoch, validate, macro_recall
from src.scheduler import GradualWarmupScheduler
from src.layers import ResidualBlock
from src.image_bengali import rand_bbox, cutmix, mixup, cutmix_criterion, mixup_criterion
from src.trainer_bengali import train_one_epoch_mixup_cutmix, train_one_epoch_mixup_cutmix_for_single_output, validate_for_single_output, train_one_epoch_simple_for_single_output
from src.iterative_stratification import MultilabelStratifiedKFold


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


EXP_ID = "exp27_keroppinet_45epoch_4e-4"
LOGGER_PATH = f"logs/log_{EXP_ID}.txt"
setup_logger(out_file=LOGGER_PATH)
LOGGER.info("seed={}".format(SEED))


SIZE = 128
HEIGHT=137
WIDTH=236
OUT_DIR = 'models'


from albumentations.core.transforms_interface import DualTransform
class GridMask(DualTransform):
    """GridMask augmentation for image classification and object detection.
    
    Author: Qishen Ha
    Email: haqishen@gmail.com
    2020/01/29

    Args:
        num_grid (int): number of grid in a row or column.
        fill_value (int, float, lisf of int, list of float): value for dropped pixels.
        rotate ((int, int) or int): range from which a random angle is picked. If rotate is a single int
            an angle is picked from (-rotate, rotate). Default: (-90, 90)
        mode (int):
            0 - cropout a quarter of the square of each grid (left top)
            1 - reserve a quarter of the square of each grid (left top)
            2 - cropout 2 quarter of the square of each grid (left top & right bottom)

    Targets:
        image, mask

    Image types:
        uint8, float32

    Reference:
    |  https://arxiv.org/abs/2001.04086
    |  https://github.com/akuxcw/GridMask
    """

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


# https://albumentations.readthedocs.io/en/latest/api/augmentations.html
data_transforms = albumentations.Compose([
    # albumentations.Flip(p=0.2),
    # albumentations.Rotate(limit=15, p=0.2),
    albumentations.ShiftScaleRotate(rotate_limit=15, p=0.2),
    # albumentations.Cutout(p=0.2),
    GridMask(num_grid=3, rotate=15, p=0.3),
    ])

data_transforms_test = albumentations.Compose([
    albumentations.Flip(p=0),                                           
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

        if self.transform is not None:
            image = np.array(image)
            image = threshold_image(image)
            image = self.transform(image=image)['image'] 
            image = (image.astype(np.float32) - 0.0692) / 0.2051
            image = image_to_tensor(image, normalize=None) 
        else:
            image = np.array(image)
            image = (image.astype(np.float32) - 0.0692) / 0.2051
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


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.act = Swish(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.act(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.act = Swish(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.act(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.act(out)

        return out


class ResMagicNet(nn.Module):

    def __init__(self, block, layers, num_classes=186):
        self.inplanes = 64
        super(ResMagicNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.act = Swish(inplace=True)
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        # self.avgpool = nn.AvgPool2d(7)
        self.avgpool = nn.AvgPool2d(1)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act(x)
        # x = self.maxpool(x)

        x = self.layer1(x)
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = self.layer2(x)
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = self.layer3(x)
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def keroppinet34(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResMagicNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        raise NotImplementedError()
    return model


batch_size_list = [36, 42, 64]


with timer('load csv data'):
    fold_id = 0
    epochs = 45
    batch_size = batch_size_list[1]
    
    train = pd.read_csv('input/train.csv')
    
    y = train[["grapheme_root", "vowel_diacritic", "consonant_diacritic"]]

    num_folds = 5    

    kf = MultilabelStratifiedKFold(n_splits = num_folds, random_state = SEED)
    splits = list(kf.split(X=train, y=y))
    train_idx = splits[fold_id][0]
    val_idx = splits[fold_id][1]
    # train_idx, val_idx = train_test_split(train.index.tolist(), test_size=0.15, random_state=SEED, stratify=train["vowel_diacritic"])

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


with timer('split data for pretraining'):
    df_train = train.iloc[train_idx]
    data_train = data.iloc[train_idx]
    y_train = y.iloc[train_idx]

    df_val = train.iloc[val_idx]
    data_val = data.iloc[val_idx]
    y_val = y.iloc[val_idx]


with timer('prepare data'):
    train_dataset = BengaliAIDataset(data_train, y=y_train, transform=data_transforms)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size*4, shuffle=True, num_workers=0, pin_memory=True)

    val_dataset = BengaliAIDataset(data_val, y=y_val, transform=data_transforms_test)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size*2, shuffle=False, num_workers=0, pin_memory=True)
    
    del train_dataset, val_dataset
    gc.collect()


with timer('create model'):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    model = keroppinet34()

    model = model.to(device)
    # model.load_state_dict(torch.load("models/exp24_pretrained_true_mixup_cutmix_45epoch_4e-4_fold0.pth"))

    criterion = nn.CrossEntropyLoss(reduction='mean').to(device)
    optimizer = optim.Adam(model.parameters(), lr=4e-4)
    scheduler = GradualWarmupScheduler(optimizer, multiplier=1.1, total_epoch=5,
                                       after_scheduler=None)


with timer('training loop'):

    best_score = -999
    best_epoch = 0
    for epoch in range(1, epochs + 1):
      
        LOGGER.info("Starting {} epoch...".format(epoch))
        tr_loss = train_one_epoch_mixup_cutmix_for_single_output(model, train_loader, criterion, optimizer, device)
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

