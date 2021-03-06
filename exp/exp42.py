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

sys.path.append("/usr/src/app/Kaggle_Bengali2019")

from src.augmentations import GridMask
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
                                train_one_epoch_cutmix_for_single_output, train_one_epoch_mixup_cutmix_plane_for_single_output

from src.resnet_model import resnext50_32x4d
from src.model import CnnModel

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


EXP_ID = "exp42_takuoko_cbam_net50_use_pretrain_cutmix_crop_not_threshold_75epoch_1e-4"
LOGGER_PATH = f"logs/log_{EXP_ID}.txt"
setup_logger(out_file=LOGGER_PATH)
LOGGER.info("seed={}".format(SEED))


SIZE = 128
HEIGHT=137
WIDTH=236
OUT_DIR = 'models'

# https://albumentations.readthedocs.io/en/latest/api/augmentations.html
data_transforms = albumentations.Compose([
    albumentations.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=(15,30), p=0.5),
    albumentations.CenterCrop(96, 96, p=1),
    # albumentations.Cutout(p=0.3),
    albumentations.Resize(128, 128, p=1),
    albumentations.Cutout(num_holes=1, max_h_size=50, max_w_size=25, p=0.3),
    albumentations.OneOf([
            GridMask(num_grid=3, rotate=(15, 30), p=0.3),
            GridMask(num_grid=4, rotate=(5, 15), p=0.3),
        ], p=0.3),
    ])

data_transforms_test = albumentations.Compose([
    albumentations.Flip(p=0),     
    albumentations.CenterCrop(96, 96, p=1),
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

        if self.transform is not None:
            image = np.array(image)
            image = self.transform(image=image)['image'] 
            image = (image.astype(np.uint8) - 0.0692) / 0.2051
            image = image_to_tensor(image, normalize=None) 
        else:
            image = np.array(image)
            image = (image.astype(np.uint8) - 0.0692) / 0.2051
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


# replace relu to prelu
def convert_model_ReLU2PReLU(module):
    mod = module
    if isinstance(module, torch.nn.ReLU):
        mod = nn.PReLU()
    for name, child in module.named_children():
        mod.add_module(name, convert_model_ReLU2PReLU(child))
    return mod


batch_size_list = [12, 36, 42, 48, 64]


with timer('load csv data'):
    fold_id = 0
    epochs = 75
    batch_size = batch_size_list[3]
    
    train = pd.read_csv('input/train.csv')
    
    y = train[["grapheme_root", "vowel_diacritic", "consonant_diacritic"]]

    num_folds = 5
    kf = MultilabelStratifiedKFold(n_splits = num_folds, random_state = SEED)
    splits = list(kf.split(X=train, y=y))
    train_idx = splits[fold_id][0]
    val_idx = splits[fold_id][1]
    
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


with timer('create model'):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    model = CnnModel(num_classes=186, encoder="resnet50_cbam", pretrained="imagenet", pool_type="gem")
    # model = CnnModel(num_classes=186, encoder="se_resnext50_32x4d", pretrained="imagenet", pool_type="concat")
    # model = convert_model_ReLU2Swish(model)
    model = model.to(device)
    # model.load_state_dict(torch.load("models/exp19_custom_fc_mixup_cutmix_45epoch_4e-4_fold0.pth"))
    # LOGGER.info("exp19 model loaded")

    criterion = nn.CrossEntropyLoss(reduction='mean').to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, eps=1e-4)
    scheduler = GradualWarmupScheduler(optimizer, multiplier=1.1, total_epoch=5,
                                       after_scheduler=None)


with timer('training loop'):
    best_score = -999
    best_epoch = 0
    for epoch in range(1, epochs + 1):

        LOGGER.info("Starting {} epoch...".format(epoch))
        # tr_loss = train_one_epoch_mixup_cutmix_for_single_output(model, train_loader, criterion, optimizer, device)
        tr_loss = train_one_epoch_mixup_cutmix_plane_for_single_output(model, train_loader, criterion, optimizer, device)
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

