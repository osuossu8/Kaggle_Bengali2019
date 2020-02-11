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
from src.trainer_bengali import train_one_epoch_with_weighted_criterion, validate_with_weighted_criterion


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


EXP_ID = "exp15_mixup_cutmix_weightedCELoss_30epoch_4e-4"
LOGGER_PATH = f"logs/log_{EXP_ID}.txt"
setup_logger(out_file=LOGGER_PATH)
LOGGER.info("seed={}".format(SEED))


SIZE = 128
HEIGHT=137
WIDTH=236
OUT_DIR = 'models'

# https://albumentations.readthedocs.io/en/latest/api/augmentations.html
data_transforms = albumentations.Compose([
    albumentations.Flip(p=0.2),
    albumentations.Rotate(limit=15, p=0.2),
    albumentations.ShiftScaleRotate(rotate_limit=15, p=0.2),
    albumentations.Cutout(p=0.2),
    # albumentations.RandomGridShuffle(grid=(3, 3), p=0.2),
    ])

data_transforms_test = albumentations.Compose([
    albumentations.Flip(p=0),                                           
    ])


class BengaliAIDataset(torch.utils.data.Dataset):
    
    def __init__(self, df, y=None, transform=None):
        self.df = df
        self.y = y
        self.transform = transform
        self.size = 128

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):

        input_dic = {}
        row = self.df.iloc[idx]

        image = self.df.iloc[idx][1:].values.reshape(128,128).astype(np.float)

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


hidden_size = 64
channel_size = 1

class ResNet18(nn.Module):
    def __init__(self):
        super(ResNet18,self).__init__()
        
        self.block1 = nn.Sequential(
            nn.Conv2d(channel_size,hidden_size,kernel_size=2,stride=2,padding=3,bias=False),
            nn.BatchNorm2d(hidden_size),
            nn.ReLU(True)
        )
        
        self.block2 = nn.Sequential(
            nn.MaxPool2d(1,1),
            ResidualBlock(hidden_size,hidden_size),
            ResidualBlock(hidden_size,hidden_size,2)
        )
        
        self.block3 = nn.Sequential(
            ResidualBlock(hidden_size,hidden_size*2),
            ResidualBlock(hidden_size*2,hidden_size*2,2)
        )
        
        self.block4 = nn.Sequential(
            ResidualBlock(hidden_size*2,hidden_size*4),
            ResidualBlock(hidden_size*4,hidden_size*4,2)
        )
        self.block5 = nn.Sequential(
            ResidualBlock(hidden_size*4,hidden_size*8),
            ResidualBlock(hidden_size*8,hidden_size*8,2)
        )
        
        self.avgpool = nn.AvgPool2d(2)   
        self.fc = nn.Linear(512*4,512)     
        # vowel_diacritic
        self.fc1 = nn.Linear(512,11)
        # grapheme_root
        self.fc2 = nn.Linear(512,168)
        # consonant_diacritic
        self.fc3 = nn.Linear(512,7)
        
    def forward(self,x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.avgpool(x)
        x = x.view(x.size(0),-1)
        x = self.fc(x)
        x1 = self.fc1(x)
        x2 = self.fc2(x)
        x3 = self.fc3(x)
        return x1,x2,x3


with timer('load csv data'):
    fold_id = 0
    epochs = 30
    batch_size = 64
    
    train = pd.read_csv('input/train.csv')
    
    y = train[["grapheme_root", "vowel_diacritic", "consonant_diacritic"]]

    num_folds = 5
    train_idx, val_idx = train_test_split(train.index.tolist(), test_size=0.2, random_state=SEED, stratify=train["grapheme_root"])
    
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


# https://pytorch.org/docs/stable/nn.html#loss-functions
def get_weight(component):
    d = pd.DataFrame(y[component].value_counts()/len(y[component])*100).reset_index()
    d['weight'] = d[component].values[::-1]
    d = d.sort_values('index')
    return torch.tensor(d['weight'].values.tolist()).to(device)


with timer('create model'):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # model = torchvision.models.resnet50(pretrained=True)
    # model.load_state_dict(torch.load("../input/pytorch-pretrained-models/resnet101-5d3b4d8f.pth"))
    
    model = ResNet18()
    model = model.to(device)
    # model.load_state_dict(torch.load("models/exp13_mixup_cutmix_15epoch_4e-4_fold0.pth"))
    # LOGGER.info("exp13 model loaded")

    vowel_weights = get_weight('vowel_diacritic')
    grapheme_weights = get_weight('grapheme_root')
    consonant_weights = get_weight('consonant_diacritic')

    criterion1 = nn.CrossEntropyLoss(weight=vowel_weights, reduction='mean').to(device)
    criterion2 = nn.CrossEntropyLoss(weight=grapheme_weights, reduction='mean').to(device)
    criterion3 = nn.CrossEntropyLoss(weight=consonant_weights, reduction='mean').to(device)
    criterion = [criterion1, criterion2, criterion3]
    optimizer = optim.Adam(model.parameters(), lr=4e-4)
    scheduler = GradualWarmupScheduler(optimizer, multiplier=1.1, total_epoch=5,
                                       after_scheduler=None)


with timer('training loop'):
    best_score = -999
    best_epoch = 0
    for epoch in range(1, epochs + 1):

        LOGGER.info("Starting {} epoch...".format(epoch))
        tr_loss = train_one_epoch_with_weighted_criterion(model, train_loader, criterion, optimizer, device)
        LOGGER.info('Mean train loss: {}'.format(round(tr_loss, 5)))

        val_pred, y_true, val_loss = validate_with_weighted_criterion(model, val_loader, criterion, device)
        score = macro_recall(y_true, val_pred)
        LOGGER.info('Mean valid loss: {} score: {}'.format(round(val_loss, 5), round(score, 5)))
        if score > best_score:
            best_score = score
            best_epoch = epoch
            torch.save(model.state_dict(), os.path.join(OUT_DIR, '{}_fold{}.pth'.format(EXP_ID, fold_id)))
            np.save(os.path.join(OUT_DIR, "{}_fold{}.npy".format(EXP_ID, fold_id)), val_pred)
        scheduler.step()

    LOGGER.info("best score={} on epoch={}".format(best_score, best_epoch))

