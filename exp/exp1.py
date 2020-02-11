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
from torchvision import models, transforms
from contextlib import contextmanager
from collections import OrderedDict
# from nltk.stem import PorterStemmer
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
from PIL import Image

from tqdm import tqdm, tqdm_notebook, trange
#from tqdm._tqdm_notebook import tqdm_notebook as tqdm
#tqdm.pandas()
import warnings
warnings.filterwarnings('ignore')
# from apex import amp

# import sys
# sys.path.append("drive/My Drive/transformers")

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
from src.image_util import resize_to_square_PIL, pad_PIL, \
                           image_to_tensor, train_one_epoch, validate, predict

#from src.logger import setup_logger
#from src.scheduler import GradualWarmupScheduler
#from src.trainer import spearmanr_score, train_one_epoch, validate
#from src.utility import seed_everything

SEED = 1129
seed_everything(SEED)
