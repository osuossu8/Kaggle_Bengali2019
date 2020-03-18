import os
import gc
import logging
import random
import numpy as np
import torch
import cv2
import PIL
import sklearn.metrics
from torchvision.transforms import functional as F
from PIL import Image
from tqdm import tqdm

import sys
sys.path.append("/usr/src/app/kaggle/bengaliai-cv19")

from src.image_bengali import rand_bbox, cutmix, mixup, \
                              cutmix_criterion, mixup_criterion, \
                              cutmix_criterion_focal, mixup_criterion_focal, \
                              mixup_criterion_weighted, cutmix_criterion_weighted                               


LOGGER = logging.getLogger()
SIZE=128
HEIGHT=137
WIDTH=236
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def train_one_epoch_mixup_cutmix(model, train_loader, criterion, optimizer, device, steps_upd_logging=500, accumulation_steps=1,
                                 multi_loss=None):
    model.train()

    total_loss = 0.0
    for step, (input_dic, targets1, targets2, targets3) in tqdm(enumerate(train_loader), total=len(train_loader)):
        for k in input_dic.keys():
            input_dic[k] = input_dic[k].to(device)
        targets1 = targets1.to(device)
        targets2 = targets2.to(device)
        targets3 = targets3.to(device)
        optimizer.zero_grad()

        images = input_dic["image"].unsqueeze(1)

        if np.random.rand()<0.5:
            images, targets = mixup(images, targets1, targets2, targets3, 0.4)
            logits1, logits2, logits3 = model(images)
            loss = mixup_criterion(logits1, logits2, logits3, targets) 
        else:
            images, targets = cutmix(images, targets1, targets2, targets3, 0.4)
            logits1, logits2, logits3 = model(images)
            loss = cutmix_criterion(logits1, logits2, logits3, targets) 

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

        if (step + 1) % steps_upd_logging == 0:
            LOGGER.info('Train loss on step {} was {}'.format(step + 1, round(total_loss / (step + 1), 5)))


    return total_loss / (step + 1)


def train_one_epoch_mixup_cutmix_for_single_output(model, train_loader, criterion, optimizer, device, steps_upd_logging=500, accumulation_steps=1,
                                 multi_loss=None):
    model.train()

    total_loss = 0.0
    for step, (input_dic, targets1, targets2, targets3) in tqdm(enumerate(train_loader), total=len(train_loader)):
        for k in input_dic.keys():
            input_dic[k] = input_dic[k].to(device)
        targets1 = targets1.to(device)
        targets2 = targets2.to(device)
        targets3 = targets3.to(device)
        optimizer.zero_grad()

        images = input_dic["image"].unsqueeze(1)

        if np.random.rand()<0.5:
            images, targets = mixup(images, targets1, targets2, targets3, 0.4)
            logits = model(images)
            logits1, logits2, logits3 = logits[:,:11], logits[:,11:11+168], logits[:, 11+168:]
            loss = mixup_criterion(logits1, logits2, logits3, targets)
        else:
            images, targets = cutmix(images, targets1, targets2, targets3, 0.4)
            logits = model(images)
            logits1, logits2, logits3 = logits[:,:11], logits[:,11:11+168], logits[:, 11+168:]
            loss = cutmix_criterion(logits1, logits2, logits3, targets)

        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        if (step + 1) % steps_upd_logging == 0:
            LOGGER.info('Train loss on step {} was {}'.format(step + 1, round(total_loss / (step + 1), 5)))

    return total_loss / (step + 1)


def train_one_epoch_mixup_cutmix_for_single_output_weighted(model, train_loader, criterion, optimizer, device, steps_upd_logging=500, accumulation_steps=1,
                                 multi_loss=None):
    model.train()

    total_loss = 0.0
    for step, (input_dic, targets1, targets2, targets3) in tqdm(enumerate(train_loader), total=len(train_loader)):
        for k in input_dic.keys():
            input_dic[k] = input_dic[k].to(device)
        targets1 = targets1.to(device)
        targets2 = targets2.to(device)
        targets3 = targets3.to(device)
        optimizer.zero_grad()

        images = input_dic["image"].unsqueeze(1)

        if np.random.rand()<0.5:
            images, targets = mixup(images, targets1, targets2, targets3, 0.4)
            logits = model(images)
            logits1, logits2, logits3 = logits[:,:11], logits[:,11:11+168], logits[:, 11+168:]
            loss = mixup_criterion_weighted(logits1, logits2, logits3, targets, criterion)
        else:
            images, targets = cutmix(images, targets1, targets2, targets3, 0.4)
            logits = model(images)
            logits1, logits2, logits3 = logits[:,:11], logits[:,11:11+168], logits[:, 11+168:]
            loss = cutmix_criterion_weighted(logits1, logits2, logits3, targets, criterion)

        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        if (step + 1) % steps_upd_logging == 0:
            LOGGER.info('Train loss on step {} was {}'.format(step + 1, round(total_loss / (step + 1), 5)))

    return total_loss / (step + 1)


def train_one_epoch_mixup_for_single_output(model, train_loader, criterion, optimizer, device, steps_upd_logging=500, accumulation_steps=1,
                                 multi_loss=None, flg=False):
    model.train()

    total_loss = 0.0
    for step, (input_dic, targets1, targets2, targets3) in tqdm(enumerate(train_loader), total=len(train_loader)):
        for k in input_dic.keys():
            input_dic[k] = input_dic[k].to(device)
        targets1 = targets1.to(device)
        targets2 = targets2.to(device)
        targets3 = targets3.to(device)
        optimizer.zero_grad()

        images = input_dic["image"].unsqueeze(1)

        if np.random.rand()<0.5 and flg==False:
            images, targets = mixup(images, targets1, targets2, targets3, 0.4)
            logits = model(images)
            logits1, logits2, logits3 = logits[:,:11], logits[:,11:11+168], logits[:, 11+168:]
            loss = mixup_criterion(logits1, logits2, logits3, targets)
        else:
            logits = model(images)
            logits1, logits2, logits3 = logits[:,:11], logits[:,11:11+168], logits[:, 11+168:]
            loss = criterion(logits1, targets1) * 0.25 +criterion(logits2, targets2) * 0.5 +criterion(logits3, targets3) * 0.25

        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        if (step + 1) % steps_upd_logging == 0:
            LOGGER.info('Train loss on step {} was {}'.format(step + 1, round(total_loss / (step + 1), 5)))

    return total_loss / (step + 1)


def train_one_epoch_cutmix_for_single_output(model, train_loader, criterion, optimizer, device, steps_upd_logging=500, accumulation_steps=1,
                                 multi_loss=None):
    model.train()

    total_loss = 0.0
    for step, (input_dic, targets1, targets2, targets3) in tqdm(enumerate(train_loader), total=len(train_loader)):
        for k in input_dic.keys():
            input_dic[k] = input_dic[k].to(device)
        targets1 = targets1.to(device)
        targets2 = targets2.to(device)
        targets3 = targets3.to(device)
        optimizer.zero_grad()

        images = input_dic["image"].unsqueeze(1)

        images, targets = cutmix(images, targets1, targets2, targets3, 0.4)
        logits = model(images)
        logits1, logits2, logits3 = logits[:,:11], logits[:,11:11+168], logits[:, 11+168:]
        loss = cutmix_criterion_weighted(logits1, logits2, logits3, targets, criterion)

        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        if (step + 1) % steps_upd_logging == 0:
            LOGGER.info('Train loss on step {} was {}'.format(step + 1, round(total_loss / (step + 1), 5)))

    return total_loss / (step + 1)


def train_one_epoch_for_single_output(model, train_loader, criterion, optimizer, device, steps_upd_logging=500, accumulation_steps=1,
                                 multi_loss=None):
    model.train()

    total_loss = 0.0
    for step, (input_dic, targets1, targets2, targets3) in tqdm(enumerate(train_loader), total=len(train_loader)):
        for k in input_dic.keys():
            input_dic[k] = input_dic[k].to(device)
        targets1 = targets1.to(device)
        targets2 = targets2.to(device)
        targets3 = targets3.to(device)
        optimizer.zero_grad()

        images = input_dic["image"].unsqueeze(1)

        logits = model(images)
        logits1, logits2, logits3 = logits[:,:11], logits[:,11:11+168], logits[:, 11+168:]
        loss = criterion(logits1, targets1) * 0.25 +criterion(logits2, targets2) * 0.5 +criterion(logits3, targets3) * 0.25

        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        if (step + 1) % steps_upd_logging == 0:
            LOGGER.info('Train loss on step {} was {}'.format(step + 1, round(total_loss / (step + 1), 5)))

    return total_loss / (step + 1)


def validate_for_single_output(model, val_loader, criterion, device, multi_loss=None):
    model.eval()

    val_loss = 0.0
    true_ans_list1 = []
    preds_cat1 = []
    true_ans_list2 = []
    preds_cat2 = []
    true_ans_list3 = []
    preds_cat3 = []
    for step, (input_dic, targets1, targets2, targets3) in tqdm(enumerate(val_loader), total=len(val_loader)):
        for k in input_dic.keys():
            input_dic[k] = input_dic[k].to(device)
        targets1 = targets1.to(device)
        targets2 = targets2.to(device)
        targets3 = targets3.to(device)

        images = input_dic["image"].unsqueeze(1)

        logits = model(images)
        logits1, logits2, logits3 = logits[:,:11], logits[:,11:11+168], logits[:, 11+168:]

        # loss = criterion(logits1, targets1)+criterion(logits2, targets2)+criterion(logits3, targets3)
        loss = criterion(logits1, targets1) * 0.25 +criterion(logits2, targets2) * 0.5 +criterion(logits3, targets3) * 0.25

        val_loss += loss.item()

        targets1 = targets1.float().cpu().detach().numpy()
        targets2 = targets2.float().cpu().detach().numpy()
        targets3 = targets3.float().cpu().detach().numpy()
        logits1 = logits1.argmax(1).float().cpu().detach().numpy().astype("float32")
        logits2 = logits2.argmax(1).float().cpu().detach().numpy().astype("float32")
        logits3 = logits3.argmax(1).float().cpu().detach().numpy().astype("float32")

        true_ans_list1.append(targets1)
        preds_cat1.append(logits1)
        true_ans_list2.append(targets2)
        preds_cat2.append(logits2)
        true_ans_list3.append(targets3)
        preds_cat3.append(logits3)
        del input_dic
        del targets1, targets2, targets3, logits1, logits2, logits3
        gc.collect()

    all_true_ans1 = np.concatenate(true_ans_list1, axis=0)
    all_preds1 = np.concatenate(preds_cat1, axis=0)
    all_true_ans2 = np.concatenate(true_ans_list2, axis=0)
    all_preds2 = np.concatenate(preds_cat2, axis=0)
    all_true_ans3 = np.concatenate(true_ans_list3, axis=0)
    all_preds3 = np.concatenate(preds_cat3, axis=0)

    all_true_ans = [all_true_ans1, all_true_ans2, all_true_ans3] # v, g, c
    all_preds = [all_preds1, all_preds2, all_preds3] # v, g, c

    return all_preds, all_true_ans, val_loss / (step + 1)


def validate_for_single_output_weighted(model, val_loader, criterion, device, multi_loss=None):
    model.eval()

    val_loss = 0.0
    true_ans_list1 = []
    preds_cat1 = []
    true_ans_list2 = []
    preds_cat2 = []
    true_ans_list3 = []
    preds_cat3 = []
    for step, (input_dic, targets1, targets2, targets3) in tqdm(enumerate(val_loader), total=len(val_loader)):
        for k in input_dic.keys():
            input_dic[k] = input_dic[k].to(device)
        targets1 = targets1.to(device)
        targets2 = targets2.to(device)
        targets3 = targets3.to(device)

        images = input_dic["image"].unsqueeze(1)

        logits = model(images)
        logits1, logits2, logits3 = logits[:,:11], logits[:,11:11+168], logits[:, 11+168:]

        loss = criterion(logits1, targets1) * 0.25 +criterion(logits2, targets2) * 0.5 +criterion(logits3, targets3) * 0.25

        val_loss += loss.item()

        targets1 = targets1.float().cpu().detach().numpy()
        targets2 = targets2.float().cpu().detach().numpy()
        targets3 = targets3.float().cpu().detach().numpy()
        logits1 = logits1.argmax(1).float().cpu().detach().numpy().astype("float32")
        logits2 = logits2.argmax(1).float().cpu().detach().numpy().astype("float32")
        logits3 = logits3.argmax(1).float().cpu().detach().numpy().astype("float32")

        true_ans_list1.append(targets1)
        preds_cat1.append(logits1)
        true_ans_list2.append(targets2)
        preds_cat2.append(logits2)
        true_ans_list3.append(targets3)
        preds_cat3.append(logits3)
        del input_dic
        del targets1, targets2, targets3, logits1, logits2, logits3
        gc.collect()

    all_true_ans1 = np.concatenate(true_ans_list1, axis=0)
    all_preds1 = np.concatenate(preds_cat1, axis=0)
    all_true_ans2 = np.concatenate(true_ans_list2, axis=0)
    all_preds2 = np.concatenate(preds_cat2, axis=0)
    all_true_ans3 = np.concatenate(true_ans_list3, axis=0)
    all_preds3 = np.concatenate(preds_cat3, axis=0)

    all_true_ans = [all_true_ans1, all_true_ans2, all_true_ans3] # v, g, c
    all_preds = [all_preds1, all_preds2, all_preds3] # v, g, c

    return all_preds, all_true_ans, val_loss / (step + 1)


def train_one_epoch_with_weighted_criterion(model, train_loader, criterion, optimizer, device, steps_upd_logging=500, accumulation_steps=1,
                                       multi_loss=None):
    model.train()

    total_loss = 0.0
    for step, (input_dic, targets1, targets2, targets3) in tqdm(enumerate(train_loader), total=len(train_loader)):
        for k in input_dic.keys():
            input_dic[k] = input_dic[k].to(device)
        targets1 = targets1.to(device) # v
        targets2 = targets2.to(device) # g
        targets3 = targets3.to(device) # c
        optimizer.zero_grad()

        images = input_dic["image"].unsqueeze(1)

        if np.random.rand()<0.5:
            images, targets = mixup(images, targets1, targets2, targets3, 0.4)
            logits1, logits2, logits3 = model(images)
            loss = mixup_criterion_weighted(logits1, logits2, logits3, targets, criterion) 
        else:
            images, targets = cutmix(images, targets1, targets2, targets3, 0.4)
            logits1, logits2, logits3 = model(images)
            loss = cutmix_criterion_weighted(logits1, logits2, logits3, targets, criterion) 

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

        if (step + 1) % steps_upd_logging == 0:
            LOGGER.info('Train loss on step {} was {}'.format(step + 1, round(total_loss / (step + 1), 5)))


    return total_loss / (step + 1)


def validate_with_weighted_criterion(model, val_loader, criterion, device, multi_loss=None):
    model.eval()

    val_loss = 0.0
    true_ans_list1 = []
    preds_cat1 = []
    true_ans_list2 = []
    preds_cat2 = []
    true_ans_list3 = []
    preds_cat3 = []
    for step, (input_dic, targets1, targets2, targets3) in tqdm(enumerate(val_loader), total=len(val_loader)):
        for k in input_dic.keys():
            input_dic[k] = input_dic[k].to(device)
        targets1 = targets1.to(device) # v
        targets2 = targets2.to(device) # g
        targets3 = targets3.to(device) # c

        logits1, logits2, logits3 = model(input_dic["image"].unsqueeze(1)) # v, g, c

        loss = criterion[0](logits1, targets1)+criterion[1](logits2, targets2)+criterion[2](logits3, targets3)
        
        val_loss += loss.item()
        
        targets1 = targets1.float().cpu().detach().numpy()
        targets2 = targets2.float().cpu().detach().numpy()
        targets3 = targets3.float().cpu().detach().numpy()
        logits1 = logits1.argmax(1).float().cpu().detach().numpy().astype("float32")
        logits2 = logits2.argmax(1).float().cpu().detach().numpy().astype("float32")
        logits3 = logits3.argmax(1).float().cpu().detach().numpy().astype("float32")

        true_ans_list1.append(targets1)
        preds_cat1.append(logits1)
        true_ans_list2.append(targets2)
        preds_cat2.append(logits2)
        true_ans_list3.append(targets3)
        preds_cat3.append(logits3)
        del input_dic
        del targets1, targets2, targets3, logits1, logits2, logits3
        gc.collect()

    all_true_ans1 = np.concatenate(true_ans_list1, axis=0)
    all_preds1 = np.concatenate(preds_cat1, axis=0)
    all_true_ans2 = np.concatenate(true_ans_list2, axis=0)
    all_preds2 = np.concatenate(preds_cat2, axis=0)
    all_true_ans3 = np.concatenate(true_ans_list3, axis=0)
    all_preds3 = np.concatenate(preds_cat3, axis=0)
    
    all_true_ans = [all_true_ans1, all_true_ans2, all_true_ans3] # v, g, c
    all_preds = [all_preds1, all_preds2, all_preds3] # v, g, c
    
    return all_preds, all_true_ans, val_loss / (step + 1)


def train_one_epoch_with_focalloss(model, train_loader, criterion, optimizer, device, steps_upd_logging=500, accumulation_steps=1,
                                       multi_loss=None):
    model.train()

    total_loss = 0.0
    for step, (input_dic, targets1, targets2, targets3) in tqdm(enumerate(train_loader), total=len(train_loader)):
        for k in input_dic.keys():
            input_dic[k] = input_dic[k].to(device)
        targets1 = targets1.to(device) # v
        targets2 = targets2.to(device) # g
        targets3 = targets3.to(device) # c
        optimizer.zero_grad()

        images = input_dic["image"].unsqueeze(1)

        if np.random.rand()<0.5:
            images, targets = mixup(images, targets1, targets2, targets3, 0.4)
            logits1, logits2, logits3 = model(images)
            loss = mixup_criterion_focal(logits1, logits2, logits3, targets)
        else:
            images, targets = cutmix(images, targets1, targets2, targets3, 0.4)
            logits1, logits2, logits3 = model(images)
            loss = cutmix_criterion_focal(logits1, logits2, logits3, targets)

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

        if (step + 1) % steps_upd_logging == 0:
            LOGGER.info('Train loss on step {} was {}'.format(step + 1, round(total_loss / (step + 1), 5)))

    return total_loss / (step + 1)


def validate_with_focalloss(model, val_loader, criterion, device, multi_loss=None):
    model.eval()

    val_loss = 0.0
    true_ans_list1 = []
    preds_cat1 = []
    true_ans_list2 = []
    preds_cat2 = []
    true_ans_list3 = []
    preds_cat3 = []
    for step, (input_dic, targets1, targets2, targets3) in tqdm(enumerate(val_loader), total=len(val_loader)):
        for k in input_dic.keys():
            input_dic[k] = input_dic[k].to(device)
        targets1 = targets1.to(device) # v
        targets2 = targets2.to(device) # g
        targets3 = targets3.to(device) # c

        logits1, logits2, logits3 = model(input_dic["image"].unsqueeze(1)) # v, g, c

        loss = criterion(logits1, targets1)+criterion(logits2, targets2)+criterion(logits3, targets3)

        val_loss += loss.item()

        targets1 = targets1.float().cpu().detach().numpy()
        targets2 = targets2.float().cpu().detach().numpy()
        targets3 = targets3.float().cpu().detach().numpy()
        logits1 = logits1.argmax(1).float().cpu().detach().numpy().astype("float32")
        logits2 = logits2.argmax(1).float().cpu().detach().numpy().astype("float32")
        logits3 = logits3.argmax(1).float().cpu().detach().numpy().astype("float32")

        true_ans_list1.append(targets1)
        preds_cat1.append(logits1)
        true_ans_list2.append(targets2)
        preds_cat2.append(logits2)
        true_ans_list3.append(targets3)
        preds_cat3.append(logits3)
        del input_dic
        del targets1, targets2, targets3, logits1, logits2, logits3
        gc.collect()

    all_true_ans1 = np.concatenate(true_ans_list1, axis=0)
    all_preds1 = np.concatenate(preds_cat1, axis=0)
    all_true_ans2 = np.concatenate(true_ans_list2, axis=0)
    all_preds2 = np.concatenate(preds_cat2, axis=0)
    all_true_ans3 = np.concatenate(true_ans_list3, axis=0)
    all_preds3 = np.concatenate(preds_cat3, axis=0)

    all_true_ans = [all_true_ans1, all_true_ans2, all_true_ans3] # v, g, c
    all_preds = [all_preds1, all_preds2, all_preds3] # v, g, c

    return all_preds, all_true_ans, val_loss / (step + 1)
