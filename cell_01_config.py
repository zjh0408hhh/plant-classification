import os, glob, random, warnings
import numpy as np
import pandas as pd
import cv2

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, classification_report

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2

warnings.filterwarnings("ignore")

# 数据路径和训练参数
CFG = {
    "train_dir": "/kaggle/input/datasets/zhangjunhanzzzz/plant-classification/train",
    "test_dir":  "/kaggle/input/datasets/zhangjunhanzzzz/plant-classification/test",
    "work_dir":  "/kaggle/working",

    "model_name": "convnext_tiny",
    "img_size": 288,
    "batch_size": 16,
    "epochs": 30,
    "lr": 1e-4,
    "weight_decay": 1e-4,
    "n_folds": 5,
    "seed": 42,
    "num_workers": 2,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}

CLASSES = ["Black-grass", "Common wheat", "Loose Silky-bent",
           "Scentless Mayweed", "Sugar beet"]
class2idx = {c: i for i, c in enumerate(CLASSES)}
idx2class = {i: c for c, i in class2idx.items()}

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(CFG["seed"])

print("Device:", CFG["device"])
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
print("albumentations:", A.__version__)
