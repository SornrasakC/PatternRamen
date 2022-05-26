import argparse
import numpy as np
from pathlib import Path
from PIL import Image
from scipy.ndimage import gaussian_filter
import sys
import cv2
import os
import random

import torch
from torch import nn
from torchvision import transforms
from torch.utils.data import DataLoader

from PIL import Image
from skimage.util import random_noise

from src.fixed_noise import fixed_noise

# PARAM = {"gamma": 0.95, "phi": 1e9, "eps": -1, "k": 4.5, "sigma": 0.3}
PARAM = {
    'gamma' : 0.95,
    'phi' : 1e9,
    'eps' : -0.4,
    'k' : 4.5,
    'sigma' : 0.15 #0.15-0.2
}


def dog(img, size=(0, 0), k=1.6, sigma=0.5, gamma=1):
    img1 = cv2.GaussianBlur(img, size, sigma)
    img2 = cv2.GaussianBlur(img, size, sigma * k)
    return img1 - gamma * img2


def xdog(img, sigma=0.5, k=1.6, gamma=1, epsilon=1, phi=1):
    imgColor = img[:, : int(img.shape[1] / 2)]
    imgLine = img[:, int(img.shape[1] / 2) :]
    # aux = dog(imgLine, sigma=sigma, k=k, gamma=gamma) / 255
    norm_image = cv2.normalize(imgColor, None, norm_type=cv2.NORM_INF, dtype=cv2.CV_32F) #cv2.NORM_INF
    # norm_image = imgColor
    aux = dog(cv2.cvtColor(norm_image, cv2.COLOR_BGR2GRAY),sigma=sigma,k=k,gamma=gamma)/255
    aux = np.where(aux < epsilon, 1 * 255, 255 * (1 + np.tanh(phi * (aux))))
    aux = cv2.cvtColor(aux.astype('float32'),cv2.COLOR_GRAY2RGB)
    # aux = cv2.merge([aux,aux,aux])
    return aux, imgColor


def draw_random_line(img, line_range):
    width = np.random.randint(*line_range)
    start = np.random.randint(0, 256 - width)
    end = np.random.randint(0 + start, start + width)
    spray_color = np.random.randint(0, 255, 3)
    spray_color = np.tile(spray_color, (256, 1))
    spray_color = np.tile(spray_color, (end - start, 1, 1))
    spray_color = np.transpose(spray_color, (2, 1, 0))
    img[:, :, start:end] = torch.Tensor(spray_color)
    return img


class XDoGData(torch.utils.data.Dataset):
    def __init__(self, param, folder_path, is_validate=False, use_xdog=True, disable_random_line=False):
        super(XDoGData, self).__init__()
        
        self.folder_path = folder_path
        self.data = sorted(os.listdir(folder_path))
        self.gamma = param["gamma"]
        self.phi = param["phi"]
        self.epsilon = param["eps"]
        self.k = param["k"]
        self.sigma = param["sigma"]
        self.is_validate = is_validate
        self.use_xdog = use_xdog
        self.disable_random_line = disable_random_line
        self.line_width_range = (25,40)

        train_transform = nn.Sequential(
            transforms.RandomRotation(60, fill=255),
            transforms.RandomPerspective(distortion_scale=0.6, p=1.0, fill=255),
            transforms.RandomResizedCrop((256, 256), scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            # transforms.RandomVerticalFlip(p=0.5),
        )
        self.train_transform = torch.jit.script(train_transform)

        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )
        # rotate = nn.Sequential(transforms.RandomRotation(60,fill=255))
        # self.rotate = torch.jit.script(rotate)

    def __getitem__(self, idx):
        img = Image.open(os.path.join(self.folder_path, self.data[idx])).convert("RGB")
        img = np.asarray(img, dtype=np.uint8)
        sigma_rand = np.random.uniform(self.sigma, self.sigma + 0.2)
        # noise = np.random.normal(0, 1, 256)
        noise = fixed_noise
        is_xdog = random.choice([True, False]) if self.use_xdog else False
        img = cv2.resize(img, (512, 256))

        if is_xdog:  ## return xdog image
            line, color = xdog(
                img,
                sigma=sigma_rand,
                k=self.k,
                gamma=self.gamma,
                epsilon=self.epsilon,
                phi=self.phi,
            )
        else:  ## return original image
            line, color = (
                img[:, int(img.shape[1] / 2) :],
                img[:, : int(img.shape[1] / 2)],
            )

        line = self.transform(line.astype(np.uint8))
        if self.is_validate:
            color = self.transform(color)
            return line, color, color, noise
        tran_color = self.train_transform(torch.Tensor(np.transpose(color, (2, 0, 1))))

        if not self.disable_random_line:
            ### draw random line on picture
            rotate_angle = np.random.uniform(-60, 60, 1)[0]
            tran_color = transforms.functional.rotate(tran_color, rotate_angle, fill=255)
            tran_color = draw_random_line(tran_color, self.line_width_range)
            tran_color = transforms.functional.rotate(tran_color, -rotate_angle, fill=255)

        # color, tran_color = self.transform(color), self.transform(tran_color)
        color, tran_color = self.transform(color), self.transform(
            np.transpose(tran_color.numpy().astype(np.uint8), (1, 2, 0))
        )

        # line, color, tran_color = np.transpose(line,(1,2,0)), np.transpose(color,(1,2,0)), np.transpose(tran_color,(1,2,0))
        return line, color, tran_color, noise

    def __len__(self):
        return len(self.data)


def gen_data_loader(data_path, is_validate=False, use_xdog=True, disable_random_line=False, **kw):
    data = XDoGData(PARAM, data_path, is_validate, use_xdog=use_xdog, disable_random_line=disable_random_line)
    defaults = {
        'shuffle': True,
        'batch_size': 16,
        'num_workers': 4,
    }
    opt = {**defaults, **kw}
    data_loader = DataLoader(data, **opt)
    return data_loader

class WganDataset(torch.utils.data.Dataset):
    def __init__(self, data_len):
        self.data_len = data_len

    def __getitem__(self, index):
        return torch.randn(1, 1, 1)

    def __len__(self):
        return self.data_len

def gen_etc_loader(**kw):
    data = WganDataset(int(1e6))
    defaults = {
        'shuffle': False,
        'batch_size': 16,
        'num_workers': 4,
    }
    opt = {**defaults, **kw}

    return DataLoader(data, **opt)


class InstanceNoise:
    def __init__(self):
        self.noise_start_var = 0.1 ** 2
        self.noise_mean = 0
    
    def cal_var(self, current_step, total_step):
        total_step = 20_000
        ratio = (total_step - current_step) / total_step
        var = self.noise_start_var * max(ratio, 0)
        return var

    def add_noise(self, color, current_step, total_step):
        var = self.cal_var(current_step, total_step)
        if var == 0:
            return color
        color_for_dis = random_noise(color.cpu(), mode='gaussian', mean=self.noise_mean, var=var)
        color_for_dis = torch.from_numpy(color_for_dis)
        return color_for_dis.cuda().to(dtype=torch.float32)
