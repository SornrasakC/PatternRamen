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

PARAM = {
    'gamma' : 0.95,
    'phi' : 1e9,
    'eps' : -1,
    'k' : 4.5,
    'sigma' : 0.3
}

def dog(img,size=(0,0),k=1.6,sigma=0.5,gamma=1):
  img1 = cv2.GaussianBlur(img,size,sigma)
  img2 = cv2.GaussianBlur(img,size,sigma*k)
  return (img1-gamma*img2)
 
def xdog(img,sigma=0.5,k=1.6, gamma=1,epsilon=1,phi=1):
  imgColor = img[:,:int(img.shape[1]/2)]
  imgLine = img[:,int(img.shape[1]/2):]
  aux = dog(imgLine,sigma=sigma,k=k,gamma=gamma)/255
  aux = np.where(aux < epsilon,1*255,255*(1 + np.tanh(phi*(aux))))
  return aux,imgColor

def draw_random_line(img,line_range):
    width = np.random.randint(*line_range)
    start = np.random.randint(0,256-width)
    end = np.random.randint(0+start,start+width)
    spray_color = np.random.randint(0,255,3)
    spray_color = np.tile(spray_color,(256,1))
    spray_color = np.tile(spray_color,(end-start,1,1))
    spray_color = np.transpose(spray_color,(2,1,0))
    img[:,:,start:end] = torch.Tensor(spray_color)
    return img

class XDoGData():
    def __init__(self,param,folder_path,is_validate = False):
        self.folder_path = folder_path
        self.data = os.listdir(folder_path)
        self.gamma = param['gamma']
        self.phi = param['phi']
        self.epsilon = param['eps']
        self.k = param['k']
        self.sigma = param['sigma']
        self.is_validate = is_validate
        transform = nn.Sequential(transforms.Resize((256,256)),
                                  transforms.RandomRotation(60,fill=255),
                                  transforms.RandomPerspective(distortion_scale=0.6,p=1.0,fill=255),
                                  transforms.RandomResizedCrop((256,256),scale=(0.8, 1.0)),
                                  transforms.RandomHorizontalFlip(p=0.5),
                                  # transforms.RandomVerticalFlip(p=0.5),
                                  )
        self.transform = torch.jit.script(transform)
        # rotate = nn.Sequential(transforms.RandomRotation(60,fill=255))
        # self.rotate = torch.jit.script(rotate)

    def __getitem__(self,idx):
        img = cv2.imread(self.folder_path + '/' + self.data[idx],cv2.COLOR_BGR2RGB)
        sigma_rand = np.random.uniform(self.sigma, self.sigma+0.2)
        noise = np.random.normal(0, 1, 256)
        is_xdog = random.choice([True, False])
        if is_xdog: ## return xdog image
            line, color = xdog(img,sigma=sigma_rand,k=self.k,gamma=self.gamma,epsilon=self.epsilon,phi=self.phi)
        else: ## return original image
            line, color = img[:,int(img.shape[1]/2):], img[:,:int(img.shape[1]/2)]
        line, color = cv2.resize(line,(256,256)), cv2.resize(color,(256,256))
        line, color = np.transpose(line,(2,0,1)), np.transpose(color,(2,0,1))
        if self.is_validate:
            return line, color, noise
        tran_color = torch.Tensor(color)
        tran_color = self.transform(tran_color)
        
        ### draw random line on picture
        rotate_angle = np.random.uniform(-60,60,1)[0]
        tran_color = transforms.functional.rotate(tran_color,rotate_angle,fill=255)
        tran_color = draw_random_line(tran_color,(50,75))
        tran_color = transforms.functional.rotate(tran_color,-rotate_angle,fill=255)


        # line, color, tran_color = np.transpose(line,(1,2,0)), np.transpose(color,(1,2,0)), np.transpose(tran_color,(1,2,0))
        return line, color, tran_color, noise
    
    def __len__(self):
        return len(self.data)

# class XDoGValData():
#     def __init__(self,folder_path):
#         self.folder_path = folder_path
#         self.data = os.listdir(folder_path)

#     def __getitem__(self,idx):
#         img = cv2.imread(self.folder_path + '/' + self.data[idx],cv2.COLOR_BGR2RGB)
#         noise = np.random.normal(0, 1, 256)
#         line, color = img[:,int(img.shape[1]/2):], img[:,:int(img.shape[1]/2)]
#         return line, color, noise
    
#     def __len__(self):
#         return len(self.data)       