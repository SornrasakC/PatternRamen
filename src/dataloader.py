import argparse
import numpy as np
from pathlib import Path
from PIL import Image
from scipy.ndimage import gaussian_filter
import sys 
import cv2
import os
import random


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

class XDoGTrainData():
    def __init__(self,param,folder_path):
        self.folder_path = folder_path
        self.data = os.listdir(folder_path)
        self.gamma = param['gamma']
        self.phi = param['phi']
        self.epsilon = param['eps']
        self.k = param['k']
        self.sigma = param['sigma']

    def __getitem__(self,idx):
        img = cv2.imread(self.folder_path + '/' + self.data[idx],cv2.COLOR_BGR2RGB)
        sigma_rand = np.random.uniform(self.sigma, self.sigma+0.2)
        noise = np.random.normal(0, 1, 256)
        is_xdog = random.choice([True, False])
        if is_xdog: ## return xdog image
            x, y = xdog(img,sigma=sigma_rand,k=self.k,gamma=self.gamma,epsilon=self.epsilon,phi=self.phi)
        else: ## return original image
            x, y = img[:,int(img.shape[1]/2):], img[:,:int(img.shape[1]/2)]
        return x, noise,y
    
    def __len__(self):
        return len(self.data)

class XDoGValData():
    def __init__(self,folder_path):
        self.folder_path = folder_path
        self.data = os.listdir(folder_path)

    def __getitem__(self,idx):
        img = cv2.imread(self.folder_path + '/' + self.data[idx],cv2.COLOR_BGR2RGB)
        noise = np.random.normal(0, 1, 256)
        x, y = img[:,int(img.shape[1]/2):], img[:,:int(img.shape[1]/2)]
        return x, noise,y
    
    def __len__(self):
        return len(self.data)        