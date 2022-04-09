
from model import Discriminator, Generator
from torch import optim
from util import random_transform, show_image
from torch.nn import functional as F
from torch import nn
import torch
import wandb

class Training():
  def __init__(self):
    self.discriminator_line = Discriminator()
    self.discriminator_color = Discriminator()
    self.generator = Generator()
    self.vgg16 = torch.hub.load('pytorch/vision:v0.10.0', 'vgg16', pretrained=True).features[:12]
    print(self.vgg16.features)
    self.vgg16.eval()
    self.discriminator_line.cuda()
    self.discriminator_color.cuda()
    self.generator.cuda()
    self.perceptual_criterion = nn.L1Loss()
    wandb.init(project="colorization")
    wandb.watch(self.discriminator)
    wandb.watch(self.generator)

  def train(self, dataLoader, iterations, valDataLoader):
    g_optimizer = optim.Adam(self.generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    d_optimizer_line = optim.Adam(self.discriminator_line.parameters(), lr=0.0002, betas=(0.5, 0.999))
    d_optimizer_color = optim.Adam(self.discriminator_color.parameters(), lr=0.0002, betas=(0.5, 0.999))
    
    for _it in range(iterations):
      line, color, noise = next(dataLoader)
      d_optimizer_line.zero_grad()
      generated_image = self.generator(line, random_transform(color), noise)
      d_loss_line = (self.discriminator_line(line, color)-1)**2 + self.discriminator_line(line, generated_image)**2
      d_loss_line.backward()
      d_optimizer_line.step()
      
      d_optimizer_color.zero_grad()
      d_loss_color = (self.discriminator_color(color, color)-1)**2 + self.discriminator_color(color, generated_image)**2
      d_loss_color.backward()
      d_optimizer_color.step()
      
      d_loss = d_loss_line + d_loss_color
      
      g_optimizer.zero_grad()
      p_loss = self.perceptual_criterion(self.vgg16(color), self.vgg16(generated_image), p=1, dim=2)
      g_loss = (self.discriminator(line, generated_image) - 1)**2 + p_loss
      g_loss.backward()
      g_optimizer.step()
      
      wandb.log({"g_loss": g_loss, "d_loss": d_loss})
      if(_it % 500==0):
        print("Iteration: {}/{}".format(_it, iterations), "g_loss: {:.4f}".format(g_loss), "d_loss: {:.4f}".format(d_loss))
      
      if(_it % 2000==0):
        pics = self.inference(valDataLoader)
        for pic in pics[:10]:
          show_image(pic)
    wandb.finish()
  
  def inference(self, dataLoader):
    self.generator.eval()
    pics = []
    for line, ref, noise in next(dataLoader):
      pics.append(self.generator(line, ref, noise).detach().cpu())
    return pics