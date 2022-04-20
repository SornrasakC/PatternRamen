
from src.model import Discriminator, Generator
from torch import optim
from src.util import show_image, show_image_row
from torch.nn import functional as F
from torch import nn
import torch
import wandb
import torchvision.models as models
from tqdm import tqdm

from src.model import Discriminator, Generator
from torch import optim
from src.util import show_image, show_image_row
from torch.nn import functional as F
from torch import nn
import torch
import wandb
import torchvision.models as models
from tqdm import tqdm

class Training():
  def __init__(self):
    self.discriminator_line = Discriminator()
    self.discriminator_color = Discriminator()
    self.generator = Generator()
    self.vgg16 = models.vgg16(pretrained=True).features[:25]
    self.vgg16.cuda()
    self.vgg16.eval()
    self.discriminator_line.cuda()
    self.discriminator_color.cuda()
    self.generator.cuda()
    self.perceptual_criterion = nn.L1Loss()

  def train(self, dataLoader, valDataLoader, iterations):
    wandb.init(project="colorization")
    wandb.watch(self.discriminator_line)
    wandb.watch(self.discriminator_color)
    wandb.watch(self.generator)
    
    g_optimizer = optim.Adam(self.generator.parameters(), lr=0.0001, betas=(0.5, 0.999))
    d_optimizer_line = optim.Adam(self.discriminator_line.parameters(), lr=0.0004, betas=(0.5, 0.999))
    d_optimizer_color = optim.Adam(self.discriminator_color.parameters(), lr=0.0004, betas=(0.5, 0.999))
    
    for _it in tqdm(range(iterations)):
      self.generator.train()
      line, color, transform_color, noise = next(dataLoader)
      line = line.cuda().to(dtype=torch.float32)
      color = color.cuda().to(dtype=torch.float32)
      transform_color = transform_color.cuda().to(dtype=torch.float32)
      noise = noise.cuda().to(dtype=torch.float32)
      
      d_optimizer_line.zero_grad()
      d_optimizer_color.zero_grad()

      generated_image = self.generator(line, transform_color, noise)

      d_loss_line = torch.mean((self.discriminator_line(line, color)-1)**2 + self.discriminator_line(line, generated_image.detach())**2)
      d_loss_color = torch.mean((self.discriminator_color(color, color)-1)**2 + self.discriminator_color(color, generated_image.detach())**2)
      
      d_loss = (d_loss_line + d_loss_color) / 2
      d_loss.backward()
      d_optimizer_color.step()      
      d_optimizer_line.step()
      
      g_optimizer.zero_grad()
      p_loss = torch.mean(self.perceptual_criterion(self.vgg16(color), self.vgg16(generated_image)))
      pure_g_loss = torch.mean((self.discriminator_line(line, generated_image) - 1)**2) + torch.mean((self.discriminator_color(color, generated_image) - 1)**2)
      g_loss = pure_g_loss + 1 * p_loss / (16 * 16) # [BS, 512, 16, 16]
      g_loss.backward()
      g_optimizer.step()
      
      wandb.log({"g_loss": g_loss, "d_loss": d_loss})
      if(_it % 100==0):
        print("Iteration: {}/{}".format(_it, iterations), "g_loss: {:.4f}".format(g_loss), "d_loss: {:.4f}".format(d_loss))
      
      if(_it % 100==0):
        pic_rows = self.inference(valDataLoader)
        for pic_row in pic_rows[:10]:
          show_image_row(pic_row)
    wandb.finish()
  
  def inference(self, dataLoader):
    self.generator.eval()
    pics = []
    for _it in range(10):
      line, color, noise = next(dataLoader)
      line = line.cuda().to(dtype=torch.float32)
      color = color.cuda().to(dtype=torch.float32)
      noise = noise.cuda().to(dtype=torch.float32)
      generated_images = self.generator(line, color, noise)
      for line_im, color_im, gen_im in zip(line, color, generated_images):
        format_im = lambda im: im.squeeze().permute(1,2,0).detach().cpu()
        gen_im = format_im(gen_im)
        line_im = format_im(line_im).type(torch.int)
        color_im = format_im(color_im).type(torch.int)
        pics.append([line_im, color_im, gen_im])
    return pics
