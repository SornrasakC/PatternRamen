from src.model import Discriminator, Generator
from torch import optim
from src.util import show_image, show_image_row, pack_checkpoint, gen_checkpoint_path
from torch.nn import functional as F
from torch import nn
import torch
import wandb
import torchvision.models as models
from tqdm import tqdm
import os

class Training():
  def __init__(self, checkpoint_path=None):
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

    self.init_optimizers()

    self.iteration = 0

    if checkpoint_path is not None:
      self.load_checkpoint(checkpoint_path)

  def init_optimizers(self):
    self.g_optimizer = optim.Adam(self.generator.parameters(), lr=1e-3, betas=(0.5, 0.999)) # paper lr 1e-4
    self.d_optimizer_line = optim.Adam(self.discriminator_line.parameters(), lr=1e-4, betas=(0.5, 0.999)) # paper lr 4e-4
    self.d_optimizer_color = optim.Adam(self.discriminator_color.parameters(), lr=1e-4, betas=(0.5, 0.999)) # paper lr 4e-4

  def train(self, dataLoader, valDataLoader, iterations):
    wandb.init(project="colorization")
    wandb.watch(self.discriminator_line)
    wandb.watch(self.discriminator_color)
    wandb.watch(self.generator)
    
    for _it in tqdm(range(iterations)):
      self.generator.train()
      line, color, transform_color, noise = next(dataLoader)
      line = line.cuda().to(dtype=torch.float32)
      color = color.cuda().to(dtype=torch.float32)
      transform_color = transform_color.cuda().to(dtype=torch.float32)
      noise = noise.cuda().to(dtype=torch.float32)
      
      self.d_optimizer_line.zero_grad()
      self.d_optimizer_color.zero_grad()

      generated_image = self.generator(line, transform_color, noise)

      d_loss_line = torch.mean((self.discriminator_line(line, color)-1)**2 + self.discriminator_line(line, generated_image.detach())**2)
      d_loss_color = torch.mean((self.discriminator_color(color, color)-1)**2 + self.discriminator_color(color, generated_image.detach())**2)
      
      d_loss = (d_loss_line + d_loss_color) / 2
      d_loss.backward()
      self.d_optimizer_color.step()      
      self.d_optimizer_line.step()
      
      self.g_optimizer.zero_grad()
      p_loss = torch.mean(self.perceptual_criterion(self.vgg16(color), self.vgg16(generated_image)))
      pure_g_loss = torch.mean((self.discriminator_line(line, generated_image) - 1)**2) + torch.mean((self.discriminator_color(color, generated_image) - 1)**2)
      g_loss = pure_g_loss + 1 * p_loss #/ (16 * 16) # [BS, 512, 16, 16]
      g_loss.backward()
      self.g_optimizer.step()
      
      wandb.log({"g_loss": g_loss, "d_loss": d_loss})
      if(_it % 100==0):
        print("Iteration: {}/{}".format(_it + self.iteration, iterations + self.iteration), "g_loss: {:.4f}".format(g_loss), "d_loss: {:.4f}".format(d_loss))
      
      if(_it % 100==0):
        pic_rows = self.inference(valDataLoader)
        for pic_row in pic_rows[:10]:
          show_image_row(pic_row)

      if(_it % 100==0):
        self.save_checkpoint(iteration=self.iteration + _it)

    wandb.finish()
    self.iteration += iterations
    self.save_checkpoint()
  
  def inference(self, dataLoader):
    self.generator.eval()
    pic_row = []
    with torch.no_grad():
      for _it in range(10):
        if len(pic_row) > 10:
          break
        line, color, noise = next(dataLoader)
        line = line.cuda().to(dtype=torch.float32)
        color = color.cuda().to(dtype=torch.float32)
        noise = noise.cuda().to(dtype=torch.float32)
        generated_images = self.generator(line, color, noise)
        for line_im, color_im, gen_im in zip(line, color, generated_images):
          format_im = lambda im: im.squeeze().permute(1,2,0).detach().cpu()
          un_norm = lambda data: (0.5 * data + 0.5)#[..., ::-1]
          un_norm_im = lambda im: un_norm(format_im(im))

          gen_im = un_norm_im(gen_im)
          line_im = un_norm_im(line_im)#.type(torch.int)
          color_im = un_norm_im(color_im)#.type(torch.int)
          pic_row.append([line_im, color_im, gen_im])
    return pic_row

  def save_checkpoint(self, iteration=..., filepath=...):
    if iteration is ...:
      iteration = self.iteration

    if filepath is ...:
      filepath = gen_checkpoint_path(iteration)
    
    pack_dict = pack_checkpoint(self.discriminator_line, self.discriminator_color, self.generator, self.g_optimizer, self.d_optimizer_line, self.d_optimizer_color, iteration)

    torch.save(pack_dict, filepath)

  def load_checkpoint(self, checkpoint_path):
    assert os.path.isfile(checkpoint_path)

    pack_dict = torch.load(checkpoint_path, location='cpu')

    self.discriminator_line.load_state_dict(pack_dict['discriminator_line'])
    self.discriminator_color.load_state_dict(pack_dict['discriminator_color'])
    self.generator.load_state_dict(pack_dict['generator'])

    self.init_optimizers()

    self.g_optimizer.load_state_dict(pack_dict['g_optimizer'])
    self.d_optimizer_line.load_state_dict(pack_dict['d_optimizer_line'])
    self.d_optimizer_color.load_state_dict(pack_dict['d_optimizer_color'])

    self.iteration = pack_dict['iteration']

