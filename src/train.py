from src.model import Discriminator, Generator
import src.util as util
from src.logger import Logger, TimeLogger
from src.dataloader import gen_data_loader

import torch
from torch import optim
from torch.nn import functional as F
from torch import nn
import torchvision

from tqdm import tqdm
import os

class Trainer():
  def __init__(self,
      wandb_run_id=None, disable_time_logger=False,
      checkpoint_path=None, checkpoint_folder_parent=None, checkpoint_interval=100,
      data_path_train=None, data_path_val=None, batch_size=16,
    ):

    self.discriminator_line = Discriminator()
    self.discriminator_color = Discriminator()
    self.generator = Generator()
    self.vgg16 = torchvision.models.vgg16(pretrained=True).features[:25]
    self.vgg16.cuda()
    self.vgg16.eval()
    self.discriminator_line.cuda()
    self.discriminator_color.cuda()
    self.generator.cuda()
    self.perceptual_criterion = nn.L1Loss()

    self.init_optimizers()

    self.iteration = 0
    self.checkpoint_interval = checkpoint_interval
    self.logger = Logger(wandb_run_id=wandb_run_id, checkpoint_path=checkpoint_path)
    self.time_logger = TimeLogger(disabled=disable_time_logger)
    self.inference_size = 2
    self.checkpoint_base = util.get_checkpoint_base(checkpoint_folder_parent)
    self.data_path_train = data_path_train
    self.data_path_val = data_path_val
    self.batch_size = self.batch_size

    self.load_checkpoint(checkpoint_path)
        

  def init_optimizers(self):
    self.g_optimizer = optim.Adam(self.generator.parameters(), lr=1e-4, betas=(0.5, 0.999)) # paper lr 1e-4
    self.d_optimizer_line = optim.Adam(self.discriminator_line.parameters(), lr=4e-4, betas=(0.5, 0.999)) # paper lr 4e-4
    self.d_optimizer_color = optim.Adam(self.discriminator_color.parameters(), lr=4e-4, betas=(0.5, 0.999)) # paper lr 4e-4

  def train(self, iterations):
    self.logger.watch(self)

    data_loader_train = gen_data_loader(self.data_path_train, shuffle=True, batch_size=self.batch_size)
    it_train = util.loader_cycle_it(data_loader_train)
    
    print(f'Starting on iteration: {self.iteration}')
    total_it = self.iteration + iterations
    for _it in tqdm(range(self.iteration + 1, total_it + 1)):
      self.generator.train()

      self.time_logger.start()
      line, color, transform_color, noise = next(it_train)
      self.time_logger.check('Data loading')

      line = line.cuda().to(dtype=torch.float32)
      color = color.cuda().to(dtype=torch.float32)
      transform_color = transform_color.cuda().to(dtype=torch.float32)
      noise = noise.cuda().to(dtype=torch.float32)
      
      self.d_optimizer_line.zero_grad()
      self.d_optimizer_color.zero_grad()
      self.time_logger.check('Preparation')

      generated_image = self.generator(line, transform_color, noise)
      self.time_logger.check('Generator forward')

      d_loss_line = torch.mean((self.discriminator_line(line, color)-1)**2 + self.discriminator_line(line, generated_image.detach())**2)
      d_loss_color = torch.mean((self.discriminator_color(color, color)-1)**2 + self.discriminator_color(color, generated_image.detach())**2)
      
      d_loss = (d_loss_line + d_loss_color) / 2
      self.time_logger.check('D Loss Calculation')

      d_loss.backward()
      self.time_logger.check('D Loss Backward')

      self.d_optimizer_color.step()      
      self.d_optimizer_line.step()
      self.time_logger.check('D Optim Steps')
      
      self.g_optimizer.zero_grad()
      p_loss = torch.mean(self.perceptual_criterion(self.vgg16(color), self.vgg16(generated_image)))
      pure_g_loss = torch.mean((self.discriminator_line(line, generated_image) - 1)**2) + torch.mean((self.discriminator_color(color, generated_image) - 1)**2)
      g_loss = pure_g_loss + 1 * p_loss #/ (16 * 16) # [BS, 512, 16, 16]
      self.time_logger.check('G Loss Calculation')

      g_loss.backward()
      self.time_logger.check('G Loss Backward')

      self.g_optimizer.step()
      self.time_logger.check('G Optim Steps')
      
      self.logger.log_losses(g_loss=g_loss, d_loss=d_loss, iteration=_it)
      self.time_logger.check('Wandb Logging')

      if _it % self.checkpoint_interval == 0:
        print(f"[Iteration: {_it}/{total_it}] g_loss: {g_loss:.4f} d_loss: {d_loss:.4f}")

        self.save_checkpoint(iteration=_it)
        self.time_logger.check('Save Checkpoint')

        self.evaluate(it_train, it_val, iteration=_it, total_it=total_it)
        self.time_logger.check('Evaluation')
      
    self.logger.finish()
    self.iteration += iterations
    self.save_checkpoint()
  
  def evaluate(self, iteration, total_it):
    self.generator.eval()

    log_kw = {'caption': f'Iteration: {iteration}', 'commit': False, 'iteration': iteration}

    opt = {'batch_size': self.inference_size}
    it_train_s = iter(gen_data_loader(self.data_path_train, shuffle=True, **opt))
    it_train = iter(gen_data_loader(self.data_path_train, shuffle=False, **opt))

    opt = {'batch_size': self.inference_size, 'is_validate': True}
    it_val_s = iter(gen_data_loader(self.data_path_val, shuffle=True, **opt))
    it_val = iter(gen_data_loader(self.data_path_val, shuffle=False, **opt))

    def _evaluate(it_data, log_msg, **inf_kw):
      print(f'''
      ------------
        {log_msg}
      ------------
      ''')
      pic_row_list = self.inference(it_data, **inf_kw)
      self.logger.log_image_row_list(pic_row_list, log_msg=log_msg, **log_kw)
      for pic_row in pic_row_list:
        util.show_image_row(pic_row)
    
    _evaluate(it_train, log_msg='train_images_fixed', lock_line=True)
    _evaluate(it_train_s, log_msg='train_images_shuffle', lock_line=True)
    
    _evaluate(it_val, log_msg='val_images_fixed', lock_line=True)
    _evaluate(it_val_s, log_msg='val_images_shuffle', lock_line=True)
    
    self.generator.train()
  
  def inference(self, it_data_loader, lock_line=False, lock_color=False):
    self.generator.eval()
    pic_rows = []
    with torch.no_grad():
      for _it in range(self.inference_size):
        line, color, _transform_color, noise = next(it_data_loader)
        if lock_line:
          line = util.lock_batch(line)
        if lock_color:
          color = util.lock_batch(color)

        line = line.cuda().to(dtype=torch.float32)
        color = color.cuda().to(dtype=torch.float32)
        noise = noise.cuda().to(dtype=torch.float32)
        generated_images = self.generator(line, color, noise)

        for line_im, color_im, gen_im in zip(line, color, generated_images):
          gen_im = util.denorm_image(gen_im)
          line_im = util.denorm_image(line_im)
          color_im = util.denorm_image(color_im)

          pic_rows.append([line_im, color_im, gen_im])
          if len(pic_rows) >= self.inference_size:
            return pic_rows[:self.inference_size]
    # Shouldn't reach here
    return pic_rows[:self.inference_size]

  def save_checkpoint(self, iteration=..., filepath=...):
    if iteration is ...:
      iteration = self.iteration

    if filepath is ...:
      filepath = util.gen_checkpoint_path(self.checkpoint_base, iteration)
    
    pack_dict = util.pack_checkpoint(self.discriminator_line, self.discriminator_color, self.generator, self.g_optimizer, self.d_optimizer_line, self.d_optimizer_color, iteration, self.logger.wandb_run_id)

    print(f'Saving Checkpoint: {filepath}')
    torch.save(pack_dict, filepath)

  def load_checkpoint(self, checkpoint_path):
    if checkpoint_path != None:
      self._load_checkpoint(checkpoint_path)

    if checkpoint_path == None:
      try:
        util.get_latest_checkpoint(self.checkpoint_base)
        raise Exception('Checkpoints not empty')
      except AssertionError:
        pass

  def _load_checkpoint(self, checkpoint_path):
    if checkpoint_path == 'latest':
      checkpoint_path = util.get_latest_checkpoint(self.checkpoint_base)
    else:
      assert os.path.isfile(checkpoint_path)

    print(f'Loading Checkpoint: {checkpoint_path}')

    pack_dict = torch.load(checkpoint_path, map_location='cpu')

    if 'wandb_run_id' in pack_dict:
      wandb_run_id = pack_dict['wandb_run_id']
      print(f'Loading checkpoint of wandb_run_id: {wandb_run_id}')
      assert wandb_run_id == self.logger.wandb_run_id
    else:
      input(f'''
      ======================================================================
      ======================================================================
        [WARNING] 'wandb_run_id' does not exists in the current checkpoint.
        Enter to continue:
      ''')

    self.discriminator_line.load_state_dict(pack_dict['discriminator_line'])
    self.discriminator_color.load_state_dict(pack_dict['discriminator_color'])
    self.generator.load_state_dict(pack_dict['generator'])

    self.init_optimizers()

    self.g_optimizer.load_state_dict(pack_dict['g_optimizer'])
    self.d_optimizer_line.load_state_dict(pack_dict['d_optimizer_line'])
    self.d_optimizer_color.load_state_dict(pack_dict['d_optimizer_color'])

    self.iteration = pack_dict['iteration']
