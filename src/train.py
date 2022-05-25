from src.model import Discriminator, Generator
import src.util as util
from src.logger import Logger, TimeLogger
from src.dataloader import gen_data_loader, InstanceNoise, gen_etc_loader

import torch
from torch import optim
from torch.nn import functional as F
from torch import nn
import torchvision

import numpy as np
from tqdm import tqdm, trange
import os
from PIL import Image
from pytorch_fid.fid_score import calculate_fid_given_paths


class Trainer():
  def __init__(self,
      wandb_run_id=None, disable_time_logger=False, use_xdog=True, disable_random_line=False,
      checkpoint_path=None, checkpoint_folder_parent=None, checkpoint_interval=100,
      data_path_train=None, data_path_val=None, batch_size=16, 
      g_lr=1e-4, d_line_lr=4e-4, d_color_lr=1e-4, inference_size=3,
      add_noise=False, n_critics_line=1, n_critics_color=1, p_loss_weight=1, 
      use_gp_loss_color=True, gp_lambda_color=10, 
      use_gp_loss_line=True, gp_lambda_line=10, 
      with_encoder_first_layer_norm=True, gan_loss_type='lsgan',
      use_vgg_cache=False, rgan_mode=False,
    ):
    self.discriminator_line = Discriminator(input_num=2, with_encoder_first_layer_norm=with_encoder_first_layer_norm, gan_loss_type=gan_loss_type, rgan_mode=rgan_mode)
    self.discriminator_color = Discriminator(input_num=1, with_encoder_first_layer_norm=with_encoder_first_layer_norm, gan_loss_type=gan_loss_type, rgan_mode=rgan_mode)
    self.generator = Generator(with_encoder_first_layer_norm=with_encoder_first_layer_norm)
    self.vgg16 = torchvision.models.vgg16(pretrained=True).features[:25]
    self.vgg16.cuda()
    self.vgg16.eval()
    self.discriminator_line.cuda()
    self.discriminator_color.cuda()
    self.generator.cuda()
    self.perceptual_criterion = nn.L1Loss()
    self.p_loss_weight = p_loss_weight

    self.n_critics_line = n_critics_line
    self.n_critics_color = n_critics_color

    self.use_gp_loss_color = use_gp_loss_color
    self.gp_lambda_color = gp_lambda_color
    self.use_gp_loss_line = use_gp_loss_line
    self.gp_lambda_line = gp_lambda_line

    self.g_lr, self.d_line_lr, self.d_color_lr = g_lr, d_line_lr, d_color_lr
    self.init_optimizers(g_lr, d_line_lr, d_color_lr)

    self.iteration = 0
    self.checkpoint_interval = checkpoint_interval
    self.logger = Logger(wandb_run_id=wandb_run_id, checkpoint_path=checkpoint_path)
    self.time_logger = TimeLogger(disabled=disable_time_logger)
    self.inference_size = inference_size
    self.checkpoint_base = util.get_checkpoint_base(checkpoint_folder_parent)
    self.data_path_train = data_path_train
    self.data_path_val = data_path_val
    self.batch_size = batch_size
    self.use_xdog = use_xdog
    self.disable_random_line = disable_random_line

    self.add_noise = add_noise
    self.instance_noise = InstanceNoise()

    self.load_checkpoint(checkpoint_path)

    self.with_encoder_first_layer_norm = with_encoder_first_layer_norm
    self.gan_loss_type = gan_loss_type
    self.rgan_mode = rgan_mode
    # self.use_vgg_cache = use_vgg_cache
        
  def init_optimizers(self, g_lr, d_line_lr, d_color_lr):
    print(f'Starting Optims g_lr: {g_lr:.0e}, d_line_lr: {d_line_lr:.0e}, d_color_lr: {d_color_lr:.0e}')

    self.g_optimizer = optim.Adam(self.generator.parameters(), lr=g_lr, betas=(0.5, 0.999)) # paper lr 1e-4
    self.d_optimizer_line = optim.Adam(self.discriminator_line.parameters(), lr=d_line_lr, betas=(0.5, 0.999)) # paper lr 4e-4
    self.d_optimizer_color = optim.Adam(self.discriminator_color.parameters(), lr=d_color_lr, betas=(0.5, 0.999)) # paper lr 4e-4

  def train(self, iterations):
    self.logger.watch(self)

    data_loader_train = gen_data_loader(
      self.data_path_train,
      shuffle=True,
      batch_size=self.batch_size,
      use_xdog=self.use_xdog,
      disable_random_line=self.disable_random_line
    )
    it_train = util.loader_cycle_it(data_loader_train, cuda_float32=True)
    
    self.generator.train()
    self.discriminator_color.train()
    self.discriminator_line.train()

    print(f'Starting on iteration: {self.iteration}')
    total_it = self.iteration + iterations
    self.total_step = total_it
    for _it in tqdm(range(self.iteration + 1, total_it + 1)):
      self.current_step = _it
      self.time_logger.start()

      line, color, transform_color, noise = next(it_train)
      self.time_logger.check('Data loading')
      
      generated_image = self.generator(line, transform_color, noise)
      self.time_logger.check('Generator forward')
      
      pack_d_loss = self.optimize_d(line, color, generated_image)
      pack_g_loss = self.optimize_g(line, color, transform_color, noise, generated_image)
      
      pack_lr = util.pack_learning_rate(self.g_lr, self.d_line_lr, self.d_color_lr)

      self.logger.log_losses(pack_loss={**pack_d_loss, **pack_g_loss}, pack_lr=pack_lr, iteration=_it)
      self.time_logger.check('Wandb Logging')

      if _it % self.checkpoint_interval == 0:
        print(f"[Iteration: {_it}/{total_it}] g_loss: {pack_g_loss['g_loss']:.4f} d_loss: {pack_d_loss['d_loss']:.4f}")

        self.save_checkpoint(iteration=_it)
        self.time_logger.check('Save Checkpoint')

        self.evaluate(iteration=_it, total_it=total_it)
        self.time_logger.check('Evaluation')

        self.logger.log_etc({
          'p_loss_weight': self.p_loss_weight,
          'n_critics_line': self.n_critics_line,
          'n_critics_color': self.n_critics_color
        }, _it, commit=False)
      
      pass
    self.logger.finish()
    self.iteration += iterations
    self.save_checkpoint()

  def optimize_d(self, line, color, generated_image):
    generated_image = generated_image.detach()

    for _ in range(self.n_critics_line):
      pack_d_loss_line = self.optimize_d_line(line, color, generated_image)
    self.time_logger.check(f'D Line Optimized n: {self.n_critics_line}')

    for _ in range(self.n_critics_color):
      pack_d_loss_color = self.optimize_d_color(color, generated_image)
    self.time_logger.check(f'D Color Optimized n: {self.n_critics_color}')
    
    d_loss = pack_d_loss_line['d_loss_line'] + pack_d_loss_color['d_loss_color']

    return util.pack_d_loss(d_loss, pack_d_loss_line, pack_d_loss_color)
  
  def optimize_d_line(self, line, color, generated_image):
    self.d_optimizer_line.zero_grad()

    crit_res = self.discriminator_line.criticise(color, generated_image, line, with_gp=self.use_gp_loss_line)
    d_loss_line_real, d_loss_line_fake, gradient_penalty_line = crit_res
      
    d_loss_line = d_loss_line_real + d_loss_line_fake + self.gp_lambda_line * gradient_penalty_line
    
    d_loss_line.backward()
    self.d_optimizer_line.step()

    rets = filter(None, [d_loss_line, d_loss_line_real, d_loss_line_fake, gradient_penalty_line])
    return util.pack_d_loss_line(*map(lambda x: x.item(), rets))

  def optimize_d_color(self, color, generated_image):
    self.d_optimizer_color.zero_grad()

    if self.add_noise:
      current_step, total_step = self.current_step, self.total_step

      color = self.instance_noise.add_noise(color, current_step, total_step)
      generated_image = self.instance_noise.add_noise(generated_image, current_step, total_step)

    crit_res = self.discriminator_color.criticise(color, generated_image, with_gp=self.use_gp_loss_color)
    d_loss_color_real, d_loss_color_fake, gradient_penalty_color = crit_res

    d_loss_color = d_loss_color_real + d_loss_color_fake + self.gp_lambda_color * gradient_penalty_color

    d_loss_color.backward()
    self.d_optimizer_color.step()

    rets = filter(None, [d_loss_color, d_loss_color_real, d_loss_color_fake, gradient_penalty_color])
    return util.pack_d_loss_color(*map(lambda x: x.item(), rets))

  def optimize_g(self, line, color, transform_color, noise, generated_image):
    self.g_optimizer.zero_grad()

    # if self.use_vgg_cache:
    #   TODO cache "self.vgg16(color)" - random batch problem?

    p_loss = torch.mean(self.perceptual_criterion(self.vgg16(color), self.vgg16(generated_image)))

    _, g_loss_line, _ = self.discriminator_line.criticise(color, generated_image, line, only_fake=True)
    _, g_loss_color, _ = self.discriminator_color.criticise(color, generated_image, only_fake=True)

    g_loss = g_loss_line + g_loss_color + self.p_loss_weight * p_loss
    self.time_logger.check('G Loss Calculation')

    g_loss.backward()
    self.time_logger.check('G Loss Backward')

    self.g_optimizer.step()
    self.time_logger.check('G Optim Steps')

    rets = [g_loss, g_loss_line, g_loss_color, p_loss]
    return util.pack_g_loss(*map(lambda x: x.item(), rets))

  def evaluate(self, iteration, total_it):
    self.generator.eval()

    log_kw = {'caption': f'Iteration: {iteration}', 'commit': False, 'iteration': iteration}

    opt = {'batch_size': self.inference_size, 'use_xdog': self.use_xdog, 'disable_random_line': self.disable_random_line, 'is_validate': False}
    it_train = iter(gen_data_loader(self.data_path_train, shuffle=False, **opt))
    it_train_s = iter(gen_data_loader(self.data_path_train, shuffle=True, **opt))

    opt = {'batch_size': self.inference_size, 'disable_random_line': self.disable_random_line, 'is_validate': True}
    it_val = iter(gen_data_loader(self.data_path_val, shuffle=False, **opt))
    it_val_s = iter(gen_data_loader(self.data_path_val, shuffle=True, **opt))

    def _evaluate(it_data, log_msg, **inf_kw):
      print(f'''
      ------------
        {log_msg}
      ------------
      ''')
      pic_row_list = self.inference(it_data, lock_line=True, **inf_kw)
      self.logger.log_image_row_list(pic_row_list, log_msg=f'{log_msg}_lock_line', **log_kw)
      for pic_row in pic_row_list:
        util.show_image_row(pic_row)

      pic_row_list = self.inference(it_data, lock_color=True, **inf_kw)
      self.logger.log_image_row_list(pic_row_list, log_msg=f'{log_msg}_lock_color', **log_kw)
      for pic_row in pic_row_list:
        util.show_image_row(pic_row)
    
    _evaluate(it_train, log_msg='train_images_fixed')
    _evaluate(it_train_s, log_msg='train_images_shuffle')
    
    _evaluate(it_val, log_msg='val_images_fixed')
    _evaluate(it_val_s, log_msg='val_images_shuffle')
    
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

  def calculate_fid(
    self,
    color_save_path='results/color/',
    gen_save_path='results/generated/',
    fid_cal_batch_size=50,
    data_loader=None,
    force_inference=False,
  ):

    os.makedirs(color_save_path, exist_ok=True)
    os.makedirs(gen_save_path, exist_ok=True)

    num_exist_color = len(os.listdir(color_save_path))
    num_exist_gen = len(os.listdir(gen_save_path))

    if not force_inference and (num_exist_color > 0 or num_exist_gen > 0):
      assert num_exist_color == num_exist_gen
      print(f"Found {num_exist_gen} entries in {color_save_path} and {gen_save_path}. Skip inference.")

    else:
      self.generator.eval()

      opt = {
        'batch_size': self.inference_size,
        'use_xdog': False,
        'disable_random_line': True,
        'is_validate': True
      }
      it_test = iter(gen_data_loader(self.data_path_val, shuffle=False, **opt)) if data_loader is None else data_loader

      with torch.no_grad():
        for _it in trange(len(it_test)):
          line, color, _, noise = next(it_test)
          line = line.cuda().to(dtype=torch.float32)
          color = color.cuda().to(dtype=torch.float32)
          noise = noise.cuda().to(dtype=torch.float32)
          generated_images = self.generator(line, color, noise)

          for batch_idx, (color_image, generated_image) in enumerate(zip(color, generated_images)):
            color_image = np.uint8(util.denorm_image(color_image).numpy() * 255)
            color_image = Image.fromarray(color_image).convert('RGB')

            generated_image = np.uint8(util.denorm_image(generated_image).numpy() * 255)
            generated_image = Image.fromarray(generated_image).convert('RGB')

            file_index = _it * opt['batch_size'] + batch_idx
            color_file_path = os.path.join(color_save_path, f'{file_index:04}.png')
            gen_file_path = os.path.join(gen_save_path, f'{file_index:04}.png')

            color_image.save(color_file_path)
            generated_image.save(gen_file_path)

      self.generator.train()
    
    num_avail_cpus = len(os.sched_getaffinity(0))
    num_workers = min(num_avail_cpus, 8)

    fid_score = calculate_fid_given_paths(
      paths=[color_file_path, gen_file_path],
      batch_size=fid_cal_batch_size,
      device=torch.device('cuda' if (torch.cuda.is_available()) else 'cpu'),
      dims=2048,
      num_workers=num_workers
    )
    
    return fid_score

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

    self.init_optimizers(self.g_lr, self.d_line_lr, self.d_color_lr)

    self.g_optimizer.load_state_dict(pack_dict['g_optimizer'])
    self.d_optimizer_line.load_state_dict(pack_dict['d_optimizer_line'])
    self.d_optimizer_color.load_state_dict(pack_dict['d_optimizer_color'])

    self.iteration = pack_dict['iteration']
