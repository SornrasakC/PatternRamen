import matplotlib.pyplot as plt
import os
from pathlib import Path


def show_image(img):
    imgplot = plt.imshow(img)
    plt.show()


def show_image_row(imgs):
    f, axs = plt.subplots(1, len(imgs))
    f.set_figheight(6)
    f.set_figwidth(15)
    for ax, img in zip(axs, imgs):
        ax.imshow(img)
        ax.axis('off')
    plt.show()  # or display.display(plt.gcf()) if you prefer


def pack_checkpoint(discriminator_line, discriminator_color, generator, g_optimizer, d_optimizer_line, d_optimizer_color, iteration, wandb_run_id):
    pack_dict = {
        'discriminator_line': discriminator_line.state_dict(),
        'discriminator_color': discriminator_color.state_dict(),
        'generator': generator.state_dict(),
        'g_optimizer': g_optimizer.state_dict(),
        'd_optimizer_line': d_optimizer_line.state_dict(),
        'd_optimizer_color': d_optimizer_color.state_dict(),
        'iteration': iteration,
        'wandb_run_id': wandb_run_id
    }

    return pack_dict

def _get_checkpoint_base(checkpoint_folder_parent):
    if checkpoint_folder_parent == None:
        return Path.cwd().parent / 'checkpoints'

    if checkpoint_folder_parent == 'parent':
        return Path.cwd().parent / 'checkpoints'

    if checkpoint_folder_parent == 'cwd':
        return Path.cwd() / 'checkpoints'

    assert os.path.isdir(checkpoint_folder_parent)
    return Path(checkpoint_folder_parent) / 'checkpoints'

def get_checkpoint_base(checkpoint_folder_parent):
    checkpoint_base = _get_checkpoint_base(checkpoint_folder_parent)
    checkpoint_base.mkdir(exist_ok=True)
    return checkpoint_base

def get_latest_checkpoint(checkpoint_base: Path):
    filename_list = os.listdir(checkpoint_base)
    cp_list = [fn for fn in filename_list if 'checkpoint_' in fn]
    assert len(cp_list) > 0
    
    return checkpoint_base / max(cp_list, key=lambda cp: int(cp.split('_')[1]) )

def gen_checkpoint_path(checkpoint_base: Path, iteration):
    checkpoint_base.mkdir(exist_ok=True)
    checkpoint_fp = checkpoint_base / f'checkpoint_{iteration}'
    return str(checkpoint_fp)

def loader_cycle_it(data_loader):
    it = iter(data_loader)
    while True:
      try:
        yield next(it)
      except StopIteration:
        it = iter(data_loader)

def denorm_image(image):
    format_im = lambda im: im.squeeze().permute(1,2,0).detach().cpu()
    un_norm = lambda data: (0.5 * data + 0.5)
    un_norm_im = lambda im: un_norm(format_im(im))

    return un_norm_im(image)