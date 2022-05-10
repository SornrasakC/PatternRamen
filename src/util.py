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


def pack_checkpoint(discriminator_line, discriminator_color, generator, g_optimizer, d_optimizer_line, d_optimizer_color, iteration):
    pack_dict = {
        'discriminator_line': discriminator_line.state_dict(),
        'discriminator_color': discriminator_color.state_dict(),
        'generator': generator.state_dict(),
        'g_optimizer': g_optimizer.state_dict(),
        'd_optimizer_line': d_optimizer_line.state_dict(),
        'd_optimizer_color': d_optimizer_color.state_dict(),
        'iteration': iteration,
    }

    return pack_dict

def get_latest_checkpoint():
    checkpoint_base = Path.cwd().parent / 'checkpoints'
    _root, _subfolder, cp_list = next(os.walk(checkpoint_base))
    assert len(cp_list) > 0
    
    return checkpoint_base / max(cp_list, key=lambda cp: int(cp.split('-')[1]) )

def gen_checkpoint_path(iteration):
    checkpoint_base = Path.cwd().parent / 'checkpoints'
    checkpoint_base.mkdir(exist_ok=True)
    checkpoint_fp = checkpoint_base / f'checkpoint-{iteration}'
    return str(checkpoint_fp)
