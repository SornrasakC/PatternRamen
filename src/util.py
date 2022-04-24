import matplotlib.pyplot as plt
import os

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
  plt.show() # or display.display(plt.gcf()) if you prefer


def pack_checkpoint(discriminator_line, discriminator_color, generator, g_optimizer, d_optimizer_line, d_optimizer_color, iteration):
  pack_dict = {
    'discriminator_line': discriminator_line,
    'discriminator_color': discriminator_color,
    'generator': generator,
    'g_optimizer': g_optimizer.state_dict(),
    'd_optimizer_line': d_optimizer_line.state_dict(),
    'd_optimizer_color': d_optimizer_color.state_dict(),
    'iteration': iteration,
  }

  return pack_dict

def gen_checkpoint_path(iteration):
  target = f'checkpoints/checkpoint-{iteration}'
  checkpoint_fp = os.path.join(os.getcwd(), target)
  return checkpoint_fp