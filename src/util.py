import matplotlib.pyplot as plt


def show_image(img):
  imgplot = plt.imshow(img)
  plt.show()


def show_image_row(imgs):
  f, axs = plt.subplots(1, len(imgs))
  f.set_figheight(6)
  f.set_figwidth(15)
  for ax, img in zip(axs, imgs):
      ax.imshow(img)
  plt.show() # or display.display(plt.gcf()) if you prefer