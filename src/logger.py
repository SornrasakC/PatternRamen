import wandb
import numpy as np

class Logger:
    def __init__(self, wandb_run_id=None, checkpoint_path=None, disable_wandb=False):
        self.disable_wandb = disable_wandb
        self.wandb_run_id = wandb_run_id
        self.checkpoint_path = checkpoint_path
        self._watch_disable_flag__()
    
    def init_wandb(self):
        options = {
            'project': 'colorization',
            'id': self.wandb_run_id,
            'resume': 'must' if self.checkpoint_path != None else None
        }
        wandb.init(**options)


    def watch(self, trainer):
        self.init_wandb()
        wandb.watch(trainer.discriminator_line)
        wandb.watch(trainer.discriminator_color)
        wandb.watch(trainer.generator)

    def log_losses(self, g_loss, d_loss):
        wandb.log({"g_loss": g_loss, "d_loss": d_loss})

    def log_image(self, np_image, log_msg='Validation image'):
        image = wandb.Image(np_image)
        wandb.log({log_msg: image})

    def log_image_row(self, np_image_row, **kw):
        np_image = np.concatenate(np_image_row, axis=1)
        self.log_image(np_image, **kw)

    def finish(self):
        wandb.finish()

    def _watch_disable_flag__(self):
        method_list = [method for method in dir(
            self) if '__' not in method and callable(getattr(self, method))]
        for method in method_list:
            func = getattr(self, method)

            def gen_func(func):
                def new_func(*args, **kw):
                    if self.disable_wandb:
                        return ...
                    return func(*args, **kw)
                return new_func
            setattr(self, method, gen_func(func))
