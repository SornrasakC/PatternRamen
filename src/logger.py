import wandb
import numpy as np
import time


class Logger:
    def __init__(self, wandb_run_id=None, checkpoint_path=None, disable_wandb=False):
        self.disable_wandb = disable_wandb
        self.wandb_run_id = wandb_run_id
        self.checkpoint_path = checkpoint_path
        self._watch_disable_flag__()

    def init_wandb(self, trainer):
        config = self.get_args_config(trainer)
        
        options = {
            'entity': 'pattern-ramen',
            'project': 'colorization',
            'config': config,
            'id': self.wandb_run_id,
            'resume': 'must' if self.checkpoint_path != None else None
        }
        wandb.init(**options)

        wandb.define_metric('iteration')
        wandb.define_metric("*", step_metric="iteration")

    def watch(self, trainer):
        self.init_wandb(trainer)
        wandb.watch(trainer.discriminator_line)
        wandb.watch(trainer.discriminator_color)
        wandb.watch(trainer.generator)

    def log_losses(self, pack_loss, pack_lr, iteration, **kw):
        wandb.log({**pack_loss, **pack_lr, "iteration": iteration}, **kw)

    def log_image(self, np_image, iteration, log_msg='Validation image', caption=None, is_img_list=False, **kw):
        if is_img_list:
            image = [wandb.Image(im, caption=caption) for im in np_image]
        else:
            image = wandb.Image(np_image, caption=caption)

        wandb.log({log_msg: image, 'iteration': iteration}, **kw)

    def log_etc(self, whatever_dict, iteration, **kw):
        wandb.log({**whatever_dict, "iteration": iteration}, **kw)

    def log_image_row_list(self, np_image_rows, **kw):
        np_images = [
            np.concatenate(np_image_row, axis=1)
            for np_image_row in np_image_rows
        ]
        self.log_image(np_images, is_img_list=True, **kw)

    def get_args_config(self, trainer):
        attr_list = trainer.__init__.__code__.co_varnames
        
        config = {}
        for attr in attr_list:
            try:
                val = getattr(trainer, attr)
                config[attr] = repr(val)
            except AttributeError:
                print(f'Out of config: {func_name}')

        return config

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


class TimeLogger:
    def __init__(self, disabled=False):
        self.start()
        self.disabled = disabled

    def start(self):
        self.start_time = time.time()
        return self.start_time

    def check(self, msg='', reset=True):
        if self.disabled:
            return ...

        time_passed = time.time() - self.start_time

        self.print(msg, time_passed)

        if reset:
            self.start()

        return time_passed

    def print(self, msg, time_passed):
        print(f'[Time Logger] {{ {msg} }} {time_passed:.2f} sec')
