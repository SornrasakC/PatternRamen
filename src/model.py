from torch import nn
from src.block import DiscriminatorBlock, GeneratorEncoderBlock, SPADEResBlock
import torch
import src.util as util

class Discriminator(nn.Module):
    def __init__(self, input_num=1, gan_loss_type='lsgan', rgan_mode=False):
        super().__init__()
        # with_norm = with_encoder_first_layer_norm
        dis_block_options = {"kernel_size": 3, "stride": 2, "padding": 1}

        self.input_num = input_num
        self.blocks = nn.Sequential(
            DiscriminatorBlock(3 * input_num, 64, **dis_block_options, with_norm=False, spec_norm=False),
            DiscriminatorBlock(64, 128, **dis_block_options),
            DiscriminatorBlock(128, 256, **dis_block_options),
            DiscriminatorBlock(256, 512, **dis_block_options),
            nn.Conv2d(512, 1, 4, 1, 1),
        )

        target_fake_label = -1.0 if self.rgan_mode in ['rgan', 'ragan'] else 0.0
        self.loss = GANLoss(gan_loss_type=gan_loss_type, target_fake_label=target_fake_label)
        self.gan_loss_type = gan_loss_type
        self.rgan_mode = rgan_mode
        if rgan_mode not in [False, 'rgan', 'ragan']:
            raise NotImplementedError(f'rgan_mode: {rgan_mode}')

    def forward(self, *x):
        x = torch.cat(x, dim=1)
        x = self.blocks(x)
        return x

    def forward_raw(self, x):
        x = self.blocks(x)
        return x
    
    def criticise(self, real, fake, *conds, with_gp=False, only_fake=False):
        if self.rgan_mode == False:
            d_loss_real = self.criterion(*conds, real, label=1) if not only_fake else None
            d_loss_fake = self.criterion(*conds, fake, label=0 if not only_fake else 1)
            
        if self.rgan_mode in ['rgan', 'ragan']:
            d_real = self(*conds, real) # if not only_fake else None
            d_fake = self(*conds, fake)

            if self.rgan_mode == 'ragan':
                d_real_diff = d_real.mean() # if not only_fake else None
                d_fake_diff = d_fake.mean()
            elif self.rgan_mode == 'rgan':
                d_real_diff = d_real
                d_fake_diff = d_fake
        
            d_loss_real = self._criterion(d_real - d_fake_diff, label=1) if not only_fake else None
            d_loss_fake = self._criterion(d_fake - d_real_diff, label=0 if not only_fake else 1)
        
        gp = self._calc_gp(real, fake, *conds) if with_gp else False

        return d_loss_real, d_loss_fake, gp
    
    def _calc_gp(self, real, fake, *conds):
        ratio = torch.rand(1).cuda()
        interpolated_image = (ratio * real + (1 - ratio) * fake)
        dis_input_fake = torch.cat([*conds, interpolated_image], axis=1).requires_grad_(True)
        d_fake = self(dis_input_fake)

        gradient_penalty = util.calc_gradient_penalty(d_fake, dis_input_fake)
        return gradient_penalty

    def criterion(self, *x, label: int):
        assert len(x) == self.input_num

        pred = self(*x)
        return self._criterion(pred, label)
    
    def _criterion(self, pred, label: int):
        if self.gan_loss_type == 'lsgan':
            return self.loss( pred, target_label ) / 2

        if self.gan_loss_type == 'wgan-gp':
            return ( 1 if bool(label) else -1 ) * pred.mean()

        if self.gan_loss_type == 'sgan':
            return self.loss( pred, bool(label) )
    

class GANLoss(nn.Module):
    # https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py#L209
    def __init__(self, gan_loss_type='lsgan', target_real_label=1.0, target_fake_label=0.0):
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))

        if gan_loss_type == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_loss_type == 'wgan-gp':
            self.loss = None
        elif gan_loss_type == 'sgan':
            self.loss = nn.BCEWithLogitsLoss()
        else:
            raise NotImplementedError(f'gan_loss_type: {gan_loss_type}')

    def get_target_tensor(self, prediction, target_is_real):
        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real):
        target_tensor = self.get_target_tensor(prediction, target_is_real)
        loss = self.loss(prediction, target_tensor)
        return loss


class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        # with_norm = with_encoder_first_layer_norm
        gen_block_options = {"kernel_size": 3, "stride": 2, "padding": 1}

        self.sketch_encoder_blocks = nn.ModuleList(
            [
                GeneratorEncoderBlock(3, 64, **gen_block_options, with_norm=False, spec_norm=False),
                GeneratorEncoderBlock(64, 128, **gen_block_options),
                GeneratorEncoderBlock(128, 256, **gen_block_options),
                GeneratorEncoderBlock(256, 512, **gen_block_options),
                GeneratorEncoderBlock(512, 512, **gen_block_options),
                GeneratorEncoderBlock(512, 512, **gen_block_options),
            ]
        )

        self.reference_encoder_blocks = nn.ModuleList(
            [
                GeneratorEncoderBlock(3, 64, **gen_block_options, with_norm=False, spec_norm=False),
                GeneratorEncoderBlock(64, 128, **gen_block_options),
                GeneratorEncoderBlock(128, 256, **gen_block_options),
                GeneratorEncoderBlock(256, 512, **gen_block_options),
                GeneratorEncoderBlock(512, 512, **gen_block_options),
            ]
        )

        res_blocks = [
            SPADEResBlock(1024, 1024, segmap_channels=512, scale_factor=2),
            SPADEResBlock(1024, 1024, segmap_channels=512),
            SPADEResBlock(1024, 1024, segmap_channels=512, scale_factor=2),
            SPADEResBlock(1024, 512 , segmap_channels=512),
            SPADEResBlock(512 , 512 , segmap_channels=512, scale_factor=2),
            SPADEResBlock(512 , 256 , segmap_channels=256),
            SPADEResBlock(256 , 256 , segmap_channels=256, scale_factor=2),
            SPADEResBlock(256 , 128 , segmap_channels=128),
            SPADEResBlock(128 , 128 , segmap_channels=128, scale_factor=2),
            SPADEResBlock(128 , 64  , segmap_channels=64 , scale_factor=2),
        ]

        self.decoder_blocks = nn.ModuleList(
            [
                nn.Linear(256, 16384),
                *res_blocks,
                nn.Sequential(
                    nn.Conv2d(64, 3, 3, 1, 1),
                    nn.Tanh()
                )
            ]
        )

    def passer(self, module_list, x):
        outputs = []
        for block in module_list:
            x = block(x)
            outputs.append(x)
        return outputs

    def forward(self, sketch, reference, noise):
        sketches = self.passer(self.sketch_encoder_blocks, sketch)
        references = self.passer(self.reference_encoder_blocks, reference)

        seg_pairs = zip(sketches[2:][::-1], references[1:][::-1])
        segs = [seg for seg_pair in seg_pairs for seg in seg_pair]
        
        linear_layer, *res_blocks, final_layer = self.decoder_blocks
        *common_res_blocks, res_block_128, res_block_64 = res_blocks
        
        x = noise
        x = linear_layer(x)
        x = x.view(-1, 1024, 4, 4)

        for res_block, seg in zip(common_res_blocks, segs):
            x = res_block(x, seg)
        
        x = res_block_128(x, sketches[1])
        x = res_block_64(x, sketches[0])
        x = final_layer(x)

        return x
