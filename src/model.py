import torch
from torch import nn
from src.block import DiscriminatorBlock, GeneratorEncoderBlock, SPADEResBlock


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        dis_block_options = {"kernel_size": 3, "stride": 2, "padding": 1}
        
        self.block_1 = DiscriminatorBlock(3, 64, **dis_block_options)
        self.block_2 = DiscriminatorBlock(64, 128, **dis_block_options)
        self.block_3 = DiscriminatorBlock(128, 256, **dis_block_options)
        self.block_4 = DiscriminatorBlock(256, 512, **dis_block_options)
        self.conv = nn.Conv2d(512, 1, 4, 1)

    def forward(self, x):
        x = self.block_1(x)
        x = self.block_2(x)
        x = self.block_3(x)
        x = self.block_4(x)
        x = self.conv(x)
        return x


class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        gen_block_options = {"kernel_size": 3, "stride": 2, "padding": 1}

        self.sketch_encoder_blocks = nn.ModuleList(
            [
                GeneratorEncoderBlock(3, 64, **gen_block_options),
                GeneratorEncoderBlock(64, 128, **gen_block_options),
                GeneratorEncoderBlock(128, 256, **gen_block_options),
                GeneratorEncoderBlock(256, 512, **gen_block_options),
                GeneratorEncoderBlock(512, 512, **gen_block_options),
                GeneratorEncoderBlock(512, 512, **gen_block_options),
            ]
        )

        self.reference_encoder_blocks = nn.ModuleList(
            [
                GeneratorEncoderBlock(3, 64, **gen_block_options),
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
                nn.Linear(512, 16384),
                *res_blocks,
                nn.Sequential([
                    nn.Conv(64, 3, 3, 1, 1),
                    nn.Tanh()
                ])
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
