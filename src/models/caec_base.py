import numpy as np
import torch
import torch.nn as nn

from ..utils import load_patches, smooth_image, save_imgs


class CAECBase(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.size = cfg.model_size
        self.device = cfg.device
        self.latent = {
            'small': (16, 8, 8),
            'medium': (16, 16, 16),
            'big': (32, 32, 32)
        }[self.size]

        # ENCODER

        # 64x64x64
        self.e_conv_1 = nn.Sequential(
            nn.ZeroPad2d((1, 2, 1, 2)),
            nn.Conv2d(
                in_channels=3, out_channels=64, kernel_size=(5, 5), stride=(2, 2)
            ),
            nn.LeakyReLU(),
        )

        # 128x32x32
        self.e_conv_2 = nn.Sequential(
            nn.ZeroPad2d((1, 2, 1, 2)),
            nn.Conv2d(
                in_channels=64, out_channels=128, kernel_size=(5, 5), stride=(2, 2)
            ),
            nn.LeakyReLU(),
        )

        # 128x32x32
        self.e_block_1 = nn.Sequential(
            nn.ZeroPad2d((1, 1, 1, 1)),
            nn.Conv2d(
                in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1)
            ),
            nn.LeakyReLU(),
            nn.ZeroPad2d((1, 1, 1, 1)),
            nn.Conv2d(
                in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1)
            ),
        )

        if self.size == 'small':
            self.e_pool_1 = nn.Sequential(nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)))

        self.e_block_2 = nn.Sequential(
            nn.ZeroPad2d((1, 1, 1, 1)),
            nn.Conv2d(
                in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1)
            ),
            nn.LeakyReLU(),
            nn.ZeroPad2d((1, 1, 1, 1)),
            nn.Conv2d(
                in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1)
            ),
        )

        self.e_block_3 = nn.Sequential(
            nn.ZeroPad2d((1, 1, 1, 1)),
            nn.Conv2d(
                in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1)
            ),
            nn.LeakyReLU(),
            nn.ZeroPad2d((1, 1, 1, 1)),
            nn.Conv2d(
                in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1)
            ),
        )

        if self.size == 'big':
            self.e_conv_3 = nn.Sequential(
                nn.Conv2d(
                    in_channels=128,
                    out_channels=32,
                    kernel_size=(5, 5),
                    stride=(1, 1),
                    padding=(2, 2),
                ),
                nn.Tanh(),
            )
        else:
            self.e_conv_3 = nn.Sequential(
                nn.ZeroPad2d((1, 2, 1, 2)),
                nn.Conv2d(
                    in_channels=128,
                    out_channels=16,
                    kernel_size=(5, 5),
                    stride=(2, 2)
                ),
                nn.Tanh(),
            )

        # DECODER

        self.d_up_conv_1 = nn.Sequential(
            nn.Conv2d(
                # in_channels different for 32x...
                in_channels=32 if self.size == 'big' else 16, out_channels=64, kernel_size=(3, 3), stride=(1, 1)
            ),
            nn.LeakyReLU(),
            nn.ZeroPad2d((1, 1, 1, 1)),
            nn.ConvTranspose2d(
                in_channels=64, out_channels=128, kernel_size=(2, 2), stride=(2, 2)
            ),
        )

        self.d_block_1 = nn.Sequential(
            nn.ZeroPad2d((1, 1, 1, 1)),
            nn.Conv2d(
                in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1)
            ),
            nn.LeakyReLU(),
            nn.ZeroPad2d((1, 1, 1, 1)),
            nn.Conv2d(
                in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1)
            ),
        )

        if self.size == 'small':
            self.d_up_1 = nn.Sequential(nn.Upsample(scale_factor=2, mode="nearest"))

        self.d_block_2 = nn.Sequential(
            nn.ZeroPad2d((1, 1, 1, 1)),
            nn.Conv2d(
                in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1)
            ),
            nn.LeakyReLU(),
            nn.ZeroPad2d((1, 1, 1, 1)),
            nn.Conv2d(
                in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1)
            ),
        )

        self.d_block_3 = nn.Sequential(
            nn.ZeroPad2d((1, 1, 1, 1)),
            nn.Conv2d(
                in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1)
            ),
            nn.LeakyReLU(),
            nn.ZeroPad2d((1, 1, 1, 1)),
            nn.Conv2d(
                in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1)
            ),
        )

        self.d_up_conv_2 = nn.Sequential(
            nn.Conv2d(
                in_channels=128, out_channels=32, kernel_size=(3, 3), stride=(1, 1)
            ),
            nn.LeakyReLU(),
            nn.ZeroPad2d((1, 1, 1, 1)),
            nn.ConvTranspose2d(
                in_channels=32, out_channels=256, kernel_size=(2, 2), stride=(2, 2)
            ),
        )

        if self.size == 'big':
            self.d_up_conv_3 = nn.Sequential(
                nn.Conv2d(
                    in_channels=256, out_channels=16, kernel_size=(3, 3), stride=(1, 1)
                ),
                nn.LeakyReLU(),
                nn.ReflectionPad2d((2, 2, 2, 2)),
                nn.Conv2d(
                    in_channels=16, out_channels=3, kernel_size=(3, 3), stride=(1, 1)
                ),
                nn.Tanh(),
            )
        else:
            self.d_up_conv_3 = nn.Sequential(
                nn.Conv2d(
                    in_channels=256, out_channels=16, kernel_size=(3, 3), stride=(1, 1)
                ),
                nn.LeakyReLU(),
                nn.ZeroPad2d((1, 1, 1, 1)),
                nn.ConvTranspose2d(
                    in_channels=16, out_channels=3, kernel_size=(2, 2), stride=(2, 2)
                ),
                nn.Tanh(),
            )

    def quantize_soft(self, encoded):
        raise NotImplementedError()

    def quantize_hard(self, encoded):
        raise NotImplementedError()

    def predecode(self, quantized):
        raise NotImplementedError()

    def to_binary(self, quantized):
        raise NotImplementedError()

    def from_binary(self, binary):
        raise NotImplementedError()

    def forward(self, x):
        encoded = self.encode(x)
        quantized = self.quantize_soft(encoded)
        predecoded = self.predecode(quantized)
        return self.decode(predecoded)

    def encode(self, x):
        x = self.e_conv_1(x)
        x = self.e_conv_2(x)
        x = self.e_block_1(x) + x
        if self.size == 'small':
            x = self.e_pool_1(x)
        x = self.e_block_2(x) + x
        x = self.e_block_3(x) + x
        x = self.e_conv_3(x)  # in [-1, 1] from tanh activation
        return x

    def decode(self, predecoded):
        predecoded = self.d_up_conv_1(predecoded)
        predecoded = self.d_block_1(predecoded) + predecoded
        if self.size == 'small':
            predecoded = self.d_up_1(predecoded)
        predecoded = self.d_block_2(predecoded) + predecoded
        predecoded = self.d_block_3(predecoded) + predecoded
        predecoded = self.d_up_conv_2(predecoded)
        predecoded = self.d_up_conv_3(predecoded)
        return predecoded

    @torch.no_grad()
    def compress(self, source_name: str, target_name: str):
        img, patches = load_patches(source_name)
        patches = patches.to(self.device)
        patches = patches.unsqueeze(0)
        output = []
        for i in range(6):
            for j in range(10):
                x = patches[:, :, i, j, :, :]
                encoded = self.encode(x)
                quantized = self.quantize_hard(encoded)
                quantized = quantized.view(-1)
                output.append(quantized)
        output = torch.cat(output)
        output = self.to_binary(output)
        with open(target_name, 'wb') as f:
            f.write(output)

    @torch.no_grad()
    def decompress(self, source_name: str, target_name: str, smooth: bool = True, ws: int = 6):
        with open(source_name, 'rb') as f:
            input = f.read()
        input = self.from_binary(input)
        input = input.to(self.device)
        input = input.view(60, -1)
        input = input.view(60, 1, *self.latent)
        input = input.permute(1, 0, 2, 3, 4)
        input = input.view(1, 6, 10, *self.latent)
        output = torch.zeros(6, 10, 3, 128, 128)
        for i in range(6):
            for j in range(10):
                quantized = input[:, i, j, :, :]
                predecoded = self.predecode(quantized)
                decoded = self.decode(predecoded)
                output[i, j] = decoded.data

        output = np.transpose(output, (0, 3, 1, 4, 2))
        output = np.reshape(output, (768, 1280, 3))
        output = np.transpose(output, (2, 0, 1))

        save_imgs(
            imgs=output.unsqueeze(0),
            to_size=(3, 768, 1280),
            name=target_name,
        )
        if smooth:
            smooth_image(target_name, ws)
