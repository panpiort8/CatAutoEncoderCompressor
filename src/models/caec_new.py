import bitarray
import numpy as np
import torch
import torch.nn
import torch.nn as nn

# from src.data_loader import load_patches
from src.data_loader import load_patches
from src.utils import smooth_image
from src.utils.utils import save_imgs


class CAEC_NEW(nn.Module):
    """
    This AE module will be fed 3x128x128 patches from the original image
    Shapes are (batch_size, channels, height, width)

    Latent representation:
        - small: 16x8x8 bits per patch => 7.5KB per image (for 720p)
        - medium: 16x16x16 bits per patch => 30KB per image (for 720p)
        - big: 32x32x32 bits per patch => 240KB per image (for 720p)
    """

    def __init__(self, cfg):
        super().__init__()
        self.size = cfg.model_size
        self.d = 4
        self.latent_dim = {'small': 1024, 'medium': 4096, 'big': 32768}[self.size]
        self.mult = torch.tensor(2 ** self.d, requires_grad=False, device=cfg.device)
        self.arrange = torch.arange(0, 2 ** self.d, device=cfg.device)
        assert self.size in ['small', 'medium', 'big']
        self.encoded = None

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

    @torch.no_grad()
    def compress(self, source_name: str, target_name: str):
        img, patches = load_patches(source_name)
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

    def to_binary(self, x):
        bits = bitarray.bitarray()
        for i in x:
            v = format(i, f'0{self.d}b')
            if len(v) != 4:
                print(i, v)
            bits.extend(format(i, f'0{self.d}b'))
        return bits.tobytes()

    def from_binary(self, x):
        bits = bitarray.bitarray()
        bits.frombytes(x)
        size = 60 * self.latent_dim
        output = torch.zeros((size), dtype=torch.float)
        for i in range(size):
            idx = i * self.d
            output[i] = int(bits[idx: idx + self.d].to01(), 2)
        return output

    @torch.no_grad()
    def decompress(self, source_name: str, target_name: str, smooth: bool = True, ws: int = 16):
        with open(source_name, 'rb') as f:
            input = f.read()
        input = self.from_binary(input)
        input = input.view(60, -1)
        input = input.view(60, 1, 16, 8, 8)
        input = input.permute(1, 0, 2, 3, 4)
        input = input.view(1, 6, 10, 16, 8, 8)
        out = torch.zeros(6, 10, 3, 128, 128)
        for i in range(6):
            for j in range(10):
                x = input[:, i, j, :, :]
                y = self.decode(x)
                out[i, j] = y.data

        out = np.transpose(out, (0, 3, 1, 4, 2))
        out = np.reshape(out, (768, 1280, 3))
        out = np.transpose(out, (2, 0, 1))

        save_imgs(
            imgs=out.unsqueeze(0),
            to_size=(3, 768, 1280),
            name=target_name,
        )
        if smooth:
            smooth_image(target_name, ws)

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

    def quantize_soft(self, encoded):
        q = 0.5 * (encoded + 1)  # q in [0, 1]
        q = q * self.mult
        q = q.view(q.shape[0], -1).unsqueeze(-1)
        exp = torch.exp(-torch.abs(q - self.arrange))
        sum_u = torch.sum(exp * self.arrange, dim=2)
        sum_d = torch.sum(exp, dim=2)
        q = sum_u / sum_d
        q = q.view(*encoded.shape)
        return q

    @torch.no_grad()
    def quantize_hard(self, encoded):
        q = 0.5 * (encoded + 1)  # q in [0, 1]
        q = torch.floor(q * self.mult).long()
        q[q == 16] = 15
        return q

    def forward(self, x):
        encoded = self.encode(x)
        quantized_soft = self.quantize_soft(encoded)
        return self.decode(quantized_soft)

    def decode(self, enc):
        x = enc / self.mult  # x in [0, 1]
        x = x * 2.0 - 1  # x in [-1, 1]

        x = self.d_up_conv_1(x)
        x = self.d_block_1(x) + x
        if self.size == 'small':
            x = self.d_up_1(x)
        x = self.d_block_2(x) + x
        x = self.d_block_3(x) + x
        x = self.d_up_conv_2(x)
        x = self.d_up_conv_3(x)
        return x
