import torch
import torch.nn as nn


# from src.data_loader import load_patches


class CAEC(nn.Module):
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

    # @torch.no_grad()
    # def compress(self, source_name: str, target_name: str):
    #     img, patches = load_patches(source_name)
    #     patches = patches.unsqueeze(0)
    #     output = []
    #     for i in range(6):
    #         for j in range(10):
    #             x = patches[:, :, i, j, :, :]
    #             encoded = self.encode(x)
    #             quantized = self.quantize(encoded)
    #             print(quantized.shape)
    #             quantized = quantized.view(-1)
    #             output.append(quantized)
    #     output = torch.cat(output)
    #     pickle.dump(output, open(target_name, 'wb'))
    #
    # def decompress(self, source_name: str, target_name: str):
    #     input = pickle.load(open(source_name, 'rb'))
    #     input = input.view(60, -1)
    #     input = input.view(60, 1, 16, 8, 8)
    #     print(input.shape)

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

    def quantize(self, encoded):
        with torch.no_grad():
            rand = torch.rand(encoded.shape, device=encoded.device)
            prob = (1 + encoded) / 2
            eps = torch.zeros(encoded.shape, device=encoded.device)
            eps[rand <= prob] = (1 - encoded)[rand <= prob]
            eps[rand > prob] = (-encoded - 1)[rand > prob]
        eps = 0.5 * (encoded + eps + 1)
        return eps

    def forward(self, x):
        encoded = self.encode(x)
        quantized = self.quantize(encoded)
        return self.decode(quantized)

    def decode(self, enc):
        x = enc * 2.0 - 1  # (0|1) -> (-1, 1)

        x = self.d_up_conv_1(x)
        x = self.d_block_1(x) + x
        if self.size == 'small':
            x = self.d_up_1(x)
        x = self.d_block_2(x) + x
        x = self.d_block_3(x) + x
        x = self.d_up_conv_2(x)
        x = self.d_up_conv_3(x)
        return x
