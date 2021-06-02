import pickle
from functools import reduce

import bitarray
import torch
import torch.nn

from .caec_base import CAECBase


class CAECUniform(CAECBase):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.d = cfg.d
        self.latent_dim = reduce(lambda x, y: x * y, self.latent)
        self.mult = torch.tensor(2 ** self.d, requires_grad=False, device=cfg.device)
        self.arrange = torch.arange(0, 2 ** self.d, device=cfg.device)

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

    def predecode(self, quantized):
        quantized = quantized / self.mult  # x in [0, 1]
        quantized = quantized * 2.0 - 1  # x in [-1, 1]
        return quantized

    # def to_binary(self, quantized):
    #     bits = bitarray.bitarray()
    #     for i in quantized:
    #         bits.extend(format(i, f'0{self.d}b'))
    #     return bits.tobytes()
    #
    # def from_binary(self, binary):
    #     bits = bitarray.bitarray()
    #     bits.frombytes(binary)
    #     size = 60 * self.latent_dim
    #     output = torch.zeros((size), dtype=torch.float)
    #     for i in range(size):
    #         idx = i * self.d
    #         output[i] = int(bits[idx: idx + self.d].to01(), 2)
    #     return output

    def to_binary(self, quantized):
        return pickle.dumps(quantized)

    def from_binary(self, binary):
        return pickle.loads(binary)