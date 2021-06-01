import pickle

import torch

from src.models.caec_base import CAECBase


class CAECStochastic(CAECBase):

    def to_binary(self, quantized):
        return pickle.dumps(quantized)

    def from_binary(self, binary):
        return pickle.loads(binary)

    def quantize_soft(self, encoded):
        with torch.no_grad():
            rand = torch.rand(encoded.shape, device=encoded.device)
            prob = (1 + encoded) / 2
            eps = torch.zeros(encoded.shape, device=encoded.device)
            eps[rand <= prob] = (1 - encoded)[rand <= prob]
            eps[rand > prob] = (-encoded - 1)[rand > prob]
        eps = 0.5 * (encoded + eps + 1)
        return eps

    @torch.no_grad()
    def quantize_hard(self, encoded):
        return self.quantize_soft(encoded)

    def predecode(self, quantized):
        quantized = quantized * 2.0 - 1  # (0|1) -> (-1, 1)
        return quantized
