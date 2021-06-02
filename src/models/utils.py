import torch

from src.models import CAECStochastic, CAECUniform


def make_model(cfg):
    model_cls = CAECStochastic if cfg.model_cls == 'CAECStochastic' else CAECUniform
    model = model_cls(cfg)
    if hasattr(cfg, 'checkpoint'):
        model.load_state_dict(torch.load(cfg.checkpoint, map_location='cpu'))
    model.eval()
    model.to(cfg.device)
    return model