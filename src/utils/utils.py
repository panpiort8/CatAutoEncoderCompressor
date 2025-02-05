import argparse
import struct

import numpy as np
import yaml
from bagoftools.namespace import Namespace
from torchvision.utils import save_image


def save_imgs(imgs, to_size, name) -> None:
    # x = np.array(x)
    # x = np.transpose(x, (1, 2, 0)) * 255
    # x = x.astype(np.uint8)
    # imsave(name, x)

    # x = 0.5 * (x + 1)

    # to_size = (C, H, W)
    imgs = imgs.clamp(0, 1)
    imgs = imgs.view(imgs.size(0), *to_size)
    save_image(imgs, name)


def save_encoded(enc: np.ndarray, fname: str) -> None:
    enc = np.reshape(enc, -1)
    sz = str(len(enc)) + "d"

    with open(fname, "wb") as fp:
        fp.write(struct.pack(sz, *enc))


def make_default_argparse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--device", type=str, default='cpu')
    return parser


def make_cfg(args):
    with open(args.config, "rt") as fp:
        cfg = Namespace(**yaml.safe_load(fp))
    setattr(cfg, 'device', args.device)
    return cfg
