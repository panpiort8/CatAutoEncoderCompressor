import os
from pathlib import Path

import numpy as np
import torch
import torch as T
import torch.nn as nn
from bagoftools.logger import Logger
from bagoftools.namespace import Namespace
from torch.utils.data import DataLoader

import models
from src.models.utils import make_model
from src.utils.data_loader import ImageFolder720p
from src.utils.utils import save_imgs, make_default_argparse, make_cfg

ROOT_EXP_DIR = Path(__file__).resolve().parents[1] / "experiments"

logger = Logger(__name__, colorize=True)


def test(cfg: Namespace) -> None:
    assert cfg.checkpoint not in [None, ""]
    device = torch.device(cfg.device)

    exp_dir = ROOT_EXP_DIR / cfg.exp_name
    os.makedirs(exp_dir / "out", exist_ok=True)
    cfg.to_file(exp_dir / "test_config.json")
    logger.info(f"[exp dir={exp_dir}]")

    model = make_model(cfg)
    model = model.to(device)
    logger.info(f"[model={cfg.checkpoint}] on {cfg.device}")

    dataloader = DataLoader(
        dataset=ImageFolder720p(cfg.dataset_path), batch_size=1, shuffle=cfg.shuffle
    )
    logger.info(f"[dataset={cfg.dataset_path}]")

    loss_criterion = nn.MSELoss()

    for batch_idx, data in enumerate(dataloader, start=1):
        img, patches, _ = data
        patches = patches.to(device)

        if batch_idx % cfg.batch_every == 0:
            pass

        out = T.zeros(6, 10, 3, 128, 128)
        avg_loss = 0

        for i in range(6):
            for j in range(10):
                x = patches[:, :, i, j, :, :]
                y = model(x)
                out[i, j] = y.data

                loss = loss_criterion(y, x)
                avg_loss += (1 / 60) * loss.item()

        logger.debug("[%5d/%5d] avg_loss: %f", batch_idx, len(dataloader), avg_loss)

        # save output
        out = np.transpose(out, (0, 3, 1, 4, 2))
        out = np.reshape(out, (768, 1280, 3))
        out = np.transpose(out, (2, 0, 1))

        y = T.cat((img[0], out), dim=2)
        save_imgs(
            imgs=y.unsqueeze(0),
            to_size=(3, 768, 2 * 1280),
            name=exp_dir / f"out/test_{batch_idx}.png",
        )


if __name__ == "__main__":
    parser = make_default_argparse()
    args = parser.parse_args()
    cfg = make_cfg(args)

    test(cfg)
