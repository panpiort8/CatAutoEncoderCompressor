import argparse

import torch as T
import yaml
from bagoftools.namespace import Namespace

from src.models import CAEC, CAEC_NEW

parser = argparse.ArgumentParser()
parser.add_argument("-c", "--config", type=str, required=True)
parser.add_argument("-s", "--source", type=str, required=True)
parser.add_argument("-t", "--target", type=str, required=True)
args = parser.parse_args()

with open(args.config, "rt") as fp:
    cfg = Namespace(**yaml.safe_load(fp))
setattr(cfg, 'device', 'cpu')

model = CAEC_NEW(cfg)
model.load_state_dict(T.load(cfg.checkpoint, map_location='cpu'))

model.compress(source_name=args.source, target_name=args.target)
