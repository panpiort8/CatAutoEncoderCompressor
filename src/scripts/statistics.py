import os
import tempfile
from math import log10, sqrt

import cv2
import numpy as np
from skimage.metrics import structural_similarity
from PIL import Image
from src.models.utils import make_model
from src.utils.utils import make_default_argparse, make_cfg


class JPEG:
    def compress(self, source, target):
        img = Image.open(source)
        img.save(target, format='JPEG')

    def decompress(self, source, target, ws=None):
        img = Image.open(source)
        img.save(target)


parser = make_default_argparse()
parser.add_argument("-d", "--dir", type=str, required=True)
parser.add_argument("--ws", type=int, default=6)
args = parser.parse_args()
cfg = make_cfg(args)

if cfg.model_cls == 'jpeg':
    model = JPEG()
else:
    model = make_model(cfg)
    model.eval()


def psnr(original, compressed):
    mse = np.mean((original - compressed) ** 2)
    if mse == 0:
        return 100
    max_pixel = 255.0
    return 20 * log10(max_pixel / sqrt(mse))


def ssim(original, compressed):
    original = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    compressed = cv2.cvtColor(compressed, cv2.COLOR_BGR2GRAY)
    return structural_similarity(original, compressed)


def print_stats(name, stats):
    print(f'{name}: {np.mean(stats):.4f} \u00b1 {np.std(stats):.4f}')


psnr_list = []
ssim_list = []
sizes_list = []
n = len(os.listdir(args.dir))
for i, name in enumerate(os.listdir(args.dir)):
    print(f'{i}/{n}')
    original_path = os.path.join(args.dir, name)
    with tempfile.NamedTemporaryFile(suffix='.bmp') as f:
        compressed_path = f.name
        model.compress(original_path, compressed_path)
        sizes_list.append(os.path.getsize(compressed_path) / 1024)
        model.decompress(compressed_path, compressed_path, ws=args.ws)
        original = cv2.imread(original_path)
        compressed = cv2.imread(compressed_path)
        psnr_list.append(psnr(original, compressed))
        ssim_list.append(ssim(original, compressed))

print_stats('PSNR', psnr_list)
print_stats('SSIM', ssim_list)
print_stats('Size', sizes_list)
