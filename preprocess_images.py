import argparse
import os

from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument("-s", "--source_dir", type=str, required=True)
parser.add_argument("-t", "--target_dir", type=str, required=True)
args = parser.parse_args()

source_dir = args.source_dir
target_dir = args.target_dir

os.makedirs(target_dir, exist_ok=True)
n = len(os.listdir(source_dir))
for i, name in enumerate(os.listdir(source_dir)):
    source_path = os.path.join(source_dir, name)
    target_path = os.path.join(target_dir, name.replace('.jpg', '.bmp'))
    print(f'{source_path} -> {target_path}  ({i + 1}/{n})')
    image = Image.open(source_path)
    if image.height > image.width:
        image = image.transpose(Image.ROTATE_90)
    image = image.resize((1280, 768), Image.LANCZOS)
    image.save(target_path)
