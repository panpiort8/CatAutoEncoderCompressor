import argparse
import os
import random
import shutil

parser = argparse.ArgumentParser()
parser.add_argument("-s", "--source_dir", type=str, required=True)
parser.add_argument("-t1", "--target1_dir", type=str, required=True)
parser.add_argument("-t2", "--target2_dir", type=str, required=True)
parser.add_argument("-r", "--ratio", type=float, required=True)
args = parser.parse_args()

source_dir = args.source_dir
target1_dir = args.target1_dir
target2_dir = args.target2_dir
ratio = args.ratio

os.makedirs(target1_dir, exist_ok=True)
os.makedirs(target2_dir, exist_ok=True)
n = len(os.listdir(source_dir))
n1 = n - int(round(n * ratio))

names = list(os.listdir(source_dir))
random.seed(42)
random.shuffle(names)


def move(files, source, target):
    for name in files:
        shutil.move(os.path.join(source, name), os.path.join(target, name))


move(names[:n1], source_dir, target1_dir)
move(names[n1:], source_dir, target2_dir)
