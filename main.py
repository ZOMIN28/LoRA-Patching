import argparse

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', '1', 'True'):
        return True
    elif v.lower() in ('no', 'false', 'f', '0', 'False'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser(description='manual to this script')
parser.add_argument("--gpus", type=int, default=0)
parser.add_argument("--epochs", type=int, default=5)
parser.add_argument("--batch_size", type=int, default=4)
parser.add_argument("--deepfake", type=str, default="stargan")
parser.add_argument("--mode", type=str, default="train")
parser.add_argument("--warning", type=str2bool, default=False)
parser.add_argument("--rank", type=int, default=8)
parser.add_argument("--leakage", type=str2bool, default=False)
parser.add_argument("--data_size", type=int, default=1000)
parser.add_argument("--lambda_feat", type=float, default=0.1)
parser.add_argument("--lambda_blip", type=float, default=0.1)
parser.add_argument("--image_dir", type=str, default="F:/paper/ComGAN/data/celeba-256/images/")
parser.add_argument("--attr_path", type=str, default="F:/paper/ComGAN/data/celeba-256/list.txt")
args = parser.parse_args()

import os
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpus)
import warnings
warnings.filterwarnings("ignore", category=Warning)
import torch
import numpy as np
import random
from lora_patching import LoRA_patching
from utils.utils import getDataloader
seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

def main(args):
    device = torch.device("cuda" if (torch.cuda.is_available()) else "cpu")
    train_dataloader, test_dataloader = getDataloader(
        image_dir=args.image_dir,
        attr_path=args.attr_path,
        batch_size=args.batch_size,
        data_size=args.data_size,
        model_type=args.deepfake)
    lp = LoRA_patching(device, args)

    if args.mode == "train":
        lp.train(train_dataloader)
    elif args.mode == "test":
        lp.test(test_dataloader)

if __name__ == '__main__':
    main(args)