import argparse
import os
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import gc
from networks.ScaleFormer import ScaleFormer as DualViT_seg
from bd4h_finalproject.networks.ScaleFormer import ScaleFormerUnet as Baseline
from bd4h_finalproject.networks.ScaleFormer import ScaleFormerUnetIntra as Intra
from bd4h_finalproject.networks.ScaleFormer import ScaleFormerUnetInter as Inter



import shutil
from trainer_synapse import trainer_synapse
parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='./project_TransUNet/data/Synapse/train_npz', help='root dir for data')
parser.add_argument('--dataset', type=str,
                    default='Synapse', help='experiment_name')
parser.add_argument('--list_dir', type=str,
                    default='./lists/lists_Synapse', help='list dir')
parser.add_argument('--num_classes', type=int,
                    default=9, help='output channel of network')
parser.add_argument('--max_iterations', type=int,
                    default=600000, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int,
                    default=8, help='batch_size per gpu')
parser.add_argument('--n_gpu', type=int, default=1, help='total gpu')
parser.add_argument('--deterministic', type=int,  default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.001,
                    help='segmentation network learning rate')
parser.add_argument('--img_size', type=int,
                    default=224, help='input patch size of network input')
parser.add_argument('--seed', type=int,
                    default=0, help='random seed')
parser.add_argument('--max_epochs', type=int,
                    default=30, help='maximum epoch number to train')
parser.add_argument('--snapshot_path', type=str,
                    default="/Users/jackycheng/Desktop/omscs/BD4H/Final_Project/test/", help='vit_patches_size, default is 16')
parser.add_argument('--isDeep', type=bool,
                    default=False, help='vit_patches_size, default is 16')
parser.add_argument('--save_interval', type=int,
                    default=3, help='vit_patches_size, default is 16')
parser.add_argument('--start_save', type=int,
                    default=50, help='vit_patches_size, default is 16')
parser.add_argument('--isLoad', type=bool,
                    default=False, help='vit_patches_size, default is 16')
parser.add_argument('--name', type=str)

args = parser.parse_args()

if __name__ == "__main__":

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    # torch.cuda.manual_seed(args.seed)
    torch.mps.manual_seed(args.seed)

    dataset_name = args.dataset
    dataset_config = {
        'Synapse': {
            'root_path': '/Users/jackycheng/Desktop/omscs/BD4H/Final_Project/test/data/Synapse/train_npz',
            'list_dir': '/Users/jackycheng/Desktop/omscs/BD4H/Final_Project/test/lists/lists_Synapse',
            'num_classes': 9,
        },
    }
    if args.batch_size != 24 and args.batch_size % 6 == 0:
        args.base_lr *= args.batch_size / 24
        
    args.num_classes = dataset_config[dataset_name]['num_classes']
    args.root_path = dataset_config[dataset_name]['root_path']
    args.list_dir = dataset_config[dataset_name]['list_dir']
    args.is_pretrain = False

    # net = Intra(n_classes=args.num_classes)
    if args.name == "Baseline":
        net = Baseline(n_classes=args.num_classes)
    elif args.name == "Intra":
        net = Intra(n_classes=args.num_classes)
    elif args.name == "Inter":
        net = Inter(n_classes=args.num_classes)
    print('# generator parameters:', 1.0 * sum(param.numel() for param in net.parameters()) / 1000000)
    trainer = {'Synapse': trainer_synapse}
    trainer[dataset_name](args, net, args.snapshot_path)