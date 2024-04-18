import argparse
import os
import random
import numpy as np
import torch
from bd4h_finalproject.networks.ScaleFormer import ScaleFormerUnetIntra as Intra
from bd4h_finalproject.networks.ScaleFormer import ScaleFormerUnet as Baseline
from trainer_synapse import trainer_synapse

def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.mps.manual_seed(seed)

def adjust_learning_rate(args):
    if args.batch_size != 24 and args.batch_size % 6 == 0:
        args.base_lr *= args.batch_size / 24


def get_dataset_config():
    return {
        'Synapse': {
            'root_path': '/Users/jackycheng/Desktop/omscs/BD4H/Final_Project/test/data/Synapse/train_npz',
            'list_dir': '/Users/jackycheng/Desktop/omscs/BD4H/Final_Project/test/lists/lists_Synapse',
            'num_classes': 9,
        },
    }

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', type=str, default='./project_TransUNet/data/Synapse/train_npz')
    parser.add_argument('--dataset', type=str, default='Synapse')
    parser.add_argument('--list_dir', type=str, default='./lists/lists_Synapse')
    parser.add_argument('--num_classes', type=int, default=9)
    parser.add_argument('--max_iterations', type=int, default=600000)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--n_gpu', type=int, default=1)
    parser.add_argument('--deterministic', type=int, default=1)
    parser.add_argument('--base_lr', type=float, default=0.001)
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--max_epochs', type=int, default=100)
    parser.add_argument('--snapshot_path', type=str, default="./")
    parser.add_argument('--save_interval', type=int, default=10)
    return parser.parse_args()

def main():
    args = parse_arguments()
    setup_seed(args.seed)
    
    dataset_config = get_dataset_config()
    dataset_name = args.dataset
    args.num_classes = dataset_config[dataset_name]['num_classes']
    args.root_path = dataset_config[dataset_name]['root_path']
    args.list_dir = dataset_config[dataset_name]['list_dir']

    adjust_learning_rate(args)
    breakpoint()
    # net = Intra(n_classes=args.num_classes)
    net = Baseline(n_classes=args.num_classes)

    
    # trainer_dict = {'Synapse': trainer_synapse}
    # trainer_dict[dataset_name](args, net, args.snapshot_path)

    trainer_synapse(args, net, args.snapshot_path)

if __name__ == "__main__":
    main()

def get_network(args):
    net = Intra(n_classes=args.num_classes)
    print(f'# generator parameters: {sum(param.numel() for param in net.parameters()) / 1e6:.2f}M')
    return net