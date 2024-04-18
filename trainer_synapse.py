import logging
import os
import random
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import DiceLoss
from torchvision import transforms
import matplotlib.pyplot as plt
import os
import sys
# from datasets.dataset_synapse import Synapse_dataset, RandomGenerator



def drawLoss(train_loss, val_loss, epoch_num, name, snapshot_path):
    fig1, ax1 = plt.subplots(figsize=(11, 8))
    ax1.plot(range(epoch_num + 1), train_loss, label='Train Loss', color='blue')  # Train loss in blue
    ax1.plot(range(epoch_num + 1), val_loss, label='Validation Loss', color='red') 
    ax1.set_title("Average Train/Validation Loss vs Epochs")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Current loss")
    ax1.legend()
    plt.savefig(os.path.join(snapshot_path, 'loss_' + name + '_vs_epochs.png'))

    plt.clf()
    plt.close()
    
def setup_logging(name, snapshot_path):
    log_format = '[%(asctime)s.%(msecs)03d] %(message)s'
    logging.basicConfig(filename=os.path.join(snapshot_path, f"{name}_training_log.txt"),
                        level=logging.INFO, format=log_format, datefmt='%H:%M:%S')
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    formatter = logging.Formatter(log_format)
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

def initialize_dataset_and_loader(args):
    transform = transforms.Compose([RandomGenerator(output_size=[args.img_size, args.img_size])])
    dataset = Synapse_dataset(dataset_directory=args.root_path, list_directory=args.list_dir,
                             dataset_split="train", data_transform=transform)
    loader = DataLoader(dataset, batch_size=args.batch_size * args.n_gpu, shuffle=True, num_workers=4, pin_memory=True)
    return dataset, loader

def initialize_testdataset_and_loader(args):
    # transform = transforms.Compose([RandomGenerator(output_size=[args.img_size, args.img_size])])
    dataset = Synapse_dataset(dataset_directory=args.root_path, list_directory=args.list_dir,
                             dataset_split="test")
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)
    return dataset, loader

def train_epoch(loss_all, model, train_loader, optimizer, ce_loss, dice_loss, args, epoch, writer, scheduler=None, valid=False):
    model.train()
    if valid:
        model.eval()
    total_loss = 0.0
    
    for batch_data in train_loader:
        image_batch, label_batch = batch_data['image'], batch_data['label']
        optimizer.zero_grad()
        outputs = model(image_batch)
        loss_ce = ce_loss(outputs, label_batch.long())
        loss_dice = dice_loss(outputs, label_batch, softmax=True)
        print("loss_ce: ", loss_ce)
        print("loss_dice: ", loss_dice)
        loss = 0.5 * (loss_ce + loss_dice)
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
    avg_loss = total_loss / len(train_loader)
    loss_all.append(avg_loss)
    logging.info(f"Epoch {epoch}: Average Loss = {avg_loss:.4f}")
    writer.add_scalar('Loss/train', avg_loss, epoch)
    if valid:
        model.train()
    return loss_all
    

def trainer_synapse(args, model, snapshot_path):
    loss_all = []
    loss_all_valid = []
    setup_logging(args.name,snapshot_path)
    logging.info(f"Training parameters: {args}")

    _, train_loader = initialize_dataset_and_loader(args)
    import copy
    args_val = copy.deepcopy(args)
    args_val.list_dir = '/Users/jackycheng/Desktop/omscs/BD4H/Final_Project/lists_val/lists_Synapse'
    _, valid_loader = initialize_dataset_and_loader(args_val)


    #hyper-params
    optimizer = optim.SGD(model.parameters(), lr=args.base_lr, momentum=0.9, weight_decay=0.0001)
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: (1 - epoch / args.max_epochs) ** 0.9)
    ce_loss = nn.CrossEntropyLoss()
    dice_loss = DiceLoss(args.num_classes)


    writer = SummaryWriter(log_dir=os.path.join(snapshot_path, 'logs'))
    for epoch in range(args.max_epochs - 1):
        train_loss = train_epoch(loss_all, model, train_loader, optimizer, ce_loss, dice_loss, args, epoch, writer, scheduler)
        val_loss = train_epoch(loss_all_valid, model, valid_loader, optimizer, ce_loss, dice_loss, args, epoch, writer, scheduler, valid=True)
        drawLoss(train_loss, val_loss, epoch, name=args.name, snapshot_path=args.snapshot_path)
        if epoch % args.save_interval == 0 or epoch == args.max_epochs:
            model_save_path = os.path.join(snapshot_path, f"{args.name}_model_epoch_{epoch}.pth")
            torch.save(model.state_dict(), model_save_path)
            logging.info(f"Saved model checkpoint to '{model_save_path}'.")

    writer.close()
    logging.info("Training completed successfully.")


import os
import random
import h5py
import numpy as np
import torch
from scipy import ndimage
from scipy.ndimage.interpolation import zoom
from torch.utils.data import Dataset


def random_rot_flip(image, label):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label


def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label


class RandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        elif random.random() > 0.5:
            image, label = random_rotate(image, label)
        x, y = image.shape
        if x != self.output_size[0] or y != self.output_size[1]:
            image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=3)  # why not 3?
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        image = image / 1000000
        label = torch.from_numpy(label.astype(np.float32))
        sample = {'image': image, 'label': label.long()}
        return sample


class Synapse_dataset(Dataset):
    def __init__(self, dataset_directory, list_directory, dataset_split, data_transform=None):
        self.data_transform = data_transform
        self.dataset_split = dataset_split
        self.sample_filenames = [line.strip() for line in open(os.path.join(list_directory, f"{dataset_split}.txt"))]
        self.dataset_directory = dataset_directory

    def __len__(self):
        return len(self.sample_filenames)

    def __getitem__(self, index):
        filename = self.sample_filenames[index]
        if self.dataset_split == "train":
            file_path = os.path.join(self.dataset_directory, f"{filename}.npz")
            data = np.load(file_path)
        else:
            file_path = os.path.join(self.dataset_directory, f"{filename}.npy.h5")
            data = h5py.File(file_path, 'r')

        image = data['image'][:]
        label = data['label'][:]
        sample = {'image': image, 'label': label, 'case_name': filename}

        if self.data_transform:
            sample = self.data_transform(sample)

        return sample
