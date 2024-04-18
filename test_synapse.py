import logging
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from trainer_synapse import Synapse_dataset
# from utils import test_single_volume
from networks.ScaleFormer import ScaleFormer
from bd4h_finalproject.networks.ScaleFormer import ScaleFormerUnet as Baseline
from bd4h_finalproject.networks.ScaleFormer import ScaleFormerUnetIntra as Intra
from bd4h_finalproject.networks.ScaleFormer import ScaleFormerUnetInter as Inter

import argparse

# Configure logging to file
logging.basicConfig(filename='test.log', level=logging.INFO,
                    format='%(asctime)s %(levelname)s:%(message)s', datefmt='%Y-%m-%d %H:%M:%S')

def inference(args, model, test_save_path=None):
    db_test = Synapse_dataset(dataset_directory=args.volume_path, dataset_split="test_vol", list_directory=args.list_dir)
    testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=0)
    logging.info("{} test iterations per epoch".format(len(testloader)))
    model.eval()
    metric_list = 0.0
    with torch.no_grad():
        for i_batch, sampled_batch in tqdm(enumerate(testloader)):
            h, w = sampled_batch["image"].size()[2:]
            image, label, case_name = sampled_batch["image"], sampled_batch["label"], sampled_batch['case_name'][0]
            metric_i = test_single_volume(image, label, model, classes=args.num_classes, patch_size=[args.img_size, args.img_size],
                                          test_save_path=test_save_path, case=case_name, z_spacing=args.z_spacing)
            metric_i =  np.array(metric_i)
            metric_list += metric_i
            logging.info('idx %d case %s mean_dice %f mean_hd95 %f' % (i_batch, case_name, np.mean(metric_i, axis=0)[0], np.mean(metric_i, axis=0)[1]))
        metric_list = metric_list / len(db_test)
        for i in range(1, args.num_classes):
            logging.info('Mean class %d mean_dice %f' % (i, metric_list[i-1][0]))
            logging.info('Mean class %d mean_hd95 %f' % (i, metric_list[i-1][1]))

            # print('Mean class %d mean_dice %f' % (i, metric_list[i-1][0]))
            # print('Mean class %d mean_hd95 %f' % (i, metric_list[i-1][1]))
            

        performance = np.mean(metric_list, axis=0)[0]
        mean_hd95 = np.mean(metric_list, axis=0)[1]
        logging.info('Testing performance in best val model: mean_dice : %f mean_hd95 : %f' % (performance, mean_hd95))
        print('Testing performance in best val model: mean_dice : %f mean_hd95 : %f' % (performance, mean_hd95))

        logging.info("Testing Finished!")
        return performance


from medpy import metric
from scipy.ndimage import zoom

def test_single_volume(image, label, net, classes, patch_size=[256, 256], test_save_path=None, case=None, z_spacing=1):
    image, label = image.squeeze(0).cpu().detach().numpy(), label.squeeze(0).cpu().detach().numpy()
    if len(image.shape) == 3:
        prediction = np.zeros_like(label)
        for ind in range(image.shape[0]):
            slice = image[ind, :, :]
            x, y = slice.shape[0], slice.shape[1]
            if x != patch_size[0] or y != patch_size[1]:
                slice = zoom(slice, (patch_size[0] / x, patch_size[1] / y), order=3)  # previous using 0
            # input = torch.from_numpy(slice).unsqueeze(0).unsqueeze(0).float().cuda()
            input = torch.from_numpy(slice).unsqueeze(0).unsqueeze(0).float()

            net.eval()
            with torch.no_grad():
                outputs = net(input)
                out = torch.argmax(torch.softmax(outputs, dim=1), dim=1).squeeze(0)
                out = out.cpu().detach().numpy()
                if x != patch_size[0] or y != patch_size[1]:
                    pred = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
                else:
                    pred = out
                prediction[ind] = pred
            if ind == 1:
                break
    else:
        # input = torch.from_numpy(image).unsqueeze(0).unsqueeze(0).float().cuda()
        input = torch.from_numpy(image).unsqueeze(0).unsqueeze(0).float()

        net.eval()
        with torch.no_grad():
            out = torch.argmax(torch.softmax(net(input), dim=1), dim=1).squeeze(0)
            prediction = out.cpu().detach().numpy()
    metric_list = []
    for i in range(1, classes):
        metric_list.append(calculate_metric_percase(prediction == i, label == i))
    return metric_list

def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() > 0 and gt.sum()>0:
        dice = metric.binary.dc(pred, gt)
        hd95 = metric.binary.hd95(pred, gt)
        return dice, hd95
    elif pred.sum() > 0 and gt.sum()==0:
        return 1, 0
    else:
        return 0, 0

if __name__ =="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', default=9)
    parser.add_argument('--model_path', default=None)
    parser.add_argument('--root_path', type=str,
                        default='/Users/jackycheng/Desktop/omscs/BD4H/Final_Project/ScaleFormer/', help='root dir for data')
    parser.add_argument('--dataset', type=str,
                        default='Synapse', help='experiment_name')
    parser.add_argument('--list_dir', type=str,
                        default='./lists/lists_Synapse', help='list dir')
    parser.add_argument('--test_save_dir', default='./predictions_Synapse', help='saving prediction as nii!')
    parser.add_argument("--volume_path", default="./data/Synapse/test_vol_h5")
    parser.add_argument('--img_size', type=int,
                        default=224, help='input patch size of network input')
    parser.add_argument("--z_spacing", default=1)
    parser.add_argument('--name', type=str)
    parser.add_argument('--epoch', type=str)

    args = parser.parse_args()

    args.model_path = f"./{args.name}_model_epoch_{args.epoch}.pth"

    if args.name == "Baseline":
        model = Baseline(n_classes=args.num_classes)
    elif args.name == "Intra":
        model = Intra(n_classes=args.num_classes)
    elif args.name == "Inter":
        model = Inter(n_classes=args.num_classes)
    
    print("name: ", args.name)
    print("path: ", args.model_path)
    model.load_state_dict(torch.load(args.model_path))
    inference(args, model, args.test_save_dir)


