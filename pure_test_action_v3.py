import argparse
import os
from decimal import Decimal
from typing import Tuple

import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

from src.utils.action_dataset_v0 import ActionDatasetV0
from src.utils.equipment import Equipment
from src.utils.configuration import elec_mlp_config
from src.utils.knife import setup_seed, get_timestr
from src.utils.constant import model_folder_path, data_folder_path
from src.models.manet import MANet


parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=11)
parser.add_argument('--batch', type=int, default=64)
parser.add_argument('--model-name', type=str,
                    default='0.9986_11_15_0.0001_72_2023-07-13|16:29:46.pt')
parser.add_argument('--train-scale', type=float, default=0.7)

args = parser.parse_args()
setup_seed(args.seed)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)


manet = MANet()
manet.load_state_dict(torch.load(
    os.path.join(model_folder_path, args.model_name)))
manet.to(device)
manet.eval()

data_prefix = '20230718'
dataset = ActionDatasetV0(data_prefix)
train_size = int(len(dataset) * args.train_scale)
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(
    dataset, [train_size, test_size])

train_loader = DataLoader(
    train_dataset, batch_size=args.batch, shuffle=True, num_workers=0)
test_loader = DataLoader(
    test_dataset, batch_size=args.batch, shuffle=True, num_workers=0)
all_loader = DataLoader(
    dataset, batch_size=args.batch, shuffle=True, num_workers=0)


def test(loader: DataLoader) -> Tuple[torch.Tensor, torch.Tensor]:
    test_accuracy_numerator = 0
    test_accuracy_denominator = 0
    test_recall_numerator = 0
    test_recall_denominator = 0
    test_loss_list = []

    # for feature, label in loader:
    for feature, label in tqdm(loader):
        action_label = label.to(device)
        chuanlian_feature = feature[Equipment.chuanlian.value].to(device)
        rongkang_feature = feature[Equipment.rongkang.value].to(device)
        binya_feature = feature[Equipment.bianya.value].to(device)
        xiandian_feature = feature[Equipment.xiandian.value].to(device)
        jiaoxian_feature = feature[Equipment.jiaoxian.value].to(device)
        fuhe_feature = feature[Equipment.fuhe.value].to(device)
        fadian_feature = feature[Equipment.fadian.value].to(device)
        muxian_feature = feature[Equipment.muxian.value].to(device)
        changzhan_feature = feature[Equipment.changzhan.value].to(device)
        

        with torch.no_grad():
            result = manet(chuanlian_feature, rongkang_feature, binya_feature, xiandian_feature,
                           jiaoxian_feature, fuhe_feature, fadian_feature, muxian_feature, changzhan_feature)
            x = torch.sigmoid(result)
            mask = action_label.ge(0.5)
            prediction = torch.where(torch.sigmoid(result) >= 0.5, 1.0, 0.0)

            test_accuracy_numerator += int((prediction == action_label).sum())
            test_accuracy_denominator += action_label.numel()
            test_recall_numerator += int(
                (prediction[mask] == action_label[mask]).sum())
            test_recall_denominator += int(mask.sum())

                


    print('TAcc:\t{}\tTRec:\t{}'.format('%.6f' % (test_accuracy_numerator /
                                                  test_accuracy_denominator), '%.6f' % (test_recall_numerator / test_recall_denominator)))
    
    return

test(all_loader)

