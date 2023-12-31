import argparse
import os
from decimal import Decimal
from typing import Tuple

from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

from src.utils.action_dataset_v0 import ActionDatasetV0
from src.utils.equipment import Equipment
from src.utils.configuration import elec_mlp_config
from src.utils.knife import setup_seed, get_timestr
from src.utils.constant import model_folder_path
from src.models.manet import MANet

# 希望复现训练时的数据集划分

parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default=0.0001)
parser.add_argument('--epoch', type=int, default=500)
parser.add_argument('--saved-epoch', type=int, default=61)
parser.add_argument('--seed', type=int, default=11)
parser.add_argument('--batch', type=int, default=64)
parser.add_argument('--pos-weight', type=int, default=15)
parser.add_argument('--train-scale', type=float, default=0.7)
parser.add_argument('--model-name', type=str,
                    default='0.9986_11_15_0.0001_72_2023-07-13|16:29:46.pt')

args = parser.parse_args()
setup_seed(args.seed)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)


manet = MANet().to(device)
pos_weight = torch.tensor([args.pos_weight]).to(device)
criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction='mean')
optimizer = torch.optim.Adam(manet.parameters(), lr=args.lr)

dataset = ActionDatasetV0('20230710')
train_size = int(len(dataset) * args.train_scale)
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(
    dataset, [train_size, test_size])

train_loader = DataLoader(
    train_dataset, batch_size=args.batch, shuffle=True, num_workers=0)
test_loader = DataLoader(
    test_dataset, batch_size=args.batch, shuffle=True, num_workers=0)

manet.load_state_dict(torch.load(
    os.path.join(model_folder_path, args.model_name)))
manet.to(device)
manet.eval()

print(train_size)
print(test_size)

def test(loader: DataLoader) -> Tuple[torch.Tensor, torch.Tensor]:
    test_accuracy_numerator = 0
    test_accuracy_denominator = 0
    test_recall_numerator = 0
    test_recall_denominator = 0

    test_celue_0_accuracy_numerator = 0
    test_celue_0_accuracy_denominator = 0
    test_celue_0_recall_numerator = 0
    test_celue_0_recall_denominator = 0

    test_celue_1_accuracy_numerator = 0
    test_celue_1_accuracy_denominator = 0
    test_celue_1_recall_numerator = 0
    test_celue_1_recall_denominator = 0

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

            false_tensor = torch.tensor(
                np.zeros(shape=(mask.shape[0], 3619, 1)), dtype=torch.bool)
            true_tensor = torch.tensor(
                np.ones(shape=(mask.shape[0], 3619, 1)), dtype=torch.bool)

            celue_0_mask = torch.cat((
                true_tensor,
                false_tensor,
            ), 2).to(device)

            celue_1_mask = torch.cat((
                false_tensor,
                true_tensor,
            ), 2).to(device)

            celue_0_mask_recall = celue_0_mask.logical_and(mask)
            celue_1_mask_recall = celue_1_mask.logical_and(mask)

            test_celue_0_accuracy_numerator += int(
                (prediction[celue_0_mask] == action_label[celue_0_mask]).sum())
            test_celue_0_accuracy_denominator += action_label[celue_0_mask].numel(
            )
            test_celue_1_accuracy_numerator += int(
                (prediction[celue_1_mask] == action_label[celue_1_mask]).sum())
            test_celue_1_accuracy_denominator += action_label[celue_1_mask].numel(
            )

            test_celue_0_recall_numerator += int(
                (prediction[celue_0_mask_recall] == action_label[celue_0_mask_recall]).sum())
            test_celue_0_recall_denominator += int(celue_0_mask_recall.sum())
            test_celue_1_recall_numerator += int(
                (prediction[celue_1_mask_recall] == action_label[celue_1_mask_recall]).sum())
            test_celue_1_recall_denominator += int(celue_1_mask_recall.sum())

    print('TAcc:\t{}\tTRec:\t{}'.format('%.6f' % (test_accuracy_numerator /
                                                  test_accuracy_denominator), '%.6f' % (test_recall_numerator / test_recall_denominator)))
    print('TAcc0:\t{}\tTRec0:\t{}'.format('%.6f' % (test_celue_0_accuracy_numerator /
                                                  test_celue_0_accuracy_denominator), '%.6f' % (test_celue_0_recall_numerator / test_celue_0_recall_denominator)))
    print('TAcc1:\t{}\tTRec1:\t{}'.format('%.6f' % (test_celue_1_accuracy_numerator /
                                                  test_celue_1_accuracy_denominator), '%.6f' % (test_celue_1_recall_numerator / test_celue_1_recall_denominator)))

    return


test(train_loader)
test(test_loader)