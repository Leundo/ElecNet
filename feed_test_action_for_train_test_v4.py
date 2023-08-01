import argparse
import os
from decimal import Decimal
from typing import Tuple

from tqdm import tqdm
import numpy as np
import torch
import json
from torch.utils.data import DataLoader
import torch.nn.functional as F

from src.utils.action_dataset_v0 import ActionDatasetV0, ActionDatasetV1
from src.utils.equipment import Equipment
from src.utils.configuration import elec_mlp_config
from src.utils.knife import setup_seed, get_timestr
from src.utils.constant import model_folder_path
from src.utils.porter import load_signifiant_mask
from src.models.manet import MANet

# 希望复现训练时的数据集划分

parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default=0.0001)
parser.add_argument('--epoch', type=int, default=500)
parser.add_argument('--saved-epoch', type=int, default=180)
parser.add_argument('--seed', type=int, default=11)
parser.add_argument('--batch', type=int, default=64)
parser.add_argument('--pos-weight', type=int, default=1)
parser.add_argument('--train-scale', type=float, default=0.7)
parser.add_argument('--model-name', type=str,
                    default='a_0.9575_11_1_0.0001_223_2023-07-24|15:19:10.pt')

args = parser.parse_args()
setup_seed(args.seed)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

prefix = '20230710'

signifiant_mask = torch.from_numpy(load_signifiant_mask('20230710')).to(device)

manet = MANet().to(device)
pos_weight = torch.tensor([args.pos_weight]).to(device)
criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction='none')
optimizer = torch.optim.Adam(manet.parameters(), lr=args.lr)

dataset = ActionDatasetV1(prefix)
train_size = int(len(dataset) * args.train_scale)
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(
    dataset, [train_size, test_size])

train_loader = DataLoader(
    train_dataset, batch_size=args.batch, shuffle=True, num_workers=0)
test_loader = DataLoader(
    test_dataset, batch_size=args.batch, shuffle=True, num_workers=0)
# all_loader = DataLoader(
#     dataset, batch_size=args.batch, shuffle=True, num_workers=0)

manet = MANet()
manet.load_state_dict(torch.load(
    os.path.join(model_folder_path, args.model_name)))
manet.to(device)
manet.eval()


@torch.no_grad()
def test(model: torch.nn.Module, action_label: torch.Tensor, chuanlian_feature: torch.Tensor, rongkang_feature: torch.Tensor, binya_feature: torch.Tensor, xiandian_feature: torch.Tensor, jiaoxian_feature: torch.Tensor, fuhe_feature: torch.Tensor, fadian_feature: torch.Tensor, muxian_feature: torch.Tensor, changzhan_feature: torch.Tensor):
    model.eval()
    output = torch.sigmoid(model(chuanlian_feature, rongkang_feature, binya_feature, xiandian_feature,
                           jiaoxian_feature, fuhe_feature, fadian_feature, muxian_feature, changzhan_feature))
    # print(output)
    print(F.binary_cross_entropy(output, action_label))


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
    
    feeds = {
        'row': [],
        'prediction': [],
        'action': [],
    }
    batch_count = 0

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
            mask = action_label.ge(0.5)
            prediction = torch.where(torch.sigmoid(result) >= 0.5, 1.0, 0.0)

            temp_signifiant_mask = signifiant_mask.expand(mask.shape[0], -1, -1)
            test_accuracy_numerator += int((prediction[temp_signifiant_mask] == action_label[temp_signifiant_mask]).sum())
            test_accuracy_denominator += action_label[temp_signifiant_mask].numel()
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

            celue_0_temp_signifiant_mask = celue_0_mask.logical_and(temp_signifiant_mask)
            celue_1_temp_signifiant_mask = celue_1_mask.logical_and(temp_signifiant_mask)
            celue_0_mask_recall = celue_0_mask.logical_and(mask)
            celue_1_mask_recall = celue_1_mask.logical_and(mask)

            test_celue_0_accuracy_numerator += int(
                (prediction[celue_0_temp_signifiant_mask] == action_label[celue_0_temp_signifiant_mask]).sum())
            test_celue_0_accuracy_denominator += action_label[celue_0_temp_signifiant_mask].numel(
            )
            test_celue_1_accuracy_numerator += int(
                (prediction[celue_1_temp_signifiant_mask] == action_label[celue_1_temp_signifiant_mask]).sum())
            test_celue_1_accuracy_denominator += action_label[celue_1_temp_signifiant_mask].numel(
            )

            test_celue_0_recall_numerator += int(
                (prediction[celue_0_mask_recall] == action_label[celue_0_mask_recall]).sum())
            test_celue_0_recall_denominator += int(celue_0_mask_recall.sum())
            test_celue_1_recall_numerator += int(
                (prediction[celue_1_mask_recall] == action_label[celue_1_mask_recall]).sum())
            test_celue_1_recall_denominator += int(celue_1_mask_recall.sum())
            
        with torch.no_grad():
            result = manet(chuanlian_feature, rongkang_feature, binya_feature, xiandian_feature,
                           jiaoxian_feature, fuhe_feature, fadian_feature, muxian_feature, changzhan_feature)
            mask = action_label.ge(0.5)
            prediction = torch.where(torch.sigmoid(result) >= 0.5, 1.0, 0.0)
            
            temp_signifiant_mask = signifiant_mask.expand(mask.shape[0], -1, -1)
            result_np = result.cpu().detach().numpy()
            row = feature['row'].detach().numpy().tolist()
            prediction_nonzero_indices = np.nonzero((prediction * temp_signifiant_mask).cpu().detach().numpy())
            prediction_nonzero_transposed = np.transpose(prediction_nonzero_indices)
            prediction_value = result_np[prediction_nonzero_indices].reshape(-1, 1)
            action_nonzero_transposed_list = np.transpose(np.nonzero(action_label.cpu().detach().numpy())).tolist()
            prediction_list = np.concatenate((prediction_nonzero_transposed, prediction_value), 1).tolist()
            
            for index in range(len(prediction_list)):
                prediction_list[index][0] += batch_count
            for index in range(len(action_nonzero_transposed_list)):
                action_nonzero_transposed_list[index][0] += batch_count
            feeds['row'] += row
            feeds['prediction'] += prediction_list
            feeds['action'] += action_nonzero_transposed_list
            batch_count += len(row)
            
            

    print('TAcc:\t{}\tTRec:\t{}'.format('%.6f' % (test_accuracy_numerator /
                                                  test_accuracy_denominator), '%.6f' % (test_recall_numerator / test_recall_denominator)))
    print('TAcc0:\t{}\tTRec0:\t{}'.format('%.6f' % (test_celue_0_accuracy_numerator /
                                                  test_celue_0_accuracy_denominator), '%.6f' % (test_celue_0_recall_numerator / test_celue_0_recall_denominator)))
    print('TAcc1:\t{}\tTRec1:\t{}'.format('%.6f' % (test_celue_1_accuracy_numerator /
                                                  test_celue_1_accuracy_denominator), '%.6f' % (test_celue_1_recall_numerator / test_celue_1_recall_denominator)))

    with open('feeds_test_v4.json', 'w') as outfile:
        json.dump(feeds, outfile, indent = 4) 
    return

    

# test(train_loader)
test(test_loader)
# test(all_loader)