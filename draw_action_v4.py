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
from src.utils.porter import load_signifiant_mask
from src.models.manet import MANet


parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=11)
parser.add_argument('--batch', type=int, default=64)
parser.add_argument('--model-name', type=str,
                    default='a_0.9575_11_1_0.0001_223_2023-07-24|15:19:10.pt')
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

data_prefix = '20230710'
dataset = ActionDatasetV0(data_prefix)
train_size = int(len(dataset) * args.train_scale)
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(
    dataset, [train_size, test_size])

signifiant_mask = torch.from_numpy(load_signifiant_mask('20230710')).to(device)

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

    status_embedding = None
    celue_for_0_label = None

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
        
        if celue_for_0_label is None:
            celue_for_0_label = action_label[:, :, 0]
        else:
            celue_for_0_label = torch.cat((
                celue_for_0_label,
                action_label[:, :, 0],
            ), 0)

        with torch.no_grad():
            result = manet(chuanlian_feature, rongkang_feature, binya_feature, xiandian_feature,
                           jiaoxian_feature, fuhe_feature, fadian_feature, muxian_feature, changzhan_feature)
            x = torch.sigmoid(result)
            mask = action_label.ge(0.5)
            prediction = torch.where(torch.sigmoid(result) >= 0.5, 1.0, 0.0)

            temp_signifiant_mask = signifiant_mask.expand(mask.shape[0], -1, -1)
            test_accuracy_numerator += int((prediction[temp_signifiant_mask] == action_label[temp_signifiant_mask]).sum())
            test_accuracy_denominator += action_label[temp_signifiant_mask].numel()
            test_recall_numerator += int(
                (prediction[mask] == action_label[mask]).sum())
            test_recall_denominator += int(mask.sum())

        with torch.no_grad():
            current_status_embedding = manet.forward_to_get_status_embedding(chuanlian_feature, rongkang_feature, binya_feature, xiandian_feature,
                                                                     jiaoxian_feature, fuhe_feature, fadian_feature, muxian_feature, changzhan_feature)
            if status_embedding is None:
                status_embedding = current_status_embedding
            else:
                status_embedding = torch.cat((
                    status_embedding,
                    current_status_embedding,
                ), 0)
                


    print('TAcc:\t{}\tTRec:\t{}'.format('%.6f' % (test_accuracy_numerator /
                                                  test_accuracy_denominator), '%.6f' % (test_recall_numerator / test_recall_denominator)))
    
    return status_embedding, celue_for_0_label

def get_bound_input(loader: DataLoader) -> torch.Tensor:
    bound_input = None
    for feature, label in tqdm(loader):
        chuanlian_feature = feature[Equipment.chuanlian.value]
        rongkang_feature = feature[Equipment.rongkang.value]
        binya_feature = feature[Equipment.bianya.value]
        xiandian_feature = feature[Equipment.xiandian.value]
        jiaoxian_feature = feature[Equipment.jiaoxian.value]
        fuhe_feature = feature[Equipment.fuhe.value]
        fadian_feature = feature[Equipment.fadian.value]
        muxian_feature = feature[Equipment.muxian.value]
        changzhan_feature = feature[Equipment.changzhan.value]
        
        count = chuanlian_feature.shape[0]
        current_input = torch.cat((
            chuanlian_feature.reshape([count, -1]),
            rongkang_feature.reshape([count, -1]),
            binya_feature.reshape([count, -1]),
            xiandian_feature.reshape([count, -1]),
            jiaoxian_feature.reshape([count, -1]),
            fuhe_feature.reshape([count, -1]),
            fadian_feature.reshape([count, -1]),
            muxian_feature.reshape([count, -1]),
            changzhan_feature.reshape([count, -1]),
        ), 1)
        
        if bound_input is None:
            bound_input = current_input
        else:
            bound_input = torch.cat((
                bound_input,
                current_input,
            ), 0)
    return bound_input


status_embedding, celue_for_0_label = test(all_loader)
status_embedding = status_embedding.to(torch.device('cpu'))
celue_for_0_label = celue_for_0_label.to(torch.device('cpu'))

np.save(os.path.join(data_folder_path, 'np', '%s_status.npy' % data_prefix), status_embedding.numpy())
np.save(os.path.join(data_folder_path, 'np', '%s_celue_for_0.npy' % data_prefix), celue_for_0_label.numpy())


# bound_input = get_bound_input(all_loader)
# np.save(os.path.join(data_folder_path, 'np', '%s_bound_input.npy' % data_prefix), bound_input.numpy())