import argparse

import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

from src.utils.action_dataset_v0 import ActionDatasetV0
from src.utils.equipment import Equipment
from src.utils.configuration import elec_mlp_config
from src.utils.knife import setup_seed
from src.models.manet import MANet


parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--epoch', type=int, default=100)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--batch', type=int, default=32)
parser.add_argument('--pos-weight', type=int, default=300)

args = parser.parse_args()
setup_seed(args.seed)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)


manet = MANet().to(device)
pos_weight = torch.tensor([args.pos_weight]).to(device)
criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction='mean')
optimizer = torch.optim.Adam(manet.parameters(), lr=args.lr)
dataset = ActionDatasetV0('20230704')
dataloader = DataLoader(dataset, batch_size=args.batch, shuffle=True)


@torch.no_grad()
def test(model: torch.nn.Module, action_label: torch.Tensor, chuanlian_feature: torch.Tensor, rongkang_feature: torch.Tensor, binya_feature: torch.Tensor, xiandian_feature: torch.Tensor, jiaoxian_feature: torch.Tensor, fuhe_feature: torch.Tensor, fadian_feature: torch.Tensor, muxian_feature: torch.Tensor, changzhan_feature: torch.Tensor):
    model.eval()
    output = F.sigmoid(model(chuanlian_feature, rongkang_feature, binya_feature, xiandian_feature, jiaoxian_feature, fuhe_feature, fadian_feature, muxian_feature, changzhan_feature))
    # print(output)
    print(F.binary_cross_entropy(output, action_label))
    

for epoch in range(0, args.epoch):
    for feature, label in dataloader:
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
        
        result = manet(chuanlian_feature, rongkang_feature, binya_feature, xiandian_feature, jiaoxian_feature, fuhe_feature, fadian_feature, muxian_feature, changzhan_feature)
        loss = criterion(result, action_label)
        with torch.no_grad():
            x = F.sigmoid(result)
            mask = action_label.ge(0.5)
            prediction = torch.where(F.sigmoid(result) >= 0.5, 1.0, 0.0)
            accuracy = int((prediction == action_label).sum()) / action_label.numel()
            recall = int((prediction[mask] == action_label[mask]).sum()) / int(mask.sum())
            print('Acc:\t{}\nRec:\t{}'.format(accuracy, recall))
        loss.backward()
        optimizer.step()
        
        print(loss.item())