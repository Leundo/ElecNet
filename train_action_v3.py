import argparse
import os
from decimal import Decimal

from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

from src.utils.action_dataset_v0 import ActionDatasetV0
from src.utils.equipment import Equipment
from src.utils.configuration import elec_mlp_config
from src.utils.knife import setup_seed, get_timestr
from src.utils.constant import model_folder_path
from src.models.manet import MANet


parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default=0.0001)
parser.add_argument('--epoch', type=int, default=500)
parser.add_argument('--saved-epoch', type=int, default=61)
parser.add_argument('--seed', type=int, default=11)
parser.add_argument('--batch', type=int, default=64)
parser.add_argument('--pos-weight', type=int, default=15)
parser.add_argument('--train-scale', type=float, default=0.7)

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

timestr = get_timestr()
best_recall = 0

@torch.no_grad()
def test(model: torch.nn.Module, action_label: torch.Tensor, chuanlian_feature: torch.Tensor, rongkang_feature: torch.Tensor, binya_feature: torch.Tensor, xiandian_feature: torch.Tensor, jiaoxian_feature: torch.Tensor, fuhe_feature: torch.Tensor, fadian_feature: torch.Tensor, muxian_feature: torch.Tensor, changzhan_feature: torch.Tensor):
    model.eval()
    output = torch.sigmoid(model(chuanlian_feature, rongkang_feature, binya_feature, xiandian_feature,
                           jiaoxian_feature, fuhe_feature, fadian_feature, muxian_feature, changzhan_feature))
    # print(output)
    print(F.binary_cross_entropy(output, action_label))


for epoch in range(0, args.epoch):

    train_accuracy_numerator = 0
    train_accuracy_denominator = 0
    train_recall_numerator = 0
    train_recall_denominator = 0
    train_loss_list = []

    manet.train()
    for feature, label in tqdm(train_loader):
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

        result = manet(chuanlian_feature, rongkang_feature, binya_feature, xiandian_feature,
                       jiaoxian_feature, fuhe_feature, fadian_feature, muxian_feature, changzhan_feature)
        loss = criterion(result, action_label)
        with torch.no_grad():
            x = torch.sigmoid(result)
            mask = action_label.ge(0.5)
            prediction = torch.where(torch.sigmoid(result) >= 0.5, 1.0, 0.0)

            train_accuracy_numerator += int((prediction == action_label).sum())
            train_accuracy_denominator += action_label.numel()
            train_recall_numerator += int(
                (prediction[mask] == action_label[mask]).sum())
            train_recall_denominator += int(mask.sum())

        loss.backward()
        optimizer.step()

        train_loss_list.append(loss.item())
        # print(loss.item())

    print('Epo:\t{}\tLos:\t{}\tAcc:\t{}\tRec:\t{}'.format(epoch, '%.8f' % (sum(train_loss_list) / len(train_loss_list)),
          '%.4f' % (train_accuracy_numerator / train_accuracy_denominator), '%.4f' % (train_recall_numerator / train_recall_denominator)))

    test_accuracy_numerator = 0
    test_accuracy_denominator = 0
    test_recall_numerator = 0
    test_recall_denominator = 0
    test_loss_list = []
    manet.eval()
    for feature, label in test_loader:
    # for feature, label in tqdm(test_loader):
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

    if (args.saved_epoch < epoch and best_recall < test_recall_numerator / test_recall_denominator):
        best_recall = test_recall_numerator / test_recall_denominator
        model_name = '{}_{}_{}_{}_{}_{}.pt'.format(Decimal(best_recall).quantize(Decimal('0.0000')), args.seed, args.pos_weight, args.lr, epoch + 1, timestr)
        torch.save(manet.state_dict(), os.path.join(model_folder_path, model_name))
        print('Model saved in {}'.format(model_name))
    