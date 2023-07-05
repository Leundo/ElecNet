import argparse

import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

from src.utils.action_dataset_v0 import ActionDatasetV0
from src.utils.equipment import Equipment
from src.utils.configuration import elec_mlp_config
from src.utils.knife import setup_seed
from src.models.MLP import MLP


parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--seed', type=int, default=0)

args = parser.parse_args()

setup_seed(args.seed)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)


chuanlian_embedding_net = MLP.new_for_embedding(Equipment.chuanlian).to(device)
rongkang_embedding_net = MLP.new_for_embedding(Equipment.rongkang).to(device)
binya_embedding_net = MLP.new_for_embedding(Equipment.bianya).to(device)
xiandian_embedding_net = MLP.new_for_embedding(Equipment.xiandian).to(device)
jiaoxian_embedding_net = MLP.new_for_embedding(Equipment.jiaoxian).to(device)
fuhe_embedding_net = MLP.new_for_embedding(Equipment.fuhe).to(device)
fadian_embedding_net = MLP.new_for_embedding(Equipment.fadian).to(device)
muxian_embedding_net = MLP.new_for_embedding(Equipment.muxian).to(device)
changzhan_embedding_net = MLP.new_for_embedding(Equipment.changzhan).to(device)

status_net = MLP.new_from_dict(elec_mlp_config['status']).to(device)

classification_net = MLP.new_from_dict(elec_mlp_config['classification']).to(device)

dataset = ActionDatasetV0('20230627')
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

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

    chuanlian_embedding = chuanlian_embedding_net(chuanlian_feature)
    rongkang_embedding = rongkang_embedding_net(rongkang_feature)
    binya_embedding = binya_embedding_net(binya_feature)
    xiandian_embedding = xiandian_embedding_net(xiandian_feature)
    jiaoxian_embedding = jiaoxian_embedding_net(jiaoxian_feature)
    fuhe_embedding = fuhe_embedding_net(fuhe_feature)
    fadian_embedding = fadian_embedding_net(fadian_feature)
    muxian_embedding = muxian_embedding_net(muxian_feature)
    changzhan_embedding = changzhan_embedding_net(changzhan_feature)

    concat_aggregation_embedding = torch.cat((
        torch.sum(chuanlian_embedding, dim=1),
        torch.sum(rongkang_embedding, dim=1),
        torch.sum(binya_embedding, dim=1),
        torch.sum(xiandian_embedding, dim=1),
        torch.sum(jiaoxian_embedding, dim=1),
        torch.sum(fuhe_embedding, dim=1),
        torch.sum(fadian_embedding, dim=1),
        torch.sum(muxian_embedding, dim=1),
        torch.sum(changzhan_embedding, dim=1)
    ), 1)

    concat_target_embedding = torch.cat((
        fadian_embedding, changzhan_embedding
    ), 1)

    status_and_embedding = torch.cat((
        status_net(concat_aggregation_embedding).repeat(3619, 1, 1).permute(1, 0, 2),
        concat_target_embedding,
    ), 2)
    
    result = classification_net(status_and_embedding)
    loss = F.binary_cross_entropy_with_logits(result, action_label)
    
    print(result.shape)
    print(concat_aggregation_embedding.shape)
