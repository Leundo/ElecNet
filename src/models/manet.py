from typing import List

import torch
from torch import nn

from src.models.MLP import MLP
from src.utils.equipment import Equipment
from src.utils.configuration import elec_mlp_config


class MANet(nn.Module):
    def __init__(self):
        super().__init__()
        self.chuanlian_embedding_net = MLP.new_for_embedding(
            Equipment.chuanlian)
        self.rongkang_embedding_net = MLP.new_for_embedding(Equipment.rongkang)
        self.binya_embedding_net = MLP.new_for_embedding(Equipment.bianya)
        self.xiandian_embedding_net = MLP.new_for_embedding(Equipment.xiandian)
        self.jiaoxian_embedding_net = MLP.new_for_embedding(Equipment.jiaoxian)
        self.fuhe_embedding_net = MLP.new_for_embedding(Equipment.fuhe)
        self.fadian_embedding_net = MLP.new_for_embedding(Equipment.fadian)
        self.muxian_embedding_net = MLP.new_for_embedding(Equipment.muxian)
        self.changzhan_embedding_net = MLP.new_for_embedding(
            Equipment.changzhan)

        self.status_net = MLP.new_from_dict(elec_mlp_config['status'])

        self.classification_net = MLP.new_from_dict(
            elec_mlp_config['classification'])

    def forward(self, chuanlian_feature: torch.Tensor, rongkang_feature: torch.Tensor, binya_feature: torch.Tensor, xiandian_feature: torch.Tensor, jiaoxian_feature: torch.Tensor, fuhe_feature: torch.Tensor, fadian_feature: torch.Tensor, muxian_feature: torch.Tensor, changzhan_feature: torch.Tensor) -> torch.Tensor:
        chuanlian_embedding = self.chuanlian_embedding_net(chuanlian_feature)
        rongkang_embedding = self.rongkang_embedding_net(rongkang_feature)
        binya_embedding = self.binya_embedding_net(binya_feature)
        xiandian_embedding = self.xiandian_embedding_net(xiandian_feature)
        jiaoxian_embedding = self.jiaoxian_embedding_net(jiaoxian_feature)
        fuhe_embedding = self.fuhe_embedding_net(fuhe_feature)
        fadian_embedding = self.fadian_embedding_net(fadian_feature)
        muxian_embedding = self.muxian_embedding_net(muxian_feature)
        changzhan_embedding = self.changzhan_embedding_net(changzhan_feature)
        
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
            self.status_net(concat_aggregation_embedding).repeat(3619, 1, 1).permute(1, 0, 2),
            concat_target_embedding,
        ), 2)
        
        return self.classification_net(status_and_embedding)
