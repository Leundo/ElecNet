import torch
from torch import nn
from torch.nn import functional as F
from typing import Dict, List, Optional, Type

from src.utils.equipment import Equipment
from src.utils.configuration import elec_mlp_config


def create_miniblock(
    input_count: int,
    output_count: int = 0,
    channel_count: Optional[int] = None,
    should_normalize: bool = True
) -> List[nn.Module]:
    if channel_count is None:
        channel_count = output_count
    layers: List[nn.Module] = [nn.Linear(input_count, output_count)]
    if should_normalize is not None:
        layers += [nn.BatchNorm1d(channel_count)]
    layers += [nn.ReLU()]
    return layers


class MLP(nn.Module):
    def __init__(self, input_count: int, hidden_counts: List[int], output_count: int, channel_count: Optional[int] = None, should_normalize: bool = True, should_flatten: bool = False):
        super().__init__()

        models = []
        if should_flatten:
            models += [nn.Flatten()]
        counts = [input_count] + hidden_counts
        for mini_input_count, mini_output_count in zip(counts[:-1], counts[1:]):
            models += create_miniblock(mini_input_count, mini_output_count,
                                       channel_count=channel_count, should_normalize=should_normalize)
        models += [nn.Linear(hidden_counts[-1], output_count)]
        self.layers = nn.Sequential(*models)

    def forward(self, x):
        return self.layers(x)

    @staticmethod
    def new_for_embedding(equipment: Equipment, should_normalize: bool = True) -> 'MLP':
        embedding_dict = elec_mlp_config['embedding']
        return MLP(embedding_dict[equipment.value]['input_count'], embedding_dict['hidden_counts'], embedding_dict['output_count'], channel_count=embedding_dict[equipment.value]['channel_count'], should_normalize=embedding_dict['should_normalize'])

    @staticmethod
    def new_from_dict(dict: Dict) -> 'MLP':
        return MLP(dict['input_count'], dict['hidden_counts'], dict['output_count'], channel_count=dict['channel_count'], should_normalize=dict['should_normalize'])