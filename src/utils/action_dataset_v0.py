from typing import Dict, Tuple

from torch.utils.data import Dataset
import numpy as np

from src.utils.equipment import Equipment
from src.utils.porter import load_equipment, load_celue


class ActionDatasetV0(Dataset):
    def __init__(self, prefix: str):
        super().__init__()
        self.prefix = prefix
        self.inventory = {}
        for equipment in Equipment:
            self.inventory[equipment.value] = load_equipment(equipment, prefix)
        self.celue = load_celue(prefix)

    def __getitem__(self, index) -> Tuple[Dict, np.ndarray]:
        input = {}
        for equipment in Equipment:
            input[equipment.value] = self.inventory[equipment.value][index]
        return input, self.celue[index]

    def __len__(self) -> int:
        return self.celue.shape[0]


# 带有序列号
class ActionDatasetV1(Dataset):
    def __init__(self, prefix: str):
        super().__init__()
        self.prefix = prefix
        self.inventory = {}
        for equipment in Equipment:
            self.inventory[equipment.value] = load_equipment(equipment, prefix)
        self.celue = load_celue(prefix)
        self.row = np.arange(0, self.celue.shape[0])
        # print(self.celue.shape[0])


    def __getitem__(self, index) -> Tuple[Dict, np.ndarray]:
        input = {}
        for equipment in Equipment:
            input[equipment.value] = self.inventory[equipment.value][index]
        input['row'] = self.row[index]
        return input, self.celue[index]

    def __len__(self) -> int:
        return self.celue.shape[0]
