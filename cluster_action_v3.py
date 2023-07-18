import argparse
import os
from decimal import Decimal
from typing import Tuple
from collections import Counter

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


data_prefix = '20230710'
celue = np.load(os.path.join(data_folder_path, 'np', '%s_celue_for_0.npy' % data_prefix))


celue_dict = {}
taipu_allocator = 0
celue_taipu = []
celue_str = []

for index in range(celue.shape[0]):
    key = ' '.join([str(item) for item in celue[index].tolist()])
    celue_str.append(key)
    taipu = celue_dict.get(key)
    if taipu is None:
        celue_dict[key] = taipu_allocator
        celue_taipu.append(taipu_allocator)
        taipu_allocator += 1
    else:
        celue_taipu.append(taipu)

# celue_taipu = np.asarray(celue_taipu)
# np.save(os.path.join(data_folder_path, 'np', '%s_celue_taipu.npy' % data_prefix), celue_taipu)


cluster_count = 40
counter = Counter(celue_str)
most_common_keys = [item[0] for item in counter.most_common()[0:cluster_count-1]]
celue_taipu = []

for index in range(celue.shape[0]):
    key = ' '.join([str(item) for item in celue[index].tolist()])
    if key in most_common_keys:
        taipu = most_common_keys.index(key)
        celue_taipu.append(taipu)
    else:
        celue_taipu.append(cluster_count-1)
        

celue_taipu = np.asarray(celue_taipu)
np.save(os.path.join(data_folder_path, 'np', '%s_celue_taipu.npy' % data_prefix), celue_taipu)