import random
from typing import List

import numpy as np
import torch


def fold(l: List, n: int) -> List:
    return [l[i:i+n] for i in range(0, len(l), n)]


def setup_seed(seed: int = 3407):
    random.seed(seed)  # Python的随机性
    # os.environ['PYTHONHASHSEED'] = str(seed)  # 设置Python哈希种子，为了禁止hash随机化，使得实验可复现
    np.random.seed(seed)  # numpy的随机性
    torch.manual_seed(seed)  # torch的CPU随机性，为CPU设置随机种子
    torch.cuda.manual_seed(seed)  # torch的GPU随机性，为当前GPU设置随机种子
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.   torch的GPU随机性，为所有GPU设置随机种子