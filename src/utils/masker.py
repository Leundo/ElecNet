import os
from typing import List

import numpy as np

from src.utils.porter import load_celue
from src.utils.constant import data_folder_path
from src.utils.porter import load_equipment, load_celue
from collections import Counter


text_folder_path = os.path.join(data_folder_path, 'text')
np_folder_path = os.path.join(data_folder_path, 'np')

if __name__ == '__main__':
    prefix = '20230710'
    celue = load_celue(prefix)
    print(celue.shape[0])
    signifiant_mask = np.zeros((celue.shape[1], celue.shape[2]), dtype=np.bool)

    for index in range(celue.shape[0]):
        new_mask = celue[index] > 0
        signifiant_mask = np.logical_or(signifiant_mask, new_mask)

    
    print(signifiant_mask.sum())
    print(np.nonzero(signifiant_mask))
    np.save(os.path.join(np_folder_path, '%s_signifiant_mask.npy' % prefix), signifiant_mask)
