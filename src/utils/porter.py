import os

import torch
import numpy as np

from src.utils.constant import data_folder_path
from src.utils.knife import fold
from src.utils.configuration import elec_mlp_config
from src.utils.equipment import Equipment


text_folder_path = os.path.join(data_folder_path, 'text')
np_folder_path = os.path.join(data_folder_path, 'np')


def save_equipment_text_to_np(equipment: Equipment, prefix: str) -> np.ndarray:
    file_path = os.path.join(
        text_folder_path, '{}_{}.txt'.format(prefix, equipment.value))
    with open(os.path.join(file_path), 'r') as fp:
        # list = [line.rstrip().split(' ') for line in fp]
        list = [[float(x) for x in line.rstrip().split(' ')] for line in fp]
    shape_dict = elec_mlp_config['embedding'][equipment.value]
    feature = np.asarray(list).astype('float32').reshape(
        (-1, shape_dict['channel_count'], shape_dict['input_count'] + 1))[:, :, 1:]
    np.save(os.path.join(np_folder_path, '{}_{}.npy'.format(
        prefix, equipment.value)), feature)
    return feature


def save_celue_text_to_np(prefix: str) -> np.ndarray:
    taipu_7_count = 1935
    taipu_9_count = 1684
    file_path = os.path.join(text_folder_path, '%s_celue.txt' % prefix)
    with open(os.path.join(file_path), 'r') as fp:
        list = [line.rstrip().split(' ') for line in fp]
    new_list = []
    for line_list in list:
        new_list.append([[0, 0] for _ in range(taipu_7_count + taipu_9_count)])
        if line_list == ['']:
            continue
        for item in fold(line_list, 3):
            if item[1] == '7':
                index = int(item[2])
            elif item[1] == '9':
                index = int(item[2]) + taipu_7_count
            new_list[len(new_list) - 1][index][int(item[0])] = 1
    label = np.asarray(new_list).astype('float32')
    np.save(os.path.join(np_folder_path, '%s_celue.npy' % prefix), label)
    return label


def save_guzhang_text_to_np(prefix: str) -> np.ndarray:
    file_path = os.path.join(text_folder_path, '%s_guzhang.txt' % prefix)
    with open(os.path.join(file_path), 'r') as fp:
        list = [[float(x) for x in line.rstrip().split(' ')] for line in fp]
    output_count = elec_mlp_config['guzhang']['output_count']
    feature = np.asarray(list).astype('float32').reshape(
        (-1, output_count))
    np.save(os.path.join(np_folder_path, '{}_guzhang.npy'.format(
        prefix)), feature)
    return feature

def load_equipment(equipment: Equipment, prefix: str) -> np.ndarray:
    file_path = os.path.join(
        np_folder_path, '{}_{}.npy'.format(prefix, equipment.value))
    return np.load(file_path)


def load_celue(prefix: str) -> np.ndarray:
    file_path = os.path.join(np_folder_path, '%s_celue.npy' % prefix)
    return np.load(file_path)

def load_guzhang(prefix: str) -> np.ndarray:
    file_path = os.path.join(np_folder_path, '%s_guzhang.npy' % prefix)
    return np.load(file_path)