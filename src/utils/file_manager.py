import os

import torch
import numpy as np

from src.utils.constant import data_folder_path
from src.utils.knife import fold


def create_one_hot(batch: int, indices: np.ndarray) -> torch.Tensor:
    one_hot = np.zeros((indices.size, indices.max() + 1))
    one_hot[np.arange(indices.size), indices] = 1
    return np.tile(one_hot, (batch, 1, 1))


def load_chuanlian_text(prefix: str) -> np.ndarray:
    file_path = os.path.join(data_folder_path, 'text', '%s_Chuanlian.txt' % prefix)
    with open(os.path.join(file_path), 'r') as fp:
        list = [line.rstrip().split(' ') for line in fp]
    feature = np.asarray(list).astype('float32').reshape((-1, 11, 7))[:, :, 1:]
    np.save(os.path.join(data_folder_path, 'np', '%s_Chuanlian.npy' % prefix), feature)
    return feature


def load_rongkang_text(prefix: str) -> np.ndarray:
    file_path = os.path.join(data_folder_path, 'text', '%s_Rongkang.txt' % prefix)
    with open(os.path.join(file_path), 'r') as fp:
        list = [line.rstrip().split(' ') for line in fp]
    feature = np.asarray(list).astype(
        'float32').reshape((-1, 3257, 7))[:, :, 1:]
    np.save(os.path.join(data_folder_path, 'np', '%s_Rongkang.npy' % prefix), feature)
    return feature


def load_bianya_text(prefix: str) -> np.ndarray:
    file_path = os.path.join(data_folder_path, 'text', '%s_Bianya.txt' % prefix)
    with open(os.path.join(file_path), 'r') as fp:
        list = [line.rstrip().split(' ') for line in fp]
    feature = np.asarray(list).astype(
        'float32').reshape((-1, 3279, 3))[:, :, 1:]
    np.save(os.path.join(data_folder_path, 'np', '%s_Bianya.npy' % prefix), feature)
    return feature


def load_xiandian_text(prefix: str) -> np.ndarray:
    file_path = os.path.join(data_folder_path, 'text', '%s_Xiandian.txt' % prefix)
    with open(os.path.join(file_path), 'r') as fp:
        list = [line.rstrip().split(' ') for line in fp]
    feature = np.asarray(list).astype(
        'float32').reshape((-1, 7661, 9))[:, :, 1:]
    np.save(os.path.join(data_folder_path, 'np', '%s_Xiandian.npy' % prefix), feature)
    return feature


def load_jiaoxian_text(prefix: str) -> np.ndarray:
    file_path = os.path.join(data_folder_path, 'text', '%s_Jiaoxian.txt' % prefix)
    with open(os.path.join(file_path), 'r') as fp:
        list = [line.rstrip().split(' ') for line in fp]
    feature = np.asarray(list).astype(
        'float32').reshape((-1, 3830, 3))[:, :, 1:]
    np.save(os.path.join(data_folder_path, 'np', '%s_Jiaoxian.npy' % prefix), feature)
    return feature


def load_fuhe_text(prefix: str) -> np.ndarray:
    file_path = os.path.join(data_folder_path, 'text', '%s_Fuhe.txt' % prefix)
    with open(os.path.join(file_path), 'r') as fp:
        list = [line.rstrip().split(' ') for line in fp]
    feature = np.asarray(list).astype(
        'float32').reshape((-1, 6044, 6))[:, :, 1:]
    np.save(os.path.join(data_folder_path, 'np', '%s_Fuhe.npy' % prefix), feature)
    return feature


def load_fadian_text(prefix: str) -> np.ndarray:
    file_path = os.path.join(data_folder_path, 'text', '%s_Fadian.txt' % prefix)
    with open(os.path.join(file_path), 'r') as fp:
        list = [line.rstrip().split(' ') for line in fp]
    feature = np.asarray(list).astype(
        'float32').reshape((-1, 1935, 9))[:, :, 1:]
    np.save(os.path.join(data_folder_path, 'np', '%s_Fadian.npy' % prefix), feature)
    return feature


def load_muxian_text(prefix: str) -> np.ndarray:
    file_path = os.path.join(data_folder_path, 'text', '%s_Muxian.txt' % prefix)
    with open(os.path.join(file_path), 'r') as fp:
        list = [line.rstrip().split(' ') for line in fp]
    feature = np.asarray(list).astype(
        'float32').reshape((-1, 5870, 7))[:, :, 1:]
    np.save(os.path.join(data_folder_path, 'np', '%s_Muxian.npy' % prefix), feature)
    return feature


def load_changzhan_text(prefix: str) -> np.ndarray:
    file_path = os.path.join(data_folder_path, 'text', '%s_Changzhan.txt' % prefix)
    with open(os.path.join(file_path), 'r') as fp:
        list = [line.rstrip().split(' ') for line in fp]
    feature = np.asarray(list).astype(
        'float32').reshape((-1, 1684, 2))[:, :, 1:]
    np.save(os.path.join(data_folder_path, 'np', '%s_Changzhan.npy' % prefix), feature)
    return feature


def load_celue_text(prefix: str) -> np.ndarray:
    taipu_7_count = 1935
    taipu_9_count = 1684
    file_path = os.path.join(data_folder_path, 'text', '%s_Celue.txt' % prefix)
    with open(os.path.join(file_path), 'r') as fp:
        list = [line.rstrip().split(' ') for line in fp]
    new_list = []
    for line_list in list:
        new_list.append([[0, 0] for _ in range(taipu_7_count + taipu_9_count)])
        for item in fold(line_list, 3):
            if item[1] == '7':
                index = int(item[2])
            elif item[1] == '9':
                index = int(item[2]) + taipu_7_count
            new_list[len(new_list) - 1][index][int(item[0])] = 1
    label = np.asarray(new_list).astype('float32')
    np.save(os.path.join(data_folder_path, 'np', '%s_Celue.npy' % prefix), label)
    return label

def load_chuanlian_np(prefix: str) -> np.ndarray:
    file_path = os.path.join(data_folder_path, 'np', '%s_Chuanlian.npy' % prefix)
    return np.load(file_path)


def load_rongkang_np(prefix: str) -> np.ndarray:
    file_path = os.path.join(data_folder_path, 'np', '%s_Rongkang.npy' % prefix)
    return np.load(file_path)


def load_bianya_np(prefix: str) -> np.ndarray:
    file_path = os.path.join(data_folder_path, 'np', '%s_Bianya.npy' % prefix)
    return np.load(file_path)


def load_xiandian_np(prefix: str) -> np.ndarray:
    file_path = os.path.join(data_folder_path, 'np', '%s_Xiandian.npy' % prefix)
    return np.load(file_path)


def load_jiaoxian_np(prefix: str) -> np.ndarray:
    file_path = os.path.join(data_folder_path, 'np', '%s_Jiaoxian.npy' % prefix)
    return np.load(file_path)


def load_fuhe_np(prefix: str) -> np.ndarray:
    file_path = os.path.join(data_folder_path, 'np', '%s_Fuhe.npy' % prefix)
    return np.load(file_path)


def load_fadian_np(prefix: str) -> np.ndarray:
    file_path = os.path.join(data_folder_path, 'np', '%s_Fadian.npy' % prefix)
    return np.load(file_path)


def load_muxian_np(prefix: str) -> np.ndarray:
    file_path = os.path.join(data_folder_path, 'np', '%s_Muxian.npy' % prefix)
    return np.load(file_path)


def load_changzhan_np(prefix: str) -> np.ndarray:
    file_path = os.path.join(data_folder_path, 'np', '%s_Changzhan.npy' % prefix)
    return np.load(file_path)


def load_celue_np(prefix: str) -> np.ndarray:
    file_path = os.path.join(data_folder_path, 'np', '%s_Celue.npy' % prefix)
    return np.load(file_path)