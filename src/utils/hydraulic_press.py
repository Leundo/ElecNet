import os
from typing import List

import numpy as np
import matplotlib.pyplot as plt
from sklearn import manifold, datasets
from scipy.stats import gaussian_kde
import matplotlib.cm as cm

from src.utils.constant import image_folder_path, data_folder_path
from src.utils.equipment import Equipment
from src.utils.porter import load_equipment, load_celue, load_guzhang

class HydraulicPress:
    def __init__(self, taipu: str, prefix: str):
        self.random_state = 0
        self.taipu = taipu
        self.prefix = prefix
        if taipu == 'celue':
            self.data = load_celue(prefix)
            self.data = self.data.reshape([self.data.shape[0], -1])
        elif taipu == 'guzhang':
            self.data = load_guzhang(prefix)
        elif taipu == 'status':
            self.data = np.load(os.path.join(data_folder_path, 'np', '%s_status.npy' % prefix))
            self.label = np.load(os.path.join(data_folder_path, 'np', '%s_celue_taipu.npy' % prefix))
        elif taipu == 'celue_for_0':
            self.data = np.load(os.path.join(data_folder_path, 'np', '%s_celue_for_0.npy' % prefix))
        elif taipu == 'input':
            chuanlian = load_equipment(Equipment.chuanlian, prefix)
            count = chuanlian.shape[0]
            self.data = np.concatenate(
                (
                    chuanlian.reshape([count, -1]),
                    load_equipment(Equipment.rongkang, prefix).reshape([count, -1]),
                    load_equipment(Equipment.bianya, prefix).reshape([count, -1]),
                    load_equipment(Equipment.xiandian, prefix).reshape([count, -1]),
                    load_equipment(Equipment.jiaoxian, prefix).reshape([count, -1]),
                    load_equipment(Equipment.fuhe, prefix).reshape([count, -1]),
                    load_equipment(Equipment.fadian, prefix).reshape([count, -1]),
                    load_equipment(Equipment.muxian, prefix).reshape([count, -1]),
                    load_equipment(Equipment.changzhan, prefix).reshape([count, -1]),
                ),
                axis=1
            )
            
    @staticmethod
    def get_k(matrix: List[float], dim: int = 3, bw: float = 0.1):
        tmp = np.array(matrix).reshape(-1, dim).T
        k = gaussian_kde(tmp, bw_method=bw)(tmp)
        return k 
    
    def pca_2d(self, alpha: float = 0.01, shouldK: bool = True):
        tsne = manifold.TSNE(n_components=2, init='pca', random_state=self.random_state)
        tsned_data = tsne.fit_transform(self.data)
        print('Org data dimension is {}.Embedded data dimension is {}'.format(self.data.shape[-1], tsned_data.shape[-1]))
        
        tsned_min, tsned_max = tsned_data.min(0), tsned_data.max(0)
        tsned_norm = (tsned_data - tsned_min) / (tsned_max - tsned_min)
        
        if shouldK:
            k = HydraulicPress.get_k(tsned_norm, dim=2)
        else:
            k = '#1f77b4'
        plt.scatter(tsned_norm[:,0], tsned_norm[:,1], c=k, alpha=alpha)
        plt.xticks([])
        plt.yticks([])
        plt.savefig(os.path.join(image_folder_path, 'pca_2d_{}_{}.png'.format(self.taipu, self.prefix)))
        
    # https://stackoverflow.com/questions/53826201/how-to-represent-density-information-on-a-matplotlib-3-d-scatter-plot
    def pca_3d(self, alpha: float = 0.1, shouldK: bool = True):
        tsne = manifold.TSNE(n_components=3, init='pca', random_state=self.random_state)
        tsned_data = tsne.fit_transform(self.data)
        print('Org data dimension is {}.Embedded data dimension is {}'.format(self.data.shape[-1], tsned_data.shape[-1]))
        
        tsned_min, tsned_max = tsned_data.min(0), tsned_data.max(0)
        tsned_norm = (tsned_data - tsned_min) / (tsned_max - tsned_min)
        
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        if shouldK:
            k = HydraulicPress.get_k(tsned_norm)
        else:
            k = '#1f77b4'
        ax.scatter(tsned_norm[:,0], tsned_norm[:,1], tsned_norm[:,2], c=k, alpha=0.1)
        plt.savefig(os.path.join(image_folder_path, 'pca_3d_{}_{}.png'.format(self.taipu, self.prefix)))
        
        # fig = plt.figure()
        # ax = fig.add_subplot(projection='3d')
        # for item in tsned_norm:
        #     ax.scatter(item[0], item[1], item[2], marker='o', c='limegreen')
        # plt.savefig(os.path.join(image_folder_path, 'pca_3d_{}.png'.format(self.taipu)))
        
    def pca_2d_with_label(self, alpha: float = 0.01):
        tsne = manifold.TSNE(n_components=2, init='pca', random_state=self.random_state)
        tsned_data = tsne.fit_transform(self.data)
        print('Org data dimension is {}.Embedded data dimension is {}'.format(self.data.shape[-1], tsned_data.shape[-1]))
        
        tsned_min, tsned_max = tsned_data.min(0), tsned_data.max(0)
        tsned_norm = (tsned_data - tsned_min) / (tsned_max - tsned_min)

        unique_label = np.unique(self.label)
        colors = cm.rainbow(np.linspace(0, 1, len(unique_label)))
        
        for index in range(len(self.label)):
            plt.scatter(tsned_norm[index,0], tsned_norm[index,1], c=colors[self.label[index]].reshape(1,-1), alpha=alpha)
                    
        plt.xticks([])
        plt.yticks([])
        plt.savefig(os.path.join(image_folder_path, 'pca_2d_label_{}_{}.png'.format(self.taipu, self.prefix)))
        
    def pca_3d_with_label(self, alpha: float = 0.1):
        tsne = manifold.TSNE(n_components=3, init='pca', random_state=self.random_state)
        tsned_data = tsne.fit_transform(self.data)
        print('Org data dimension is {}.Embedded data dimension is {}'.format(self.data.shape[-1], tsned_data.shape[-1]))
        
        tsned_min, tsned_max = tsned_data.min(0), tsned_data.max(0)
        tsned_norm = (tsned_data - tsned_min) / (tsned_max - tsned_min)
        
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        
        unique_label = np.unique(self.label)
        colors = cm.rainbow(np.linspace(0, 1, len(unique_label)))
        
        for index in range(len(self.label)):
            ax.scatter(tsned_norm[index,0], tsned_norm[index,1], tsned_norm[index,2], c=colors[self.label[index]].reshape(1,-1), alpha=0.1)
             
        
        plt.savefig(os.path.join(image_folder_path, 'pca_3d_label_{}_{}.png'.format(self.taipu, self.prefix)))
        
        # fig = plt.figure()
        # ax = fig.add_subplot(projection='3d')
        # for item in tsned_norm:
        #     ax.scatter(item[0], item[1], item[2], marker='o', c='limegreen')
        # plt.savefig(os.path.join(image_folder_path, 'pca_3d_{}.png'.format(self.taipu)))
        
        
    
if __name__ == '__main__':
    # press = HydraulicPress('celue', '20230704')
    # press = HydraulicPress('input', '20230704')
    # press = HydraulicPress('celue', '20230710')
    # press = HydraulicPress('input', '20230710')
    # press = HydraulicPress('guzhang', '20230710')
    
    press = HydraulicPress('status', '20230710')
    # press = HydraulicPress('celue_for_0', '20230710')

    # press.pca_2d(alpha=0.1, shouldK=False)
    # press.pca_3d(alpha=0.1, shouldK=False)
    
    # press.pca_2d_with_label(alpha=1)
    press.pca_3d_with_label(alpha=1)