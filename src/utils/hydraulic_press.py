import os
from typing import List

import numpy as np
import matplotlib.pyplot as plt
from sklearn import manifold, datasets
from scipy.stats import gaussian_kde
from src.utils.constant import image_folder_path

from src.utils.equipment import Equipment
from src.utils.porter import load_equipment, load_celue

class HydraulicPress:
    def __init__(self, taipu: str, prefix: str):
        self.random_state = 0
        self.taipu = taipu
        if taipu == 'celue':
            self.data = load_celue(prefix)
            self.data = self.data.reshape([self.data.shape[0], -1])
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
    def get_k(matrix: List[float], bw: float = 0.1):
        tmp = np.array(matrix).reshape(-1, 3).T
        k = gaussian_kde(tmp, bw_method=bw)(tmp)
        return k 
    
    def pca_2d(self):
        tsne = manifold.TSNE(n_components=2, init='pca', random_state=self.random_state)
        tsned_data = tsne.fit_transform(self.data)
        print('Org data dimension is {}.Embedded data dimension is {}'.format(self.data.shape[-1], tsned_data.shape[-1]))
        
        tsned_min, tsned_max = tsned_data.min(0), tsned_data.max(0)
        tsned_norm = (tsned_data - tsned_min) / (tsned_max - tsned_min)
                   
        plt.scatter(tsned_norm[:,0], tsned_norm[:,1])
        plt.xticks([])
        plt.yticks([])
        plt.savefig(os.path.join(image_folder_path, 'pca_2d_{}.png'.format(self.taipu)))
        
    # https://stackoverflow.com/questions/53826201/how-to-represent-density-information-on-a-matplotlib-3-d-scatter-plot
    def pca_3d(self):
        tsne = manifold.TSNE(n_components=3, init='pca', random_state=self.random_state)
        tsned_data = tsne.fit_transform(self.data)
        print('Org data dimension is {}.Embedded data dimension is {}'.format(self.data.shape[-1], tsned_data.shape[-1]))
        
        tsned_min, tsned_max = tsned_data.min(0), tsned_data.max(0)
        tsned_norm = (tsned_data - tsned_min) / (tsned_max - tsned_min)
        
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        k = HydraulicPress.get_k(tsned_norm)
        ax.scatter(tsned_norm[:,0], tsned_norm[:,1], tsned_norm[:,2], c=k, alpha=0.2)
        plt.savefig(os.path.join(image_folder_path, 'pca_3d_{}.png'.format(self.taipu)))
        
        # fig = plt.figure()
        # ax = fig.add_subplot(projection='3d')
        # for item in tsned_norm:
        #     ax.scatter(item[0], item[1], item[2], marker='o', c='limegreen')
        # plt.savefig(os.path.join(image_folder_path, 'pca_3d_{}.png'.format(self.taipu)))
        
        
        
        
    
if __name__ == '__main__':
    # press = HydraulicPress('celue', '20230704')
    press = HydraulicPress('input', '20230704')

    # press.pca_2d()
    press.pca_3d()