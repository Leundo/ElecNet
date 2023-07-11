import os
from typing import List

from src.utils.porter import load_celue
from src.utils.constant import data_folder_path
from collections import Counter


text_folder_path = os.path.join(data_folder_path, 'text')

if __name__ == '__main__':
    prefix = '20230710'
    file_path = os.path.join(text_folder_path, '%s_celue.txt' % prefix)
    
    dic = {}
    number = 0
    with open(os.path.join(file_path), 'r') as fp:
        list = [line for line in fp]
    
    mapedList = []
    for line in list:
        if line not in dic.keys():
            dic[line] = number
            mapedList.append(number)
            number += 1
        else:
            mapedList.append(dic[line])
            
    
    counter = Counter(mapedList)
    print(len(dic))
    print(counter)