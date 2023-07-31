import os
from typing import List
import json

from src.utils.porter import load_celue
from src.utils.constant import data_folder_path
from collections import Counter


text_folder_path = os.path.join(data_folder_path, 'text')

def count_with_number():
    prefix = '20230710'
    file_path = os.path.join(text_folder_path, '%s_celue.txt' % prefix)
    
    dic = {}
    number = 0
    with open(os.path.join(file_path), 'r') as fp:
        list = [line.strip() for line in fp]
    
    mapedList = []
    for line in list:
        if line not in dic.keys():
            dic[line] = number
            mapedList.append(number)
            number += 1
        else:
            mapedList.append(dic[line])
            
    
    counter = Counter(mapedList)
    print(dic)
    print(counter)


def count_and_save():
    prefix = '20230710'
    file_path = os.path.join(text_folder_path, '%s_celue.txt' % prefix)
    
    dic = {}
    number = 0
    with open(os.path.join(file_path), 'r') as fp:
        list = [line.strip() for line in fp]
    
    mapedList = []
    for line in list:
        if line not in dic.keys():
            dic[line] = number
            mapedList.append(number)
            number += 1
        else:
            mapedList.append(dic[line])
            
    counter = Counter(mapedList)
    
    
    results = []
    for key in dic.keys():
        fingerprint = []
        tmp = [x for x in key.split(' ')]
        tmp = tmp if len(tmp) > 1 else []
        if len(tmp) > 0:
            assert(len(tmp) % 3 == 0)
            for iter in range(int(len(tmp) / 3)):
                fingerprint.append([int(tmp[iter*3]), int(tmp[iter*3+1]), int(tmp[iter*3+2])])
        results.append({
            'count': counter[dic[key]],
            'fingerprint': fingerprint,
        })
    # for result in results:
        
    # print(results)
    with open('counter_v4.json', 'w') as outfile:
        json.dump(results, outfile, indent = 4) 
    
if __name__ == '__main__':
    count_and_save()