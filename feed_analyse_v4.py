import argparse
import os
from decimal import Decimal
import json
import math
from typing import Tuple, List

import numpy as np

from src.utils.porter import load_equipment, load_celue, load_signifiant_mask, load_equipment_map

signifiant_mask = load_signifiant_mask('20230710')
equipment_map = load_equipment_map()

def remain_guzhang_0(value):
    return value[1] < 0.5

def remain_guzhang_1(value):
    return value[1] > 0.5

def convert_vector_to_equipment(vector: List[List[int]]) -> dict:
    results = []
    vector = [[int(item[0]), int(item[1])] for item in vector]
    for item in vector:
        for key, value in equipment_map.items():
            if item[0] < 1935 and value['taipu'] == 7 and item[0] == value['number']:
                result = {}
                result['id'] = key
                result['payload'] = value
                result['guzhang'] = item[1]
                results.append(result)
                break
            elif item[0] >= 1935 and value['taipu'] == 7 and item[0] - 1935  == value['number']:
                result = {}
                result['id'] = key
                result['payload'] = value
                result['guzhang'] = item[1]
                results.append(result)
                break
    return results

def print_equipment_vector(vector: List[List[int]]):
    equipments = convert_vector_to_equipment(vector)
    for equipment in equipments:
        print('{}\t{}\t{}'.format(equipment['id'], equipment['payload']['name'], equipment['guzhang']))

def print_all_equipment():
    print_equipment_vector(np.transpose(np.nonzero(signifiant_mask)).tolist())

with open('feeds_v4.json', 'r') as f:
    feeds = json.load(f)

    
rows = feeds['row']
predictions = feeds['prediction']
actions = feeds['action']

batch_predictions = [[] for _ in range(len(rows))]
batch_actions = [[] for _ in range(len(rows))]

for prediction in predictions:
    row = rows[int(prediction[0])]
    prediction[3] = 1 / (1 + math.e ** - prediction[3])
    batch_predictions[row].append(prediction[1:])
    
for action in actions:
    row = rows[int(action[0])]
    batch_actions[row].append(action[1:])
    
    
def create_recall_and_accurate_sheet():
    non_100_accurate_count = 0
    guzhang_0_non_100_accurate_count = 0
    guzhang_1_non_100_accurate_count = 0
    for index in range(len(rows)):
        predictions_0 = list(filter(remain_guzhang_0, batch_predictions[index]))
        predictions_1 = list(filter(remain_guzhang_1, batch_predictions[index]))
        actions_0 = list(filter(remain_guzhang_0, batch_actions[index]))
        actions_1 = list(filter(remain_guzhang_1, batch_actions[index]))
        
        recall_0 = 0
        recall_1 = 0
        accurate_0 = 0
        accurate_1 = 0
        
        for action in actions_0:
            recall_0 += any(x for x in predictions_0 if int(x[0]) == action[0])
        for prediction in predictions_0:
            accurate_0 += any(x for x in actions_0 if int(x[0]) == prediction[0])
        for action in actions_1:
            recall_1 += any(x for x in predictions_1 if int(x[0]) == action[0])
        for prediction in predictions_1:
            accurate_1 += any(x for x in actions_1 if int(x[0]) == prediction[0])
        
        recall_0 = recall_0 / len(actions_0) if len(actions_0) != 0 else 1.0
        recall_1 = recall_1 / len(actions_1) if len(actions_1) != 0 else 1.0
        accurate_0 = accurate_0 / len(predictions_0) if len(predictions_0) != 0 else 1.0
        accurate_1 = accurate_1 / len(predictions_1) if len(predictions_1) != 0 else 1.0
        
        print('index:\t{}\tR0:\t{}\tR1:\t{}\tA0:\t{}\tA1:\t{}'.format(index, '%.6f' % recall_0, '%.6f' % recall_1, '%.6f' % accurate_0, '%.6f' % accurate_1))

        if accurate_0 < 1.0 or accurate_1 < 1.0:
            non_100_accurate_count += 1
        if accurate_0 < 1.0:
            guzhang_0_non_100_accurate_count += 1
        if accurate_1 < 1.0:
            guzhang_1_non_100_accurate_count += 1
        # elif len(actions_0) > 0 or len(actions_1) > 0:
        #     print(index)
        
        
    print('total_A:\t{}'.format(1 - non_100_accurate_count / len(rows)))
    print('0_A:\t{}'.format(1 - guzhang_0_non_100_accurate_count / len(rows)))
    print('1_A:\t{}'.format(1 - guzhang_1_non_100_accurate_count / len(rows)))
    
create_recall_and_accurate_sheet()


print_all_equipment()

# print(len(batch_actions[0]))
# print(len(batch_predictions[0]))
# print_equipment_vector(batch_actions[0])
# print('======')
# print_equipment_vector(batch_predictions[0])


# print(len(batch_actions[118]))
# print(len(batch_predictions[118]))
# print_equipment_vector(batch_actions[118])
# print('======')
# print_equipment_vector(batch_predictions[118])


# print(len(batch_actions[237]))
# print(len(batch_predictions[237]))
# print_equipment_vector(batch_actions[237])
# print('======')
# print_equipment_vector(batch_predictions[237])


# print(len(batch_actions[171]))
# print(len(batch_predictions[171]))
# print_equipment_vector(batch_actions[171])
# print('======')
# print_equipment_vector(batch_predictions[171])


print(len(batch_actions[87]))
print(len(batch_predictions[87]))
print_equipment_vector(batch_actions[87])
print('======')
print_equipment_vector(batch_predictions[87])