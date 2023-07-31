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
    
    accumulated_recall_0 = 0
    accumulated_recall_1 = 0
    accumulated_precision_0 = 0
    accumulated_precision_1 = 0
    accumulated_accuracy_0 = 0
    accumulated_accuracy_1 = 0
    
    for index in range(len(rows)):
        predictions_0 = list(filter(remain_guzhang_0, batch_predictions[index]))
        predictions_1 = list(filter(remain_guzhang_1, batch_predictions[index]))
        actions_0 = list(filter(remain_guzhang_0, batch_actions[index]))
        actions_1 = list(filter(remain_guzhang_1, batch_actions[index]))
        
        
        TP_0 = 0
        TN_0 = 0
        FP_0 = 0
        FN_0 = 0
        
        TP_1 = 0
        TN_1 = 0
        FP_1 = 0
        FN_1 = 0
        
        N_0 = 25
        N_1 = 2
        
        for prediction in predictions_0:
            is_prediction_in_action = any(x for x in actions_0 if int(x[0]) == prediction[0])
            TP_0 += 1 if is_prediction_in_action else 0
            FP_0 += 1 if not is_prediction_in_action else 0
        for prediction in predictions_1:
            is_prediction_in_action = any(x for x in actions_1 if int(x[0]) == prediction[0])
            TP_1 += 1 if is_prediction_in_action else 0
            FP_1 += 1 if not is_prediction_in_action else 0
        for action in actions_0:
            is_action_in_prediction = any(x for x in predictions_0 if int(x[0]) == action[0])
            FN_0 += 1 if not is_action_in_prediction else 0
        for action in actions_1:
            is_action_in_prediction = any(x for x in predictions_1 if int(x[0]) == action[0])
            FN_1 += 1 if not is_action_in_prediction else 0
            
        TN_0 = N_0 - TP_0 - FP_0 - FN_0
        TN_1 = N_1 - TP_1 - FP_1 - FN_1
        assert(TN_0 >= 0 and TN_1 >= 0)
        
        recall_0 = TP_0 / (TP_0 + FN_0) if TP_0 + FN_0 != 0 else 1.0
        recall_1 = TP_1 / (TP_1 + FN_1) if TP_1 + FN_1 != 0 else 1.0
        precision_0 = TP_0 / (TP_0 + FP_0) if TP_0 + FP_0 != 0 else 1.0
        precision_1 = TP_1 / (TP_1 + FP_1) if TP_1 + FP_1 != 0 else 1.0
        accuracy_0 = (TP_0 + TN_0) / N_0
        accuracy_1 = (TP_1 + TN_1) / N_1
        
        # recall_0 = 0
        # recall_1 = 0
        # precision_0 = 0
        # precision_1 = 0
        # accuracy_0 = 0
        # accuracy_1 = 0
        
        # for action in actions_0:
        #     recall_0 += any(x for x in predictions_0 if int(x[0]) == action[0])
        # for prediction in predictions_0:
        #     precision_0 += any(x for x in actions_0 if int(x[0]) == prediction[0])
        # for prediction in predictions_0:
        #     accuracy_0 += any(x for x in actions_0 if int(x[0]) == prediction[0])
        
        # for action in actions_1:
        #     recall_1 += any(x for x in predictions_1 if int(x[0]) == action[0])
        # for prediction in predictions_1:
        #     precision_1 += any(x for x in actions_1 if int(x[0]) == prediction[0])
        # for prediction in predictions_1:
        #     accuracy_1 += any(x for x in actions_1 if int(x[0]) == prediction[0])
        

        # recall_0 = recall_0 / len(actions_0) if len(actions_0) != 0 else 1.0
        # recall_1 = recall_1 / len(actions_1) if len(actions_1) != 0 else 1.0
        # precision_0 = precision_0 / len(predictions_0) if len(predictions_0) != 0 else 1.0
        # precision_1 = precision_1 / len(predictions_1) if len(predictions_1) != 0 else 1.0
        
        
        accumulated_recall_0 += recall_0
        accumulated_recall_1 += recall_1
        accumulated_precision_0 += precision_0
        accumulated_precision_1 += precision_1
        accumulated_accuracy_0 += accuracy_0
        accumulated_accuracy_1 += accuracy_1
        
        print('{}\tR0:\t{}\tR1:\t{}\tP0:\t{}\tP1:\t{}\tA0:\t{}\tA1:\t{}'.format(index, '%.4f' % recall_0, '%.4f' % recall_1, '%.4f' % precision_0, '%.4f' % precision_1, '%.4f' % accuracy_0, '%.4f' % accuracy_1))

    print('total_recall_0:\t{}'.format(accumulated_recall_0 / len(rows)))
    print('total_recall_1:\t{}'.format(accumulated_recall_1 / len(rows)))
    print('total_precision_0:\t{}'.format(accumulated_precision_0 / len(rows)))
    print('total_precision_1:\t{}'.format(accumulated_precision_1 / len(rows)))
    print('total_accuracy_0:\t{}'.format(accumulated_accuracy_0 / len(rows)))
    print('total_accuracy_1:\t{}'.format(accumulated_accuracy_1 / len(rows)))
 
create_recall_and_accurate_sheet()


# print_all_equipment()
