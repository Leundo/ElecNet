import argparse
import os
from decimal import Decimal
import json
from typing import Tuple

def remain_guzhang_0(value):
    return value[1] < 0.5

def remain_guzhang_1(value):
    return value[1] > 0.5

with open('feeds.json', 'r') as f:
    feeds = json.load(f)
    
rows = feeds['row']
predictions = feeds['prediction']
actions = feeds['action']

batch_predictions = [[] for _ in range(len(rows))]
batch_actions = [[] for _ in range(len(rows))]

for prediction in predictions:
    row = rows[int(prediction[0])]
    batch_predictions[row].append(prediction[1:])
    
for action in actions:
    row = rows[int(action[0])]
    batch_actions[row].append(action[1:])
    
    
def create_recall_and_accurate_sheet():
    non_100_accurate_count = 0
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
        # print(predictions_0)
        # print(actions_0)
        _ = 0
        
    print('total_A:\t{}'.format(1 - non_100_accurate_count / len(rows)))
    
create_recall_and_accurate_sheet()