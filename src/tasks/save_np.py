import torch

from src.utils.equipment import Equipment
from src.utils.porter import save_equipment_text_to_np, save_celue_text_to_np, save_guzhang_text_to_np


if __name__ == '__main__':
    save_celue_text_to_np('20230710')
    save_guzhang_text_to_np('20230710')
    
    # save_equipment_text_to_np(Equipment.chuanlian, '20230704')
    # save_equipment_text_to_np(Equipment.rongkang, '20230704')
    # save_equipment_text_to_np(Equipment.bianya, '20230704')
    # save_equipment_text_to_np(Equipment.xiandian, '20230704')
    # save_equipment_text_to_np(Equipment.jiaoxian, '20230704')
    # save_equipment_text_to_np(Equipment.fuhe, '20230704')
    # save_equipment_text_to_np(Equipment.fadian, '20230704')
    # save_equipment_text_to_np(Equipment.muxian, '20230704')
    # save_equipment_text_to_np(Equipment.changzhan, '20230704')
    for equipment in Equipment:
        save_equipment_text_to_np(equipment, '20230710')