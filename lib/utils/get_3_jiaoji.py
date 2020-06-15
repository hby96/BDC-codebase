import os
from tqdm import tqdm
import pandas as pd
import numpy as np
import shutil


total_root_path = '/home/ecg/Documents/Dataset/Huawei_Big_Data/chusai/split_train_wo_duplicate'
total_total_list = os.listdir(total_root_path)

save_path = '/home/ecg/Documents/Dataset/Huawei_Big_Data/chusai/useful_train_by_anchor_start_port'

action_root_path = '/home/ecg/Documents/Dataset/Huawei_Big_Data/chusai/useful_train_by_anchor_feature'
action_list = os.listdir(action_root_path)

port_path = '/home/ecg/Documents/Projects/HuaWei_Big_Data/Demo/order_destination_GPS_repaired.csv'
port_list = pd.read_csv(port_path, )['loadingOrder'].values.tolist()
port_list = [(nr + '.csv') for nr in port_list]

start_path = '/home/ecg/Documents/Projects/HuaWei_Big_Data/Demo/order_destination_GPS_repaired_begin(2)(1).csv'
start_list = pd.read_csv(start_path, )['loadingOrder'].values.tolist()
start_list = [(nr + '.csv') for nr in start_list]

for idx, port_csv_path in tqdm(enumerate(port_list)):
    if (port_csv_path in start_list) and (port_csv_path in action_list):
        src = os.path.join(total_root_path, port_csv_path)
        dst = os.path.join(save_path, port_csv_path)
        shutil.copy(src, dst)

