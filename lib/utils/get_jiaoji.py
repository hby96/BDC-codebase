import os
from tqdm import tqdm
import pandas as pd
import numpy as np
import shutil


action_root_path = '/home/ecg/Documents/Dataset/Huawei_Big_Data/chusai/split_train_wo_duplicate_feature'
action_total_list = os.listdir(action_root_path)
# total_root_path = '/home/ecg/Documents/Dataset/Huawei_Big_Data/chusai/split_train_wo_duplicate'
# total_list = os.listdir(total_root_path)

port_path = '/home/ecg/Documents/Projects/HuaWei_Big_Data/Demo/order_testport2.csv'
port_list = pd.read_csv(port_path, )['loadingOrder'].values.tolist()
port_list = [(nr + '.csv') for nr in port_list]

# save_path = '/home/ecg/Documents/Dataset/Huawei_Big_Data/chusai/is_both_feature'
save_path = '/home/ecg/Documents/Dataset/Huawei_Big_Data/chusai/useful_train_by_test_2_feature'

# for idx, action_csv_path in tqdm(enumerate(action_total_list)):
#     # if action_csv_path in port_list:
#     #     src = os.path.join(action_root_path, action_csv_path)
#     #     tar = os.path.join(save_path, action_csv_path)
#     #     shutil.copy(src, tar)

for idx, port_csv_path in tqdm(enumerate(port_list)):
    if port_csv_path in action_total_list:
        src = os.path.join(action_root_path, port_csv_path)
        tar = os.path.join(save_path, port_csv_path)
        shutil.copy(src, tar)

