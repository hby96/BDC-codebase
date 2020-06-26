import pandas as pd
import pickle
import shutil
import os
from tqdm import tqdm
import random


def get_test_trace_order_from_pkl(test_trace_path):
    with open(test_trace_path, 'rb') as f:
        order_info_dict = pickle.load(f)
    return order_info_dict


if __name__ == "__main__":

    src_root_path = '/home/ecg/Documents/Dataset/Huawei_Big_Data/chusai/0624_data/all_cal_trace_w_anchor_split'
    dst_root_path = '/home/ecg/Documents/Dataset/Huawei_Big_Data/chusai/0624_data/train_specified_trace'

    test_trace_path = '/home/ecg/Documents/Projects/HuaWei_Big_Data/Demo/assistant_data/test_trace_order.pkl'
    trace_time_info = get_test_trace_order_from_pkl(test_trace_path)

    for key in tqdm(trace_time_info.keys()):
        order_list = trace_time_info[key]
        if key == 'CNSHK-SGSIN':
            random.seed(147 + 4*2080)
            random.shuffle(order_list)
            order_list = order_list[:300]
        for order in order_list:
            src_path = os.path.join(src_root_path, order)
            dst_path = os.path.join(dst_root_path, order)
            shutil.copy(src_path, dst_path)
