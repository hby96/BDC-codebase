import pandas as pd
import pytz
import os
from datetime import datetime, timedelta, timezone
from tqdm import tqdm
import numpy as np


# get port gps information
def get_port_gps_info(port_gps_path):
    port_info_dict = dict()
    port_df = pd.read_csv(port_gps_path, header=0)
    for i in range(port_df.shape[0]):
        latitude = float(port_df.iloc[i]['manual_latitude'])
        longtitude = float(port_df.iloc[i]['manual_longtitude'])
        port_info_dict[port_df.iloc[i]['Port']] = (latitude, longtitude)
    return port_info_dict


def get_trace_time(test_trace_path):
    port_info_dict = dict()
    port_df = pd.read_csv(test_trace_path, header=0)
    for i in range(port_df.shape[0]):
        time = float(port_df.iloc[i]['time'])
        port_info_dict[port_df.iloc[i]['TRANSPORT_TRACE']] = time  # * 24 * 3600
    return port_info_dict
# save csv to gps feature


def save_to_feature(root_path, save_root_path):
    dis_list = list()

    total_list = os.listdir(root_path)

    converters = {
        'longitude': np.float,
        'latitude': np.float,
    }

    for idx, csv_path in tqdm(enumerate(total_list)):
        df = pd.read_csv(os.path.join(root_path, csv_path), header=0, converters=converters)

        with open(os.path.join(root_path, csv_path), 'r') as f:
            total_length = len(f.readlines())

        df['timestamp'] = pd.to_datetime(df['timestamp'], infer_datetime_format=True)
        group_data = df.groupby('loadingOrder')['timestamp'].agg(mmax='max', mmin='min').reset_index()
        # 读取数据的最大值-最小值，即确认时间间隔为label
        group_data['label'] = (group_data['mmax'] - group_data['mmin']).dt.total_seconds()
        group_data.drop(['mmax', 'mmin'], axis=1, inplace=True)

        length_1 = int(0.1 * total_length)
        length_2 = int(0.2 * total_length)
        length_3 = int(0.5 * total_length)

        df_1 = df.iloc[:length_1]
        df_1 = df_1.merge(group_data, on='loadingOrder', how='left')
        df_2 = df.iloc[:length_2]
        df_2 = df_2.merge(group_data, on='loadingOrder', how='left')
        df_3 = df.iloc[:length_3]
        df_3 = df_3.merge(group_data, on='loadingOrder', how='left')

        df_1.to_csv(os.path.join(save_root_path, csv_path.replace('.csv', '_1.csv')), index=False)
        df_2.to_csv(os.path.join(save_root_path, csv_path.replace('.csv', '_2.csv')), index=False)
        df_3.to_csv(os.path.join(save_root_path, csv_path.replace('.csv', '_5.csv')), index=False)


if __name__ == "__main__":

    root_path = '/home/ecg/Documents/Dataset/Huawei_Big_Data/chusai/0624_data/train_specified_trace'
    save_root_path = '/home/ecg/Documents/Dataset/Huawei_Big_Data/chusai/0624_data/random_cut_train_specified_trace'

    save_to_feature(root_path, save_root_path)
