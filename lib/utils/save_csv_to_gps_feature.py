import pandas as pd
import os
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


def get_test_trace_time(test_trace_path):
    port_info_dict = dict()
    port_df = pd.read_csv(test_trace_path, header=0)
    for i in range(port_df.shape[0]):
        time = float(port_df.iloc[i]['time'])
        port_info_dict[port_df.iloc[i]['TRANSPORT_TRACE']] = time
    return port_info_dict
# save csv to gps feature


def get_rad(d):
    return d * np.pi / 180.0


def get_distance(A, B):
    LatA, LonA = A[0], A[1]
    LatB, LonB = B[0], B[1]
    EARTH_RADIUS = 6378.137  # 千米
    radLatA = get_rad(LatA)
    radLatB = get_rad(LatB)
    a = radLatA - radLatB
    b = get_rad(LonA) - get_rad(LonB)
    s = 2 * np.arcsin(np.sqrt(np.power(np.sin(a / 2), 2) + np.cos(radLatA) * np.cos(radLatB)*np.power(np.sin(b / 2), 2)))
    s = s * EARTH_RADIUS
    #  保留两位小数
    s = np.round(s * 100)/100
    # s = s * 1000  # 转换成m
    return s



def save_to_feature(port_gps_info, test_trace_time_info, root_path, save_root_path):
    dis_list = list()

    total_list = os.listdir(root_path)

    converters = {
        'longitude': np.float,
        'latitude': np.float,
    }

    for idx, csv_path in tqdm(enumerate(total_list)):
        df = pd.read_csv(os.path.join(root_path, csv_path), header=0, converters=converters)

        # print(csv_path)
        # print(test_trace_time_info[df['TRANSPORT_TRACE'].iloc[0]])

        df['timestamp'] = pd.to_datetime(df['timestamp'], infer_datetime_format=True)
        start_port = df['TRANSPORT_TRACE'].iloc[0].split('-')[0]
        end_port = df['TRANSPORT_TRACE'].iloc[0].split('-')[1]
        start_gps = port_gps_info[start_port]
        end_gps = port_gps_info[end_port]
        # dis = get_distance(start_gps, end_gps)

        if ('train' in root_path) or ('trace' in root_path):
            group_data = df.groupby('loadingOrder')['timestamp'].agg(mmax='max', mmin='min').reset_index()
            # 读取数据的最大值-最小值，即确认时间间隔为label
            group_data['label'] = (group_data['mmax'] - group_data['mmin']).dt.total_seconds()
            group_data['day'] = int(group_data['label'] / 3600 / 24)
        elif 'test' in root_path:
            group_data = df.groupby('loadingOrder')['timestamp'].agg(count='count').reset_index()
            group_data['onboardDate'] = df['onboardDate']
            group_data['day'] = int(test_trace_time_info[df['TRANSPORT_TRACE'].iloc[0]])

        # group_data['dis'] = dis
        group_data['start_lat'] = start_gps[0]
        group_data['start_long'] = start_gps[1]
        group_data['end_lat'] = end_gps[0]
        group_data['end_long'] = end_gps[1]

        group_data.to_csv(os.path.join(save_root_path, csv_path), index=False)


if __name__ == "__main__":

    port_gps_path = '/home/ecg/Documents/Projects/HuaWei_Big_Data/Demo/GPS_LUT.csv'

    test_trace_path = '/home/ecg/Documents/Projects/HuaWei_Big_Data/Demo/test_trace_time.csv'

    # root_path = '/home/ecg/Documents/Dataset/Huawei_Big_Data/chusai/two_trace_cal_head_end_file/'
    # save_root_path = '/home/ecg/Documents/Dataset/Huawei_Big_Data/chusai/two_trace_cal_head_end_gps_feature'
    root_path = '/home/ecg/Documents/Dataset/Huawei_Big_Data/chusai/split_test_file/'
    save_root_path = '/home/ecg/Documents/Dataset/Huawei_Big_Data/chusai/split_test_gps_feature/'

    port_gps_info = get_port_gps_info(port_gps_path)

    test_trace_time_info = get_test_trace_time(test_trace_path)

    save_to_feature(port_gps_info, test_trace_time_info, root_path, save_root_path)
