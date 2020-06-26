import pandas as pd
import pytz
import pickle
import os
from datetime import datetime, timedelta, timezone
from tqdm import tqdm
import numpy as np
import math


# get port gps information
def get_port_gps_info(port_gps_path):
    port_info_dict = dict()
    port_df = pd.read_csv(port_gps_path, header=0)

    for i in range(port_df.shape[0]):
        latitude = float(port_df.iloc[i]['manual_latitude'])
        longtitude = float(port_df.iloc[i]['manual_longitude'])
        port_info_dict[port_df.iloc[i]['Port']] = (latitude, longtitude)
    return port_info_dict


def get_trace_time_from_csv(test_trace_path):
    port_info_dict = dict()
    port_df = pd.read_csv(test_trace_path, header=0)
    for i in range(port_df.shape[0]):
        time = float(port_df.iloc[i]['time'])
        port_info_dict[port_df.iloc[i]['TRANSPORT_TRACE']] = time  # * 24 * 3600
    return port_info_dict
# save csv to gps feature


def get_trace_time_from_pkl(test_trace_path):
    with open(test_trace_path, 'rb') as f:
        port_info_dict = pickle.load(f)
    port_info_dict['CNSHK-ESALG'] = 1900800  # 22
    port_info_dict['CNYTN-MTMLA'] = 1857600
    port_info_dict['CNSHA-PAMIT'] = 1900800  # 22
    return port_info_dict


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


def save_to_feature(port_gps_info, trace_time_info, root_path, save_root_path):
    dis_list = list()

    total_list = os.listdir(root_path)

    converters = {
        'longitude': np.float,
        'latitude': np.float,
    }

    time_50000 = 0
    time_200000 = 0
    time_500000 = 0

    for idx, csv_path in tqdm(enumerate(total_list)):
        df = pd.read_csv(os.path.join(root_path, csv_path), header=0, converters=converters)

        if not (df['TRANSPORT_TRACE'].iloc[0] in trace_time_info):
            print(csv_path)
            print(df['TRANSPORT_TRACE'].iloc[0])
            print(trace_time_info)
            assert False
            continue

        # print(csv_path)
        # print(test_trace_time_info[df['TRANSPORT_TRACE'].iloc[0]])

        df['timestamp'] = pd.to_datetime(df['timestamp'], infer_datetime_format=True)
        start_port = df['TRANSPORT_TRACE'].iloc[0].split('-')[0]
        end_port = df['TRANSPORT_TRACE'].iloc[0].split('-')[1]
        start_gps = port_gps_info[start_port]
        end_gps = port_gps_info[end_port]
        # dis = get_distance(start_gps, end_gps)

        # df['diff_secs'] = df.groupby('loadingOrder')['timestamp'].diff(1).dt.total_seconds()
        # df['old_direc_diff'] = df.groupby('loadingOrder')['direction'].diff(1)
        # df['direc_diff'] = df['old_direc_diff'].copy(True)
        # df['direc_diff'][df['old_direc_diff'] < 0] = df['old_direc_diff'] + 36000
        # df['angular_speed'] = df['direc_diff'] / df['diff_secs']

        # value = pd.Timedelta(seconds=truth[trace])
        yiqing_start = pd.to_datetime('2020/1/20 00:00:00', infer_datetime_format=True).tz_localize(pytz.UTC)

        mean_time = int(trace_time_info[df['TRANSPORT_TRACE'].iloc[0]])
        mean_time_day = int(mean_time / 24 / 3600)

        if ('train' in root_path) or ('trace' in root_path):
            group_data = df.groupby('loadingOrder')['timestamp'].agg(mmax='max', mmin='min').reset_index()
            # 读取数据的最大值-最小值，即确认时间间隔为label
            if 'label' in df.columns:
                group_data['label'] = df['label'] - mean_time
            else:
                group_data['label'] = (group_data['mmax'] - group_data['mmin']).dt.total_seconds() - mean_time


            # if (group_data['mmax'].iloc[0] - yiqing_start).days > 0:
            #     group_data['yiqing'] = 1
            # else:
            #     group_data['yiqing'] = 0
        elif 'test' in root_path:
            group_data = df.groupby('loadingOrder')['timestamp'].agg(count='count').reset_index()
            group_data['onboardDate'] = df['onboardDate']
            test_start = pd.to_datetime(df['onboardDate'].iloc[0], infer_datetime_format=True).tz_localize(pytz.UTC)
            # if (test_start - yiqing_start).days > -mean_time_day:
            #     group_data['yiqing'] = 1
            # else:
            #     group_data['yiqing'] = 0

        # agg_function = ['min', 'max', 'mean', 'median']
        # agg_col = ['latitude', 'speed', 'direc_diff', 'angular_speed']
        # group = df.groupby('loadingOrder')[agg_col].agg(agg_function).reset_index()
        # group.columns = ['loadingOrder'] + ['{}_{}'.format(i, j) for i in agg_col for j in agg_function]
        # group_data = group_data.merge(group, on='loadingOrder', how='left')

        # group_data['dis'] = dis
        group_data['start_lat'] = start_gps[0]
        group_data['start_long'] = start_gps[1]
        group_data['end_lat'] = end_gps[0]
        group_data['end_long'] = end_gps[1]
        group_data['TRANSPORT_TRACE'] = df['TRANSPORT_TRACE'].iloc[0]

        lat_max_idx = np.argmax(df['latitude'])
        lat_min_idx = np.argmin(df['latitude'])
        lon_max_idx = np.argmax(df['longitude'])
        lon_min_idx = np.argmin(df['longitude'])
        lat_diff = df['latitude'].iloc[lat_max_idx] - df['latitude'].iloc[lat_min_idx]
        lat_hours = math.ceil(abs(df['timestamp'].iloc[lat_max_idx] - df['timestamp'].iloc[lat_min_idx]).total_seconds() / 3600)
        if lat_hours == 0:
            group_data['lat_per_hour'] = 0
        else:
            group_data['lat_per_hour'] = lat_diff / lat_hours

        # group_data['start_err_dis'] = get_distance((df['latitude'].iloc[0], df['longitude'].iloc[0]), (start_gps[0], start_gps[1]))
        # print(group_data.head())
        # assert False
        group_data['mean_speed'] = get_distance((df['latitude'].iloc[0], df['longitude'].iloc[0]), (df['latitude'].iloc[-1], df['longitude'].iloc[-1]))\
                                   / math.ceil(abs(df['timestamp'].iloc[-1] - df['timestamp'].iloc[0]).total_seconds() / 3600)

        group_data.to_csv(os.path.join(save_root_path, csv_path), index=False)


if __name__ == "__main__":

    # port_gps_path = '/home/ecg/Documents/Projects/HuaWei_Big_Data/Demo/GPS_LUT.csv'
    port_gps_path = '/home/ecg/Documents/Projects/HuaWei_Big_Data/Demo/assistant_data/Port_Dic.csv'

    # trace_time_path = '/home/ecg/Documents/Projects/HuaWei_Big_Data/Demo/test_trace_time.csv'
    trace_time_path = '/home/ecg/Documents/Projects/HuaWei_Big_Data/Demo/assistant_data/truth_n.pkl'

    # root_path = '/home/ecg/Documents/Dataset/Huawei_Big_Data/chusai/random_cut_two_trace_cal_head_end_file/'
    # save_root_path = '/home/ecg/Documents/Dataset/Huawei_Big_Data/chusai/random_cut_two_trace_cal_head_end_res_feature'
    # root_path = '/home/ecg/Documents/Dataset/Huawei_Big_Data/chusai/0624_data/random_cut_train_specified_trace'
    # save_root_path = '/home/ecg/Documents/Dataset/Huawei_Big_Data/chusai/0624_data/random_cut_train_specified_trace_feature'

    # root_path = '/home/ecg/Documents/Dataset/Huawei_Big_Data/chusai/split_test_file/'
    # save_root_path = '/home/ecg/Documents/Dataset/Huawei_Big_Data/chusai/split_test_res_feature/'
    root_path = '/home/ecg/Documents/Dataset/Huawei_Big_Data/chusai/0624_data/split_test_uniform'
    save_root_path = '/home/ecg/Documents/Dataset/Huawei_Big_Data/chusai/0624_data/split_test_uniform_feature'

    port_gps_info = get_port_gps_info(port_gps_path)

    # trace_time_info = get_trace_time_from_csv(trace_time_path)
    trace_time_info = get_trace_time_from_pkl(trace_time_path)

    save_to_feature(port_gps_info, trace_time_info, root_path, save_root_path)
