import pandas as pd
import numpy as np
import os
from tqdm import tqdm
import shutil


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
    s = s * 1000  # 转换成m
    return s


def get_port_pos():
    return (0, 0), 0


if __name__ == "__main__":
    columns = ['loadingOrder', 'carrierName', 'timestamp', 'longitude',
               'latitude', 'vesselMMSI', 'speed', 'direction', 'vesselNextport',
               'vesselNextportETA', 'vesselStatus', 'vesselDatasource', 'TRANSPORT_TRACE']
    converters = {
        'loadingOrder': str,
        'carrierName': str,
        'timestamp': str,
        'longitude': np.float,
        'latitude': np.float,
        'vesselMMSI': str,
        'speed': np.float,
        'direction': np.float,
        'vesselNextport': str,
        'vesselNextportETA': str,
        'vesselStatus': str,
        'vesselDatasource': str,
        'TRANSPORT_TRACE': str,
    }

    root_path = '/home/ecg/Documents/Dataset/Huawei_Big_Data/chusai/split_train_wo_duplicate/'
    total_list = os.listdir(root_path)

    root_save_path = '/home/ecg/Documents/Dataset/Huawei_Big_Data/chusai/useful_train_by_anchor/'

    is_anchor_num, no_anchor_num = 0, 0
    is_near_num, no_near_num = 0, 0
    is_both_num, no_both_num = 0, 0
    for idx, csv_path in tqdm(enumerate(total_list)):
        df = pd.read_csv(os.path.join(root_path, csv_path), names=columns, header=0, converters=converters)
        df['timestamp'] = pd.to_datetime(df['timestamp'], infer_datetime_format=True)

        df['lat_diff'] = df.groupby('loadingOrder')['latitude'].diff(1)
        df['lon_diff'] = df.groupby('loadingOrder')['longitude'].diff(1)
        df['speed_diff'] = df.groupby('loadingOrder')['speed'].diff(1)
        df['diff_minutes'] = df.groupby('loadingOrder')['timestamp'].diff(1).dt.total_seconds() // 60
        df['anchor'] = (
                (df['lat_diff'] <= 0.03) & (df['lon_diff'] <= 0.03) & (df['speed_diff'] <= 0.3) & (
                df['diff_minutes'] <= 10)).astype('int')

        latitude = df.iloc[df.shape[0]-1]['latitude']
        longitude = df.iloc[df.shape[0]-1]['longitude']
        is_anchor = df.iloc[df.shape[0]-1]['anchor']

        # port_pos, radius = get_port_pos()
        # dis = get_distance((longitude, longitude), port_pos)
        # if dis < radius:
        #     is_near = 1
        # else:
        #     is_near = 0

        # counting
        if is_anchor == 1:
            is_anchor_num += 1
        elif is_anchor == 0:
            no_anchor_num += 1

        # if is_near == 1:
        #     is_near_num += 1
        # elif is_near_num == 0:
        #     no_near_num += 1
        #
        # if (is_anchor == 1) and (is_near == 1):
        #     is_both_num += 1
        # else:
        #     no_both_num += 1

        if is_anchor == 1:
            save_path = os.path.join(root_save_path, csv_path)
            shutil.copy(os.path.join(root_path, csv_path), save_path)

    print('total_is_anchor_num:', is_anchor_num)
    print('total_no_anchor_num:', no_anchor_num)
    # print('total_is_near_num:', is_near_num)
    # print('total_no_near_num:', no_near_num)
    # print('total_is_both_num:', is_both_num)
    # print('total_no_both_num:', no_both_num)
