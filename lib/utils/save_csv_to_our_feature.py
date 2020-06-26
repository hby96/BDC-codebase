import pandas as pd
import os
from tqdm import tqdm
import numpy as np

root_path = '/home/ecg/Documents/Dataset/Huawei_Big_Data/chusai/split_test_file/'
save_root_path = '/home/ecg/Documents/Dataset/Huawei_Big_Data/chusai/split_test_feature/'
# root_path = '/home/ecg/Documents/Dataset/Huawei_Big_Data/chusai/train_wo_head_end_file/'
# save_root_path = '/home/ecg/Documents/Dataset/Huawei_Big_Data/chusai/train_wo_head_end_feature'
total_list = os.listdir(root_path)


# columns = ['loadingOrder', 'carrierName', 'timestamp', 'longitude',
#                   'latitude', 'vesselMMSI', 'speed', 'direction', 'vesselNextport',
#                   'vesselNextportETA', 'vesselStatus', 'vesselDatasource', 'TRANSPORT_TRACE']
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
        'anchor': np.float,
        'new_anchor': np.float,
        'lat_diff': np.float,
        'lon_diff': np.float,
        'speed_diff': np.float,
        'diff_secs': np.float,
        'lat_diff_per': np.float,
        'lon_diff_per': np.float,
        'diff_minutes': np.float,
    }


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


for idx, csv_path in tqdm(enumerate(total_list)):
    if ('train' in root_path) or ('valid' in root_path) or ('filter' in root_path):
        df = pd.read_csv(os.path.join(root_path, csv_path), header=0, converters=converters)
    elif 'test' in root_path:
        df = pd.read_csv(os.path.join(root_path, csv_path), header=0)
    euclid_dis = get_distance((df.iloc[0]['latitude'], df.iloc[0]['longitude']), (df.iloc[len(df) - 1]['latitude'], df.iloc[len(df) - 1]['longitude']))

    abs_long = max(df.iloc[0]['longitude'], df.iloc[len(df) - 1]['longitude']) - min(df.iloc[0]['longitude'], df.iloc[len(df) - 1]['longitude'])
    if abs_long > 180:
        abs_long = 360 - abs_long
    abs_lati = abs(df.iloc[0]['latitude'] - df.iloc[len(df) - 1]['latitude'])
    manha_dis = abs_lati + abs_long
    ratio_dis = abs_lati / np.sqrt(np.array(abs_lati ** 2) + np.array(abs_long ** 2))
    df['timestamp'] = pd.to_datetime(df['timestamp'], infer_datetime_format=True)

    df.sort_values(['loadingOrder', 'timestamp'], inplace=True)
    # 特征只选择经纬度、速度、方向
    # df['lat_diff'] = df.groupby('loadingOrder')['latitude'].diff(1)
    # df['lon_diff'] = df.groupby('loadingOrder')['longitude'].diff(1)
    # df['speed_diff'] = df.groupby('loadingOrder')['speed'].diff(1)
    # df['diff_minutes'] = df.groupby('loadingOrder')['timestamp'].diff(1).dt.total_seconds() // 60
    # df['old_direc_diff'] = df.groupby('loadingOrder')['direction'].diff(1)
    # df['direc_diff'] = df['old_direc_diff'].copy(True)
    # df['direc_diff'][df['old_direc_diff'] < 0] = df['old_direc_diff'] + 36000

    # df['anchor'] = ((df['lat_diff'] <= 0.03) & (df['lon_diff'] <= 0.03) & (df['speed_diff'] <= 0.3) &
    #                 (df['diff_minutes'] <= 10)).astype('int')

    df['diff_secs'] = df.groupby('loadingOrder')['timestamp'].diff(1).dt.total_seconds()
    df['old_direc_diff'] = df.groupby('loadingOrder')['direction'].diff(1)
    df['direc_diff'] = df['old_direc_diff'].copy(True)
    df['direc_diff'][df['old_direc_diff'] < 0] = df['old_direc_diff'] + 36000

    df['angular_speed'] = df['direc_diff'] / df['diff_secs']
    # df['angular_speed'] = df['direc_diff'] / df['diff_secs']
    # df['anchor'] = (
    #         (df['lat_diff'] <= 0.03) & (df['lon_diff'] <= 0.03) & (df['speed_diff'] <= 0.3) & (df['speed'] <= 3) & (
    #         abs(df['lat_diff_per']) <= 0.001) & (abs(df['lon_diff_per']) <= 0.001)).astype('int')
    # df['new_anchor'] = df['anchor'].copy(True)
    # win_size = 20
    # for i in range(10, df.shape[0] - 10, 1):
    #     min_win = max(i - win_size / 2, 0)
    #     max_win = min(i + win_size / 2, df.shape[0])
    #     value = df['anchor'].iloc[int(min_win): int(max_win)].mean()
    #     if value >= 0.5:
    #         df['new_anchor'].iloc[i] = 1
    #     else:
    #         df['new_anchor'].iloc[i] = 0

    if 'train' or 'valid' in root_path:
        group_data = df.groupby('loadingOrder')['timestamp'].agg(mmax='max', count='count',
                                                                        mmin='min').reset_index()
        # 读取数据的最大值-最小值，即确认时间间隔为label
        group_data['label'] = (group_data['mmax'] - group_data['mmin']).dt.total_seconds()
    elif 'test' in root_path:
        group_data = df.groupby('loadingOrder')['timestamp'].agg(count='count').reset_index()
        group_data['onboardDate'] = df['onboardDate']

    anchor_data = df.groupby('loadingOrder')['new_anchor'].agg('sum').reset_index()
    anchor_data.columns = ['loadingOrder', 'anchor_cnt']
    group_data = group_data.merge(anchor_data, on='loadingOrder', how='left')
    group_data['anchor_ratio'] = group_data['anchor_cnt'] / group_data['count']

    agg_function = ['min', 'max', 'mean', 'median']
    agg_col = ['latitude', 'speed', 'direc_diff', 'angular_speed']

    group = df.groupby('loadingOrder')[agg_col].agg(agg_function).reset_index()
    group.columns = ['loadingOrder'] + ['{}_{}'.format(i, j) for i in agg_col for j in agg_function]
    group_data = group_data.merge(group, on='loadingOrder', how='left')
    group_data['euclid_dis'] = euclid_dis
    group_data['manha_dis'] = manha_dis
    group_data['ratio_dis'] = ratio_dis

    group_data.to_csv(os.path.join(save_root_path, csv_path), index=False)
