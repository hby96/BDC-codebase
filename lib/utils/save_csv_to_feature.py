import pandas as pd
import os
from tqdm import tqdm
import numpy as np

root_path = '/home/ecg/Documents/Dataset/Huawei_Big_Data/chusai/useful_train_by_anchor_start_port_series/'
save_root_path = '/home/ecg/Documents/Dataset/Huawei_Big_Data/chusai/useful_train_by_anchor_start_port_series_feature/'
# root_path = '/home/ecg/Documents/Dataset/Huawei_Big_Data/chusai/split_valid/'
# save_root_path = '/home/ecg/Documents/Dataset/Huaweis_Big_Data/chusai/split_valid_feature/'
total_list = os.listdir(root_path)


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

for idx, csv_path in tqdm(enumerate(total_list)):
    if 'train' or 'valid' in root_path:
        df = pd.read_csv(os.path.join(root_path, csv_path), names=columns, header=0, converters=converters)
        df['vesselNextportETA'] = pd.to_datetime(df['vesselNextportETA'], infer_datetime_format=True)
        df.columns = columns
    elif 'test' in root_path:
        df = pd.read_csv(os.path.join(root_path, csv_path))

    df['timestamp'] = pd.to_datetime(df['timestamp'], infer_datetime_format=True)

    df.sort_values(['loadingOrder', 'timestamp'], inplace=True)
    # 特征只选择经纬度、速度、方向
    df['lat_diff'] = df.groupby('loadingOrder')['latitude'].diff(1)
    df['lon_diff'] = df.groupby('loadingOrder')['longitude'].diff(1)
    df['speed_diff'] = df.groupby('loadingOrder')['speed'].diff(1)
    df['diff_minutes'] = df.groupby('loadingOrder')['timestamp'].diff(1).dt.total_seconds() // 60
    df['anchor'] = (
            (df['lat_diff'] <= 0.03) & (df['lon_diff'] <= 0.03) & (df['speed_diff'] <= 0.3) & (
            df['diff_minutes'] <= 10)).astype('int')

    if 'train' or 'valid' in root_path:
        group_data = df.groupby('loadingOrder')['timestamp'].agg(mmax='max', count='count',
                                                                      mmin='min').reset_index()
        # 读取数据的最大值-最小值，即确认时间间隔为label
        group_data['label'] = (group_data['mmax'] - group_data['mmin']).dt.total_seconds()
    elif 'test' in root_path:
        group_data = df.groupby('loadingOrder')['timestamp'].agg(count='count').reset_index()
        group_data['onboardDate'] = df['onboardDate']

    anchor_data = df.groupby('loadingOrder')['anchor'].agg('sum').reset_index()
    anchor_data.columns = ['loadingOrder', 'anchor_cnt']
    group_data = group_data.merge(anchor_data, on='loadingOrder', how='left')
    group_data['anchor_ratio'] = group_data['anchor_cnt'] / group_data['count']

    agg_function = ['min', 'max', 'mean', 'median']
    agg_col = ['latitude', 'longitude', 'speed', 'direction']

    group = df.groupby('loadingOrder')[agg_col].agg(agg_function).reset_index()
    group.columns = ['loadingOrder'] + ['{}_{}'.format(i, j) for i in agg_col for j in agg_function]
    group_data = group_data.merge(group, on='loadingOrder', how='left')

    group_data.to_csv(os.path.join(save_root_path, csv_path), index=False)
