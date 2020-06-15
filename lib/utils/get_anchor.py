import pandas as pd
import os
from tqdm import tqdm
import numpy as np

root_path = '/home/ecg/Documents/Dataset/Huawei_Big_Data/chusai/filter_anchor_recal/'
# anchor_root_path = '/home/ecg/Documents/Projects/HuaWei_Big_Data/Demo/filter_anchor/'
anchor_root_path = '/home/ecg/Documents/Dataset/Huawei_Big_Data/chusai/filter_anchor'
total_list = os.listdir(root_path)


# columns = ['loadingOrder', 'carrierName', 'timestamp', 'longitude',
#                   'latitude', 'vesselMMSI', 'speed', 'direction', 'vesselNextport',
#                   'vesselNextportETA', 'vesselStatus', 'vesselDatasource', 'TRANSPORT_TRACE',]# 'anchor', 'new_anchor']
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
        # 'anchor': np.float,
        # 'new_anchor': np.float,
    }

for idx, csv_path in tqdm(enumerate(total_list)):
    if 'train' or 'valid' in root_path:
        df = pd.read_csv(os.path.join(root_path, csv_path), header=0, converters=converters)
        df['vesselNextportETA'] = pd.to_datetime(df['vesselNextportETA'], infer_datetime_format=True)
        # df.columns = columns
    elif 'test' in root_path:
        df = pd.read_csv(os.path.join(root_path, csv_path))

    df['timestamp'] = pd.to_datetime(df['timestamp'], infer_datetime_format=True)

    df.sort_values(['loadingOrder', 'timestamp'], inplace=True)
    # 特征只选择经纬度、速度、方向
    df['lat_diff'] = df.groupby('loadingOrder')['latitude'].diff(1)
    df['lon_diff'] = df.groupby('loadingOrder')['longitude'].diff(1)
    df['speed_diff'] = df.groupby('loadingOrder')['speed'].diff(1)
    df['diff_secs'] = df.groupby('loadingOrder')['timestamp'].diff(1).dt.total_seconds()
    df['lat_diff_per'] = 100 * df['lat_diff'] / (df['diff_secs'])
    df['lon_diff_per'] = 100 * df['lon_diff'] / (df['diff_secs'])
    df['diff_minutes'] = df.groupby('loadingOrder')['timestamp'].diff(1).dt.total_seconds() // 60
    df['anchor'] = df.apply(lambda x: 1 if x['lat_diff'] <= 0.03 and x['lon_diff'] <= 0.03
                                           and x['speed_diff'] <= 0.3 and x['speed'] <= 3 and abs(x['lat_diff_per']) <= 0.001 and abs(x['lon_diff_per']) <= 0.001 else 0, axis=1)
    df['new_anchor'] = df['anchor'].copy(True)
    win_size = 20
    for i in range(10, df.shape[0]-10, 1):
        min_win = max(i - win_size/2, 0)
        max_win = min(i + win_size/2, df.shape[0])
        value = df['anchor'].iloc[int(min_win): int(max_win)].mean()
        if value >= 0.5:
            df['new_anchor'].iloc[i] = 1
        else:
            df['new_anchor'].iloc[i] = 0
    # df.drop(['lat_diff', 'lon_diff', 'speed_diff', 'diff_minutes'], axis=1, inplace=True)
    save_path = os.path.join(anchor_root_path, csv_path)
    df.to_csv(save_path, index=False)
    assert False
