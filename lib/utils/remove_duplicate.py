import pandas as pd
import os
from tqdm import tqdm
import numpy as np

root_path = '/home/ecg/Documents/Dataset/Huawei_Big_Data/chusai/split_test/'
save_root_path = '/home/ecg/Documents/Dataset/Huawei_Big_Data/chusai/split_test_wo_duplicate/'
total_list = os.listdir(root_path)

if 'train' in root_path:
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
elif 'test' in root_path:
    columns = ['loadingOrder', 'timestamp', 'longitude', 'latitude', 'speed', 'direction', 'carrierName',
        'vesselMMSI', 'onboardDate', 'TRANSPORT_TRACE']

    converters = {
        'loadingOrder': str,
        'timestamp': str,
        'longitude': np.float,
        'latitude': np.float,
        'speed': np.float,
        'direction': np.float,
        'carrierName': str,
        'vesselMMSI': str,
        'onboardDate': str,
        'TRANSPORT_TRACE': str,
    }


for idx, csv_path in tqdm(enumerate(total_list)):
    csv_file = pd.read_csv(os.path.join(root_path, csv_path), names=columns, header=0, converters=converters, index_col=False)
    csv_file.drop_duplicates(subset=None, keep='first', inplace=True)
    csv_file.to_csv(os.path.join(save_root_path, csv_path), index=False)
