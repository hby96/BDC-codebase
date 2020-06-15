import pandas as pd
# import modin.pandas as pd
import numpy as np
from tqdm import tqdm
import sys
import os


class LoadByOrder():

    def __init__(self, mode, cfg):
        assert mode == 'train' or mode == 'test'

        self.mode = mode
        self.nrow = cfg.DATASET.LOADER.NROWS
        self.columns = ['loadingOrder', 'carrierName', 'timestamp', 'longitude',
                  'latitude', 'vesselMMSI', 'speed', 'direction', 'vesselNextport',
                  'vesselNextportETA', 'vesselStatus', 'vesselDatasource', 'TRANSPORT_TRACE']
        self.converters = {
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
        if self.mode == 'train':
            total_list = os.listdir(cfg.DATASET.TRAIN_GPS_PATH)
            if self.nrow == -1:
                read_list = [os.path.join(cfg.DATASET.TRAIN_GPS_PATH, csv) for csv in total_list]
                self.read_list = read_list
            else:
                total_list = total_list[:self.nrow]
                read_list = [os.path.join(cfg.DATASET.TRAIN_GPS_PATH, csv) for csv in total_list]
                self.read_list = read_list

        else:
            self.read_list = [cfg.DATASET.TEST_DATA_PATH]

    def get_data(self):
        if self.mode == 'train':
            df_tmp = []
            for idx, csv_path in tqdm(enumerate(self.read_list)):
                df_tmp.append(pd.read_csv(csv_path, names=self.columns, header=0, converters=self.converters))
            df = pd.concat(df_tmp, ignore_index=True)
        else:
            df = pd.read_csv(self.read_list[0])

        if self.mode == 'train':
            df['vesselNextportETA'] = pd.to_datetime(df['vesselNextportETA'], infer_datetime_format=True)
            df.columns = self.columns
        elif self.mode == 'test':
            df['temp_timestamp'] = df['timestamp']
            df['onboardDate'] = pd.to_datetime(df['onboardDate'], infer_datetime_format=True)
        df['timestamp'] = pd.to_datetime(df['timestamp'], infer_datetime_format=True)
        # df['longitude'] = df['longitude'].astype(float)
        # df['loadingOrder'] = df['loadingOrder'].astype(str)
        # df['latitude'] = df['latitude'].astype(float)
        # df['speed'] = df['speed'].astype(float)
        # df['direction'] = df['direction'].astype(float)

        return df