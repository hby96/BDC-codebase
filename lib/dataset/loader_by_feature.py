import pandas as pd
import numpy as np
import os
from tqdm import tqdm

class LoadByFeature():

    def __init__(self, mode, cfg):
        assert mode == 'train' or mode == 'valid' or mode == 'test'

        self.mode = mode
        self.nrow = cfg.DATASET.LOADER.NROWS
        self.columns = ['loadingOrder', 'mmax', 'count', 'mmin', 'label', 'anchor_cnt', 'anchor_ratio',
                        'latitude_min', 'latitude_max', 'latitude_mean', 'latitude_median', 'longitude_min',
                        'longitude_max', 'longitude_mean', 'longitude_median', 'speed_min', 'speed_max', 'speed_mean',
                        'speed_median' 'direction_min', 'direction_max', 'direction_mean', 'direction_median']
        self.converters = {
            'loadingOrder': str,
            'mmax': str,
            'count': np.float,
            'mmin': str,
            'label': np.float,
            'anchor_cnt': np.float,
            'anchor_ratio': np.float,
            'latitude_min': np.float,
            'latitude_max': np.float,
            'latitude_mean': np.float,
            'latitude_median': np.float,
            'longitude_min': np.float,
            'longitude_max': np.float,
            'longitude_mean': np.float,
            'longitude_median': np.float,
            'speed_min': np.float,
            'speed_max': np.float,
            'speed_mean': np.float,
            'speed_median': np.float,
            'direction_min': np.float,
            'direction_max': np.float,
            'direction_mean': np.float,
            'direction_median': np.float,
        }
        if self.mode == 'train' or self.mode == 'valid':
            if self.mode == 'train':
                total_list = os.listdir(cfg.DATASET.TRAIN_GPS_PATH)
                if self.nrow == -1:
                    read_list = [os.path.join(cfg.DATASET.TRAIN_GPS_PATH, csv) for csv in total_list]
                    self.read_list = read_list
                else:
                    total_list = total_list[:self.nrow]
                    read_list = [os.path.join(cfg.DATASET.TRAIN_GPS_PATH, csv) for csv in total_list]
                    self.read_list = read_list
            elif self.mode == 'valid':
                total_list = os.listdir(cfg.DATASET.VALID_GPS_PATH)
                read_list = [os.path.join(cfg.DATASET.VALID_GPS_PATH, csv) for csv in total_list]
                self.read_list = read_list
        else:
            total_list = os.listdir(cfg.DATASET.TEST_DATA_PATH)
            read_list = [os.path.join(cfg.DATASET.TEST_DATA_PATH, csv) for csv in total_list]
            self.read_list = read_list

    def get_data(self):
        if self.mode == 'train' or self.mode == 'valid':
            df_tmp = []
            for idx, csv_path in tqdm(enumerate(self.read_list)):
                df_tmp.append(pd.read_csv(csv_path, header=0, converters=self.converters))
                # df_tmp.append(pd.read_csv(csv_path))
            df = pd.concat(df_tmp, ignore_index=True)
            df.names = self.columns
        elif self.mode == 'test':
            df_tmp = []
            for idx, csv_path in tqdm(enumerate(self.read_list)):
                df_tmp.append(pd.read_csv(csv_path))
            df = pd.concat(df_tmp, ignore_index=True)

        return df
