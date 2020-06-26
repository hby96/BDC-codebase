import pandas as pd
import numpy as np
import os
from tqdm import tqdm

class LoadByOurFeature():

    def __init__(self, mode, cfg):
        assert mode == 'train' or mode == 'valid' or mode == 'test'

        self.mode = mode
        self.nrow = cfg.DATASET.LOADER.NROWS
        self.columns = ['loadingOrder',
            'mmax',
            'count',
            'mmin',
            'label',
            'anchor_cnt',
            'anchor_ratio',
            'latitude_min'
            'latitude_max',
            'latitude_mean',
            'latitude_median',
            'speed_min',
            'speed_max',
            'speed_mean',
            'speed_median',
            # 'direc_diff_min',
            # 'direc_diff_max',
            # 'direc_diff_mean',
            # 'direc_diff_median',
            # 'angular_speed_min',
            # 'angular_speed_max',
            # 'angular_speed_mean',
            # 'angular_speed_median',
            # 'euclid_dis',
            # 'manha_dis',
            # 'ratio_dis',
        ]
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
            'speed_min': np.float,
            'speed_max': np.float,
            'speed_mean': np.float,
            'speed_median': np.float,
            # 'direc_diff_min': np.float,
            # 'direc_diff_max': np.float,
            # 'direc_diff_mean': np.float,
            # 'direc_diff_median': np.float,
            # 'angular_speed_min': np.float,
            # 'angular_speed_max': np.float,
            # 'angular_speed_mean': np.float,
            # 'angular_speed_median': np.float,
            # 'euclid_dis': np.float,
            # 'manha_dis': np.float,
            # 'ratio_dis': np.float,
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
                # df_tmp.append(pd.read_csv(csv_path, names=self.columns, header=0, converters=self.converters, index_col=False))
                df_tmp.append(pd.read_csv(csv_path))
            df = pd.concat(df_tmp, ignore_index=True)
            df.names = self.columns
        elif self.mode == 'test':
            df_tmp = []
            for idx, csv_path in tqdm(enumerate(self.read_list)):
                df_tmp.append(pd.read_csv(csv_path))
            df = pd.concat(df_tmp, ignore_index=True)
        return df
