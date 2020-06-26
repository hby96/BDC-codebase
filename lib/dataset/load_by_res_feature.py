import pandas as pd
import numpy as np
import os
from tqdm import tqdm

class LoadByResFeature():

    def __init__(self, mode, cfg):
        assert mode == 'train' or mode == 'valid' or mode == 'test'

        self.mode = mode
        self.nrow = cfg.DATASET.LOADER.NROWS
        self.converters = {
            'loadingOrder': str,
            'mmax': str,
            'mmin': str,
            'yiqing': np.float,
            'label': np.float,
            'start_lat': np.float,
            'start_long': np.float,
            'end_lat': np.float,
            'end_long': np.float,
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
                df_tmp.append(pd.read_csv(csv_path, header=0))
            df = pd.concat(df_tmp, ignore_index=True)
        elif self.mode == 'test':
            df_tmp = []
            for idx, csv_path in tqdm(enumerate(self.read_list)):
                df_tmp.append(pd.read_csv(csv_path, header=0))
            df = pd.concat(df_tmp, ignore_index=True)
        return df
