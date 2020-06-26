import pandas as pd
import numpy as np
import os
from tqdm import tqdm

class LoadByMultiTimeFeature():

    def __init__(self, mode, cfg):
        assert mode == 'train' or mode == 'valid' or mode == 'test'

        self.mode = mode
        self.nrow = cfg.DATASET.LOADER.NROWS
        self.converters = {
            'loadingOrder': str,
            'mmax': str,
            'mmin': str,
            'label': np.float,
            'day': int,
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
            df_10_tmp = []
            # df_20_tmp = []
            df_30_tmp = []
            df_other_tmp = []
            for idx, csv_path in tqdm(enumerate(self.read_list)):
                df_tmp = pd.read_csv(csv_path, header=0)
                if int(df_tmp.iloc[0]['day']) <= 10:
                    df_10_tmp.append(df_tmp)
                # elif int(df_tmp.iloc[0]['day']) <= 20:
                #     df_20_tmp.append(df_tmp)
                elif int(df_tmp.iloc[0]['day']) <= 25:
                    df_30_tmp.append(df_tmp)
                else:
                    df_other_tmp.append(df_tmp)
            df_10 = pd.concat(df_10_tmp, ignore_index=True)
            # df_20 = pd.concat(df_20_tmp, ignore_index=True)
            df_30 = pd.concat(df_30_tmp, ignore_index=True)
            df_other = pd.concat(df_other_tmp, ignore_index=True)

            # return df_10, df_20, df_30, df_other
            return df_10, df_30, df_other

        elif self.mode == 'test':
            df_tmp = []
            for idx, csv_path in tqdm(enumerate(self.read_list)):
                df_tmp.append(pd.read_csv(csv_path, header=0))
            df = pd.concat(df_tmp, ignore_index=True)

            return df
