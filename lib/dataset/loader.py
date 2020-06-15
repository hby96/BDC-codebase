import pandas as pd
# import modin.pandas as pd
import numpy as np
from tqdm import tqdm
import sys
import os

# class Base():
#     def __init__(self, mode, cfg):
#         assert mode == 'train' or mode == 'test'
#
#         self.mode = mode
#         self.nrow = cfg.DATASET.LOADER.NROWS
#         self.columns = ['loadingOrder','carrierName','timestamp','longitude',
#                   'latitude','vesselMMSI','speed','direction','vesselNextport',
#                   'vesselNextportETA','vesselStatus','vesselDatasource','TRANSPORT_TRACE']
#         if self.mode == 'train':
#             if self.nrow == 0:
#                 self.data = pd.read_csv(cfg.DATASET.TRAIN_GPS_PATH, header=None)
#             else:
#                 self.data = pd.read_csv(cfg.DATASET.TRAIN_GPS_PATH, nrows=self.nrow, header=None)
#             self.data.columns = self.columns
#         else:
#             self.data = pd.read_csv(cfg.DATASET.TEST_DATA_PATH)
#
#     def get_data(self):
#         if self.mode == 'train':
#             self.data['vesselNextportETA'] = pd.to_datetime(self.data['vesselNextportETA'], infer_datetime_format=True)
#         elif self.mode == 'test':
#             self.data['temp_timestamp'] = self.data['timestamp']
#             self.data['onboardDate'] = pd.to_datetime(self.data['onboardDate'], infer_datetime_format=True)
#         self.data['timestamp'] = pd.to_datetime(self.data['timestamp'], infer_datetime_format=True)
#         self.data['longitude'] = self.data['longitude'].astype(float)
#         self.data['loadingOrder'] = self.data['loadingOrder'].astype(str)
#         self.data['latitude'] = self.data['latitude'].astype(float)
#         self.data['speed'] = self.data['speed'].astype(float)
#         self.data['direction'] = self.data['direction'].astype(float)
#
#         return self.data


def Base(mode, cfg):
    assert mode == 'train' or mode == 'test'

    nrow = cfg.DATASET.LOADER.NROWS
    columns = ['loadingOrder', 'carrierName', 'timestamp', 'longitude',
              'latitude', 'vesselMMSI', 'speed', 'direction', 'vesselNextport',
              'vesselNextportETA', 'vesselStatus', 'vesselDatasource', 'TRANSPORT_TRACE']
    if mode == 'train':
        if nrow == -1:
            # data = pd.read_csv(cfg.DATASET.TRAIN_GPS_PATH, header=None)

            print('start')
            # chunk_size = 10 ** 5
            # data = pd.concat([chunk for chunk in pd.read_csv(cfg.DATASET.TRAIN_GPS_PATH, chunksize=chunk_size)],
            #                ignore_index=False)

            # train_flux = pd.read_csv(cfg.DATASET.TRAIN_GPS_PATH, chunksize=100000, names=columns)
            # # train_flux = pd.read_csv(cfg.DATASET.TRAIN_GPS_PATH, chunksize=10000, nrows=100000, names=columns)
            # # print(train_flux[0].head())
            # # assert False
            # count = 0
            # loadingOrders = []  # 可以用来记录读取了多少个loadingOrders
            # data = pd.DataFrame(columns=columns)
            # # data = pd.DataFrame()
            # # data = []
            # for batch_df in tqdm(train_flux):
            #     # 这部分可以是你对data的处理
            #     data = data.append(batch_df)
            # # data = pd.concat(data)

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
            na_vals = ["\\N", " ", "", "NULL"]
            # train_flux = pd.read_csv(cfg.DATASET.TRAIN_GPS_PATH, header=None, sep=',', chunksize=200000, nrows=1000000,
            #                          error_bad_lines=False, delimiter="\t", lineterminator="\n",
            #                          keep_default_na=True, na_values=na_vals)
            train_flux = pd.read_csv(cfg.DATASET.TRAIN_GPS_PATH, header=None, names=columns, chunksize=400000)# converters=converters)
                                     # keep_default_na=True, na_values=na_vals, )
            df_tmp = []
            df_train = pd.DataFrame(columns=columns)
            for chunk in tqdm(train_flux):
                df_tmp.append(chunk)
                del chunk
                # print("the mem-cost is now: ", str(sys.getsizeof(df_tmp) / (1)), "MB \n")
            idx = 0
            # data = pd.concat(df_tmp)
            for i in range(len(df_tmp)):
                tmp = pd.DataFrame(data=df_tmp[idx])
                df_train = pd.concat([df_train, tmp], ignore_index=True)
                del df_tmp[idx], tmp
                print(i)
                # print("the frame size is: ", df_train.memory_usage().sum() / (1024 ** 2), "MB")
            print(df_train.head())
            print(data.head())
            print('have done')
            assert False
            # data = pd.read_csv(cfg.DATASET.TRAIN_GPS_PATH, nrows=100000, names=columns, header=None)
        else:
            data = pd.read_csv(cfg.DATASET.TRAIN_GPS_PATH, nrows=nrow, header=None)

        data.columns = columns
    else:
        data = pd.read_csv(cfg.DATASET.TEST_DATA_PATH)

    if mode == 'train':
        data['vesselNextportETA'] = pd.to_datetime(data['vesselNextportETA'], infer_datetime_format=True)
    elif mode == 'test':
        data['temp_timestamp'] = data['timestamp']
        data['onboardDate'] = pd.to_datetime(data['onboardDate'], infer_datetime_format=True)
    # print(data['timestamp'].head())
    data['timestamp'] = pd.to_datetime(data['timestamp'], infer_datetime_format=True)
    # print(data['timestamp'].head())
    data['longitude'] = data['longitude'].astype(float)
    data['loadingOrder'] = data['loadingOrder'].astype(str)
    data['latitude'] = data['latitude'].astype(float)
    data['speed'] = data['speed'].astype(float)
    data['direction'] = data['direction'].astype(float)

    return data


