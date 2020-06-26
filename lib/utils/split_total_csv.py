# import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import csv


ori_csv_path = '/home/ecg/Documents/Dataset/Huawei_Big_Data/chusai/A_testData0531.csv'
save_root_path = '/home/ecg/Documents/Dataset/Huawei_Big_Data/chusai/split_test/'

# columns = [
#     'loadingOrder',
#     'carrierName',
#     'timestamp',
#     'longitude',
#     'latitude',
#     'vesselMMSI',
#     'speed',
#     'direction',
#     'vesselNextport',
#     'vesseNextportETA',
#     'vesselStatus',
#     'vesselDatasource',
#     'TRANSPORT_TRACE'
# ]
columns = [
    'loadingOrder',
    'timestamp',
    'longitude',
    'latitude',
    'speed',
    'direction',
    'carrierName',
    'vesselMMSI',
    'onboardDate',
    'TRANSPORT_TRACE'
]

with open(ori_csv_path, 'r') as f:
    reader = csv.reader(f)
    # assert False
    for idx, row in enumerate(reader):
        print(idx)
        order_name = row[0]
        save_path = os.path.join(save_root_path, order_name + '.csv')
        if os.path.exists(save_path):
            with open(save_path, 'a+') as sf:
                f_csv = csv.writer(sf)
                f_csv.writerow(row)

        else:
            with open(save_path, 'a+') as sf:
                f_csv = csv.writer(sf)
                f_csv.writerow(columns)
                f_csv.writerow(row)

