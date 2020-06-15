import os
import pandas as pd
from tqdm import tqdm
import numpy as np
import shutil

query_root_path = '/home/ecg/Documents/Dataset/Huawei_Big_Data/chusai/split_test_wo_duplicate_feature/'
query_list = os.listdir(query_root_path)
query_list = [os.path.join(query_root_path, query) for query in query_list]

gallery_root_path = '/home/ecg/Documents/Dataset/Huawei_Big_Data/chusai/useful_train_by_start_port_feature/'
gallery_list = os.listdir(gallery_root_path)
gallery_list = [os.path.join(gallery_root_path, gallery) for gallery in gallery_list]

save_root_path = '/home/ecg/Documents/Dataset/Huawei_Big_Data/chusai/split_valid_0613_feature'

columns = ['loadingOrder', 'mmax', 'count', 'mmin', 'label', 'anchor_cnt', 'anchor_ratio',
                        'latitude_min', 'latitude_max', 'latitude_mean', 'latitude_median', 'longitude_min',
                        'longitude_max', 'longitude_mean', 'longitude_median', 'speed_min', 'speed_max', 'speed_mean',
                        'speed_median' 'direction_min', 'direction_max', 'direction_mean', 'direction_median']
converters = {
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
    'csv_path': str,
}


query_tmp = []
for idx, csv_path in tqdm(enumerate(query_list)):
    tmp_df = pd.read_csv(csv_path)
    tmp_df['csv_path'] = csv_path
    query_tmp.append(tmp_df)
    del tmp_df
query_df = pd.concat(query_tmp, ignore_index=True)

gallery_tmp = []
for idx, csv_path in tqdm(enumerate(gallery_list)):
    tmp_df = pd.read_csv(csv_path)
    # tmp_df['csv_path'] = csv_path.replace('_feature/', '/')
    tmp_df['csv_path'] = csv_path
    gallery_tmp.append(tmp_df)
    del tmp_df
gallery_df = pd.concat(gallery_tmp, ignore_index=True)
gallery_df.names = columns

features = [c for c in query_df.columns if c not in ['loadingOrder', 'label', 'mmin', 'mmax', 'count', 'onboardDate', 'csv_path']]

query_set = query_df[features].values
query_mean = query_set.mean(axis=0).reshape(1, query_set.shape[1])
query_std = query_set.std(axis=0).reshape(1, query_set.shape[1])
query_set = (query_set - query_mean) / query_std

gallery_set = gallery_df[features].values
# gallery_mean = gallery_set.mean(axis=0).reshape(1, gallery_set.shape[1])
# gallery_std = gallery_set.std(axis=0).reshape(1, gallery_set.shape[1])
# gallery_set = (gallery_set - gallery_mean) / gallery_std
gallery_set = (gallery_set - query_mean) / query_std

# dis_matrix = (query_set - gallery_set) ** 2

query_2 = np.sum(query_set ** 2, axis=1, keepdims=True)
gallery_2 = np.sum(gallery_set ** 2, axis=1, keepdims=True).T
dis_matrix = query_2 + gallery_2 - 2 * np.dot(query_set, gallery_set.T)
sort_index = np.argsort(dis_matrix, axis=1)

for i in range(query_set.shape[0]):
    for j in range(100):
        src = gallery_df.iloc[sort_index[i][j]]['csv_path']
        dst = os.path.join(save_root_path, src.split('/')[-1])
        if not os.path.exists(dst):
            shutil.copy(src, dst)

print('have done')

