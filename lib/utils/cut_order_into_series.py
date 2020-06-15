import os
from tqdm import tqdm
import pandas as pd
import numpy as np


root_path = '/home/ecg/Documents/Dataset/Huawei_Big_Data/chusai/useful_train_by_anchor_start_port/'
save_root_path = '/home/ecg/Documents/Dataset/Huawei_Big_Data/chusai/useful_train_by_anchor_start_port_series/'
total_list = os.listdir(root_path)


length_range = [(10, 50), (50, 200), (200, 900), (900, 2000)]
for idx, csv_path in tqdm(enumerate(total_list)):
    for i in range(4):
        nrows = np.random.randint(length_range[i][0], length_range[i][1], 1)[0]
        try:
            df = pd.read_csv(os.path.join(root_path, csv_path), nrows=nrows)
            save_csv_path = csv_path.split('.')[0] + '_' + str(i) + '.csv'
            df.to_csv(os.path.join(save_root_path, save_csv_path))
        except:
            continue