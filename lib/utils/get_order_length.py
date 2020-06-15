import os
import csv

root_path = '/home/ecg/Documents/Dataset/Huawei_Big_Data/chusai/split_train_wo_duplicate/'
length_save_path = '/home/ecg/Documents/Projects/HuaWei_Big_Data/Demo/order_length_wo_duplicate.csv'

order_files = os.listdir(root_path)

print(len(order_files))

for idx, order in enumerate(order_files):
    print(idx)
    num = 0
    with open(os.path.join(root_path, order), 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            num += 1
    with open(length_save_path, 'a+') as sf:
        f_csv = csv.writer(sf)
        f_csv.writerow([order, num])
