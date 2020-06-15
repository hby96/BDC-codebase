# import os
# import numpy as np
#
#
# save_root_path = '/home/ecg/Documents/Dataset/Huawei_Big_Data/chusai/split_valid_0613_feature'
#
# file_list = os.listdir(save_root_path)
# # file_list = [file.split('_')[0] for file in file_list]
#
# print(len(file_list))
#
# final_file_list = []
# np.random.shuffle(file_list)
# for file in file_list:
#     if file.split('_')[0] not in final_file_list:
#         final_file_list.append(file.split('_')[0])
#     else:
#         os.remove(os.path.join(save_root_path, file))
#         continue
#
# print(len(os.listdir(save_root_path)))

import numpy as np


def get_rad(d):
    return d * np.pi / 180.0


def get_distance(A, B):
    LatA, LonA = A[0], A[1]
    LatB, LonB = B[0], B[1]
    EARTH_RADIUS = 6378.137  # 千米
    radLatA = get_rad(LatA)
    radLatB = get_rad(LatB)
    a = radLatA - radLatB
    b = get_rad(LonA) - get_rad(LonB)
    s = 2 * np.arcsin(np.sqrt(np.power(np.sin(a / 2), 2) + np.cos(radLatA) * np.cos(radLatB)*np.power(np.sin(b / 2), 2)))
    s = s * EARTH_RADIUS
    #  保留两位小数
    s = np.round(s * 100)/100
    s = s * 1000  # 转换成m
    return s


A = (30, 0)
B = (30, -179)

print(get_distance(A, B))
