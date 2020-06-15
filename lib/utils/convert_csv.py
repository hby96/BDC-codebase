import csv

ori_csv_path = '/Users/hby/Documents/XJTU/场景生成与离线测试/Projects/draw_hist/ordernum.csv'
save_csv_path = '/Users/hby/Documents/XJTU/场景生成与离线测试/Projects/draw_hist/after_ordernum.csv'


save_list = []

with open(ori_csv_path, 'r') as f:
    reader = csv.reader(f)
    for idx, row in enumerate(reader):
        print(idx)
        row[1] = str(int(row[1]))
        save_list.append(row)

with open(save_csv_path, 'a+') as sf:
    f_csv = csv.writer(sf)
    f_csv.writerows(save_list)


