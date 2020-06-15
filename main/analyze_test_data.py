import pandas as pd
import matplotlib.pyplot as plt


test_df = pd.read_csv('/home/ecg/Documents/Dataset/Huawei_Big_Data/chusai/A_testData0531.csv')
test_df['timestamp'] = pd.to_datetime(test_df['timestamp'], infer_datetime_format=True)

print('总共有{}艘船'.format(test_df.vesselMMSI.nunique()))
print('{}个快递运单'.format(test_df.loadingOrder.nunique()))
print('{}个运货公司'.format(test_df.carrierName.nunique()))
print('{}条运输路径'.format(test_df.TRANSPORT_TRACE.nunique()))
print('运输路段长度：{}'.format(test_df.TRANSPORT_TRACE.apply(lambda x:len(x.split('-'))).unique()))
print('运输过程中的 spped 情况：{}'.format(test_df.speed.unique()))
print('经度跨越：{}'.format(test_df.longitude.max()-test_df.longitude.min()),'纬度跨越：{}'.format(test_df.latitude.max()-test_df.latitude.min()))
print('最小经度:{}，最大经度{}'.format(test_df.longitude.min(), test_df.longitude.max()))
print('最小纬度:{}, 最大纬度{}'.format(test_df.latitude.min(), test_df.latitude.max()))
print('测试集时间跨度:min time:{} max time:{}'.format(test_df.timestamp.min(),test_df.timestamp.max()))

group_data = test_df.groupby('loadingOrder')['timestamp'].agg(mmax='max', count='count',
                                                                      mmin='min').reset_index()
# 读取数据的最大值-最小值，即确认时间间隔为label
group_data['label'] = (group_data['mmax'] - group_data['mmin']).dt.total_seconds()
print('海运时间平均值：{}， 最小值：{}， 最大值：{}'.format(group_data['label'].mean(), group_data['label'].min(), group_data['label'].max()))
print(group_data['label'].astype(int).mode())
print((group_data['label'].astype(int) == 330220).sum())
print('测试集中，船只的运输的港口：')
print(test_df.TRANSPORT_TRACE.value_counts())

plt.figure()
commutes = pd.Series(group_data['label'])
commutes.plot.hist(grid=True, bins=20, rwidth=0.9, color='#607c8e')
plt.title('Commute Times for 1,000 Commuters')
plt.xlabel('Counts')
plt.ylabel('Commute Time')
plt.grid(axis='y', alpha=0.75)
plt.show()
2