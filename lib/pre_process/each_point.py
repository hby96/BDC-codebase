import pandas as pd


class EachPoint():
    def __init__(self, df, mode='train'):
        assert mode == 'train' or mode == 'test'
        self.mode = mode
        self.df = df

    def get_feature(self):
        self.df.sort_values(['loadingOrder', 'timestamp'], inplace=True)
        # 特征只选择经纬度、速度、方向
        self.df['lat_diff'] = self.df.groupby('loadingOrder')['latitude'].diff(1)
        self.df['lon_diff'] = self.df.groupby('loadingOrder')['longitude'].diff(1)
        self.df['speed_diff'] = self.df.groupby('loadingOrder')['speed'].diff(1)
        self.df['diff_minutes'] = self.df.groupby('loadingOrder')['timestamp'].diff(1).dt.total_seconds() // 60
        self.df['anchor'] = ((self.df['lat_diff'] <= 0.03) & (self.df['lon_diff'] <= 0.03) & (self.df['speed_diff'] <= 0.3) & (
                self.df['diff_minutes'] <= 10)).astype('int')
        if self.mode == 'train':
            # group_max = self.df.groupby('loadingOrder')['timestamp'].agg(mmax='max', count='count').reset_index()
            # group_max = self.df.groupby('loadingOrder')['timestamp'].agg(mmax='max', count='count').reset_index()
            self.df['max'] = self.df.groupby('loadingOrder')['timestamp'].transform('max')
            print(self.df.head())
            assert False
            group_list = [group_max.iloc[i]['loadingOrder'] for i in range(len(group_max))]
            # print(self.df['loadingOrder'].values.tolist())
            # print(type(self.df['loadingOrder'].values.tolist()))

            # print(group_list)
            # print(group_list.index(self.df['loadingOrder'].values.any()))
            # assert False
            # final_group_max = pd.DataFrame(columns=['mmax'])
            # for i in range(len(group_max)):
            #     a = group_max.loc[i]
            #     d = pd.DataFrame(a).T
            #     final_group_max = final_group_max.append([d] * group_max.loc[i]['count'])  # 每行复制24倍
            # print(final_group_max.shape)
            print(group_max.shape)
            group_max[['mmax']].values
            assert False
            self.df['label'] = (group_max.iloc[group_list.index(self.df['loadingOrder'])]['mmax'] - self.df['timestamp']).dt.total_seconds()
            print(group_max)
            print(group_max['loadingOrder'])
            assert False
            # 读取数据的最大值-最小值，即确认时间间隔为label
        assert False
        if self.mode == 'train':
            group_data = self.df.groupby('loadingOrder')['timestamp'].agg(mmax='max', count='count', mmin='min').reset_index()
            # 读取数据的最大值-最小值，即确认时间间隔为label
            group_data['label'] = (group_data['mmax'] - group_data['mmin']).dt.total_seconds()
        elif self.mode == 'test':
            group_data = self.df.groupby('loadingOrder')['timestamp'].agg(count='count').reset_index()

        anchor_data = self.df.groupby('loadingOrder')['anchor'].agg('sum').reset_index()
        anchor_data.columns = ['loadingOrder', 'anchor_cnt']
        group_data = group_data.merge(anchor_data, on='loadingOrder', how='left')
        group_data['anchor_ratio'] = group_data['anchor_cnt'] / group_data['count']

        agg_function = ['min', 'max', 'mean', 'median']
        agg_col = ['latitude', 'longitude', 'speed', 'direction']

        group = self.df.groupby('loadingOrder')[agg_col].agg(agg_function).reset_index()
        group.columns = ['loadingOrder'] + ['{}_{}'.format(i, j) for i in agg_col for j in agg_function]
        group_data = group_data.merge(group, on='loadingOrder', how='left')
        assert False
        return group_data
