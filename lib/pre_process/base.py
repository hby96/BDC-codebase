class Base():
    def __init__(self, data, mode='train'):
        assert mode == 'train' or mode == 'test'
        self.mode = mode
        self.data = data

    def get_feature(self):
        self.data.sort_values(['loadingOrder', 'timestamp'], inplace=True)
        # 特征只选择经纬度、速度\方向
        self.data['lat_diff'] = self.data.groupby('loadingOrder')['latitude'].diff(1)
        self.data['lon_diff'] = self.data.groupby('loadingOrder')['longitude'].diff(1)
        self.data['speed_diff'] = self.data.groupby('loadingOrder')['speed'].diff(1)
        self.data['diff_minutes'] = self.data.groupby('loadingOrder')['timestamp'].diff(1).dt.total_seconds() // 60
        self.data['anchor'] = self.data.apply(lambda x: 1 if x['lat_diff'] <= 0.03 and x['lon_diff'] <= 0.03
                                               and x['speed_diff'] <= 0.3 and x['diff_minutes'] <= 10 else 0, axis=1)
        if self.mode == 'train':
            group_data = self.data.groupby('loadingOrder')['timestamp'].agg(mmax='max', count='count', mmin='min').reset_index()
            # 读取数据的最大值-最小值，即确认时间间隔为label
            group_data['label'] = (group_data['mmax'] - group_data['mmin']).dt.total_seconds()
        elif self.mode == 'test':
            group_data = self.data.groupby('loadingOrder')['timestamp'].agg(count='count').reset_index()

        anchor_data = self.data.groupby('loadingOrder')['anchor'].agg('sum').reset_index()
        anchor_data.columns = ['loadingOrder', 'anchor_cnt']
        group_data = group_data.merge(anchor_data, on='loadingOrder', how='left')
        group_data['anchor_ratio'] = group_data['anchor_cnt'] / group_data['count']

        agg_function = ['min', 'max', 'mean', 'median']
        agg_col = ['latitude', 'longitude', 'speed', 'direction']

        group = self.data.groupby('loadingOrder')[agg_col].agg(agg_function).reset_index()
        group.columns = ['loadingOrder'] + ['{}_{}'.format(i, j) for i in agg_col for j in agg_function]
        group_data = group_data.merge(group, on='loadingOrder', how='left')

        return group_data
