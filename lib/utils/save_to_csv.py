import pandas as pd


def save_to_csv(result, test_data, cfg):

    test_data = pd.read_csv('/home/ecg/Documents/Dataset/Huawei_Big_Data/chusai/A_testData0531.csv')
    test_data['timestamp'] = pd.to_datetime(test_data['timestamp'], infer_datetime_format=True)
    test_data['onboardDate'] = pd.to_datetime(test_data['onboardDate'], infer_datetime_format=True)

    test_data = test_data.merge(result, on='loadingOrder', how='left')
    test_data['ETA'] = (test_data['onboardDate'] + test_data['output'].apply(lambda x: pd.Timedelta(seconds=x))).apply(
        lambda x: x.strftime('%Y/%m/%d  %H:%M:%S'))

    test_data.drop(['direction', 'TRANSPORT_TRACE'], axis=1, inplace=True)
    test_data['onboardDate'] = test_data['onboardDate'].apply(lambda x: x.strftime('%Y/%m/%d  %H:%M:%S'))
    test_data['creatDate'] = pd.datetime.now().strftime('%Y/%m/%d  %H:%M:%S')
    # test_data['timestamp'] = test_data['temp_timestamp']
    test_data['timestamp'] = test_data['timestamp'].apply(lambda x: x.strftime('%Y-%m-%dT%H:%M:%S.000Z'))
    # 整理columns顺序
    result = test_data[
        ['loadingOrder', 'timestamp', 'longitude', 'latitude', 'carrierName', 'vesselMMSI', 'onboardDate', 'ETA',
         'creatDate']]
    result.to_csv(cfg.CSV_SAVE_PATH, index=False)


# def save_to_csv(result, test_data, cfg):
#
#     test_data = pd.read_csv('/home/ecg/Documents/Dataset/Huawei_Big_Data/chusai/A_testData0531.csv')
#     test_data['timestamp'] = pd.to_datetime(test_data['timestamp'], infer_datetime_format=True)
#     test_data['onboardDate'] = pd.to_datetime(test_data['onboardDate'], infer_datetime_format=True)
#
#     test_data = test_data.merge(result, on='loadingOrder', how='left')
#     test_data['time_min'] = test_data.groupby('loadingOrder')['timestamp'].transform('min')
#     test_data['ETA'] = (test_data['time_min'] + test_data['output'].apply(lambda x: pd.Timedelta(seconds=x))).apply(
#         lambda x: x.strftime('%Y/%m/%d  %H:%M:%S'))
#
#     test_data.drop(['direction', 'TRANSPORT_TRACE'], axis=1, inplace=True)
#     test_data['onboardDate'] = test_data['onboardDate'].apply(lambda x: x.strftime('%Y/%m/%d  %H:%M:%S'))
#     test_data['creatDate'] = pd.datetime.now().strftime('%Y/%m/%d  %H:%M:%S')
#     # test_data['timestamp'] = test_data['temp_timestamp']
#     test_data['timestamp'] = test_data['timestamp'].apply(lambda x: x.strftime('%Y-%m-%dT%H:%M:%S.000Z'))
#     # 整理columns顺序
#     result = test_data[
#         ['loadingOrder', 'timestamp', 'longitude', 'latitude', 'carrierName', 'vesselMMSI', 'onboardDate', 'ETA',
#          'creatDate']]
#     result.to_csv(cfg.CSV_SAVE_PATH, index=False)

