from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error,explained_variance_score

import lightgbm as lgb
import numpy as np


class MultiTime():
    def __init__(self, meta):
        self.train_10 = meta['train_10']
        # self.train_20 = meta['train_20']
        self.train_30 = meta['train_30']
        self.train_other = meta['train_other']
        self.test = meta['test']
        self.pred = meta['pred']
        self.label = meta['label']
        self.seed = meta['seed']
        self.is_shuffle = meta['is_shuffle']
        self.n_splits = 10
        self.fold = KFold(n_splits=self.n_splits, shuffle=self.is_shuffle, random_state=self.seed)
        self.kf_way_10 = self.fold.split(self.train_10[self.pred])
        # self.kf_way_20 = self.fold.split(self.train_20[self.pred])
        self.kf_way_30 = self.fold.split(self.train_30[self.pred])
        self.kf_way_other = self.fold.split(self.train_other[self.pred])
        self.params = {
            'learning_rate': 1,
            'boosting_type': 'gbdt',
            'objective': 'regression',
            'num_leaves': 32,
            # 'feature_fraction': 0.6,
            # 'bagging_fraction': 0.7,
            # 'bagging_freq': 6,
            # 'seed': 8,
            # 'bagging_seed': 1,
            # 'feature_fraction_seed': 7,
            # 'min_data_in_leaf': 20,
            'nthread': 8,
            'verbose': 1,
            # 'metric': 'mse',
        }

    def mse_score_eval(self, preds, valid):
        labels = valid.get_label()
        scores = mean_squared_error(y_true=labels / 3600, y_pred=preds / 3600)
        return 'mse_score', scores, True

    def train_model(self, train, test, kf_way, number):
        train_pred = np.zeros((train.shape[0],))
        test_pred = np.zeros((test.shape[0],))
        for n_fold, (train_idx, valid_idx) in enumerate(kf_way, start=1):
            train_x, train_y = train[self.pred].iloc[train_idx], train[self.label].iloc[train_idx]
            valid_x, valid_y = train[self.pred].iloc[valid_idx], train[self.label].iloc[valid_idx]
            # 数据加载
            n_train = lgb.Dataset(train_x, label=train_y)
            n_valid = lgb.Dataset(valid_x, label=valid_y)
            clf = lgb.train(
                params=self.params,
                train_set=n_train,
                num_boost_round=3000,
                valid_sets=[n_valid],
                early_stopping_rounds=100,
                verbose_eval=100,
                feval=self.mse_score_eval
            )
            train_pred[valid_idx] = clf.predict(valid_x, num_iteration=clf.best_iteration)
            # test_pred += clf.predict(self.test[self.pred], num_iteration=clf.best_iteration) / self.fold.n_splits
            if number == '10':
                for idx in range(test_pred.shape[0]):
                    day = self.test.iloc[idx]['day']
                    if day <= 10:
                        test_pred[idx] += clf.predict(self.test[self.pred].iloc[idx], num_iteration=clf.best_iteration)[0] / self.fold.n_splits
            # elif number == '20':
            #     for idx in range(test_pred.shape[0]):
            #         day = self.test.iloc[idx]['day']
            #         if (day > 10) and (day <= 20):
            #             test_pred[idx] += clf.predict(self.test[self.pred].iloc[idx], num_iteration=clf.best_iteration)[0] / self.fold.n_splits
            elif number == '30':
                for idx in range(test_pred.shape[0]):
                    day = self.test.iloc[idx]['day']
                    if (day > 10) and (day <= 25):
                        test_pred[idx] += clf.predict(self.test[self.pred].iloc[idx], num_iteration=clf.best_iteration)[0] / self.fold.n_splits
            elif number == 'other':
                for idx in range(test_pred.shape[0]):
                    day = self.test.iloc[idx]['day']
                    if day > 25:
                        test_pred[idx] += clf.predict(self.test[self.pred].iloc[idx], num_iteration=clf.best_iteration)[0] / self.fold.n_splits
        return test_pred

    def do_train(self):

        result_10 = self.train_model(self.train_10, self.test, self.kf_way_10, '10')
        # result_20 = self.train_model(self.train_20, self.test, self.kf_way_20, '20')
        result_30 = self.train_model(self.train_30, self.test, self.kf_way_30, '30')
        result_other = self.train_model(self.train_other, self.test, self.kf_way_other, 'other')

        # self.test['output'] = result_10 + result_20 + result_30 + result_other
        self.test['output'] = result_10 + result_30 + result_other

        # return self.test[['loadingOrder', 'output', 'day']]
        return self.test[['loadingOrder', 'output']]


    # def do_train(self):
    #     for i in range(4):
    #         if i == 0:
    #             train, test, kf_way = self.train_10, self.test, self.kf_way_10
    #         elif i == 1:
    #             train, test, kf_way = self.train_20, self.test, self.kf_way_20
    #         elif i == 2:
    #             train, test, kf_way = self.train_30, self.test, self.kf_way_30
    #         elif i == 3:
    #             train, test, kf_way = self.train_other, self.test, self.kf_way_other
    #
    #         train_pred = np.zeros((train.shape[0],))
    #         test_pred = np.zeros((test.shape[0],))
    #         for n_fold, (train_idx, valid_idx) in enumerate(kf_way, start=1):
    #             train_x, train_y = train[self.pred].iloc[train_idx], train[self.label].iloc[train_idx]
    #             valid_x, valid_y = train[self.pred].iloc[valid_idx], train[self.label].iloc[valid_idx]
    #             # 数据加载
    #             n_train = lgb.Dataset(train_x, label=train_y)
    #             n_valid = lgb.Dataset(valid_x, label=valid_y)
    #
    #             if i == 0:
    #                 clf_0 = lgb.train(params=self.params, train_set=n_train, num_boost_round=3000, valid_sets=[n_valid],
    #                           early_stopping_rounds=100, verbose_eval=100, feval=self.mse_score_eval)
    #                 for idx in range(test_pred.shape[0]):
    #                     day = self.test.iloc[idx]['day']
    #                     if day <= 10:
    #                         test_pred[idx] += clf_0.predict(self.test[self.pred].iloc[idx], num_iteration=clf_0.best_iteration)[0] / self.fold.n_splits
    #             elif i == 1:
    #                 clf_1 = lgb.train(params=self.params, train_set=n_train, num_boost_round=3000, valid_sets=[n_valid],
    #                           early_stopping_rounds=100, verbose_eval=100, feval=self.mse_score_eval)
    #                 for idx in range(test_pred.shape[0]):
    #                     day = self.test.iloc[idx]['day']
    #                     if (day > 10) and (day <= 20):
    #                         test_pred[idx] += clf_1.predict(self.test[self.pred].iloc[idx], num_iteration=clf_1.best_iteration)[0] / self.fold.n_splits
    #             elif i == 2:
    #                 clf_2 = lgb.train(params=self.params, train_set=n_train, num_boost_round=3000, valid_sets=[n_valid],
    #                           early_stopping_rounds=100, verbose_eval=100, feval=self.mse_score_eval)
    #                 for idx in range(test_pred.shape[0]):
    #                     day = self.test.iloc[idx]['day']
    #                     if (day > 20) and (day <= 30):
    #                         test_pred[idx] += clf_2.predict(self.test[self.pred].iloc[idx], num_iteration=clf_2.best_iteration)[0] / self.fold.n_splits
    #             elif i == 3:
    #                 clf_3 = lgb.train(params=self.params, train_set=n_train, num_boost_round=3000, valid_sets=[n_valid],
    #                           early_stopping_rounds=100, verbose_eval=100, feval=self.mse_score_eval)
    #                 for idx in range(test_pred.shape[0]):
    #                     day = self.test.iloc[idx]['day']
    #                     if day > 30:
    #                         print(clf_3.predict(self.test[self.pred].iloc[idx], num_iteration=clf_3.best_iteration)[0])
    #                         print(clf_3.predict(self.test[self.pred].iloc[idx], num_iteration=clf_3.best_iteration)[0] / self.fold.n_splits)
    #                         assert False
    #                         test_pred[idx] += clf_3.predict(self.test[self.pred].iloc[idx], num_iteration=clf_3.best_iteration)[0] / self.fold.n_splits
    #
    #             # train_pred[valid_idx] = clf.predict(valid_x, num_iteration=clf.best_iteration)
    #             # print(train_pred.shape)
    #             # test_pred += clf.predict(test[self.pred], num_iteration=clf.best_iteration) / self.fold.n_splits
    #
    #     # for idx in range(test_pred.shape[0]):
    #     #     day = self.test.iloc[idx]['day']
    #     #
    #     #     if day <= 10:
    #     #         test_pred[idx] = clf_0.predict(self.test[self.pred].iloc[idx], num_iteration=clf_0.best_iteration)[0]
    #     #     elif day <= 20:
    #     #         test_pred[idx] = clf_1.predict(self.test[self.pred].iloc[idx], num_iteration=clf_1.best_iteration)[0]
    #     #     elif day <= 30:
    #     #         test_pred[idx] = clf_2.predict(self.test[self.pred].iloc[idx], num_iteration=clf_2.best_iteration)[0]
    #     #     else:
    #     #         test_pred[idx] = clf_3.predict(self.test[self.pred].iloc[idx], num_iteration=clf_3.best_iteration)[0]
    #
    #     self.test['output'] = test_pred
    #         # _, final_mse, _ = mean_squared_error(y_pred=test_pred.tolist(), y_true=self.test['label'].values.tolist())
    #
    #     return self.test[['loadingOrder', 'output']]
