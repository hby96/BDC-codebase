from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error,explained_variance_score

import lightgbm as lgb
import numpy as np


class Base():
    def __init__(self, meta):
        self.train = meta['train']
        self.test = meta['test']
        self.valid = meta['valid']
        self.pred = meta['pred']
        self.label = meta['label']
        self.seed = meta['seed']
        self.is_shuffle = meta['is_shuffle']
        self.n_splits = 10
        self.fold = KFold(n_splits=self.n_splits, shuffle=self.is_shuffle, random_state=self.seed)
        self.kf_way = self.fold.split(self.train[self.pred])
        self.params = {
            'learning_rate': 0.01,
            'boosting_type': 'gbdt',
            'objective': 'regression',
            'num_leaves': 36,
            'feature_fraction': 0.6,
            'bagging_fraction': 0.7,
            'bagging_freq': 6,
            'seed': 8,
            'bagging_seed': 1,
            'feature_fraction_seed': 7,
            'min_data_in_leaf': 20,
            'nthread': 8,
            'verbose': 1,
        }

    def mse_score_eval(self, preds, valid):
        labels = valid.get_label()
        scores = mean_squared_error(y_true=labels, y_pred=preds)
        return 'mse_score', scores, True

    def do_train(self):
        train_pred = np.zeros((self.train.shape[0],))
        valid_pred = np.zeros((self.valid.shape[0],))
        test_pred = np.zeros((self.test.shape[0],))
        for n_fold, (train_idx, valid_idx) in enumerate(self.kf_way, start=1):
            train_x, train_y = self.train[self.pred].iloc[train_idx], self.train[self.label].iloc[train_idx]
            valid_x, valid_y = self.train[self.pred].iloc[valid_idx], self.train[self.label].iloc[valid_idx]
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
            # print(valid_x.values[:10])
            # print(valid_y.values[:10])
            # assert False
            # print(self.valid[self.pred].shape)
            # print(self.test[self.pred].shape)
            # print(self.valid[self.pred].iloc[0])
            # print(self.test[self.pred].iloc[0])
            # assert False
            # valid_pred += clf.predict(self.valid[self.pred], num_iteration=clf.best_iteration) / self.fold.n_splits
            test_pred += clf.predict(self.test[self.pred], num_iteration=clf.best_iteration) / self.fold.n_splits


        self.test['output'] = test_pred
        # _, final_mse, _ = mean_squared_error(y_pred=test_pred.tolist(), y_true=self.test['label'].values.tolist())

        return self.test[['loadingOrder', 'output']]
