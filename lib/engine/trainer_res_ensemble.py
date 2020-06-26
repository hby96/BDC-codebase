from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, explained_variance_score

import lightgbm as lgb
import xgboost as xgb
import numpy as np

import pandas as pd


class ResEnsemble():
    def __init__(self, meta, cfg):
        self.train = meta['train']
        self.test = meta['test']
        self.pred = meta['pred']
        self.label = meta['label']
        self.seed = meta['seed']
        self.is_shuffle = meta['is_shuffle']
        self.n_splits = 10
        self.fold = KFold(n_splits=self.n_splits, shuffle=self.is_shuffle, random_state=self.seed)
        self.kf_way = self.fold.split(self.train[self.pred])
        self.trace_time_info = self.get_trace_time(cfg.DATASET.TRACE_TIME_PATH)
        self.lgb_params = {
            'learning_rate': 0.1,
            'boosting_type': 'gbdt',
            'objective': 'regression',
            'num_leaves': 32,
            'nthread': 8,
            'verbose': 1,
            'metric': 'mse',
        }
        self.xgb_params = {
            'booster': 'gbtree',
            'objective': 'reg:gamma',  # 回归问题
            'gamma': 0.1,  # 用于控制是否后剪枝的参数,越大越保守，一般0.1、0.2这样子。
            'max_depth': 12,  # 构建树的深度，越大越容易过拟合
            'lambda': 2,  # 控制模型复杂度的权重值的L2正则化项参数，参数越大，模型越不容易过拟合。
            'subsample': 0.7,  # 随机采样训练样本
            'colsample_bytree': 0.7,  # 生成树时进行的列采样
            'min_child_weight': 3,
            # 这个参数默认是 1，是每个叶子里面 h 的和至少是多少，对正负样本不均衡时的 0-1 分类而言
            # ，假设 h 在 0.01 附近，min_child_weight 为 1 意味着叶子节点中最少需要包含 100 个样本。
            # 这个参数非常影响结果，控制叶子节点中二阶导的和的最小值，该参数值越小，越容易 overfitting。
            'silent': 0,  # 设置成1则没有运行信息输出，最好是设置为0.
            'eta': 0.1,  # 如同学习率
            'seed': 1000,
            'nthread': 7,  # cpu 线程数
            # 'eval_metric': 'auc'
        }

    def get_trace_time(self, test_trace_path):
        port_info_dict = dict()
        port_df = pd.read_csv(test_trace_path, header=0)
        for i in range(port_df.shape[0]):
            time = float(port_df.iloc[i]['time'])
            port_info_dict[port_df.iloc[i]['TRANSPORT_TRACE']] = time  # * 24 * 3600
        return port_info_dict

    def mse_score_eval(self, preds, valid):
        labels = valid.get_label()
        scores = mean_squared_error(y_true=labels / 3600, y_pred=preds / 3600)
        return 'mse_score', scores, True

    def train_lgb(self):
        train_pred = np.zeros((self.train.shape[0],))
        lgb_test_pred = np.zeros((self.test.shape[0],))
        for n_fold, (train_idx, valid_idx) in enumerate(self.kf_way, start=1):
            train_x, train_y = self.train[self.pred].iloc[train_idx].copy(), self.train[self.label].iloc[train_idx].copy()
            valid_x, valid_y = self.train[self.pred].iloc[valid_idx].copy(), self.train[self.label].iloc[valid_idx].copy()
            # 数据加载
            n_train = lgb.Dataset(train_x, label=train_y)
            n_valid = lgb.Dataset(valid_x, label=valid_y)

            lgb_model = lgb.train(
                params=self.lgb_params,
                train_set=n_train,
                num_boost_round=3000,
                valid_sets=[n_valid],
                early_stopping_rounds=100,
                verbose_eval=100,
                # feval=self.mse_score_eval
            )
            train_pred[valid_idx] = lgb_model.predict(valid_x, num_iteration=lgb_model.best_iteration)
            lgb_test_pred += lgb_model.predict(self.test[self.pred].copy(), num_iteration=lgb_model.best_iteration) / self.fold.n_splits
        for i in range(lgb_test_pred.shape[0]):
            trace = self.test.iloc[i]['TRANSPORT_TRACE']
            mean_time = self.trace_time_info[trace]
            lgb_test_pred[i] += mean_time
        return lgb_test_pred

    def train_xgb(self):
        xgb_test_pred = np.zeros((self.test.shape[0],))
        for n_fold, (train_idx, valid_idx) in enumerate(self.kf_way, start=1):
            train_x, train_y = self.train[self.pred].iloc[train_idx].copy(), self.train[self.label].iloc[train_idx].copy()
            # 数据加载
            min_train_y = train_y.min()
            new_train_y = train_y - min_train_y
            n_train = xgb.DMatrix(train_x, label=new_train_y)
            xgb_model = xgb.train(
                self.xgb_params, n_train,
                num_boost_round=2000,
            )
            xgb_test_pred += (xgb_model.predict(xgb.DMatrix(self.test[self.pred]).copy()) + min_train_y) / self.fold.n_splits
        for i in range(xgb_test_pred.shape[0]):
            trace = self.test.iloc[i]['TRANSPORT_TRACE']
            mean_time = self.trace_time_info[trace]
            xgb_test_pred[i] += mean_time
        return xgb_test_pred

    def do_train(self):

        lgb_test_pred = self.train_lgb()
        xgb_test_pred = self.train_xgb()

        self.test['output'] = (lgb_test_pred + xgb_test_pred) / 2

        return self.test[['loadingOrder', 'output']]
