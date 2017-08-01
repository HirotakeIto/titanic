#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 24 15:22:45 2017

@author: hirotake_ito
"""
import os
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from .stacked_generalization import StackedGeneralization

file_directory = os.path.dirname(__file__)


class Generalizer:
    def __init__(self):
        self.model = None

    def name(self):
        raise NotImplementedError

    def guess_partial(self, sg):
        assert (isinstance(sg, StackedGeneralization))
        layer_predict = np.zeros(sg.train_target.shape)  # 結果の入れ物
        for train_index, test_index in sg.skf.split(sg.train_target):
            self.train(sg.train_data[train_index, :],
                       sg.train_target[train_index])
            layer_predict[test_index] = self.predict(sg.train_data[test_index, :])
        return layer_predict

    def guess_whole(self, sg):
        assert (isinstance(sg, StackedGeneralization))
        self.guess(sg.train_data, sg.train_target)
        return

    def guess(self, input_data, input_target):
        return self.train(input_data, input_target)

    def train(self, data, taisyou):
        raise NotImplementedError

    def predict(self, data):
        raise NotImplementedError

    def score(self, x, y_true):
        pred = self.predict(x)
        return mean_squared_error(y_true, pred)

    @staticmethod  # self使わない&クラス内で定義
    def load_partial(name):
        return pd.read_pickle(file_directory + '/trained_model/layer_' + 'partial_' + name )

    @staticmethod
    def load_whole(name):
        return pd.read_pickle(file_directory + '/trained_model/layer_' + 'whole_' + name )

    @staticmethod
    def save_partial(name, prediction):
        prediction.to_pickle(file_directory + '/trained_model/layer_' + 'partial_' + name )

    @staticmethod
    def save_whole(name, prediction):
        prediction.to_pickle(file_directory + '/trained_model/layer_' + 'whole_' + name )


class RFRegressor(Generalizer):
    def __init__(self, **argv):
        super().__init__()
        self.model = RandomForestRegressor(**argv)

    def name(self):
        return "RFRegressor"

    def train(self, data, taisyou, **argv):
        self.model = self.model.fit(data, taisyou)

    def predict(self, data):
        return self.model.predict(data)


class RFClassifier(Generalizer):
    def __init__(self, **argv):
        super().__init__()
        self.model = RandomForestClassifier(**argv)

    def name(self):
        return "RFClassifier"

    def train(self, data, taisyou, **argv):
        self.model = self.model.fit(data, taisyou)

    def predict(self, data):
        return self.model.predict(data)


class RFClassifier2(Generalizer):
    """
    予測値として連続の確率値を返すクラス
    """
    def __init__(self, **argv):
        super().__init__()
        self.model = RandomForestClassifier(**argv)

    def name(self):
        return "RFClassifier"

    def train(self, data, taisyou, **argv):
        self.model = self.model.fit(data, taisyou)

    def predict(self, data):
        return self.model.predict_proba(data)


class GBRegressor(Generalizer):
    def __init__(self, **argv):
        super().__init__()
        self.model = GradientBoostingRegressor(learning_rate=0.1,
                                               max_depth=3,
                                               random_state=0,
                                               **argv)

    def name(self):
        return "GBRegressor"

    def train(self, data, taisyou, ):
        self.model = self.model.fit(data, taisyou)

    def predict(self, data):
        return self.model.predict(data)


class GBClassifier(Generalizer):
    def __init__(self, **argv):
        super().__init__()
        self.model = GradientBoostingClassifier(**argv)

    def name(self):
        return "GBClassifier"

    def train(self, data, taisyou, **argv):
        self.model = self.model.fit(data, taisyou)

    def predict(self, data):
        return self.model.predict(data)


class LRRegressor(Generalizer):
    def __init__(self, **argv):
        super().__init__()
        self.model = LinearRegression(**argv)

    def name(self):
        return "LRRegressor"

    def train(self, data, taisyou):
        self.model = self.model.fit(data, taisyou)

    def predict(self, data):
        return self.model.predict(data)


class LRClassifier(Generalizer):
    def __init__(self, **argv):
        super().__init__()
        self.model = LogisticRegression(**argv)

    def name(self):
        return "LogisticRegression"

    def train(self, data, taisyou):
        self.model = self.model.fit(data, taisyou)

    def predict(self, data):
        return self.model.predict(data)

