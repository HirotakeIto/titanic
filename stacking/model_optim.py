#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 24 15:30:33 2017

@author: hirotake_ito
"""
import scipy
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
# import hyperopt
from hyperopt import fmin, space_eval, hp, Trials, tpe


class ModelOptimization():
    '''
    sample)
    data = pd.read_csv('./data/analysis_data.csv')
    data = data.iloc[:,2:]
    data = data.fillna(data.mean())
    data = data.values
    y = data[:,0].astype(int)
    x = data[:,1:]
    sm = SMOTE(random_state=42)
    x,y = sm.fit_sample(x,y)
    index_ = np.arange(len(x))
    np.random.shuffle(index_)
    x = x[index_]
    y = y[index_]
    RC = RandomForestClassifier()
    m = model_optim.ModelOptimization(model=RC, X=x, Y=y)
    m.set_model_space()
    # 何をターゲットにして最適化するか
    # precision は Trueラベルを与えた人間のうち、何人が本当にTrueだったか。
    # recall Trueの人を何人探すことができたか
    p = make_scorer(precision_score,pos_label=1)
    m.change_model_setting(p)
    m.optimization()
    # print...
    # {'min_samples_split': 0.05110661868217986, 'n_jobs': -1, 'max_depth': 15, 'n_estimators': 187, 'min_samples_leaf': 0.008395269301557543}

    '''

    def __init__(self,
                 model,
                 X,
                 Y):
        self.model = model
        self.X = X
        self.Y = Y

    def set_model_space(self):
        model = self.model
        if isinstance(model, RandomForestClassifier):
            print('this model is....RandomForestClassifier!')
            n_features = self.X.shape[1]
            self.model_param_dict = {
                'n_estimators':
                    hp.choice('n_estimators', range(2, 100)),
                'max_depth':
                    hp.choice('max_depth', range(1, 20)),
                'min_samples_leaf':
                    hp.uniform('min_samples_leaf', 0.01, 0.5),
                'min_samples_split':
                    hp.uniform('min_samples_split', 0.01, 1),
                'max_features':
                    hp.choice('max_features', range(1, n_features)),
                # 'oob_score':
                #     hp.choice('oob_score', [False, True]),
                #                'min_weight_fraction_leaf':
                #                    hp.uniform('min_weight_fraction_leaf', 0, 1),
                'n_jobs': -1}
            self.how_scoring = 'r2'
        if isinstance(model, GradientBoostingClassifier):
            print('this model is....GradientBoostingClassifier!')
            self.model_param_dict = {
                'n_estimators':
                    hp.choice('n_estimators', range(80, 200)),
                'max_depth':
                    hp.choice('max_depth', range(1, 20)),
                'min_samples_leaf':
                    hp.uniform('min_samples_leaf', 0.01, 0.5),
                'min_samples_split':
                    hp.uniform('min_samples_split', 0.01, 0.5),
                'learning_rate':
                    hp.uniform('learning_rate', 0.01, 0.5),
                'subsample':
                    hp.uniform('subsample', 0.5, 1),
            }
            self.how_scoring = 'r2'

    def change_model_setting(self, scoring):
        self.how_scoring = scoring

    def optimization_model(self,
                           space,
                           **args):
        model = self.model
        model = model.set_params(**space)
        scoring = cross_val_score(model, self.X, self.Y,scoring=self.how_scoring, cv=5, n_jobs=-1)
        scoring = scipy.mean(scoring)
        return -scoring

    def optimization(self):
        trials = Trials()
        best = fmin(fn=self.optimization_model,
                    space=self.model_param_dict,
                    algo=tpe.suggest,
                    max_evals=150, trials=trials)
        self.best = space_eval(self.model_param_dict, best)
        print(self.best)
        return self.best



