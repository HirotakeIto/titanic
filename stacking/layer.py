import pickle
import numpy as np
import os
file_directory = os.path.dirname(__file__)

class Layer:
    def __init__(self):
        self.sg = None
        self.models = None
        self.layer_depth = None

    def train_partial(self, return_predictin=False, model_directory_path = file_directory):
        sg = self.sg
        models = self.models
        partial = np.zeros(shape=(len(sg.train_target), 0))
        for model in models:
            partial_ = model.guess_partial(sg)
            partial = np.c_[partial, partial_]
        np.save(model_directory_path + '/trained_model/layer%s_partial' % self.layer_depth, partial)
        if return_predictin is True:
            return partial
    
    def train_whole(self, model_directory_path = file_directory):
        sg = self.sg
        models = self.models
        for model in models:
            model.guess_whole(sg)
        self.save_model(model_directory_path + '/trained_model/layer%s_whole_class' % self.layer_depth, models)

    def layer_predict(self, data , model_directory_path = file_directory):
        # self.load_model(model_directory_path + '/trained_model/layer%s_whole_class' % self.layer_depth)
        models = self.models
        whole = np.zeros(shape=(len(data), 0))
        for model in models:
            whole_ = model.predict(data)
            whole = np.c_[whole, whole_]
        return whole

    @staticmethod
    def save_model(ppath, object):
        with open(ppath, mode='wb') as f:
            pickle.dump(object, f)

    def load_model(self, ppath=file_directory):
        with open(ppath, mode='rb') as f:
            self.models = pickle.load(f)
