"""
https://github.com/sergeant-wizard/stacked_generalization/
を参考に作成
"""

from sklearn.base import BaseEstimator
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from .model_list import *
from . import layer
from .stacked_generalization import StackedGeneralization

file_directory = os.path.dirname(__file__)


class StackLearn(BaseEstimator):
    """

    こいつはworking directoryの下に勝手にモデルを保存し、predictするときもそいつを利用するからtrainを保存する際は気をつけて！

    """
    def __init__(self, 
                 depths=2, 
                 models={1: [RFRegressor()],  2: [RFRegressor()]},
                 mixed=False):
        super().__init__()
        self.depths = depths
        self.models = models  # dixtionry.{depth:UseModel...}
        self.mixed = mixed

    def set_data(self, x=None, y=None):
        self.x_orient = x
        self.y_orient = y

    def fit(self,  x,  y):
        # うまくいってれば、これで回るんだけど、
        # うまくいってなかったら、根本的にリファクタリング
        assert (isinstance(x,  np.ndarray))
        assert (isinstance(y,  np.ndarray))
        self.set_data(x=x, y=y)
        self.stack_train(x,  y)
        return print('finish!!')

    def predict(self,  x):
        assert (isinstance(x,  np.ndarray))
        self.set_data(x = x)
        return self.stack_predict(x)

    def stack_train(self,  x,  y):
        depths = self.depths
        if depths == 1:
            print('depths must be over than 1 . ')
            return
        # 最終層以外の学習。最初とそれ以外で読み込み方は違うので分岐。
        for depth in tqdm(range(1,  depths)):
            print('depth is',  depth)
            l0 = self.layer_set(depth)
            l0.train_partial()
            l0.train_whole()
        # 最終層の学習。
        else:
            depth = self.depths
            print('depth is',  depth)
            l0 = self.layer_set(depth)
            l0.train_whole()

    def load_data(self, now_depth, model_directory_path=file_directory):
        if now_depth != 1:
            x = np.load(model_directory_path + '/trained_model/' + 'layer%s_partial.npy' % (now_depth - 1))
            y = self.y_orient
            if self.mixed == True:
                x = np.c_[x,  self.x_orient]
        elif now_depth == 1:
            x = self.x_orient
            y = self.y_orient
        return x,  y

    def layer_set(self,  depth):
        x,  y = self.load_data(depth)
        l0 = layer.Layer()
        l0.sg = StackedGeneralization(5,  x,  y)
        l0.models = self.models[depth]
        l0.layer_depth = depth
        return l0

    def stack_predict(self, x, model_directory_path=file_directory):
        x_ = x
        print('please check now model type(model is mixed or not).', 
              'if you call model by mistake,  this code may end in error.')
        print('"predict model is mixed" is %s.' % self.mixed)
        depths = self.depths
        for depth in tqdm(range(1,  depths)):
            l0 = layer.Layer()
            l0.layer_depth = depth
            l0.load_model(model_directory_path +
                          '/trained_model/layer%s_whole_class' % l0.layer_depth)
            x = l0.layer_predict(x)
            if self.mixed==True:
                x = np.c_[x,  x_]
        else:
            depth = self.depths
            l0 = layer.Layer()
            l0.layer_depth = depth
            l0.load_model(model_directory_path +
                          '/trained_model/layer%s_whole_class' % l0.layer_depth)
            return l0.layer_predict(x)

    def score(self, x, y):
        y_pred = self.predict(x)
        y_pred = y_pred.astype(int)
        return accuracy_score(y, y_pred)



