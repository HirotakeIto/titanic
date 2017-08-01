from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
import importlib as imp
from src.stacking import layer
from  src.stacking import model_list
imp.reload(model_list)
imp.reload(layer)

l = load_boston()
x = l.data
y = l.target
x_tr,x_te,y_tr,y_te = train_test_split(x,y,test_size=0.05,random_state=657)
name = l.feature_names

R = model_list.RFRegressor()
R2 = model_list.RFRegressor(n_estimators=20)


lay = layer.Layer()
lay.models = [R,R2]
lay.layer_depth=2
lay.sg = model_list.StackedGeneralization(4,x_tr,y_tr)

# train_partial test

a=lay.train_partial(return_predictin=True)
lay.train_whole()
a=lay.layer_predict(x_te)


