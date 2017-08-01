from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
l = load_boston()
x = l.data
y = l.target
x_tr,x_te,y_tr,y_te = train_test_split(x,y,test_size=0.05,random_state=657)
name = l.feature_names

from src.stacking import stacking
import importlib as imp
imp.reload(stacking)

R = stacking.RFRegressor(n_estimators = 20)
G = stacking.GBRegressor(n_estimators = 20)
L = stacking.LRRegressor()

s = stacking.StackLearn()
s.mixed = True
s.models = {1: [R,G,L], 2: [R]}
s.fit(x_tr,y_tr)

p =  s.predict(x_te)
import numpy as np
np.c_[p,y_te]
