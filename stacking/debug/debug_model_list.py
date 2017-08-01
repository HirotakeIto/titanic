from  src.stacking import model_list
from src.stacking import layer


from sklearn.datasets import load_boston
import importlib as imp
imp.reload(model_list)

l = load_boston()
x = l.data
y = l.target
name = l.feature_names


sg = model_list.StackedGeneralization(4,x,y)
R = model_list.RFRegressor()


p=R.guess_partial(sg=sg)
R.guess_whole(sg=sg)
w=R.predict(x)

