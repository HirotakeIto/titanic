from lingam import lingam
from sklearn.preprocessing import StandardScaler
import pandas as pd


import pandas as pd
import numpy as np
import re
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.model_selection import KFold
from stacking import model_optim
from sklearn.metrics import make_scorer
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, f1_score, accuracy_score, roc_auc_score
from stacking.stacking import StackLearn
from stacking.model_list import *
pd.set_option("display.max_columns", 80)
# Load in the train and test datasets
train = pd.read_csv('./input/train.csv')
test = pd.read_csv('./input/test.csv')

# Store our passenger ID for easy access
PassengerId = test['PassengerId']

# Feature Engineer
data_set = [train, test]
full_data = pd.concat([train, test], axis=0)

aaa = full_data.copy()
aaa['cabin_id'] = full_data['Cabin'].str.replace('\d','')
aaa['cabin_id'] = aaa['cabin_id'].str.replace('(\s.*)', '')
aaa.groupby('cabin_id')['Survived'].mean()

# pre research


def get_title(name):
    title_search = re.search(' ([A-Za-z]+)\.', name)
    # If the title exists, extract and return it.
    if title_search:
        return title_search.group(1)
    return ""
full_data['Title'] = full_data['Name'].apply(get_title)

# get lonely sur


def get_lonely_sur(surname):
    """
    :param surname:
     surname['Name'] - ex:'Frauenthal, Mr. Isaac Gerald'
    :return:
    """
    surname['SurName'] = surname['Name'].str.split('\s')
    surname['SurName'] = surname['SurName'].apply(lambda x: x[0])
    """
    Vanで始まる人とか、ドイツ人系？もいるので、本当は別処理が必要だけれど。。。
    後、姓名と名前が区別できていない人も結構いるね
    """
    surname['SurName'] = surname['SurName'].str.replace('([^A-Za-z0-9])', '')
    surname = surname.groupby('SurName')['SurName'].agg(['count'])
    sur_lonely = surname.ix[surname['count'] == 1, :].index
    return sur_lonely

sur_lonely = get_lonely_sur(full_data[['Name', 'Survived']].copy())

def get_survive_rate_surname(surname):
    surname['SurName'] = surname['Name'].str.split('\s')
    surname['SurName'] = surname['SurName'].apply(lambda x: x[0])
    surname['SurName'] = surname['SurName'].str.replace('([^A-Za-z0-9])', '')  # Vanで始まる人とか、ドイツ人系？もいるので、本当は別処理が必要だけれど。。。
    surname = surname.groupby('SurName', as_index=True)['Survived'].agg(['mean', 'count'])
    survive_rate_surname = surname['mean'].to_dict()
    return survive_rate_surname

survive_rate_surname = get_survive_rate_surname(train[['Name', 'Survived']].copy())
mean_survive_rate_surname = np.mean(list(survive_rate_surname.values()))

# get cabin id
def get_cabin_id(cabin:pd.Series):
    cabin = cabin.str.replace('\d', '')
    cabin = cabin.str.replace('(\s.*)', '')
    return cabin


# engineer start
for dataset in data_set:
    # name length
    dataset['Name_length'] = dataset['Name'].apply(len)
    # has cabin
    dataset['Has_Cabin'] = dataset["Cabin"].apply(lambda x: 0 if type(x) == float else 1)
    dataset['cabin_id'] = get_cabin_id(dataset["Cabin"])
    dataset.ix[dataset['cabin_id'] == 'A', 'cabin_a'] = 1
    dataset.ix[dataset['cabin_id'] == 'B', 'cabin_b'] = 1
    dataset.ix[dataset['cabin_id'] == 'C', 'cabin_c'] = 1
    dataset.ix[dataset['cabin_id'] == 'D', 'cabin_d'] = 1
    dataset.ix[dataset['cabin_id'] == 'E', 'cabin_e'] = 1
    dataset.ix[dataset['cabin_id'] == 'F', 'cabin_f'] = 1
    dataset['cabin_a'] = dataset['cabin_a'].fillna(0)
    dataset['cabin_b'] = dataset['cabin_b'].fillna(0)
    dataset['cabin_c'] = dataset['cabin_c'].fillna(0)
    dataset['cabin_d'] = dataset['cabin_d'].fillna(0)
    dataset['cabin_e'] = dataset['cabin_e'].fillna(0)
    dataset['cabin_f'] = dataset['cabin_f'].fillna(0)
    # family size
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1
    # is alone dummy
    dataset['IsAlone'] = 0
    dataset.ix[dataset['FamilySize'] == 1, 'IsAlone'] = 1
    # dataset married women
    dataset['MarriedWomen'] = 0
    # # ' (Mary Eloise Hughes)'みたいなものを拾ってる。カッコの前のスペースも許容。
    dataset.ix[dataset['Name'].str.contains('\s*(\(.*\))', ''), 'MarriedWomen'] = 1
    # embarked
    dataset['Embarked'] = dataset['Embarked'].fillna('S')
    # fillna for Fare. use median
    fare_median = full_data['Fare'].median()
    dataset['Fare'] = dataset['Fare'].fillna(fare_median)
    # dummy for Fare
    rank_1st = full_data['Fare'].quantile(0.15)
    dataset['Fare_1st'] = 0
    dataset.ix[dataset['Fare'] < rank_1st, 'Fare_1st'] = 1
    rank_9st = full_data['Fare'].quantile(0.85)
    dataset['Fare_9st'] = 0
    dataset.ix[dataset['Fare'] > rank_9st, 'Fare_9st'] = 1
    # Mapping Embarked
    dataset['Embarked'] = dataset['Embarked'].map({'S': 0, 'C': 1, 'Q': 2}).astype(int)
    # Age dummy
    dataset['Children'] = 0
    dataset.ix[dataset['Age'] < 20, 'Children'] = 1
    dataset['Elder'] = 0
    dataset.ix[dataset['Age'] > 55, 'Elder'] = 1
    # Women dummy
    dataset['Women'] = 0
    dataset.ix[dataset['Sex'] =='female' ,'Women'] = 1
    dataset['Men'] = 0
    dataset.ix[dataset['Sex'] == 'male', 'Men'] = 1
    # fillna age
    age_median = full_data['Age'].median()
    dataset['Age'] = dataset['Age'].fillna(age_median)
    # get SES from Name
    dataset['Title'] = dataset['Name'].apply(get_title)
    dataset['Title'] = dataset['Title'].replace(
        ['Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')
    dataset.ix[dataset['Title'] == 'Mr', 'Mr'] = 1
    dataset.ix[dataset['Title'] == 'Miss', 'Miss'] = 1
    dataset.ix[dataset['Title'] == 'Mrs', 'Mrs'] = 1
    dataset.ix[dataset['Title'] == 'Master', 'Master'] = 1
    dataset.ix[dataset['Title'] == 'Rare', 'Rare'] = 1
    dataset[['Mr', 'Miss', 'Mrs', 'Master', 'Rare']] = dataset[['Mr', 'Miss', 'Mrs', 'Master', 'Rare']].fillna(0)
    # set surname
    dataset['SurName'] = dataset['Name'].str.split('\s')
    dataset['SurName'] = dataset['SurName'].apply(lambda x: x[0])
    dataset['SurName'] = dataset['SurName'].str.replace('([^A-Za-z0-9])', '')  # Vanで始まる人とか、ドイツ人系？もいるので、本当は別処理が必要だけれど。。。
    # get lonely surname flag
    dataset['SurLonely'] = 0
    dataset.ix[dataset['SurName'].isin(sur_lonely), 'SurLonely'] = 1
    # get surname survival rate
    dataset['SurviveRateSurname'] = np.nan
    dataset['SurviveRateSurname'] = dataset['SurName'].replace(survive_rate_surname)
    dataset['SurviveRateSurname'] = pd.to_numeric(dataset['SurviveRateSurname'], errors='coerce')  # 変換できないやつも残るので、nanに
    dataset.ix[dataset['SurName'].isin(sur_lonely), 'SurviveRateSurname'] = np.nan  # 一人しかいないやつは参考にならない
    dataset['SurviveRateSurname'] = dataset['SurviveRateSurname'].fillna(mean_survive_rate_surname)  # 平均で埋める

def optim(Model, x, y, s):
    # Model = RandomForestClassifier()
    m = model_optim.ModelOptimization(model=Model, X=x, Y=y)
    m.set_model_space()
    # 何をターゲットにして最適化するか
    # precision は Trueラベルを与えた人間のうち、何人が本当にTrueだったか。
    # recall Trueの人を何人探すことができたか
    # p = make_scorer(precision_score, pos_label=1)
    m.change_model_setting(s)
    best = m.optimization()
    return best


def cross_valid(x, y, M):
    kf = KFold(n_splits=10, shuffle=True)
    index_ = np.arange(len(y))
    result = []
    for train_index, test_index in kf.split(index_):
        # training
        x_tr = x[train_index]
        y_tr = y[train_index]
        M.fit(x_tr, y_tr)
        # test
        x_te = x[test_index]
        y_te = y[test_index]
        y_pred = M.predict(x_te)
        res = precision_recall_fscore_support(y_te, y_pred, pos_label=1, average='binary')
        print('precision,recall,fscore,support')
        print(res)
        result.append(res)
    a=confusion_matrix(y_te,y_pred,labels=[1,0])
    return result,a


# Lets fit
# ready dataset
print(train.corr())
use_columns_x = ['Pclass', 'Age', 'Parch', 'Fare',
                 'Embarked', 'Name_length', 'Has_Cabin',
                 'cabin_a', 'cabin_b', 'cabin_c', 'cabin_d', 'cabin_e', 'cabin_f',
                 'FamilySize', 'IsAlone', 'MarriedWomen',
                 'Fare_1st', 'Fare_9st', 'Children', 'Elder',
                 'Women', 'Men',
                 'Mr', 'Miss', 'Mrs', 'Master',
                 'Rare', 'SurLonely', 'SurviveRateSurname']
use_columns_x = ['Pclass', 'Age', 'Parch', 'Fare',
                 'Embarked', 'Name_length', 'Has_Cabin']
use_columns_y = ['Survived']
use_columns = use_columns_x + use_columns_y
train = train[use_columns]
test = test[use_columns_x]


# train = pd.read_csv('./input/train.csv')
# train = train[['Survived' ,'Pclass', 'Age', 'SibSp', 'Parch', 'Fare']].dropna()
S = StandardScaler()
data = S.fit_transform(train.values)
result = lingam.estimate(data)
print(result)
train.columns