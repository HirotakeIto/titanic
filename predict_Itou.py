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
PassengerId = test['PassengerId'].copy()
Partner = test[['PassengerId', 'Name']].copy()
# Feature Engineer
data_set = [train, test]
full_data = pd.concat([train, test], axis=0)
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
    count_surname = surname['count'].to_dict()
    sur_lonely = surname.ix[surname['count'] == 1, :].index
    return count_surname, sur_lonely

count_surname, sur_lonely = get_lonely_sur(full_data[['Name', 'Survived']].copy())

def get_survive_rate_surname(surname):
    surname['SurName'] = surname['Name'].str.split('\s')
    surname['SurName'] = surname['SurName'].apply(lambda x: x[0])
    surname['SurName'] = surname['SurName'].str.replace('([^A-Za-z0-9])', '')  # Vanで始まる人とか、ドイツ人系？もいるので、本当は別処理が必要だけれど。。。
    surname = surname.groupby('SurName', as_index=True)['Survived'].agg(['mean', 'count'])
    survive_rate_surname = surname.loc[surname['count']>1, 'mean'].to_dict() # ある程度名前が残っている人に限定
    return survive_rate_surname

survive_rate_surname = get_survive_rate_surname(train[['Name', 'Survived']].copy())
mean_survive_rate_surname = np.mean(list(survive_rate_surname.values()))

# trainに一つ, testにも一つという性別を探す
# get_survive_rate_surnameではのぞいた、trainに一人しかいないケースを別カラムで用意する。
# すごく恣意的なカラムで嫌なんだけど（二人以上いたって同じ扱いにしたいga...）
# と思ったけど、使えんかったわwww（そんな奴は初めからすごく確率低め）
def get_survive_partner(surname_tr, surname_te):
    data_set = [surname_tr, surname_te]
    for dataset in data_set:
        dataset['SurName'] = dataset['Name'].str.split('\s')
        dataset['SurName'] = dataset['SurName'].apply(lambda x: x[0])
        dataset['SurName'] = dataset['SurName'].str.replace('([^A-Za-z0-9])', '')  # Vanで始まる人とか、ドイツ人系？もいるので、本当は別処理が必要だけれど。。。
    surname_tr = surname_tr.groupby('SurName', as_index=True)['Survived'].agg(['mean', 'count']).reset_index(drop=False)
    surname_te = surname_te.groupby('SurName', as_index=True).size().rename('size').reset_index(drop=False)
    survive_surname = pd.merge(surname_tr, surname_te, on='SurName', how='inner')
    survive_surname = survive_surname.loc[survive_surname['count'] == 1, ['SurName', 'mean']]
    return survive_surname.set_index('SurName').to_dict()

survive_partner = get_survive_partner(train[['Name', 'Survived']].copy(),
                                      test[['Name']].copy())

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
    # age_median = full_data['Age'].median()
    # dataset['Age'] = dataset['Age'].fillna(age_median)
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
    # get surname count
    dataset['SurnameCount'] = np.nan
    dataset['SurnameCount'] = dataset['SurName'].replace(count_surname)
    dataset['SurnameCount'] = pd.to_numeric(dataset['SurnameCount'], errors='coerce')  # 変換できないやつも残るので、nanに
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

# ready dataset
print(train.corr())
use_columns_x = ['Pclass', 'Age', 'Parch', 'Fare',
                 'Embarked', 'Name_length', 'Has_Cabin',
                 'cabin_a', 'cabin_b', 'cabin_c', 'cabin_d', 'cabin_e', 'cabin_f',
                 'FamilySize', 'IsAlone', 'MarriedWomen',
                 'Fare_1st', 'Fare_9st', 'Children', 'Elder',
                 'Women', 'Men',
                 'Mr', 'Miss', 'Mrs', 'Master',
                 'Rare', 'SurLonely', 'SurnameCount', 'SurviveRateSurname']
use_columns_y = ['Survived']
use_columns = use_columns_y + use_columns_x
train = train[use_columns]
test = test[use_columns_x]
# fillna
age_x = use_columns_x.copy()
age_x.remove('Age')
import sklearn.linear_model as lm
reg = lm.LinearRegression()
full_data = pd.concat([train, test], axis=0)
indexer = full_data['Age'].isnull()
reg.fit(full_data.loc[~indexer, age_x], full_data.loc[~indexer, "Age"])
for dataset in [train, test]:
    indexer = dataset['Age'].isnull()
    predicted = reg.predict(dataset.loc[indexer, age_x])
    dataset.loc[indexer, 'Age'] = predicted

# Lets fit
# get usefull columns
R = RandomForestClassifier(n_estimators=300)
M = R
M.fit(train.ix[:,1:].values, train.ix[:,0].values)
columns_usefullness = pd.DataFrame(np.c_[M.feature_importances_, train.columns[1:]], columns=['value', 'columns'])
usefull_columns = list(columns_usefullness.loc[columns_usefullness['value']>0.02, 'columns'].values)
use_columns = use_columns_y + usefull_columns
train = train[use_columns]
test = test[usefull_columns]
# get best parametor
p = make_scorer(roc_auc_score)
best = optim(Model=M, x=train.ix[:,1:], y=train.ix[:,0], s=p)
# set  estimator
R = RandomForestClassifier( **best)
M = R
result, _ = cross_valid(x=train.ix[:,1:].values, y=train.ix[:,0].values, M=M)
# # predict
M.fit(train.ix[:,1:].values, train.ix[:,0].values)
result = M.predict(test.values)
result = pd.DataFrame(result, columns=['Survived'])
result = pd.concat([PassengerId, result],axis=1)
result.to_csv('my_submit.csv', index=False)

















# # give partner survive flag
# Partner['SurName'] = Partner['Name'].str.split('\s')
# Partner['SurName'] = Partner['SurName'].apply(lambda x: x[0])
# Partner['SurName'] = Partner['SurName'].str.replace('([^A-Za-z0-9])', '')
# Partner['PartnerSurviveFlag'] = np.nan
# Partner['PartnerSurviveFlag'] = Partner['SurName'].replace(survive_rate_surname)
# Partner['Hosei'] = 1
# Partner.loc[Partner['PartnerSurviveFlag'] == 1 , 'Hosei'] = 10
# Partner.loc[Partner['PartnerSurviveFlag'] == 0 , 'Hosei'] =  0
#
# result_a = M.predict_proba(test.values)
# result_a = pd.DataFrame(result_a, columns=['Death', 'Survive'])
# result_a = pd.concat([PassengerId, result_a],axis=1)
# result_a = pd.concat([result_a, Partner[['Hosei']],],axis=1)
# result_a['Survived'] = 0
# result_a.loc[result_a['Survive']*result_a['Hosei'] > 0.5, 'Survived'] = 1
# result_a[['PassengerId', 'Survived']]
# (result.loc[:,'Survived'] == result_a.loc[:,'Survived']).sum()
#



# {'max_depth': 3,
#  'max_features': 9,
#  'min_samples_leaf': 0.011964085560497856,
#  'min_samples_split': 0.049334878034131224,
#  'n_estimators': 73,
#  'n_jobs': -1}



# # get best parametor
# R = RandomForestClassifier()
# G = GradientBoostingClassifier()
# # p = make_scorer(f1_score, pos_label=1)
# # R = LogisticRegression()
# p = make_scorer(roc_auc_score)
# M = R
# best = optim(Model=M, x=train.ix[:,1:], y=train.ix[:,0], s=p)
# # set  estimator
# R = RandomForestClassifier(**best)
# # G = GradientBoostingClassifier(**best)
# # R = RandomForestClassifier()
# # R = RFClassifier(**{'max_depth': 8,
# #                               'max_features': 28,
# #                               'min_samples_leaf': 0.01066348,
# #                               'min_samples_split': 0.0907340,
# #                               'n_estimators': 86,
# #                               'n_jobs': -1})
# # G = GBClassifier(**{'min_samples_leaf': 0.0210847231499676,
# #                     'min_samples_split': 0.3664713039286747,
# #                     'max_depth': 11,
# #                     'n_estimators': 154,
# #                     'learning_rate': 0.13007973251085603,
# #                     'subsample': 0.6295311034249846})
# # L = LRClassifier()
# # R2 = RFClassifier(**{'n_estimators': 100})
# # S = StackLearn(depths=2,
# #                models={1: [R,G,L], 2: [R2]},
# #                mixed=True)
# # check perforamance
# M = R
# result, _ = cross_valid(x=train.ix[:,1:].values, y=train.ix[:,0].values, M=M)
# # # predict
# M.fit(train.ix[:,1:].values, train.ix[:,0].values)
# result = M.predict(test.values)
# result = pd.DataFrame(result, columns=['Survived'])
# result = pd.concat([PassengerId, result],axis=1)
# result.to_csv('my_submit.csv', index=False)
#
#
# print(np.c_[M.feature_importances_, train.columns[1:]])
#








# parametor memo
# RandomForest
# {'max_depth': 8,
#  'max_features': 28,
#  'min_samples_leaf': 0.010663481099052697,
#  'min_samples_split': 0.09073406709412284,
#  'n_estimators': 86,
#  'n_jobs': -1}
# Gradient
# {'max_depth': 14, 'min_samples_leaf': 0.010833455370681215, 'min_samples_split': 0.4408565535988071, 'n_estimators': 72}






# # trash
# train.corr()
# # aaa = train[['Ticket','Survived']].copy()
# # aaa['Ticket'] = aaa['Ticket'].str.contains('\D')
# # aaa['Ticket'] = pd.to_numeric(aaa['Ticket'] )
# # # aaa['Ticket'] = aaa['Ticket'].apply(len)
# # aaa.groupby('Survived').mean()
# G = AdaBoostClassifier(n_estimators=100)
# result, _ = cross_valid(x=train.ix[:,1:].values, y=train.ix[:,0].values, M=G)
# G .fit(train.ix[:,1:], train.ix[:,0])
# result = G.predict(test)
# result = pd.DataFrame(result, columns=['Survived'])
# result = pd.concat([PassengerId, result],axis=1)
# result.to_csv('my_submit.csv', index=False)
#
# from sklearn.preprocessing import StandardScaler
# from sklearn.model_selection import cross_val_score
# from sklearn.pipeline import Pipeline
# from sklearn.svm import SVC
#
# best_score = 0.79
# for i in range(5,20):
#     i = i / 100
#     for j in range(100,250):
#         j = j / 100
#         clf_svm = SVC(kernel='rbf',random_state=0,gamma=i,C=j)
#         pipe_svm = Pipeline([['sc', StandardScaler()],['clf', clf_svm]])
#         scores = cross_val_score(estimator=clf_svm,
#                                  X=train.ix[:, 1:].values,
#                                  y=train.ix[:, 0].values,
#                                  cv=10,
#                                  scoring='roc_auc')
#         if scores.mean() > best_score:
#             print("gamma:",i,"C:",j,scores.mean(),"+-",scores.std())
