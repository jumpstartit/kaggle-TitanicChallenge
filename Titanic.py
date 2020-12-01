# -*- coding: utf-8 -*-
"""
Author:Tharun Loknath
This is a exercise code for titanic problem on keggle
"""

import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split,cross_val_score,GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score
import warnings
warnings.filterwarnings("ignore",category=FutureWarning)

titanic = pd.read_csv('titanic.csv')
#clean
titanic["Age"].fillna(titanic["Age"].mean(),inplace=True)
titanic["family_count"] =titanic["SibSp"]+titanic["Parch"]
#continuous features
titanic.drop([ "PassengerId","SibSp","Parch"], axis=1, inplace=True)
titanic['Cabin_ind'] = np.where(titanic['Cabin'].isnull(), 0, 1)

gender_num = {'male': 0, 'female': 1}


'''
for i, col in enumerate(['Cabin_ind', 'Sex', 'Embarked']):
    plt.figure(i)
    sns.catplot(x=col, y='Survived', data=titanic, kind='point', aspect=2, )
'''

#categorial features
titanic['Sex'] = titanic['Sex'].map(gender_num)
titanic.drop(['Cabin', 'Embarked','Name', 'Ticket'], axis=1, inplace=True)

#split train - test -validation
 
feat=titanic.drop("Survived",axis=1)
lables=titanic["Survived"]

x_train, x_test_va, y_train, y_test_va = train_test_split(feat,lables,test_size=0.4,random_state=42)
x_test, x_val , y_test, y_val = train_test_split(x_test_va,y_test_va,test_size=0.5,random_state=42)

#cross validation- fitting the model on the training set by dividing 
def print_results(results):
    print('BEST PARAMS: {}\n'.format(results.best_params_))

    means = results.cv_results_['mean_test_score']
    stds = results.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, results.cv_results_['params']):
        print('{} (+/-{}) for {}'.format(round(mean, 3), round(std * 2, 3), params))

rf = RandomForestClassifier()
parameters={
        "n_estimators":[5,50,100], # total number of decison tress
        "max_depth":[2,10,20,None] #depth of the tree
        }
#scores = cross_val_score(rf,x_train,y_train.values.ravel(),cv=5) #y_train is a couloum vector so it is converted to a array
scores=GridSearchCV(rf,parameters,cv=5)
scores.fit(x_train,y_train.values.ravel())
print_results(scores)
print(scores)
#three best performance from cross validation is used to fit to the whole data
rf1=RandomForestClassifier(n_estimators=5,max_depth=10)
rf1.fit(x_train,y_train.values.ravel())

rf2=RandomForestClassifier(n_estimators=100,max_depth=10)
rf2.fit(x_train,y_train.values.ravel())

rf3=RandomForestClassifier(n_estimators=100,max_depth=None)
rf3.fit(x_train,y_train.values.ravel())

for mdl in [rf1,rf2,rf3]:
    y_pred=mdl.predict(x_val)
    acc= round(accuracy_score(y_val,y_pred),3)
    preci= round(precision_score(y_val,y_pred),3)
    recall= round(recall_score(y_val,y_pred),3)
    print('MAX_Dept:{} /# OF EST: {} --A:{} / P: {} /R:{}'.format(mdl.max_depth,mdl.n_estimators,acc,preci,recall))

#evaluation of completly fit model,now to test it with testing data
    
y_pred=rf2.predict(x_test)
acc= round(accuracy_score(y_test,y_pred),3)
preci= round(precision_score(y_test,y_pred),3)
recall= round(recall_score(y_test,y_pred),3)
print('\nFinal Robust Model MAX_Dept:{} /# OF EST: {} --A:{} / P: {} /R:{}'.format(rf2.max_depth,rf2.n_estimators,acc,preci,recall))
