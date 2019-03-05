# -*- coding: utf-8 -*-
"""
Created on Tue Feb 26 19:48:59 2019

@author: jiang
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn.model_selection import train_test_split
import warnings
#import all algorithm packages
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier 
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from datetime import datetime
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import RandomizedSearchCV
from sklearn import metrics
from sklearn.model_selection import GridSearchCV, cross_val_score
warnings.filterwarnings('ignore')

df = pd.read_csv("aac_intakes_outcomes.csv", header = 0)
print(df.shape)

print(df.groupby(["outcome_type"])["outcome_type"].count().sort_values(ascending = False))
final_df = df.loc[df.outcome_type.isin(["Adoption","Transfer","Return to Owner","Euthanasia","Died","Disposal"]), ]
def target(x):
    if x in ["Euthanasia","Died","Disposal","Transfer"]: 
        return 0
    else: 
        return 1
    
final_df.loc[:,"target"]  = final_df.outcome_type.apply(lambda x: target(x))
print("The new dataset's shape is", final_df)

def get_df(input_x, df = final_df):
    df.loc[:,"count_column"] = 1
    df.loc[: ,"avg_column"] = 1
    #print df.loc['avg_column']
    new_df = df.groupby([input_x,"target"]).agg({"count_column":'count', 'avg_column':'count'})
    new_df["count_column"] = new_df.groupby(level=0).transform(lambda x: x/x.sum())
    new_df.reset_index(inplace = True)
    return new_df

final_df['breed_group'] = final_df['breed']
dog_breed = final_df.loc[final_df.animal_type == 'Dog','breed']
replace_f = lambda x: str(x).replace(' Mix',"")
v_replace_f = np.vectorize(replace_f)
dog_breed = np.unique(v_replace_f(dog_breed))
dog_breed.sort()
dog_group = {}
current_group = dog_breed[0]
for d in dog_breed:
    if d.startswith(current_group):
        pass
    else:
        current_group = d
    dog_group[d] = current_group[:]
dog_breed_df = final_df.loc[final_df.animal_type == 'Dog',['breed','target','age_upon_outcome']]
dog_breed_df['breed'] = dog_breed_df['breed'].apply(lambda x: dog_group[replace_f(x)])
Top30dogs = dog_breed_df['breed'].value_counts().index[:30]
def transform_dog_breed(x, d):
    try:
        if dog_group[replace_f(x)] == d: 
            return 1 
        else: 
            return 0
    except:
        return 0
for d in Top30dogs:
    final_df.loc[:, 'dogbreed_' + d] = final_df['breed'].apply(transform_dog_breed,d = d)
    
cat_breed = final_df.loc[final_df.animal_type == 'Cat','breed']
replace_f = lambda x: str(x).replace(' Mix',"")
v_replace_f = np.vectorize(replace_f)
cat_breed = np.unique(v_replace_f(cat_breed))
cat_breed.sort()
cat_group = {}
current_group = cat_breed[0]

for c in cat_breed:
    if c.startswith(current_group):
        pass
    else:
        current_group = c
    cat_group[c] = current_group[:]
    
cat_breed_df = final_df.loc[final_df.animal_type == 'Cat',['breed','target','age_upon_outcome']]
cat_breed_df['breed'] = cat_breed_df['breed'].apply(lambda x: cat_group[replace_f(x)])
cat_breed_df['breed'].value_counts().sort_values(ascending = False)
Top20cats = cat_breed_df['breed'].value_counts().index[:30]
def transform_cat_breed(x, d):
    try:
        if cat_group[replace_f(x)] == d: 
            return 1 
        else: 
            return 0
    except:
        return 0
for d in Top20cats:
    final_df.loc[:, 'catbreed_' + d] = final_df['breed'].apply(transform_cat_breed,d = d)
    
    
color_list = ["Black", "Dark color", "Light color", "muti-color"]
def transform_color(x, d):
    if x == "Black" and d == "Black":
        return 1
    elif ("Tan" in x or "Brown" in x or "Black" in x or "Black" in x) and "White" not in x and d == "Dark Color":
        return 1
    elif "White" in x and d == "Light color":
        return 1
    elif "/" in x and d == "muti-color":
        return 1
    else:
        return 0

for co in color_list:
    final_df.loc[:, 'color' + co] = final_df['color'].apply(transform_color,d = co)
    
    
dummies = pd.get_dummies(final_df[['sex_upon_outcome','animal_type','intake_condition','intake_type']], drop_first=True)
final_df = pd.concat([final_df, dummies], axis = 1)
all_var = ['age_upon_outcome_(years)','outcome_hour','outcome_month'] + list(dummies.columns) + ['color' + co for co in color_list] + ['catbreed_' + d for d in Top20cats] + ['dogbreed_' + d for d in Top30dogs]
X_train, X_test, y_train, y_test = train_test_split(final_df[all_var], final_df['target'], test_size=0.33, random_state=42)
selection_list= all_var + ['target']
final_df[selection_list].to_csv('data.csv', index = None)
#To evaluate the training time, a timer function will be create
def timer(start_time=None):
    if not start_time:
        start_time = datetime.now()
        return start_time
    elif start_time:
        thour, temp_sec = divmod((datetime.now() - start_time).total_seconds(), 3600)
        tmin, tsec = divmod(temp_sec, 60)
        print('Time taken: %i hours %i minutes and %s seconds.' % (thour, tmin, round(tsec, 2)))
#DT = DecisionTreeClassifier()
NN = MLPClassifier()
#xgb = XGBClassifier()
#SM = svm.SVC()
#Knn = KNeighborsClassifier()
model_dict = {"Neural_network": NN}

hidden_layer_sizes_list=[4,5,10,20,30,40,50,100]
start_time = timer(None)
NN_clf = GridSearchCV(estimator=model_dict["Neural_network"], param_grid=dict(hidden_layer_sizes=hidden_layer_sizes_list),
                      n_jobs=-1, cv = 5, scoring = 'f1_weighted')
NN_clf.fit(X_train, y_train)
timer(start_time)
print(NN_clf.best_score_ )


hidden_layer_sizes_list=[100, 150, 200]
start_time = timer(None)
NN_clf = GridSearchCV(estimator=model_dict["Neural_network"], param_grid=dict(hidden_layer_sizes=hidden_layer_sizes_list),
                      n_jobs=-1, cv = 5, scoring = 'f1_weighted')
NN_clf.fit(X_train, y_train)
timer(start_time)
print(NN_clf.best_score_ )