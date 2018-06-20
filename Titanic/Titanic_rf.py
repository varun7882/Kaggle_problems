import pandas as pd
import math
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
# SciKitLearn is a useful machine learning utilities library
import sklearn
# The sklearn dataset module helps generating datasets
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier

dataset=pd.read_csv("train.csv")
sub=pd.read_csv("test.csv")
ds=dataset.loc[:,['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked']]
dssub=sub.loc[:,['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked']]
ds.SibSp=ds.SibSp+ds.Parch
dssub.SibSp=dssub.SibSp+dssub.Parch
ds=ds.drop(['Parch'],axis=1)
dssub=dssub.drop(['Parch'],axis=1)
pid=sub.PassengerId.values
#handling missing values
#for Age
ds.Age=ds.Age.fillna(ds.Age.median())
dssub.Age=dssub.Age.fillna(dssub.Age.median())
#for Embarked
ds.Embarked=ds.Embarked.fillna('S')
dssub.Embarked=dssub.Embarked.fillna('S')
X=ds.values
X_sub=dssub.values

X_sub[152,4]=7.7500     
X_all=np.concatenate((X,X_sub),axis=0)
y=dataset.loc[:,'Survived'].values

#Handling categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X_all[:, 1] = labelencoder_X.fit_transform(X_all[:, 1])
X_all[:, 5] = labelencoder_X.fit_transform(X_all[:, 5])
onehotencoder = OneHotEncoder(categorical_features=[0,5])
X_all = onehotencoder.fit_transform(X_all).toarray()
X=X_all[:891,:]
X_sub=X_all[891:,:]

#Splitting data
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = 0.33, random_state = 1)
#X_val, X_test, y_val, y_test = train_test_split(X_testVal, y_testVal, test_size = 0.50, random_state = 1)

#Training Classifier
from sklearn.model_selection import RandomizedSearchCV
# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]
# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}



# Use the random grid to search for best hyperparameters
# First create the base model to tune
rf = RandomForestClassifier()
# Random search of parameters, using 3 fold cross validation, 
# search across 100 different combinations, and use all available cores
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 10, cv = 2, verbose=2, random_state=42, n_jobs = -1)
# Fit the random search model
rf_random.fit(X_train, y_train)
y_pred_val=rf_random
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_val, y_pred_val)
print ('confusion matrix :')
print (cm)
from sklearn.metrics import f1_score
print ('f-score(weighted) is : ')
print (f1_score(y_val,y_pred_val,average='weighted'))

'''
y_sub=rf.predict(X_sub)
dct={'PassengerId':pid,'Survived':y_sub}
resdf=pd.DataFrame(data=dct)
resdf.to_csv('result_dt.csv',index=False)'''
