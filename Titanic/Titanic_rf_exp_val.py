import pandas as pd
import math
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pylab as py
# SciKitLearn is a useful machine learning utilities library
import sklearn
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestClassifier

dataset=pd.read_csv("train.csv")
sub=pd.read_csv("test.csv")
ds=dataset.loc[:,['Pclass','Sex','Age','SibSp','Parch','Embarked']]
dssub=sub.loc[:,['Pclass','Sex','Age','SibSp','Parch','Embarked']]
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


    
X_all=np.concatenate((X,X_sub),axis=0)
y=dataset.loc[:,'Survived'].values
#Handling categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X_all[:, 1] = labelencoder_X.fit_transform(X_all[:, 1])
X_all[:, 4] = labelencoder_X.fit_transform(X_all[:, 4])
onehotencoder = OneHotEncoder(categorical_features=[0,4])
X_all = onehotencoder.fit_transform(X_all).toarray()
X=X_all[:891,:]
X_sub=X_all[891:,:]
#Splitting data
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = 0.33, random_state = 1)
#X_val, X_test, y_val, y_test = train_test_split(X_testVal, y_testVal, test_size = 0.50, random_state = 1)
xh=[]
yf=[]
#Training Classifier
for t in range(10,2500,100):
    xh.append(t)
    clf = RandomForestClassifier(n_estimators=t)
    clf.fit(X_train, y_train)
    y_pred_val=clf.predict(X_val)
    cm = confusion_matrix(y_val, y_pred_val)
    print 'confusion matrix :'
    print cm
    print 'f-score(weighted) is : '
    f1=f1_score(y_val,y_pred_val,average='weighted')
    print f1
    print 'trees ',t,'^'
    yf.append(f1)
    #print xh
    #print yf
plt.xlabel('trees')
plt.ylabel('fscore')
plt.title('trees vs fscore')
plt.plot(xh,yf)
plt.savefig('trees vs fscore validation.png')
plt.show()

'''

 y_pred_val=clf.predict(X_val)
    cm = confusion_matrix(y_val, y_pred_val)
    print 'confusion matrix :'
    print cm
    print 'f-score(weighted) is : '
    f1=f1_score(y_val,y_pred_val,average='weighted')
    print f1




from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_val, y_pred_val)
print 'confusion matrix :'
print cm
from sklearn.metrics import f1_score
print 'f-score(weighted) is : '
print f1_score(y_val,y_pred_val,average='weighted')'''
'''
y_pred_test=clf.predict(X_test)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred_test)
print 'confusion matrix :'
print cm
from sklearn.metrics import f1_score
print 'f-score(weighted) is : '
print f1_score(y_test,y_pred_test,average='weighted')
y_sub=clf.predict(X_sub)
dct={'PassengerId':pid,'Survived':y_sub}
resdf=pd.DataFrame(data=dct)
resdf.to_csv('result.csv',index=False)'''
