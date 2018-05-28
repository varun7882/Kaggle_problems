import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pylab as py
# SciKitLearn is a useful machine learning utilities library
import sklearn
# The sklearn dataset module helps generating datasets
import sklearn.datasets
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score
df=pd.read_csv("train.csv")
colors=np.where(df.Survived==0,'r','g')
plt.scatter(df.Parch,df.Fare,c=colors)

plt.show()



