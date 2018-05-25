import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
# SciKitLearn is a useful machine learning utilities library
import sklearn
# The sklearn dataset module helps generating datasets
import sklearn.datasets
import sklearn.linear_model
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score
dataset=pd.read_csv("train.csv")
'''
X=dataset.drop(['Cultivar 1','Cultivar 2','Cultivar 3'],axis=1).values
y=dataset.loc[:,['Cultivar 1','Cultivar 2','Cultivar 3']].values

from sklearn.model_selection import train_test_split
X_train, X_testVal, y_train, y_testVal = train_test_split(X, y, test_size = 0.40, random_state = 1)

X_val, X_test, y_val, y_test = train_test_split(X_testVal, y_testVal, test_size = 0.50, random_state = 1)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_val = sc.transform(X_val)
X_test = sc.transform(X_test)'''


def sigmoid(z):
    #np.clip(z,-1,1)
    return 1/(1+np.exp(-z))

def derivative_sigmoid(z):
    #np.clip(z,-1,1)
    return z*(1-z)
    
def initialize_parameters(nn_input_dim,nn_hdim,nn_output_dim):
    print "initializaing parameters...."
    # First layer weights
    W1 = np.random.randn(nn_input_dim, nn_hdim) * 2.0/np.sqrt((nn_input_dim))
    # First layer bias
    b1 = np.random.randn(1, nn_hdim) * 2/np.sqrt((nn_input_dim))
    
    # Second layer weights
    W2=np.random.randn(nn_hdim, nn_output_dim)* 2.0/np.sqrt((nn_hdim))
    
    b2 = np.random.randn(1,nn_output_dim) * 2.0/np.sqrt((nn_hdim))
     
    # Package and return model
    model = { 'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}
    return model 

# This is the forward propagation function
def forward_prop(model,a0):
    #print 'model is ',
    #print model
    W1, b1, W2, b2 = model['W1'], model['b1'], model['W2'], model['b2']
    z1 = a0.dot(W1) + b1
    a1 = sigmoid(z1)
    z2 = a1.dot(W2) + b2
    a2 = sigmoid(z2)
    #Store all results in these values
    cache = {'a0':a0,'z1':z1,'a1':a1,'z2':z2,'a2':a2}
    return cache

# This is the BACKWARD PROPAGATION function
def backward_prop(model,cache,y):
    W1, b1, W2, b2= model['W1'], model['b1'], model['W2'], model['b2']
    a0,a1, a2 = cache['a0'],cache['a1'],cache['a2']
    E2 = y-a2
    # Calculate loss derivative with respect to output
    grad2=derivative_sigmoid(a2)
    grad1=derivative_sigmoid(a1)
    # Calculate loss derivative with respect to second layer weights
    d2=E2*grad2
    dW2 = np.dot(a1.T,d2)
    db2 = np.sum(d2, axis=0,keepdims=True)
    E1=d2.dot(W2.T)
    d1=E1*grad1
    dW1 = np.dot(a0.T,d1)
    db1 = np.sum(d1,axis=0,keepdims=True)
    # Store gradients
    grads = {'dW2':dW2,'db2':db2,'dW1':dW1,'db1':db1}
    return grads

#TRAINING PHASE


def update_parameters(model,grads,learning_rate):
    # Load parameters
    W1, b1, W2, b2= model['W1'], model['b1'], model['W2'], model['b2']

    # Update parameters
    W1 += learning_rate * grads['dW1']
    b1 += learning_rate * grads['db1']
    W2 += learning_rate * grads['dW2']
    b2 += learning_rate * grads['db2']

    
    # Store and return parameters
    model = { 'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}
    return model
def predict(model, x):
    # Do forward pass
    c = forward_prop(model,x)
    yy=np.zeros(c['a2'].shape)
    ty=np.argmax(c['a2'],axis=1)
    k=0
    for i in ty:
        yy[k,i]=1
        k=k+1
    return yy,ty
def train(model,X_,y_,learning_rate, epochs=20000, print_loss=False):
    # Gradient descent. Loop over epochs
    for i in range(0, epochs):
        #raw_input()
        # Forward propagation
        cache = forward_prop(model,X_)
        #print 'cache is ',
        #print cache
        #a1, probs = cache['a1'],cache['a2']
        # Backpropagation
        if i%100==0:
            print "Loss: \n" + str(np.sqrt(np.sum(np.square(y_ - cache['a2']))))
        grads = backward_prop(model,cache,y_)
        # Gradient descent parameter update
        # Assign new parameters to the model
        model = update_parameters(model=model,grads=grads,learning_rate=learning_rate)
    return model
np.random.seed(0)
# This is what we return at the end
'''
model = initialize_parameters(nn_input_dim=13, nn_hdim= 6, nn_output_dim= 3)


model = train(model,X_train,y_train,learning_rate=0.01,epochs=20000,print_loss=True)
y_pred_val,ty=predict(model,X_val)
from sklearn.metrics import f1_score
print 'f-score(weighted) is : '
print f1_score(y_val,y_pred_val,average='weighted')
y_pred_test,ty=predict(model,X_test)
from sklearn.metrics import f1_score
print 'f-score(weighted) is : '
print f1_score(y_test,y_pred_test,average='weighted')'''

