# -*- coding: utf-8 -*-
"""
Created on Fri May 18 12:00:42 2018

@author: gaurav.jain
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.model_selection import train_test_split
data1 = pd.read_csv("C:\\Users\\gaurav.jain\\Downloads\\machine-learning-ex1\\machine-learning-ex1\\ex1\\ex1data1.txt", header=None)
#print(data1)
X = data1.iloc[:,0]
Y = data1.iloc[:,1]
#ax = plt.scatter(X,y=Y)
#ax.set_xlabel("PROFIT")
#plt.ylabel=""
#data1.plot(kind='scatter', x=X, y=Y, figsize=(12,8))
fig, ax = plt.subplots(figsize=(12,8))
ax.scatter(X,Y,)

print(X)
print(X.shape)
X = X.values.reshape(-1,1)
#Y = Y.values.reshape(1,-1)
# Q2
X_train,X_test,Y_train,Y_test = train_test_split(X,Y)
lr = linear_model.LinearRegression()
lr.fit(X_train,Y_train)
print(Y_test)
print(lr.score(X_test,Y_test))
print(lr.predict([[15]]))
print(lr.get_params)
print(lr.coef_)
print(lr.intercept_)
'''
Gradient Descent
Q3
'''

    
    
    