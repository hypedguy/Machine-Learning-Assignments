# -*- coding: utf-8 -*-
"""
Created on Sun Mar  6 15:00:17 2016

@author: virgiltataru
"""
import scipy as sp #use this to get the text data
from sklearn import svm #use this for support vectors
import matplotlib.pyplot as plt #plot some stuff
from sklearn import cross_validation #use this to do cross validation on different settings
from sklearn.grid_search import GridSearchCV#use this find optimal parameters

"""A few notes about my implementation:
    - using no kernel was kind of a challenge, as the only way I found was trying to
        pass the kernel argument as "precomputed", which will always give either an
        error relating to sizes not matching or a space error
    - I did a comparison between a gaussian and a linear kernel instead
    - the function I used to find the optimal paramaters takes quite a well to run,
        as there are a lot of cross validations to be done in a high dimensionality,
        I think around 5 minutes, so I commented out that part of the code, feel free 
        to test it if you'd like
    - I also graphed the cross validation errors for the 3 configurations"""

def data_fetch(file): #fetch relevant data from the files
    data = sp.genfromtxt(file, delimiter=',')
    y = data[:, 0]
    x = data[:, 1:]
    return x,y

def scale (X): #function to scale the data
    for i in range (len(X)):
        for j in range (len(X[0])):
            X[i][j] = 2*X[i][j]/255 - 1
    return X

x_test, y_test = data_fetch("mnist_test.txt") #get data
x_train, y_train = data_fetch("mnist_train.txt")

scale (x_test) #scale the features
scale (x_train)

svc = svm.SVC(kernel='rbf',C=1) #gaussian kernel with normal parameters
svc.fit(x_train, y_train)

y_predict = svc.predict(x_test) #get the predictions
s = 0
for i in range (len(y_predict)):
    if (y_predict[i] == y_test[i]):
        s += 1 
gaus = ((len(y_predict)-s)/len(y_predict))*100 # error rate is 7.8% on the test data with C=1 and default gama 
print ("Error rate for C=1 and default gamma: %s" %gaus)
gaus_scores = cross_validation.cross_val_score(svc, x_test, y_predict, cv = 5) #cross validate on different parts of the test data (avg error rate is 8-10%) 

svc = svm.SVC(kernel='rbf',C=10) #linear kernel
svc.fit(x_train, y_train)

y_predict = svc.predict(x_test)
s = 0
for i in range (len(y_predict)):
    if (y_predict[i] == y_test[i]):
        s += 1 
linear = s/len(y_predict) * 10 #linear kerner gives a slightly higher error rate on test data- 9%
print ("Error rate for a linear kernel:%s" %linear)
linear_scores = cross_validation.cross_val_score(svc, x_test, y_predict, cv = 5) #cross validate on different parts of the test data (avg error rate is 8-10%) 

svc = svm.SVC(kernel='rbf',C=10, gamma = 0.001) #gaussian kernel with optimal parameters
svc.fit(x_train, y_train)

y_predict = svc.predict(x_test) #get the predictions
s = 0
for i in range (len(y_predict)):
    if (y_predict[i] == y_test[i]):
        s += 1 
optimal = ((len(y_predict)-s)/len(y_predict))*100 # error rate is 7.8% on the test data with C=1 and default gama 
print ("Error rate for optimal C and gamma: %s" %optimal)

optimal_scores = cross_validation.cross_val_score(svc, x_test, y_predict, cv = 5)

plt.plot(gaus_scores)
plt.plot(linear_scores)
plt.plot(optimal_scores)
plt.legend(["Gauss Kernel Scores", "Linear Kernel Scores", "Optimal config Scores"])
plt.show() #Gaussian Kernel gives, in general, much better results

#explore different Cs and gammas using cross validation, takes very long to run
parameters= [{'C': [1, 10, 20, 100, 1000], 'gamma': [1,0.1, 0.001, 0.0001], 'kernel': ['rbf']},]
clf = GridSearchCV(svm.SVC(C=1), parameters, cv=5) #this takes very, very long to run
clf.fit(x_train, y_train)
print (clf.best_params_) #optimal parameters are  {'C': 10, 'gamma': 0.001, 'kernel': 'rbf'}
