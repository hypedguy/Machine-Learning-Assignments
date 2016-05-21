# -*- coding: utf-8 -*-
"""
Created on Wed Feb 10 16:32:35 2016

@author: virgiltataru
"""
import math as m
import numpy as np
#Hi Nishant! Unfortunately, my code for this problem set got a bit messy, I'll try to walk you through it
def FeatureNormalization(data): #nothing too fancy, just applying the formulas to list/array to get mean,std
    mean = sum(data)/len(data)
    std = m.sqrt((1/len(data))*sum((i-mean)**2 for i in data))
    return mean, std
    
def Normalize(data):#uses FeatureNormalization to return a version with normalized data
    mean,std = FeatureNormalization(data)
    result = [(i - mean)/std for i in data ] 
    return result, mean, std  #also returns the values used for normalizing

with open("housing.txt","r") as f:
    data = f.readlines()

i = 0
s = [] #auxiliary list to procees data from file extract to from file
la = [] #living area /x1
nb = [] #nr of bedrooms /x2
pr = [] #price /y

for house in data:
    house = house.replace('\n', '') #strip the newline charcters
    s.append([int (j) for j in house.split(',') if j.isdigit()]) #takes ints of raws of data we extracted

for i in range (len(s)):
    la.append(s[i][0]) 
    nb.append(s[i][1]) #divide all data in relevant fields
    pr.append(s[i][2])

with open("normalized.txt","w+") as f: #write to the file where we keep the normalized data
    f.write(str(Normalize(la)[0]) + '\n')
    f.write(str(Normalize(nb)[0])+ '\n')
    f.write(str(Normalize(pr)[0]) + '\n')
bias = [] #I work with array multiplication, so I append this to the features in order to be able to write f = x*w 
for i in range(47):
    bias.append(1) #it's only a bunch of ones that multiply with the bias, needded it to match sizes
x = np.array([bias,Normalize(la)[0], Normalize(nb)[0]]) #create final feature vector
y = np.array(Normalize(pr)[0]) #results vector

def gradientDescent(x, y, alpha, iterr):  #itter a maximum number if iterations
    w = np.zeros(3) #initialize the weights as 0's, just as in perceptron
    i = 0 # this keeps count of the number of iterations
    cost_history = []#will use this to perform checks and compare values of alpha
    while(iterr > 0):
        """
        Quick note: I originally wrote this differently but found it to be very slow with
        the continous while loops. I got the idea to write it like this after watching
        the original lecture given by Andrew Ng at Stanford on youtube(https://www.youtube.com/watch?v=5u4G23_OohI)
        as he works with multidimensional vectors and transposes (also, this version way more math than the coursera thing)
        """
        f = np.dot(x.transpose(),w)#transpose the feature vector so it matches the weights than multiply
        cost = np.sum((y - f)** 2)/(2 * len(y)) #cost formula witn 1/2m term at the end
        cost_history.append(cost) #append current cost 
        if (i > 1): #after one itteration
            if (cost_history[i] - 0.00000000000001>cost_history[i-1]): #if the cost increased at any step (I had this problem where it would increase by something ridiculous, like 10^(-100), so I added a thresold)
                return "cost increased, gradient not working" #kill the function 
            if (cost_history[i] == cost_history[i-1]):#if current cost is the same as last one, converge 
                return w, cost_history
        print ("Current run : %d || Cost: %f"%(i,cost))
        w = w - alpha * (np.dot(x, f - y) / len(y)) #update the weight vector
        iterr -= 1 #make maximum itterations smaller
        i += 1 #increase the iterations count
    return w, cost_history

n_data1 = [Normalize(la)[1],Normalize(la)[2]]
n_data2 = [Normalize(nb)[1],Normalize(nb)[2]] #get relevat data and normalize new house info
new_house_normalized = [1, (2500 - n_data1[0])/n_data1[1],(3 - n_data2[0])/n_data2[1] ]
f = np.dot((gradientDescent(x, y, 0.8, 200000000)[0]),new_house_normalized) # write the function as product of weights and data
print ((f*Normalize(pr)[2]+Normalize(pr)[1])) #print un-normalized price of the house
