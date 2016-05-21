# -*- coding: utf-8 -*-
"""
Created on Tue Apr 12 15:29:26 2016

@author: virgiltataru
"""
import numpy as np
import operator
#Get the data form the files
test_labels = np.genfromtxt("ps5_data-labels.csv", delimiter=',')
       
test_data = np.genfromtxt("ps5_data.csv", delimiter=',')
                
weights_1 = np.genfromtxt("ps5_theta1.csv", delimiter=',')

weights_2 = np.genfromtxt("ps5_theta2.csv", delimiter=',')

def sig(x): #sigmunoid function
    return 1/(1+np.exp(-x))     

loss = 0
def neuron(prev_input,weights):
    s = []#store the values for the neurons
    for w in weights:
        result = sig((np.dot(prev_input, w[1:]) + w[0] )) #activation function
        s.append(result)
    return s

wrong = 0#count the misclassified examples
s = [0*i for i in range (len(test_data))]
loss = 0#initialize the loss
reg = 0
for i in weights_1: #compute regularization term
    for j in i:
        reg = reg + j**2
for i in weights_2:
    for j in i:
        reg = reg + j**2
reg = reg/(2*len(test_data))
for i in range(len(test_data)):
    l_1 = neuron(test_data[i], weights_1) #calculate first layer
    l_2 = neuron(l_1, weights_2) #compute second layer
    index, value = max(enumerate(l_2), key=operator.itemgetter(1) ) #find the index of the most likely outcome and  
    if (index + 1 !=  test_labels[i]): #update the count for wrong (+1 because list indexing starts at 0)
        s[i] == 1
        wrong += 1
    for j in range (len(l_2)):
        loss += s[j] * np.log(l_2[j]) + (1 - s[j])*np.log(1-l_2[j])#update the cost function  

J = - loss / len(test_data) + reg#computer the cost
print ("Error rate on the test:" + str(wrong / len(test_data) * 100) + "%") # print the erro rate 
print (J)