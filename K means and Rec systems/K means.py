# -*- coding: utf-8 -*-
"""
Created on Wed Mar 16 10:48:59 2016

@author: virgiltataru
"""
import random
import numpy as np
import pylab as plt

MAX_ITERATIONS = 20 #maximum number of k means iteratiosns
    
def fetch_features(file):#used to only to fetch the features, no labels
    features = []
    with open(file, "r") as f:
        data = f.readlines()
        for i in data:
            features.append([float(j) for j in (i.split(","))[0:4]])
    return features

def fetch_features_and_labels(file):#used to get the clusters according to the labels
    k1 = []
    k2 = []
    k3 = []
    with open(file, "r") as f:
        data = f.readlines()
        for i in data:
            if (i.split(',')[4] == 'Iris-setosa\n'):
                k1.append([float(j) for j in (i.split(","))[0:4]])
            if (i.split(',')[4] == 'Iris-versicolor\n'):
                k2.append([float(j) for j in (i.split(","))[0:4]])
            if (i.split(',')[4] == 'Iris-virginica\n'):
                k3.append([float(j) for j in (i.split(","))[0:4]])
       
    return k1,k2,k3

def kmeans(data, k):  
    cost = []#cost functions for all variations
    variations = []#clusters variations history
    for i in range (MAX_ITERATIONS):
        print ("Itteration:" + str(i))
        centroids = random.sample(list(data),k) #randomly initialize centroids
        clusters = [[] for i in range(k)] #precompute an empty cluster list
        clusters = Dist_and_loss(data, centroids, clusters)[0] #get the clusters
        cost.append(Dist_and_loss(data, centroids, clusters)[1])#append the loss value for the configuration
        variations.append(clusters)#append the clusters to variations history  
    return variations[np.array(cost).argmin()], min(cost) #return the variation corresponding to the lowest cost and the cost



def Dist_and_loss(data, centroids, clusters): #conpute clusters and loss function
    loss = 0
    for point in data:  
        temp = [] #used to computer the distance between centroids and points
        for (i,v) in enumerate(centroids): # get a list of tuples (index, value of index) to compute distance from centroids to points
            temp.append((i, np.linalg.norm(point - v))) #append the index of the point and the distance

        minValue = min(temp, key = lambda x:x[1])#get the minimum value , sorted by the second element in the toople
        index = minValue[0] #index of the centroid 
        loss = loss + minValue[1]**2 #add the squared value to the loss function
        try:
            clusters[index].append(point) #append the element to its cluster
        except KeyError:
            clusters[index] = [point]

    return clusters, (1/(len(data))) * loss #return the clusters and the loss function

    
data = fetch_features ("K_means_Data")
#loss = []
#for i in range (2,7): #plot the loss function for various features, elbow is always there
#    loss.append(kmeans(np.array(data), i)[1])
#x = [i for i in range (2,7)]
#plt.plot(x,loss)
#plt.show()

data1 = fetch_features_and_labels ("K_means_Data")

estimation = kmeans(np.array(data), 3)[0] #get the best cluster found in MAX_ITERATIONS
missclassed = 0 
for i in range (0,3):
    bm1 = 0
    bm2 = 0 #use this to see which centroid index corresponds to which data type (since it's random initialization, we don't know )
    bm3 = 0
    for i in data1[i]:
        if any((np.array(i) == x).all() for x in estimation[0]): 
            bm1 += 1
        if any((np.array(i) == x).all() for x in estimation[1]): #see in which cluster we can find the most number of elements from a classification
            bm2 += 1
        if any((np.array(i) == x).all() for x in estimation[2]):
            bm3 += 1
    missclassed += 50 - max(bm1,bm2,bm3) # 50 (what we should have gotten) -  the best number of classified elements in a cluster 
print (missclassed)#error rate is like 10%, big variance though       
    
