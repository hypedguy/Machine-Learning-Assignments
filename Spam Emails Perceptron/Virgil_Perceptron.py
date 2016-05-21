#created by Virgil
import numpy as np
import operator
import time
start_time = time.time()
Word_t = 25 #word Thresold (X)
Train_d = 4000 #email Treshold (N)

#dictionary to map each email index to wheter or not it is spam,name comes from spam/ham
sh = {}
#dictionary to help create the vocabulary
vocabd = {}
#list of the vocabulary words
vocab = []
#list of all the emails part of the training data
ems = []
#data structures to model validation data
vems = []
vsh = {}
#same of the test data
sht = {} #test data dictionary
tems = [] #test data list for emails
j = 0 #use this to control the validation data in the with down
with open("spam_train.txt", "r") as f:
    for i in range (0,5000):
        if i < Train_d:
            tdata = f.readline()
            sh[i] = int(tdata[0])
            tdata = tdata[1:]
            ems.append(tdata)
        else:
            vdata = f.readline()
            vsh[j] = int(vdata[0])
            vdata = vdata[1:]
            vems.append(vdata)
            j += 1

with open("spam_test.txt", "r") as f:
    for i in range (0, 1000):
        testdata = f.readline()
        sht [i] = int(testdata[0])
        testdata = testdata[1:]
        tems.append(testdata)
#Change the 0 in dictionaries to -1 so it matches the requirment in all the dicts
for e in range (0, Train_d):
     if sh[e] == 0:
        sh[e] = -1

for e in range (0, 5000 - Train_d):
     if vsh[e] == 0:
        vsh[e] = -1

for e in range (0, 1000):
     if sht[e] == 0:
        sht[e] = -1        
#create the vocabulary
for emails in ems:
    sw = [] #using a switch to make sure we don't count the same word multiple times in an email
    for word in emails.split():
        if word not in vocabd.keys():
            vocabd[word] = 1
        else:
            if word not in sw:
                vocabd[word] += 1
                #updating the switch so the same word can't be counted 2x
                sw.append(word)
for word in vocabd:
    #check the thresold for vocab
    if vocabd[word] >= Word_t:
        vocab.append(word)
print (len(vocab))
#creating the feature vector, feature vector for validation and feature vector for test data
fv = [[i for i in range (len(vocab))] for i in range (Train_d)]

fvv = [[i for i in range (len(vocab))] for i in range (5000 - Train_d)]

fvt = [[i for i in range (len(vocab))] for i in range (1000)]

for i in range (0, Train_d):
    for j in range (0, len(vocab)):
        if vocab[j] in ems[i]:
                    fv[i][j] = 1
        else:
                    fv[i][j] = 0

for i in range (0, 5000 - Train_d):
    for j in range (0, len(vocab)):
        if vocab[j] in vems[i]:
                    fvv[i][j] = 1
        else:
                    fvv[i][j] = 0

for i in range (0, 1000):
    for j in range (0, len(vocab)):
        if vocab[j] in tems[i]:
                    fvt[i][j] = 1
        else:
                    fvt[i][j] = 0


def classify(w, vector): #simple classify funtion 
        if np.dot(vector, w) > 0:
            return 1
        return -1

def Perceptron_error(w, data, fv): #data is a dictionary, fv a 2 dimensional vector
    global vocab #global variable to make it easier to communicate
    errors = 0
    for i in range (0, len(fv)):
        c = classify(w, fv[i]) 
        if (c == -1 and data[i] == 1 ) or (c == 1 and data[i] == -1):
             errors += 1
    print (errors)
    return float(errors/len(fv))*100
            
def Perceptron_train(data, fv, maximum_passes): 
    global vocab
    w = np.zeros(len(vocab), int) #intialize weights vector with as many 0 as features
    k = 0
    #Python said iter is already taken so I made a typo on purpose
    itter = 0
    while (itter < maximum_passes) or (maximum_passes == 0):
        sw = 0 
        itter += 1
        print ("I am on run:"+ str(itter))
        for i in range (0, len(fv)):
            c = classify(w, fv[i])
            if (c == -1 and data[i] == 1 ) or (c == 1 and data[i] == -1):
                w = np.add(w, np.multiply(data[i],fv[i]))
                sw += 1 
                k += 1
        if sw == 0: #check if it happened that there has been no missclasification last run
            break
    return w, k , itter

w = Perceptron_train(sh, np.array(fv), 0)[0] 

mapp = {} #use this to map the weitghts back to features

for i in range (len(vocab)):
    mapp[vocab[i]] = w[i]
   

sorted_mapp = sorted(mapp.items(), key=operator.itemgetter(1)) #this returnes a list of tuples of words and weights sorted by weights
print (sorted_mapp)
#print (Perceptron_train(sh, np.array(fv)))
print (Perceptron_error(w, sht, np.array(fvt)))
print("--- %s seconds ---" % (time.time() - start_time))
