# -*- coding: utf-8 -*-
"""
@author: Sandy
"""
import numpy as np
import pandas as pd
import sklearn 
import sklearn.model_selection
import sklearn.ensemble
import sklearn.tree
import sklearn.metrics

import matplotlib
import matplotlib.pyplot as plt

#read the dataSet 
data = pd.read_csv("dataset.csv")
addedcolumn = []
#I am sure there is a more elegant python way :P
for i in range(1,41):
    for j in range(0,15):
        addedcolumn.append(i)
        
        
del data['file name']
        
data.insert(loc=0, column='person category', value=addedcolumn)

all_data = data.to_numpy()


## we will take 80/20 training testing ratio = 12 rows of each   0-12  15-(15+12)
X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(all_data[:,1:], all_data[:,0], test_size=0.20, random_state=10)

# Normalizing
scaler = sklearn.preprocessing.StandardScaler().fit(X_train)
X_train= scaler.transform(X_train)
X_test= scaler.transform(X_test)

#in this cell we splitted the MFCC features and MFCC-Delta Features
MFCC_train= X_train[:,:14]
MFCC_Delta_train= X_train[:,14:]
MFCC_test= X_test[:,:14]
MFCC_Delta_test= X_test[:,14:]

######################################## Applying the random Forest Classification To identify Distinct people 

#################################################ON ALL THE COEFICCENTs
myForest = sklearn.ensemble.RandomForestClassifier(random_state=0,n_estimators=100)
myForest.fit(X_train,y_train)

#### plotting takes alooot of ram 
#for subTree in myForest.estimators_:
    #plt.figure()
    #sklearn.tree.plot_tree(subTree);
train_acc = sklearn.metrics.accuracy_score(y_train, myForest.predict(X_train))
test_acc  = sklearn.metrics.accuracy_score(y_test, myForest.predict(X_test))

print ("ON all Coefficents : Training Accuracy: %f , Testing Accuracy: %f " % (train_acc, test_acc))

##################################################ON MFCC ONLY
myForest = sklearn.ensemble.RandomForestClassifier(random_state=0,n_estimators=100)
myForest.fit(MFCC_train,y_train)
train_acc = sklearn.metrics.accuracy_score(y_train, myForest.predict(MFCC_train))
test_acc  = sklearn.metrics.accuracy_score(y_test, myForest.predict(MFCC_test))

print ("On MFCC coefficents:Training Accuracy: %f , Testing Accuracy: %f " % (train_acc, test_acc))

###################################################On MFCC DELTA
myForest = sklearn.ensemble.RandomForestClassifier(random_state=0,n_estimators=100)
myForest.fit(MFCC_Delta_train,y_train)
train_acc = sklearn.metrics.accuracy_score(y_train, myForest.predict(MFCC_Delta_train))
test_acc  = sklearn.metrics.accuracy_score(y_test, myForest.predict(MFCC_Delta_test))

print ("On MFCC DELTA coefficents:Training Accuracy: %f , Testing Accuracy: %f " % (train_acc, test_acc))


#RESULTS (INterseting how little did the mfcc delta contribute ? Is there anything wrong i did?)
#ON all Coefficents : Training Accuracy: 1.000000 , Testing Accuracy: 0.875000 
#On MFCC coefficents:Training Accuracy: 1.000000 , Testing Accuracy: 0.891667 
#On MFCC DELTA coefficents:Training Accuracy: 1.000000 , Testing Accuracy: 0.075000  




