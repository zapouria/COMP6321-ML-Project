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

import matplotlib.pyplot as plt

#read the dataSet 
data= pd.read_csv('dataset.CSV')
y= data['Speaker'].to_numpy()
all_data = data.to_numpy()
## we will take 80/20 training testing ratio
X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(all_data[:,2:], y,
                                                                            test_size=0.3, random_state=10)
#Input Normalization
scaler = sklearn.preprocessing.StandardScaler().fit(X_train)
X_train= scaler.transform(X_train)
X_test= scaler.transform(X_test)

#In this following we splitted the MFCC features and MFCC-Delta Features
MFCC_train= X_train[:,:14]
MFCC_Delta_train= X_train[:,14:]
MFCC_test= X_test[:,:14]
MFCC_Delta_test= X_test[:,14:]

######################################## Applying the random Forest Classification To identify Distinct people 



##################################################ON MFCC ONLY
myMfccForest = sklearn.ensemble.RandomForestClassifier(random_state=0,n_estimators=100)
myMfccForest.fit(MFCC_train,y_train)
train_acc = sklearn.metrics.accuracy_score(y_train, myMfccForest.predict(MFCC_train))
test_acc  = sklearn.metrics.accuracy_score(y_test, myMfccForest.predict(MFCC_test))

print ("On MFCC coefficents:Random Forest:Training Accuracy: %f , Testing Accuracy: %f " % (train_acc, test_acc))

#PLotting confusion matrix for MFCC Forest
fig, ax = plt.subplots(figsize=(20, 20))
sklearn.metrics.plot_confusion_matrix(myMfccForest, MFCC_test,y_test, ax=ax)
plt.show()

#################################################ON ALL THE COEFICCENTs inclding delta
myDeltaForest = sklearn.ensemble.RandomForestClassifier(random_state=0,n_estimators=100,max_features = "auto")
myDeltaForest.fit(X_train,y_train)

train_acc = sklearn.metrics.accuracy_score(y_train, myDeltaForest.predict(X_train))
test_acc  = sklearn.metrics.accuracy_score(y_test, myDeltaForest.predict(X_test))


print ("ON all Coefficents:Random Forest : Training Accuracy: %f , Testing Accuracy: %f " % (train_acc, test_acc))

#PLotting confusion matrix for MFCC DELTA Forest
fig, ax = plt.subplots(figsize=(20, 20))
sklearn.metrics.plot_confusion_matrix(myDeltaForest, X_test,y_test, ax=ax)
plt.show()


###############################################Applying Logisstic Regression to identify distinct people
###################ON MFCC COEF
#Since we have a multi class problem and since lbfgs is a powerful gradient descent 
#When applyiny none penality
logistic_model = sklearn.linear_model.LogisticRegression(fit_intercept=False, penalty='none', solver='lbfgs')
logistic_model.fit(MFCC_train, y_train);

log_train_acc = sklearn.metrics.accuracy_score(y_train, logistic_model.predict(MFCC_train))
log_test_acc  = sklearn.metrics.accuracy_score(y_test, logistic_model.predict(MFCC_test))

print ("ON MFCC Coefficents:Logistic Regression 'none penality' : Training Accuracy: %f , Testing Accuracy: %f " % (log_train_acc, log_test_acc))

logistic_model = sklearn.linear_model.LogisticRegression(fit_intercept=False, penalty='l2', solver='lbfgs')
logistic_model.fit(MFCC_train, y_train);

log_train_acc = sklearn.metrics.accuracy_score(y_train, logistic_model.predict(MFCC_train))
log_test_acc  = sklearn.metrics.accuracy_score(y_test, logistic_model.predict(MFCC_test))
print ("ON MFCC Coefficents:Logistic Regression 'l2 penality': Training Accuracy: %f , Testing Accuracy: %f " % (log_train_acc, log_test_acc))

###################ON MFCC DELTA

#SINCE l2 penality preformed better (reduced the over fitting we will not repeat the 'penality =none' parameter )
logistic_delta_model = sklearn.linear_model.LogisticRegression(fit_intercept=False, penalty='l2', solver='lbfgs')
logistic_delta_model.fit(X_train, y_train);

log_delta_train_acc = sklearn.metrics.accuracy_score(y_train, logistic_delta_model.predict(X_train))
log_delta_test_acc  = sklearn.metrics.accuracy_score(y_test, logistic_delta_model.predict(X_test))

print ("ON ALL (DELTA) Coefficents:Logistic Regression 'l2 penality' : Training Accuracy: %f , Testing Accuracy: %f " % (log_delta_train_acc, log_delta_test_acc))
