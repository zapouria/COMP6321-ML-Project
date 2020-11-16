# -*- coding: utf-8 -*-
"""
@author: Sandy
"""
import numpy as np
import pandas as pd
import sklearn 
import sklearn.model_selection
#read the dataSet 
data = pd.read_csv("dataset.csv")
addedcolumn = []
for i in range(1,41):
    for j in range(0,15):
        addedcolumn.append(i)
        
        
del data['file name']
        
data.insert(loc=0, column='person category', value=addedcolumn)

all_data = data.to_numpy()


## we will take 80/20 training testing ratio = 12 rows of each   0-12  15-(15+12)
X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(all_data[:,1:], all_data[:,0], test_size=0.20, random_state=10)




