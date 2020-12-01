import matplotlib.pyplot as plt
import sklearn
import sklearn.ensemble
import numpy as np   
import pandas as pd
import scipy
import sklearn.metrics
import IPython.display as ipd
import seaborn as sns
import warnings
from sklearn.manifold import TSNE
from mpl_toolkits.mplot3d import Axes3D
from train_knn import performance


def Forest(xtrn, ytrn, **kwargs):
    '''
    This function fits the best haperparameters for random forests.
    '''
    param_rand_forest = {'n_estimators':[np.random.randint(1, 100)]}
    mfcc_forest = sklearn.ensemble.RandomForestClassifier()

    rand_mfcc_forest = sklearn.model_selection.GridSearchCV(mfcc_forest,
                                                            param_rand_forest,
                                                            cv=10, **kwargs)
    rand_mfcc_forest.fit(xtrn, ytrn)

    rand_mfcc_forest_score = rand_mfcc_forest.best_score_*100

    print("The best score is: %.2f" % rand_mfcc_forest_score)
    print("The best patameters is:", rand_mfcc_forest.best_params_)
    return rand_mfcc_forest.best_estimator_


def forest_best(X_train, X_test, y_train, y_test):
    best_mfcc_model = Forest(X_train, y_train, verbose=1)
    performance(best_mfcc_model, X_train, X_test, y_train, y_test)
