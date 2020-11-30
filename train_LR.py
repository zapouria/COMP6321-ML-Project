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


def LR_random(xtrn, xtst, ytrn, ytst, **kwargs):
    '''
    Finds best hyperparameter using Random Search
    '''
    param_grid = {'C': scipy.stats.reciprocal(0.1, 1)}
    lg_model = sklearn.linear_model.LogisticRegression()
    logistic_mfcc_model = sklearn.model_selection.RandomizedSearchCV(lg_model, param_grid, cv=10, **kwargs)
    logistic_mfcc_model.fit(xtrn, ytrn)

    log_reg_score = logistic_mfcc_model.best_score_*100

    print("The best Logistic Regression score is: %.2f" % log_reg_score)
    print("The best Logistic Regression grid search  patameters is:", logistic_mfcc_model.best_params_)
    return logistic_mfcc_model.best_estimator_, logistic_mfcc_model.best_estimator_.score(xtst, ytst)*100


def LR_grid(xtrn, xtst, ytrn, ytst, **kwargs):
    '''
    Finds best hyperparameter using GridSearch
    '''
    param_grid = {'C': np.logspace(-1, 0, 10)}
    lg_model = sklearn.linear_model.LogisticRegression()
    logistic_mfcc_model = sklearn.model_selection.GridSearchCV(lg_model, param_grid,cv=10, **kwargs)
    logistic_mfcc_model.fit(xtrn, ytrn)

    log_reg_score = logistic_mfcc_model.best_score_*100
    print("The best Logistic Regression score is: %.2f" % log_reg_score)
    print("The best Logistic Regression grid search  patameters is:", logistic_mfcc_model.best_params_)
    return logistic_mfcc_model.best_estimator_, logistic_mfcc_model.best_estimator_.score(xtst, ytst)*100
