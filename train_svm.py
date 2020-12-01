import matplotlib.pyplot as plt
import sklearn
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


def SVM(xtrn, xtst, ytrn, ytst, **kwargs):
    '''
    In this function the best Linear SVC is found and applied on the data
    using RandomizedSearch. The outpout is test accuracy and best fitted model.
    '''
    SVC = sklearn.svm.LinearSVC(**kwargs)
    param_SVM = {'C': scipy.stats.reciprocal(1, 100)}
    SVM_search = sklearn.model_selection.RandomizedSearchCV(SVC, param_SVM,
                                                            random_state=0,
                                                            verbose=1,
                                                            n_iter=20, cv=10)

    SVM_search.fit(xtrn, ytrn)

    SVC_Score = SVM_search.best_score_*100
    print("The best score is: %.2f" % SVC_Score)
    print("The best patameters is:", SVM_search.best_params_)

    # Testing Scores
    SVCtest = SVM_search.best_estimator_.score(xtst, ytst)*100
    return SVCtest, SVM_search.best_estimator_


def svm_best(X_train, X_test, y_train, y_test, multi_class):
    _, SVMmodel = SVM(X_train, X_test, y_train, y_test, multi_class=multi_class) #MFCC
    performance(SVMmodel, X_train, X_test, y_train, y_test)
