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


def KNN(xtrn, xtst, ytrn, ytst, **kwargs):
    '''
    In this function the best K-Nearest-Neighborhood is found and applied using GridSearch. 
    The outpout is training and test accuracy the best model.
    '''
    param_KNN = {'n_neighbors':[1, 5, 10, 13, 15, 20, 25, 30, 40], 'weights': ['uniform', 'distance']} 
    KNN = sklearn.neighbors.KNeighborsClassifier()
    GridSearch = sklearn.model_selection.GridSearchCV(KNN, param_KNN, cv=10, **kwargs).fit(xtrn, ytrn)
    KNN_Score = GridSearch.best_score_*100
    print("The best score is: %.2f" % KNN_Score)
    print("The best patameters is:", GridSearch.best_params_)

    GStrain = GridSearch.best_estimator_.score(xtrn, ytrn)*100
    GStest = GridSearch.best_estimator_.score(xtst, ytst)*100

    return GStest, GridSearch.best_estimator_


def performance(model, xtrn, xtst, ytrn, ytst):
    '''
    In this function the confusion matrix and scores are printed.
    '''
    # Training and testing Scores
    train = model.score(xtrn, ytrn)*100
    test = model.score(xtst, ytst)*100
    print('%.1f%% train accuracy' % train)
    print('%.1f%% test accuracy' % test)
    # F1 Score
    model_F1_metric = sklearn.metrics.f1_score(ytst, model.predict(xtst), average='macro')
    print('model_F1_metric:', model_F1_metric)

    # Confusion Matrix
    fig, ax = plt.subplots(figsize=(8, 8))
    sklearn.metrics.plot_confusion_matrix(model, xtst, ytst, ax=ax)
    plt.show()


def KNN_best(X_train, X_test, y_train, y_test):
    _, model = KNN(X_train, X_test, y_train, y_test, verbose=1)
    performance(model, X_train, X_test, y_train, y_test)


def best_scores(X_train, X_test, y_train, y_test):
    # In this cell KNN is applied to the reduced dimension of X_train into different dimensions to see which one has
    # the best accuracy
    score = []
    for i in range(1, 26):
        PCA = sklearn.decomposition.PCA(n_components=i).fit(X_train)
        PCA_train = PCA.transform(X_train)
        PCA_test = PCA.transform(X_test)
        test_score, _ = KNN(PCA_train, PCA_test, y_train, y_test)
        score.append(test_score)
    print('\033[94m', "Best Dimension is", score.index(max(score)), 'the max accuracy is', max(score), '\033[0m')
