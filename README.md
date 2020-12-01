# COMP6321-ML-Project
## README for group 18

Our project relies on numpy, sklearn,librosa, scipy, IPython, pandas, pysndfx,matplotlib.
Our code submission contains the following files:

    ./environment.yaml              <-- conda environment for our project
    ./report.ipynb                  <-- notebook to generate figues, tables and examples
    ./dataset.csv                   <-- csv data set after preprocessing the voices
    ./feature_extraction_util.py    <-- some utility for feature extraction
    ./feature_to_csv.py             <-- script to preprocess the data and extract them into csv file
    ./train_forest.py               <-- script to train our Random Forest
    ./train_knn.py                  <-- script to train our KNN
    ./train_LR.py                   <-- script to train our Logistic Regression
    ./train_svm.py                  <-- script to train our SVM

* Runing the scripts by order in the report.ipynb you can see the steps that we took for this project and you can see the output. You can run Feature extraction, however it will takes time since we are working on 950 voices.
* GPU is not required.
* The report notebook saves files to an "out" directory.
* This is the link to our data sets : https://www.dropbox.com/sh/v0vivtco9krfcxr/AACUp5i3GC1mYDCKdYYbGrIUa?dl=0

The data set was downloaded from http://www.openslr.org/12/
