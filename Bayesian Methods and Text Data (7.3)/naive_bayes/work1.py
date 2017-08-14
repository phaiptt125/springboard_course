# -*- coding: utf-8 -*-
"""
Created on Sat Aug 12 02:26:21 2017

@author: phaiptt125
"""

import sys
import os
import time

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import svm
from sklearn.naive_bayes import MultinomialNB
from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report

import numpy as np
import scipy as sp
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from six.moves import range
from sklearn.model_selection import KFold

# Setup Pandas
pd.set_option('display.width', 500)
pd.set_option('display.max_columns', 100)
pd.set_option('display.notebook_repr_html', True)

# Setup Seaborn
sns.set_style("whitegrid")
sns.set_context("poster")

critics = pd.read_csv('./critics.csv')
critics = critics[~critics.quote.isnull()]
critics.head()

#...........................................................#

from sklearn.model_selection import train_test_split
_, itest = train_test_split(range(critics.shape[0]), train_size=0.7)
mask = np.zeros(critics.shape[0], dtype=np.bool)
mask[itest] = True

def make_xy(critics, vectorizer):   
    X = vectorizer.fit_transform(critics.quote)
    X = X.tocsc()  # some versions of sklearn return COO format
    y = (critics.fresh == 'fresh').values.astype(np.int)
    return X, y

def log_likelihood(clf, x, y):
    prob = clf.predict_log_proba(x)
    rotten = y == 0
    fresh = ~rotten
    return prob[rotten, 0].sum() + prob[fresh, 1].sum()

def cv_score(clf, X, y, scorefunc):
    result = 0.
    nfold = 5
    for train, test in KFold(nfold).split(X): # split data into train/test groups, 5 times
        clf.fit(X[train], y[train]) # fit the classifier, passed is as clf.
        result += scorefunc(clf, X[test], y[test]) # evaluate score function on held-out data
    return result / nfold # average

#...........................................................#

alpha_range = [0.5, 1 , 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5]
min_df_range = [1,5,10,20,30,50,100]
max_features_range = [500,1000,1500,2000,5000,10000]

best_score = -np.inf
best_min_df = None
best_max_features = None 
best_alpha = None

for min_df in min_df_range:
    for max_features in max_features_range:
        for alpha in alpha_range:
            vectorizer = TfidfVectorizer(min_df = min_df,
                                         sublinear_tf = True,
                                         ngram_range = (1, 2), 
                                         max_features = max_features,
                                         stop_words = 'english',
                                         use_idf=True)

            X, y = make_xy(critics,vectorizer)
            xtrainthis = X[mask]
            ytrainthis = y[mask]
            clf = MultinomialNB(alpha = alpha)
            cvscore = cv_score(clf, xtrainthis, ytrainthis, log_likelihood)
            if cvscore > best_score:
                best_alpha = alpha
                best_min_df = min_df
                best_max_features = max_features

#...........................................................#

vectorizer = TfidfVectorizer(min_df = best_min_df,
                             sublinear_tf = True,
                             ngram_range = (1, 2), 
                             max_features = best_max_features,
                             stop_words = 'english',
                             use_idf=True)
xtrain = X[mask]
ytrain = y[mask]
xtest = X[~mask]
ytest = y[~mask]

clf = MultinomialNB(alpha=best_alpha).fit(xtrain, ytrain)

#your turn. Print the accuracy on the test and training dataset
training_accuracy = clf.score(xtrain, ytrain)
test_accuracy = clf.score(xtest, ytest)

print('Training sets : accuracy = ' + '{0:.3f}'.format(accuracy_train))

#...........................................................#









