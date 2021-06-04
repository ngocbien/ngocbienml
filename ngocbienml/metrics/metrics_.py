from __future__ import print_function

from sklearn.metrics import *
import pandas as pd
import numpy as np
import pandas as pd


def binary_score(model, test, y_test, name='test', silent=False):

    from sklearn.metrics import accuracy_score, balanced_accuracy_score, recall_score, precision_score
    from collections import Counter
    try:
        pred_test_proba = model.predict_proba(test)
        if len(pred_test_proba.shape) > 1:
            try:
                pred_test_proba = pred_test_proba[:, 1]
            except IndexError:
                pred_test_proba = pred_test_proba.reshape(-1,)
    except AttributeError:
        pred_test_proba = model.predict(test)
    gini_test = gini(y_test, pred_test_proba)
    best_threshold = find_best_threshold(y_test, pred_test_proba)
    pred_test = pred_test_proba > best_threshold
    recall = recall_score(y_test, pred_test)
    precision = precision_score(y_test, pred_test)
    if silent:
        return gini_test, recall
    print('*'*80)
    print('Best threshold finding on this test set = %.4f' % best_threshold)
    print('on %s' % name.upper())
    print('AUC = %.3f' % ((gini_test+1)/2), end=' | ')
    print('Gini = %.3f ' % gini_test, end=' | ')
    print('Recall score=%.3f' % recall, end=' | ')
    print('Precision score=%.3f' % precision)
    print('confusion matrix:')
    df = pd.DataFrame(confusion_matrix(y_test, pred_test),
                      index=['Actual Negative', 'Actual Positive'],
                      columns=['Predict Negative', 'Predict Positive'])
    df['SUM'] = df.sum(axis=1)
    df.loc['SUM'] = df.sum()
    print(df)


def binary_score_(y_test, pred_test_proba, name='test'):

    from sklearn.metrics import accuracy_score, balanced_accuracy_score, recall_score, precision_score
    from collections import Counter
    if len(pred_test_proba.shape) > 1:
        try:
            pred_test_proba=pred_test_proba[:, 1]
        except IndexError:
            pred_test_proba=pred_test_proba.reshape(-1,)
    best_threshold = find_best_threshold(y_test, pred_test_proba)
    pred_test_label = pred_test_proba > best_threshold
    gini_test = gini(y_test, pred_test_proba)

    print('*'*80)
    print('Best threshold finding on this test set=%.4f'%best_threshold)
    print('on %s' % name.upper())
    print('AUC = %.3f ' % ((gini_test+1)/2), end=' | ')
    print('Gini = %.3f ' % gini_test, end=' | ')
    recall = recall_score(y_test, pred_test_label)
    print('Recall score=%.3f' % recall, end=' | ')
    precision = precision_score(y_test, pred_test_label)
    print('Precision score = %.3f' % precision)
    print('confusion matrix:')
    df = pd.DataFrame(confusion_matrix(y_test, pred_test_label),
                      index=['Actual Negative', 'Actual Positive'],
                      columns=['Predict Negative', 'Predict Positive'])
    df['SUM'] = df.sum(axis=1)
    df.loc['SUM'] = df.sum()
    print(df)


def multiclass_score(model, test, y_test, name):

    from sklearn.metrics import balanced_accuracy_score, accuracy_score
    pred_test = model.predict(test)
    if len(pred_test.shape) > 1:
        pred_test = np.argmax(pred_test, axis=1)
    acc_test = balanced_accuracy_score(y_test, pred_test)*100
    print('*'*80)
    print('On %s: Balanced accuracy  = %.3f %%' % (name, acc_test), end=' | ')
    print('accuracy = %.3f %%' % accuracy_score(y_test, pred_test)*100, end=' | ')
    index = ['True_%s' % str(i) for i in range(len(np.unique(y_test)))]
    columns = ['Pred_%s' % str(i) for i in range(len(np.unique(y_test)))]
    df = pd.DataFrame(confusion_matrix(y_test, pred_test), index=index, columns=columns)
    print('Confusion matrix:')
    print(df)


def gini(actual, pred):

    from sklearn.metrics import roc_curve, auc
    fpr, tpr, thresholds = roc_curve(actual, pred, pos_label=1)
    auc_score = auc(fpr,tpr)
    return 2*auc_score-1


def confusion_matrix(actual, pred):

    from sklearn.metrics import confusion_matrix
    return confusion_matrix(actual, pred)


def KfoldWithoutCv(models, test, y_test, name='test', threshold=.5):

    from sklearn.metrics import accuracy_score, balanced_accuracy_score, recall_score, precision_score
    from collections import Counter
    ginis = []
    recalls = []
    AUCS = []
    precisions = []
    df = pd.DataFrame(np.zeros((2, 2)), index=['Actual Negative', 'Actual Positive'],
                      columns=['Predict Negative', 'Predict Positive'])
    for model in models:
        try:
            pred_test_proba = model.predict_proba(test.copy())
            if len(pred_test_proba.shape) > 1:
                pred_test_proba = pred_test_proba[:, 1]
        except AttributeError:
            pred_test_proba = model.predict(test.copy())
        gini_ = gini(y_test, pred_test_proba)
        auc_ = (gini_+1)/2
        ginis.append(gini_)
        AUCS.append(auc_)
        best_threshold = find_best_threshold(y_test, pred_test_proba)
        pred_test = pred_test_proba > best_threshold
        recall_ = recall_score(y_test, pred_test)*100
        recalls.append(recall_)
        precision_ = precision_score(y_test, pred_test)*100
        precisions.append(precision_)
        df = df.add(pd.DataFrame(confusion_matrix(y_test, pred_test), index=['Actual Negative', 'Actual Positive'],
                                 columns=['Predict Negative', 'Predict Positive']))
    recalls = np.array(recalls)
    precisions = np.array(precisions)
    ginis = np.array(ginis)
    AUCS = np.array(AUCS)
    print('*'*80)
    print('Best threshold finding on this test set = %.4f' % best_threshold)
    print('on %s'%name.upper())
    print('AUC = %.3f+/-%.03f%%' % (AUCS.mean(), AUCS.std()), end=' | ')
    print('Gini = %.3f+/-%.03f' % (ginis.mean(), ginis.std()), end=' | ')
    print('Recall = %.3f+/-%.03f%%' % (recalls.mean(), recalls.std()), end=' | ')
    print('Precision = %.3f+/-%.03f%%' % (precisions.mean(), precisions.std()))
    print('confusion matrix:')
    df = df/len(models)
    df['SUM'] = df.sum(axis=1)
    df.loc['SUM'] = df.sum()
    print(df)


def binary_scoreKfold(models, cv, X, y, get_score=False):
    '''
    :param models: list of model for Cv
    :param cv: cross validation kFolds
    :param X: Data
    :param y: target
    :param get_score: If true, return the score of all data set X, from test set in each kfold
    :return: score_ if get_score is set true
    '''
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, recall_score, precision_score
    from collections import Counter
    gini_trains, gini_tests = [], []
    recall_trains, recall_tests = [], []
    auc_trains, auc_tests = [], []
    precision_trains, precision_tests = [], []
    df_train = pd.DataFrame(np.zeros((2, 2)), index=['Actual Negative', 'Actual Positive'],
                            columns=['Predict Negative', 'Predict Positive'])
    df_test = pd.DataFrame(np.zeros((2, 2)), index=['Actual Negative', 'Actual Positive'],
                           columns=['Predict Negative', 'Predict Positive'])
    if y is not None:
        if not X.index.equals(y.index):
            X = X.reset_index(drop=True)
            y = y.reset_index(drop=True)
    for model, (index_train, index_test) in zip(models, cv.split(X, y)):
        train, test = X.iloc[index_train], X.iloc[index_test]
        y_train, y_test = y.iloc[index_train], y.iloc[index_test]
        try:
            pred_test_proba = model.predict_proba(test)
            try:
                score_ = np.append(score_, pred_test_proba[:, 1])
            except NameError:
                score_ = pred_test_proba[:, 1]
            pred_train_proba = model.predict_proba(train)
            if len(pred_test_proba.shape) > 1:
                pred_test_proba = pred_test_proba[:, 1]
                pred_train_proba = pred_train_proba[:, 1]
        except AttributeError:
            pred_test_proba = model.predict(test)
            pred_train_proba = model.predict(train)
        gini_train = gini(y_train, pred_train_proba)
        gini_test = gini(y_test, pred_test_proba)
        auc_train = (gini_train+1)/2
        auc_test = (gini_test+1)/2
        auc_trains.append(auc_train)
        auc_tests.append(auc_test)
        gini_trains.append(gini_train)
        gini_tests.append(gini_test)
        best_threshold = find_best_threshold(y_test, pred_test_proba)
        pred_test = pred_test_proba > best_threshold
        pred_train = pred_train_proba > best_threshold
        recall_tests.append(recall_score(y_test, pred_test)*100)
        recall_trains.append(recall_score(y_train, pred_train)*100)
        precision_trains.append(precision_score(y_test, pred_test)*100)
        precision_tests.append(precision_score(y_train, pred_train)*100)
        df_test = df_test.add(pd.DataFrame(confusion_matrix(y_test, pred_test),
                                           index=['Actual Negative', 'Actual Positive'],
                                           columns=['Predict Negative', 'Predict Positive']))
        df_train = df_train.add(pd.DataFrame(confusion_matrix(y_train, pred_train),
                                             index=['Actual Negative', 'Actual Positive'],
                                             columns=['Predict Negative', 'Predict Positive']))
    if get_score:
        return score_
    recall_trains, recall_tests = np.array(recall_trains), np.array(recall_tests)
    auc_trains, auc_tests = np.array(auc_trains), np.array(auc_tests)
    gini_trains, gini_tests = np.array(gini_trains), np.array(gini_tests)
    precision_trains, precision_tests = np.array(precision_trains), np.array(precision_tests)

    print('*'*80)
    print('Best threshold finding on test set=%.4f'%best_threshold)
    print('ON TRAIN')
    print('AUC = %.3f+/-%.03f%%' % (auc_trains.mean(), auc_trains.std()), end=' | ')
    print('Gini = %.3f+/-%.03f' % (gini_trains.mean(), gini_trains.std()), end=' | ')
    print('Recall = %.3f+/-%.03f%%' % (recall_trains.mean(), recall_trains.std()), end=' | ')
    print('Precision = %.3f+/-%.03f%%' % (precision_trains.mean(), precision_trains.std()))
    print('confusion matrix:')
    df_train = df_train/len(models)
    df_train['sum'] = df_train.sum(axis=1)
    df_train.loc['sum'] = df_train.sum()
    print(df_train)

    print('*'*80)
    print('ON TEST')
    print('AUC = %.3f+/-%.03f%%' % (auc_tests.mean(), auc_tests.std()), end=' | ')
    print('Gini = %.3f+/-%.03f' % (gini_tests.mean(), gini_tests.std()), end=' | ')
    print('Recall = %.3f+/-%.03f%%' % (recall_tests.mean(), recall_tests.std()), end=' | ')
    print('Precision = %.3f+/-%.03f%%' % (precision_tests.mean(), precision_tests.std()))
    print('confusion matrix:')
    df_test = df_test/len(models)
    df_test['SUM'] = df_test.sum(axis=1)
    df_test.loc['SUM'] = df_test.sum()
    print(df_test)


def find_best_threshold(y_test, y_proba, metric='f1'):
    '''
    :param y_test: label of test set
    :param y_proba: predict probability
    :param metric: to find the best. It may be f1 or Gmeans.
    Actually, available only for f1
    :return: best threshold by given metric
    '''
    from numpy import argmax
    from sklearn.datasets import make_classification
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import precision_recall_curve

    if len(y_proba.shape) > 1:
        y_proba=y_proba[:, 1]
    precision, recall, thresholds = precision_recall_curve(y_test, y_proba)
    # locate the index of the largest f score
    #print('precision contains non numeric=', np.isnan(precision).any())
    #print('recall contains non numeric=',np.isnan(recall).any())
    funct = np.vectorize(lambda x, y: -1 if (x <= 0 or y <= 0) else 2*x*y/(x+y))
    fscore = funct(precision, recall)
    #print('fscore contains non numeric=', np.isnan(fscore).any())
    #print('fscore contains inf =', np.isinf(fscore).any())
    #ix = argmax(fscore)
    max_value = np.max(fscore)
    #print('max value = %.3f'%max_value)
    ix = np.where(fscore==max_value)[0][0]
    #print('Best Threshold=%f, F-Score=%.3f' %(thresholds[ix], fscore[ix]))
    best_threshold = thresholds[ix]
    return best_threshold



