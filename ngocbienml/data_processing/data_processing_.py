import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import gc
from tqdm import tqdm
from time import time
from ..utils.config import *
from ..utils.utils_ import *
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MinMaxScaler as Scale


class AssertGoodHeader(BaseEstimator, TransformerMixin):

    def __init__(self, ):
        self.columns = None

    def fit(self, X, y=None):
        self.columns = X.columns
        return self

    def transform(self, X, y=None):
        return X[self.columns]

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, y)
        return self.transform(X, y)


class Fillna(BaseEstimator, TransformerMixin):

    def __init__(self, method='mean'):
        self.num_cols = []
        self.cat_cols = []
        self.method = method
        self.fillvalues = None

    def fit(self, X, y=None):
        num_types = ['float64', 'float32', 'int8', 'int16', 'int32', 'int64', 'bool']
        self.num_cols = [col for col in X.columns if X[col].dtypes in num_types]
        self.cat_cols = [col for col in X.columns if col not in self.num_cols]
        if self.method == 'mean':
            self.fillvalues = X[self.num_cols].mean()
        elif self.method == 'median':
            self.fillvalues = X[self.num_cols].median()
        else:
            self.fillvalues = X[self.num_cols].median().map(lambda x: 0)
        return self

    def transform(self, X, y=None):
        X[self.num_cols] = X[self.num_cols].fillna(self.fillvalues)
        X[self.cat_cols] = X[self.cat_cols].fillna('missing')
        return X

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, y)
        return self.transform(X, y)


class FillnaAndDropCatFeat(BaseEstimator, TransformerMixin):

    def __init__(self):
        self.num_cols = []
        self.cat_cols = []

    def fit(self, X, y=None):
        num_types = ['float64', 'float32', 'int8', 'int16', 'int32', 'int64', 'bool']
        self.num_cols = [col for col in X.columns if X[col].dtypes in num_types]
        self.cat_cols = [col for col in X.columns if col not in self.num_cols]
        if len(self.cat_cols) > 0:
            print('Will delete %s features which are not numerics:' % len(self.cat_cols))
            print(' '.join(self.cat_cols))
        return self

    def transform(self, X, y=None):
        if len(self.cat_cols) > 0:
            X = X.drop(columns=self.cat_cols)
        X = X.replace([np.inf, -np.inf], np.nan, inplace=False)
        X = X.astype('float32')
        print('X shape=', X.shape)
        X = X.fillna(X.mean()).astype('float32')
        return X

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, y)
        return self.transform(X, y)


class LabelEncoder(BaseEstimator, TransformerMixin):
    def __init__(self):

        self.dict_ = {}
        self.cat_cols = []
        self.unknown_key = 'unknown'

    def fit(self, X, y=None):

        self.cat_cols = [col for col in X.columns if X[col].dtypes == 'object']
        if len(self.cat_cols) > 0:
            print('start to label encoder...')
        for col in self.cat_cols:
            print(col, end=' ')
            index = X[col].value_counts().index
            self.dict_[col] = dict(zip(index, range(len(index))))
            self.dict_[col][self.unknown_key] = -1
        print()
        return self

    def transform(self, X, y=None):
        # print('start to transform label encoder')
        X_ = X.copy()
        for col in self.cat_cols:
            # print(col, len(self.dict_[col]), end=' | ')
            X_[col] = X_[col].map(lambda x: x if x in self.dict_[col].keys() else self.unknown_key)
            X_[col] = X_[col].map(self.dict_[col])
        return X_

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, y)
        return self.transform(X, y)


class FeatureSelection(BaseEstimator, TransformerMixin):

    def __init__(self):
        self.to_drop = []

    def feature_selection(self, X, threshold=.01):

        def var_selection(df):
            print('start to var selection with threshold=%s' % threshold)
            list_drop = []
            for var in tqdm(df.columns):
                if df[var].std() < threshold:
                    list_drop.append(var)
            return list_drop

        def correlation_feat_selection(df):
            list_drop = []
            print('start to correlation selection feature with threshold=%s...' % (1 - threshold))
            corr = df.corr()
            for var1 in corr.index:
                for var2 in corr.columns:
                    condition = (var1 != var2) and (corr.loc[var1][var2] > 1 - threshold) \
                                and (var1 not in list_drop) and (var2 not in list_drop)
                    if condition:
                        list_drop.append(var1)
            return list_drop

        self.to_drop = correlation_feat_selection(X) + var_selection(X)
        if len(self.to_drop) > 0:
            print('to drop %s features:' % len(self.to_drop))
            print(' '.join(self.to_drop))
        return self

    def fit(self, X, y=None):

        self.feature_selection(X)
        return self

    def transform(self, X, y=None):

        X_ = X.drop(columns=self.to_drop)
        return X_

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, y)
        return self.transform(X, y)


class MinMaxScale(BaseEstimator, TransformerMixin):

    def __init__(self):
        self.index = None
        self.columns = None
        self.scale = Scale()

    def fit(self, X, y=None):
        self.columns = X.columns
        print('scale data by min max scale, X shape=', X.shape)
        self.scale.fit(X)
        return self

    def transform(self, X, y=None):
        X_ = self.scale.transform(X)
        X_ = pd.DataFrame(X_, columns=self.columns)
        return X_

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, y)
        return self.transform(X, y)


class NumericFeatureProcessing(BaseEstimator, TransformerMixin):

    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return X

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, y)
        return self.transform(X, y)
