from sklearn.base import BaseEstimator, TransformerMixin

import gc
from time import time

from ..utils.config import *
from ..visualization.plot import Plot, plot_aucKfold, plot_importance_Kfold, plot_train_test, plot_precision_recall_curve
from ..visualization.plot import __giniKfold__, __gini__
from sklearn.model_selection import train_test_split
from ..utils.utils_ import *
from ..metrics.metrics_ import binary_score, multiclass_score, binary_scoreKfold, KfoldWithoutCv
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
import os
from tqdm import tqdm
os.environ['KMP_WARNINGS'] = '0'


params_ = {'feature_fraction': 0.7319044104939286,
          'max_depth': 65,
          'min_child_weight': 1e-05,
          'min_data_in_leaf': 47,
          'n_estimators': 497,
          'num_leaves': 45,
          'reg_alpha': 0,
          'reg_lambda': 50,
          'metric': 'auc',
          'eval_metric': 'auc',
          'is_unbalance': True,
          'subsample': 0.5380272330885912}


class ModelWithPipeline(BaseEstimator, TransformerMixin):

    def __init__(self,  model_name='lgb', epochs=200, pred_leaf=False, params=None, **kwargs):

        if params is not None:
            self.params = params
        else:
            self.params = params_
        self.kwargs = kwargs
        self.model_name = model_name.lower()
        self.epochs = epochs
        self.pred_leaf = pred_leaf
        self.threshold = .5
        self.plot = True

    def fit(self, X, y=None):

        if self.model_name == 'lgb':
            self.model = lgb.LGBMClassifier(**self.params)
        elif 'regression' in self.model_name.lower():
            from sklearn.linear_model import LogisticRegression
            self.model = LogisticRegression()
        else:
            print(' please select a model to training classifier')
            # raise KeyError
        print('Input data to trainning has shape=', X.shape)
        assert len(X)==len(y)
        print('split data in train, test by 90:10')
        X1,  X2, y1, y2 = train_test_split(X, y, random_state=49, test_size=.1, stratify=y)
        if self.model_name.lower()=='dl' or 'deep' in self.model_name.lower():
            print('Using Deep learning model...')
            from keras.callbacks import ModelCheckpoint
            from keras.layers import Dropout, Dense
            from keras.models import Sequential, load_model
            from numpy.testing import assert_allclose
            import keras
            try:
                self.hidden_layers = self.kwargs['hidden_layers']
            except KeyError:
                self.hidden_layers = [128, 64]
            try:
                self.dropout = self.kwargs['dropout']
            except KeyError:
                self.dropout = 0.5
            try:
                self.learning_rate = self.kwargs['learning_rate']
            except KeyError:
                self.learning_rate = 0.01
            try:
                self.activation = self.kwargs['activation']
            except KeyError:
                self.activation = ['relu' for i in self.hidden_layers]

            if isinstance(y, pd.Series):
                class_weight = y.value_counts().to_dict()
            elif isinstance(y, np.array):
                class_weight = pd.Series(y).value_counts().to_dict()
            self.model = Sequential()
            index = 0
            for dim_i, activation_ in zip(self.hidden_layers, self.activation):
                if index == 0:
                    index += 1
                    self.model.add(Dense(output_dim=dim_i, kernel_initializer='glorot_normal', activation=activation_,
                             input_dim=X.shape[1]))
                    self.model.add(Dropout(self.dropout))
                else:
                    self.model.add(Dense(output_dim=dim_i, kernel_initializer='glorot_normal', activation=activation_))
                    self.model.add(Dropout(self.dropout))
            self.model.add(Dense(output_dim=1, activation='sigmoid'))
            self.model.compile(keras.optimizers.Adam(lr=self.learning_rate), loss='binary_crossentropy', metrics=['accuracy'])
            print(self.model.summary())
            acc, loss = [], []
            gini_train, gini_test, recall_train, recall_test = [], [], [], []
            for i in tqdm(range(100)):
                epoch = max(self.epochs//100, 1)
                history = self.model.fit(X1, y1, batch_size=1000, epochs=epoch, verbose=0, class_weight=class_weight)
                acc += history.history['acc']
                loss += history.history['loss']
                gini_train_, recall_train_ = binary_score(self.model, X1, y1, name='train', silent=True)
                gini_test_, recall_test_ = binary_score(self.model, X2, y2, name='test', silent=True)
                gini_train.append(gini_train_)
                gini_test.append(gini_test_)
                recall_train.append(recall_train_)
                recall_test.append(recall_test_)
            if self.plot:
                plot_train_test(acc, loss, name='accuracy', legend=['accuracy', 'loss normalised'])
                plot_train_test(gini_train, gini_test, name='gini')
                __gini__(X1, X2, y1, y2)
                plt.plot(acc)
                plt.title("accuracy during training")
                plt.show()
        else:
            print('start to training model by %s  model...'%self.model_name.upper())
            try:
                self.model.fit(X1, y1, eval_set=[(X1, y1), (X2, y2)], verbose=-1)
            except:
                self.model.fit(X1, y1)
            binary_score(self.model, X1, y1, name='train')
            binary_score(self.model, X2, y2, name='test')
        if self.model_name == 'lgb' and self.plot:
            self.plot_(X)
            plot_precision_recall_curve(model=self.model, X_test=X2, y_test=y2)
            __gini__(X1, X2, y1, y2)
        gc.collect()
        return self

    def plot_(self, X):

        plot__ = Plot(name="LGB classifier",
                     model=self.model,
                     feat_name = X.columns,
                     lgb = lgb)
        plot__.plot_metric_and_importance()
        return self

    def predict(self, X, **kwargs):
        return self.model.predict(X, **kwargs)

    def predict_proba(self, X):
        return self.model.predict_proba(X, pred_leaf=self.pred_leaf)

    def score(self, X, y=None):

        binary_score(self.model, X, y, name='back test')
        return self

    def predict_contribution(self, X, y):
        return self


class ModelWithPipelineAndKfold(BaseEstimator, TransformerMixin):

    def __init__(self, params= params, kfold=5, **kwargs):

        self.params = params
        self.get_list = False
        try:
            self.model_name = kwargs['model_name']
            self.model_name = self.model_name.strip().lower()
        except KeyError:
            self.model_name = 'lgb'
        self.kfold = kfold
        self.cv = StratifiedKFold(n_splits=self.kfold, random_state=49)
        if self.model_name == 'lgb':
            self.models = [lgb.LGBMClassifier(**self.params) for i in range(self.kfold)]
        if 'regression' in self.model_name:
            from sklearn.linear_model import LogisticRegression
            self.models = [LogisticRegression(class_weight='balanced') for i in range(self.kfold)]
        else:
            self.models = [lgb.LGBMClassifier(**self.params) for i in range(self.kfold)]
        self.threshold = .5
        self.plot = True

    def fit(self, X, y=None, **kwargs):

        from tqdm import tqdm
        print('Input data before split into train/test has shape=', X.shape)
        assert len(X)==len(y)
        if not X.index.equals(y.index):
            print("Warning: The index of data and target are not the same!")
        X = X.reset_index(drop=True)
        y = y.reset_index(drop=True)
        print('using %s folds'%self.kfold)
        print('start to training model by LGBClassifier...')
        for model, (train_index, test_index) in tqdm(zip(self.models, self.cv.split(X, y))):
            X1 = X.iloc[train_index]
            X2 = X.iloc[test_index]
            y1 = y.iloc[train_index]
            y2 = y.iloc[test_index]
            if self.model_name=='lgb':
                model.fit(X1, y1, eval_set=[(X1, y1), (X2, y2)], verbose=-1)
            else:
                model.fit(X1, y1)
        binary_scoreKfold(self.models, self.cv,  X, y)
        if self.plot:
            self.plot_(X, y)
        return self

    def plot_(self, X, y):
        Plot().plotKfold(self.models, self.cv, X, y)
        __giniKfold__(self.cv, X, y)
        if self.model_name=='lgb':
            plot_aucKfold(self.models)
            plot_importance_Kfold(self.models)
        return self

    def predict(self, X,  **kwargs):
        result = [model.predict(X,**kwargs) for model in self.models]
        if self.get_list:
            return result
        else:
            df = pd.DataFrame()
            for i, val in enumerate(result):
                name = 'predict_%s'%str(i)
                df[name] = val
            return df

    def predict_proba(self, X, **kwargs):
        result = [model.predict_proba(X, **kwargs) for model in self.models]
        if self.get_list:
            return result
        else:
            df = pd.DataFrame()
            for i, val in enumerate(result):
                name = 'predict_%s'%str(i)
                df[name] = val[:, 1]
            return df

    def score(self, X, y=None):
        KfoldWithoutCv(self.models,  X, y, threshold=self.threshold)
        return self

    def transform(self, X):
        y = X.apply(lambda x: np.random.randint(0, 2), axis=1)
        return binary_scoreKfold(self.models, self.cv,  X, y, get_score=True)
