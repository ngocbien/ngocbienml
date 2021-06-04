from time import time
from skopt.space import Real, Categorical, Integer
import pandas as pd
import numpy as np
from joblib import load, dump
import os
import warnings
warnings.filterwarnings("ignore")

model_file_name = 'research_cv'
params_random = {'num_leaves':[40,80,160,320],
                    'min_child_samples':[100,200,400,800,1600],
                    'min_child_weight':[1e-7,1e-6,1e-5,1e-3,1e-2],
                    'subsample':[.2,.3,.4,.5,.6,.7,.8],
                    'feature_fraction':[0.4,.5,.6,.7],
                    'reg_alpha':[0,1e-1,1,10,100],
                    'num_boost_round':[100,500,1000,5000,10000],
                    'max_depth':[10,20,40,80,160],
                    'reg_lambda':[0,1e-1,1,10,100],
                    'min_split_gain':[0,1e-5,1e-4,1e-3,1e-2,1e-1]}

params_bayes = {'num_leaves':Integer(40,320),
                   'min_child_samples':Integer(100,2000),
                   'min_child_weight':Real(1e-7,1e-2),
                   'subsample':Real(.2,.8),
                   'feature_fraction':Real(.4,.8),
                   'reg_alpha':Real(0,100),
                   'num_boost_round':Integer(100,10000),
                   'max_depth':Integer(10,160),
                   'reg_lambda':Real(0,100),
                   'min_split_gain':Real(0,1e-1)}


class SearchCv:

    def __init__(self, model_file_name=model_file_name,  method='bayes', params = None, n_iter=200):

        self.method = method.lower().strip()
        self.folder_name = 'data searchCV'
        self.create_folder()
        self.n_iter = n_iter
        self.save_every = min(20, self.n_iter)
        self.epochs = max(self.n_iter//self.save_every, 1)
        self.n_iter = self.n_iter//self.epochs   # lấy
        self.model_path = os.path.join(self.folder_name, model_file_name + self.method + '.joblib')
        # tên nhằm phân biệt 2 phương pháp
        self.params_path = os.path.join(self.folder_name, 'params.txt')
        if params is not None:
            self.params = params
        elif 'random' in self.method:
            self.params = params_random
        else:
            self.params = params_bayes
        self.__init_search_cv__()

    def create_folder(self):
        import os
        if not os.path.exists(self.folder_name):
            os.makedirs(self.folder_name)

    def break_down_step(self):
        pass

    def __research_cv__(self):
        import lightgbm as lgb
        from sklearn.model_selection import RandomizedSearchCV
        from skopt import BayesSearchCV
        import warnings
        warnings.filterwarnings("ignore")

        clf = lgb.LGBMClassifier(random_state=314, silent=True, metric='auc', n_jobs=4)
        if 'random' in self.method:
            self.rcv=RandomizedSearchCV(
                estimator=clf,
                param_distributions=self.params,
                n_iter=self.save_every,
                scoring='roc_auc',
                cv=3,
                refit=True,
                random_state=314,
                verbose=True)
        else: # bayes method
            self.rcv = BayesSearchCV(
                clf,
                self.params,
                n_iter=self.save_every,
                scoring='roc_auc',
                cv=3,
                refit=True,
                random_state=314,
                verbose=True
            )

    def __init_search_cv__(self):

        try:
            self.rcv = load(self.model_path)
            self.rcv.set_params(n_iter=self.save_every)
            print('loaded model from disk!')
        except FileNotFoundError:
            print('initial new Search CV...')
            self.__research_cv__()

    def save_all_to_file(self):

        import json
        dump(self.rcv, self.model_path)
        print('save model to file at:')
        print(self.model_path)
        with open(self.params_path, 'a') as file:
            file.write(json.dumps(self.rcv.best_params_))
            file.write('\n')

    def fit(self, data, target):
        from tqdm import tqdm
        s = time()
        for i in tqdm(range(self.epochs)):
            self.rcv.fit(data, target)
            self.save_all_to_file()
        print('Best score reached: {} with params: {} '.format(self.rcv.best_score_, self.rcv.best_params_))
        print('total running time=%s minutes'%((time()-s)//60))

    def get_best_params(self):

        print('best params:')
        print(self.rcv.best_params_)
        print('best score:')
        print(self.rcv.best_score_)

