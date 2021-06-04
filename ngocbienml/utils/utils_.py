import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from ..visualization.plot import PATH
from ..metrics import *
from .config import *

color = ['darkorange', 'cornflowerblue', 'c', 'y']


def distribution_plot(x1, x2=None, x3=None, x4=None, label1='train',
                      label2='back_test', label3=None, label4 = None,
                      title=None, xlabel=None, ylabel=None, figsize=(12, 3)):
    """
    :param x1: pd series or np array with shape (n1,)
    :param x2: pd series or np array with shape (n2,)
    :param x3: pd series or np array with shape (n3,)
    :param x4: pd series or np array with shape (n3,)
    :param label1:
    :param label2:
    :param title:
    :param xlabel:
    :param ylabel:
    :param figsize:
    :return:
    """

    fig, ax = plt.subplots(figsize=figsize)
    sns.kdeplot(x1, shade=True, color=color[0], label=label1, alpha=.6, ax=ax)
    if x2 is not None:
        sns.kdeplot(x2, shade=True, color=color[1], label=label2, alpha=.4, ax=ax)
    if label3 is None:
        label3 = label2
    if x3 is not None:
        sns.kdeplot(x3, shade=True, color=color[2], label=label3, alpha=.4, ax=ax)
    if label4 is None:
        label4 = label3
    if x4 is not None:
        sns.kdeplot(x4, shade=True, color=color[3], label=label4, alpha=.4, ax=ax)
    if title is not None:
        plt.title(title, fontsize=16)
    if xlabel is not None:
        plt.xlabel(xlabel)
    if ylabel is not None:
        plt.ylabel(ylabel)
    plt.savefig(PATH + 'phân bố ', bbox_inches='tight', bpi=1200)
    plt.show()


def show_time(seconds):
    max_ = 10**4*60*60
    if seconds>max_:
        import time
        seconds = time.time() - seconds
    hours = int(seconds // (60 * 60))
    minutes = int((seconds - hours * 60 * 60) // 60)
    seconds = int(seconds - minutes * 60 - hours * 60 * 60)
    if hours > 0:
        str = "%s hours %s mins %s secs" % (hours, minutes, seconds)
    elif minutes > 0:
        str = "%s mins, %s secs" % (minutes, seconds)
    else:
        str = "%s secs" % seconds
    print(str)


def memory_usage():
    import sys
    # These are the usual ipython objects, including this one you are creating
    x = 10 ** 9
    ipython_vars = ['In', 'Out', 'exit', 'quit', 'get_ipython', 'ipython_vars']
    # Get a sorted list of the objects and their sizes
    return sorted([(x, sys.getsizeof(globals().get(x))) for x in dir() if \
                   not x.startswith('_') and x not in sys.modules and x not in ipython_vars], \
                  key=lambda x: x[1], reverse=True)


def extract_package_from_notebook():
    import os
    this_path = os.getcwd()
    if 'notebook' in this_path:
        this_path = this_path.split('\\')[:-1]
        this_path = '\\'.join(this_path)
    os.chdir(this_path)
    print('current path now is:')
    print(os.getcwd())


def up_sample(df, target=None):
    from sklearn.utils import resample
    if target is not None:
        df['target'] = target
    df_0 = df[df.target == 0]
    df_1 = df[df.target == 1]
    # upsample minority
    if len(df_1) < len(df_0):
        df_1 = resample(df_1,
                        replace=True,  # sample with replacement
                        n_samples=len(df_0),  # match number in majority class
                        random_state=49)  # reproducible results
    else:
        df_0 = resample(df_0, replace=True,  # sample with replacement
                        n_samples=len(df_1),  # match number in majority class
                        random_state=49)  # reproducible results
    df = pd.concat([df_0, df_1])
    target = df.target
    df = df.drop(columns=['target']).astype('float32')
    return df, target


def pick_color(n=1):
    import random
    colors = ["blue", "black", "brown", "red", "yellow", "green", "orange", "beige", "turquoise", "pink"]
    random.shuffle(colors)
    if n == 1:
        return colors[0]
    else:
        colors_ = []
        for i in range(n):
            colors_.append(random.choice(colors))
        return colors_


def _random_research_cv_(data, target, train_size=.1, params=None):
    from scipy.stats import randint as sp_randint
    from scipy.stats import uniform as sp_uniform
    from sklearn.model_selection import train_test_split

    import lightgbm as lgb
    from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
    if params is None:
        param_test = {'num_leaves': sp_randint(6, 50),
                      'min_child_samples': sp_randint(100, 500),
                      'min_child_weight': [1e-5, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4],
                      'subsample': sp_uniform(loc=0.2, scale=0.8),
                      'colsample_bytree': sp_uniform(loc=0.4, scale=0.6),
                      'reg_alpha': [0, 1e-1, 1, 2, 5, 7, 10, 50, 100],
                      'num_boost_round': sp_randint(100, 5000),
                      'max_depth': sp_randint(5, 100),
                      'reg_lambda': [0, 1e-1, 1, 5, 10, 20, 50, 100]}
    else:
        param_test = params

    n_HP_points_to_test = 200
    clf = lgb.LGBMClassifier(random_state=314, silent=True, metric='auc', n_jobs=4)
    gs = RandomizedSearchCV(
        estimator=clf, param_distributions=param_test,
        n_iter=n_HP_points_to_test,
        scoring='roc_auc',
        cv=3,
        refit=True,
        random_state=314,
        verbose=True)

    train, _, y_train, _ = train_test_split(data, target, train_size=train_size, random_state=49)
    gs.fit(train, y_train)
    print('Best score reached: {} with params: {} '.format(gs.best_score_, gs.best_params_))


def bayes_research_cv(data, target, train_size=.1):
    from scipy.stats import randint as sp_randint
    from scipy.stats import uniform as sp_uniform
    from sklearn.model_selection import train_test_split

    import pandas as pd
    import numpy as np
    import lightgbm as lgb
    import gc

    from skopt.space import Real, Integer
    from skopt.utils import use_named_args
    import itertools
    from sklearn.metrics import roc_auc_score
    from skopt import gp_minimize, BayesSearchCV

    search_spaces = {'num_leaves': Integer(6, 100),
                     'min_child_samples': Integer(100, 500),
                     'min_child_weight': Real(1e-5, 1e4),
                     'subsample': Real(0.2, 1.0),
                     'colsample_bytree': Real(0.4, 1.0),
                     'reg_alpha': Real(0.0, 100.),
                     'num_boost_round': Integer(100, 5000),
                     'max_depth': Integer(5, 100),
                     'reg_lambda': Real(0.0, 100.0),
                     'max_bin': Integer(100, 500)}

    n_iter = 4

    # n_estimators is set to a "large value".
    # The actual number of trees build will depend on early stopping and 5000 define only the absolute maximum
    clf = lgb.LGBMClassifier(random_state=314, silent=True, metric='auc', n_jobs=-1, verbose=-1)
    import pickle
    from config import model_path
    import os
    path = model_path + 'bayes_searchCV.pickle'
    if os.path.isfile(path):
        with open(path, 'rb') as handle:
            print('reload ... ')
            gs = pickle.load(handle)
    else:
        print('initial new model BayesSearchCV ...')
        gs = BayesSearchCV(
            estimator=clf,
            search_spaces=search_spaces,
            n_iter=n_iter,
            scoring='roc_auc',
            cv=3,
            random_state=314,
            verbose=1
        )
    gs.n_iter = n_iter
    gs.fit(data, target)
    with open(path, 'wb') as handle:
        print('save model to file')
        pickle.dump(gs, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print('Best score reached: {} with params: {} '.format(gs.best_score_, gs.best_params_))
