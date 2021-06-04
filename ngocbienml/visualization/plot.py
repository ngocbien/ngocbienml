import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from datetime import datetime
import os
from ..utils.config import picture_path
from ..metrics.metrics_ import gini

COLOR = ['darkorange', 'cornflowerblue']
PATH = os.getcwd() + "/picture/modelling/"
from pathlib import Path

Path(PATH).mkdir(parents=True, exist_ok=True)


# from model import Model


class Plot:
    def __init__(self, **kwargs):
        # self.__init__(Model)
        try:
            self.name = kwargs['name']
        except KeyError:
            pass
        try:
            self.model = kwargs['model']
        except KeyError:
            pass
        self.feature_importance = None
        try:
            self.feature_name = kwargs['feature_name']
        except KeyError:
            self.feature_name = None
        self.args = kwargs
        self.max_feat = 50
        self.color = COLOR
        self.path = PATH

    def get_parameters(self):

        if self.name == 'lgb_model':
            self.feature_importance = self.model.feature_importance()
            self.feature_name = self.model.feature_name()
        else:
            feat_name_key = [key for key in self.args.keys() if 'feat' in key and 'name' in key]
            assert len(feat_name_key) == 1
            self.feature_name = self.args[feat_name_key[0]]

        if self.name == 'lgb_classifier' or self.name == 'lgb':
            self.feature_importance = self.model.feature_importances_
        elif self.name == 'RF':
            self.feature_importance = self.model.feature_importances_

    def simple_plot(self):

        self.get_parameters()
        dfx = pd.Series(self.feature_importance, index=self.feature_name)
        dfx = dfx.sort_values(ascending=False)
        if len(dfx) > self.max_feat:
            dfx = dfx.iloc[:self.max_feat]
        dfx.plot.bar(figsize=(12, 4), color=self.color)
        plt.title('feat importance')
        plt.show()
        plt.savefig(self.path + 'feat_importance', bbox_inches='tight', dpi=1200)

    def plot_lgb(self):

        if 'lgb' in self.args.keys():
            lgb = self.args['lgb']
            ax = lgb.plot_importance(self.model, figsize=(6, 8), \
                                     importance_type='gain', \
                                     max_num_features=40,
                                     height=.8,
                                     color=self.color,
                                     grid=False,
                                     )
            plt.sca(ax)
            plt.xticks([], [])
            plt.title('lgb model gain importance')
            plt.show()
            plt.savefig(self.path + 'feat_importance_lgb', bbox_inches='tight', dpi=1200)
        else:
            pass

    def get_path(self):

        if os.path.isdir(picture_path):
            now = datetime.now()
            current_time = now.strftime("%d %m %y %H %M")  # will add this time to the name of file distinct them
            path = picture_path + current_time + '_picture.png'
            return path
        else:
            return None

    def plot_metric(self):

        if 'lgb' in self.args.keys():
            lgb = self.args['lgb']
            ax = lgb.plot_metric(self.model, figsize=(6, 8))
            plt.savefig(self.path + 'metric', bbox_inches='tight', dpi=1200)
            plt.show()
        else:
            pass

    def plot_metric_and_importance(self):

        if 'lgb' in self.args.keys():
            lgb = self.args['lgb']
            fig, ax = plt.subplots(2, 1)
            fig.subplots_adjust(hspace=.2)
            fig.set_figheight(6)
            fig.set_figwidth(14)
            lgb.plot_metric(self.model, ax=ax[0])
            booster = self.model.booster_  # case of classifier, we must to acces to booster_ instance
            dfx = pd.DataFrame(index=booster.feature_name())
            dfx['gain'] = booster.feature_importance('gain')
            dfx['gain'] = dfx['gain'] / dfx.gain.max()
            dfx['split'] = booster.feature_importance('split')
            dfx['split'] = dfx['split'] / dfx.split.max()
            dfx = dfx.sort_values('gain', ascending=False).iloc[:self.max_feat]
            dfx.plot.bar(width=0.9, ax=ax[1], color=COLOR)
            plt.subplots_adjust(left=None, bottom=.5, right=None, top=None, wspace=None, hspace=None)
            plt.savefig(self.path + 'feat_importance lgb', bbox_inches='tight', dpi=1200)
            plt.show()
        else:
            print('nothing to plot')
            pass

    def plot_booster_lgb(self):

        booster = self.model.booster_  # case of classifier, we must to acces to booster_ instance
        dfx = pd.DataFrame(index=self.lgb.feature_name())
        dfx['gain'] = booster.feature_importance('gain')
        dfx['gain'] = dfx['gain'] / dfx.gain.max()
        dfx['split'] = booster.feature_importance('split')
        dfx['split'] = dfx['split'] / dfx.split.max()
        dfx = dfx.sort_values('split', ascending=False).iloc[:self.max_feat]
        dfx.plot.bar(width=0.9, figsize=(12, 3))
        plt.show()
        plt.savefig(self.path + 'feat_importance lgb 2', bbox_inches='tight', dpi=1200)

    def plot_rf_or_lr(self):
        if self.name.strip().lower() == 'lr':
            feature_importance = abs(self.model.coef_[0])
        elif self.name.strip().lower() == 'rf':
            feature_importance = self.model.feature_importances_
        else:
            return self
        plt.figure(figsize=(13, 4))
        df = pd.Series(feature_importance, index=self.feature_name).sort_values(ascending=False).iloc[:self.max_feat]
        plt.bar(range(len(df)), df, color=self.color)
        plt.xticks(range(len(df)), df.index, rotation=90)
        plt.title('Feature Importance Of %s Model' % (self.name.upper()), fontsize=16)
        plt.subplots_adjust(left=None, bottom=.5, right=None, top=None, wspace=None, hspace=None)
        plt.savefig(self.path + 'feat_importance rf or lr', bbox_inches='tight', dpi=1200)
        plt.show()

    def plotKfold(self, models, cv, X, y):

        import numpy as np
        import matplotlib.pyplot as plt

        from sklearn import svm, datasets
        from sklearn.metrics import auc
        from sklearn.metrics import plot_roc_curve, roc_curve
        from sklearn.model_selection import StratifiedKFold
        tprs = []
        aucs = []
        mean_fpr = np.linspace(0, 1, 10000)

        fig, ax = plt.subplots()
        fig.set_figheight(8)
        fig.set_figwidth(8)
        index = 0
        for model, (index_train, index_test) in zip(models, cv.split(X, y)):
            train, test = X.loc[index_train], X.loc[index_test]
            y_train, y_test = y.loc[index_train], y.loc[index_test]
            prediction = model.predict_proba(test)
            fpr, tpr, t = roc_curve(y_test, prediction[:, 1])
            auc_value = auc(fpr, tpr)
            interp_tpr = np.interp(mean_fpr, fpr, tpr)
            interp_tpr[0] = 0.0
            tprs.append(interp_tpr)
            aucs.append(auc_value)

            plt.plot(fpr, tpr, lw=2, alpha=0.3, label='ROC fold %d (AUC = %0.2f)' % (index, auc_value))
            index += 1

        ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
                label='Chance', alpha=.8)

        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)
        std_auc = np.std(aucs)
        ax.plot(mean_fpr, mean_tpr, color='b',
                label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
                lw=2, alpha=.8)

        std_tpr = np.std(tprs, axis=0)
        tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
        ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                        label=r'$\pm$ 1 std. dev.')

        ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05],
               title="AUC by Kfold on Test set")
        ax.legend(loc="lower right")
        ax.set_xlabel('faux positive rate')
        ax.set_ylabel('true positive rate')
        plt.savefig(self.path + 'auc kfold', bbox_inches='tight', dpi=1200)
        plt.show()
        # plt.savefig('kfold.png', dpi=1200, bbox_inches='tight')


def plot_importance_Kfold(models):
    to_plot = pd.DataFrame(index=models[0].booster_.feature_name())
    for index, model in enumerate(models):
        to_plot['%s_gain' % index] = model.booster_.feature_importance('gain')
        to_plot['%s_split' % index] = model.booster_.feature_importance('split')
    about_gains = [col for col in to_plot.columns if '_gain' in col]
    about_split = [col for col in to_plot.columns if '_split' in col]
    to_plot[about_gains] = to_plot[about_gains] / to_plot[about_gains].max().max()
    to_plot[about_split] = to_plot[about_split] / to_plot[about_split].max().max()
    to_plot['gain'] = to_plot[about_gains].mean(axis=1)
    to_plot['split'] = to_plot[about_split].mean(axis=1)
    to_plot['gain_std'] = to_plot[about_gains].std(axis=1)
    to_plot['split_std'] = to_plot[about_split].std(axis=1)
    to_plot = to_plot.sort_values('gain', ascending=False)
    total = len(to_plot)
    max_len = min(45, len(to_plot))
    height = 0.45
    x1 = to_plot.index[:max_len]
    x = np.arange(max_len)
    gain = to_plot.gain.iloc[:max_len]
    gain_err = to_plot.iloc[:max_len].gain_std
    split = to_plot.split.iloc[:max_len]
    split_err = to_plot.iloc[:max_len].split_std
    fig = plt.figure(figsize=(10, 10))
    fig.subplots_adjust(wspace=.5)
    plt.barh(x - height, gain, xerr=gain_err, height=height, color=COLOR[0], align='center',
             error_kw=dict(ecolor='gray', lw=1, capsize=1.5, capthick=.5))
    plt.barh(x, split, height=height, xerr=split_err, color=COLOR[1], align='center',
             error_kw=dict(ecolor='gray', lw=1, capsize=1.5, capthick=.5))
    plt.yticks(x, x1, rotation=0)
    plt.legend(('gain', 'split'))
    plt.title('MOST %s IMPORTANT FEATURES, FEATURES BY TOTAL =%s' % (max_len, total))
    plt.savefig(PATH + 'feat_importance kfold', bbox_inches='tight', dpi=1200)
    # plt.savefig('most_importance.png', dpi=1200, bbox_inches='tight')
    plt.show()


def plot_aucKfold(models):
    this_df = pd.DataFrame()
    for index, model in enumerate(models):
        this_df['%s_train' % index] = model.evals_result_['training']['auc']
        this_df['%s_test' % index] = model.evals_result_['valid_1']['auc']
    all_about_train = [col for col in this_df.columns if '_train' in col]
    all_about_test = [col for col in this_df.columns if '_test' in col]
    train_mean = this_df[all_about_train].mean(axis=1)
    train_std = this_df[all_about_train].std(axis=1)
    test_mean = this_df[all_about_test].mean(axis=1)
    test_std = this_df[all_about_test].std(axis=1)
    x = np.arange(len(train_mean))
    plt.plot(x, train_mean)
    plt.fill_between(x, train_mean - train_std, train_mean + train_std, color='gray', alpha=.2)
    plt.plot(x, test_mean)
    plt.fill_between(x, test_mean - test_std, test_mean + test_std, color='gray', alpha=.2)
    plt.legend(['train', 'valid', 'confidence zone'])
    plt.title('Train and Valid auc during training')
    plt.ylabel('auc')
    plt.xlabel('epoch')
    plt.savefig(PATH + 'auc kfold', bbox_inches='tight', dpi=1200)
    plt.show()


def plot_train_test(acc_train, acc_test, name, legend=('train', 'test')):
    plt.plot(acc_train)
    plt.plot(acc_test)
    plt.title(name)
    plt.ylabel(name)
    plt.xlabel('epoch')
    plt.legend(legend, loc='upper left')
    plt.savefig(PATH + 'acc train test', bbox_inches='tight', dpi=1200)
    plt.show()


def plot_precision_recall_curve(model, X_test, y_test):
    # from sklearn.metrics import average_precision_score
    # from sklearn.metrics import precision_recall_curve
    # from sklearn.metrics import plot_precision_recall_curve
    # import matplotlib.pyplot as plt
    from numpy import argmax
    from sklearn.datasets import make_classification
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import precision_recall_curve

    y_score = model.predict_proba(X_test)
    if len(y_score.shape) > 1:
        y_score = y_score[:, 1]
    precision, recall, thresholds = precision_recall_curve(y_test, y_score)
    funct = np.vectorize(lambda x, y: -1 if (x <= 0 or y <= 0) else 2 * x * y / (x + y))
    fscore = funct(precision, recall)
    # print('fscore contains non numeric=',np.isnan(fscore).any())
    # print('fscore contains inf =',np.isinf(fscore).any())
    # ix = argmax(fscore)
    max_value = np.max(fscore)
    # print('max value = %.3f'%max_value)
    ix = np.where(fscore == max_value)[0][0]
    # print('Best Threshold=%f, F-Score=%.3f'%(thresholds[ix], fscore[ix]))
    plt.plot(recall, precision, marker='.', label='model')
    plt.plot(recall[ix], precision[ix], marker='o', color='red', label='Best')
    # axis labels
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend()
    plt.grid()
    plt.title('best threshold=%.4f. Recall=%.2f, Precision = %.2f' % (thresholds[ix], recall[ix],
                                                                      precision[ix]))
    # show the plot
    plt.savefig(PATH + 'precision recall', bbox_inches='tight', dpi=1200)
    plt.show()


def __gini__(train, test, y_train, y_test, is_plot=True):
    max_len = 30
    gini_trains = []
    gini_tests = []
    for col in train.columns:
        gini_train = np.abs(gini(y_train, train[col]))
        gini_test = np.abs(gini(y_test, test[col]))
        gini_trains.append(gini_train)
        gini_tests.append(gini_test)
    dfxx = pd.DataFrame(index=train.columns)
    dfxx['train'] = gini_trains
    dfxx['test'] = gini_tests
    dfxx = dfxx.sort_values('train', ascending=False)
    if not is_plot:
        return dfxx
    is_drop = False
    total_features = len(dfxx)
    if len(dfxx) > max_len:
        dfxx = dfxx.iloc[:max_len]
        is_drop = True
    ax = dfxx.plot.barh(width=.9, rot=0, figsize=(10, 10), color=['darkorange', 'cornflowerblue'])
    rects = ax.patches

    # For each bar: Place a label
    for rect in rects:
        # Get X and Y placement of label from rect.
        x_value = rect.get_width()
        y_value = rect.get_y() + rect.get_height() / 2

        # Number of points between bar and label. Change to your liking.
        space = 5
        # Vertical alignment for positive values
        ha = 'left'

        # If value of bar is negative: Place label left of bar
        if x_value < 0:
            # Invert space to place label to the left
            space *= -1
            # Horizontally align label at right
            ha = 'right'

        # Use X value as label and format number with one decimal place
        label = '%.2f' % x_value

        # Create annotation
        plt.annotate(
            label,  # Use `label` as label
            (x_value, y_value),  # Place label at end of the bar
            xytext=(space, 0),  # Horizontally shift label by `space`
            textcoords="offset points",  # Interpret `xytext` as offset in points
            va='center',  # Vertically center label
            ha=ha)  # Horizontally align label differently for
        # positive and negative values.
    if is_drop:
        plt.title('Top %s features by gini on train and test, total features=%s' % (max_len, total_features))
    else:
        plt.title('Gini by features on train and test')
    plt.savefig(PATH + 'gini', bbox_inches='tight', dpi=1200)
    plt.show()


def __giniByUNiqueData__(data, target_name='label', is_plot=True):
    max_len = 30
    ginis = []
    cols = []
    for col in data.columns:
        if col != target_name:
            gini_ = np.abs(gini(data[target_name], data[col]))
            ginis.append(gini_)
            cols.append(col)
            # print('%s=%.3f'%(col, gini_))
    dfxx = pd.Series(ginis, index=cols)
    dfxx = dfxx.sort_values(ascending=False)
    if not is_plot:
        return dfxx
    is_drop = False
    total_features = len(dfxx)
    if len(dfxx) > max_len:
        dfxx = dfxx.iloc[:max_len]
        is_drop = True
    ax = dfxx.plot.barh(width=.9, rot=0, figsize=(10, 10), color=['darkorange', 'cornflowerblue'])
    rects = ax.patches

    # For each bar: Place a label
    for rect in rects:
        # Get X and Y placement of label from rect.
        x_value = rect.get_width()
        y_value = rect.get_y() + rect.get_height() / 2

        # Number of points between bar and label. Change to your liking.
        space = 5
        # Vertical alignment for positive values
        ha = 'left'
        label = '%.2f' % x_value

        # Create annotation
        plt.annotate(
            label,  # Use `label` as label
            (x_value, y_value),  # Place label at end of the bar
            xytext=(space, 0),  # Horizontally shift label by `space`
            textcoords="offset points",  # Interpret `xytext` as offset in points
            va='center',  # Vertically center label
            ha=ha)  # Horizontally align label differently for
        # positive and negative values.
    if is_drop:
        plt.title('Top %s features by gini on data,  total features=%s' % (max_len, total_features))
    else:
        plt.title('Gini by features vs target')
    plt.savefig(PATH + 'gini on hold data vs %s' % target_name, bbox_inches='tight', dpi=1200)
    plt.show()


def __giniKfold__(cv, X, y):
    max_len = 30
    df_train_std = pd.DataFrame()
    df_test_std = pd.DataFrame()
    folds = 0
    dfx_mean = pd.DataFrame()
    for index_train, index_test in cv.split(X, y):
        folds += 1
        try:
            train, test = X.iloc[index_train], X.iloc[index_test]
            y_train, y_test = y.iloc[index_train], y.iloc[index_test]
        except:
            train, test = X.loc[index_train], X.loc[index_test]
            y_train, y_test = y.loc[index_train], y.loc[index_test]
        this_df = __gini__(train, test, y_train, y_test, is_plot=False)
        if len(dfx_mean) == 0:
            dfx_mean = this_df.copy()
        else:
            dfx_mean += this_df
        # print('this_df')
        # print(this_df.head())
        # print('dfx mean')
        # print(dfx_mean.head())
        df_train_std['train_%s' % folds] = this_df['train']
        df_test_std['test_%s' % folds] = this_df['test']
    # print(dfx_mean.head())
    dfx_mean = dfx_mean / folds
    dfx_mean['std_train'] = df_train_std.std(axis=1)
    dfx_mean['std_test'] = df_test_std.std(axis=1)
    # print(dfx_mean.head())
    dfx_mean = dfx_mean.sort_values('train', ascending=False)
    is_drop = False
    total_features = len(dfx_mean)
    if len(dfx_mean) > max_len:
        dfx_mean = dfx_mean.iloc[:max_len]
        is_drop = True
    err = pd.DataFrame()
    err['train'] = dfx_mean['std_train']
    err['test'] = dfx_mean['std_test']
    ax = dfx_mean[['train', 'test']].plot.barh(width=.9, rot=0,
                                               figsize=(10, 10), color=['darkorange', 'cornflowerblue'],
                                               xerr=err, error_kw=dict(ecolor='gray', lw=1, capsize=1.5, capthick=.5))
    rects = ax.patches

    # For each bar: Place a label
    for rect in rects:
        # Get X and Y placement of label from rect.
        x_value = rect.get_width()
        y_value = rect.get_y() + rect.get_height() / 2

        # Number of points between bar and label. Change to your liking.
        space = 5
        # Vertical alignment for positive values
        ha = 'left'

        # If value of bar is negative: Place label left of bar
        if x_value < 0:
            # Invert space to place label to the left
            space *= -1
            # Horizontally align label at right
            ha = 'right'

        # Use X value as label and format number with one decimal place
        label = '%.2f' % x_value

        # Create annotation
        plt.annotate(
            label,  # Use `label` as label
            (x_value, y_value),  # Place label at end of the bar
            xytext=(space, 0),  # Horizontally shift label by `space`
            textcoords="offset points",  # Interpret `xytext` as offset in points
            va='center',  # Vertically center label
            ha=ha)  # Horizontally align label differently for
        # positive and negative values.
    if is_drop:
        plt.title('Top %s features by gini on train and test, total features=%s' % (max_len, total_features))
    else:
        plt.title('Gini by features on train and test')
    plt.savefig(PATH + 'gini kfold', bbox_inches='tight', dpi=1200)
    plt.show()
