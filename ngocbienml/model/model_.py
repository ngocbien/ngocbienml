import gc
from time import time

from ..utils.config import *
from ..visualization.plot import Plot
from sklearn.model_selection import train_test_split
from ..utils.utils_ import *
from ..metrics.metrics_ import binary_score,multiclass_score


class GeneralModel:

    def __init__(self):
        self.train_=None
        self.test_=None
        self.validation_=None
        self.y_train_=None
        self.y_test_=None
        self.y_validation_=None
        self.test2=None
        self.y_test2=None
        self.model=None
        self.lgb=None
        self.params_type=""  # may be config, local on else
        self.train_size=1
        self.reduce_memory=True
        self.use_down_sample=None
        self.num_class=2
        self.test_result=None
        self.train_result=None
        self.is_reset_index=False
        self.params=None


class Model(GeneralModel):

    def __init__(self, data, target,  name='lgb', up_sample=False, params=None, **kwargs):
        GeneralModel.__init__(self)
        self.feature_name = None
        self.target = target
        self.data = data
        self.dim = self.data.shape[1]
        self.up_sample = up_sample
        self.name = name.strip().lower()
        try:
            self.num_boost_round=kwargs['num_boost_round']
        except KeyError:
            self.num_boost_round=500
        self.params=params
        del data
        gc.collect()
        self.correct_type()
        self.reset_index()
        self.is_unbalance = None
        self.check_balance()

    def check_balance(self):
        frac = self.target.sum()/len(self.target)
        if .4 < frac < .6:
            self.is_unbalance = False
            print('Label is balanced')
        else:
            self.is_unbalance = True
            print('label is high unbalanced')

    def correct_type(self):

        try:
            self.feature_name=self.data.columns
        except AttributeError:
            self.data=pd.DataFrame(self.data)
        self.feature_name=self.data.columns

    def reset_index(self):

        if not self.data.index.equals(self.target.index):
            print('target, and data have not the same index. start to re index target ')
            self.target=self.target.reindex(self.data.index)
        if self.is_reset_index:
            print('reset index of data and target')
            self.data = self.data.reset_index(drop=True)
            self.target = self.target.reset_index(drop=True)
        return self

    def lgb_model(self):

        import lightgbm as lgb
        from ..utils.config import params
        params['is_unbalance'] = self.is_unbalance
        self.lgb = lgb
        if self.params is not None:
            self.params['metric'] = 'auc'
            self.params['eval_metric'] = 'auc'
            self.params['is_unbalance'] = self.is_unbalance
            #self.params['device'] = 'gpu'
            self.model=lgb.LGBMClassifier(**self.params)
            print('using params from loading from input')
        else:
            self.model = lgb.LGBMClassifier(  #num_leaves= 15,
                early_stopping_rounds=100,
                max_depth=40,
                random_state=49,
                metric='auc',
                eval_metric='auc',
                min_data=100,
                reg_alpha=.1,
                min_data_in_leaf=30,
                feature_fraction=.8,
                n_jobs=4,
                #device="gpu",
                is_unbalance=self.is_unbalance,
                n_estimators=self.num_boost_round,
                subsample=0.9,
                learning_rate=0.05)
        if self.params_type == 'config':
            print('using params from config setting')
            self.model=lgb.LGBMClassifier(**params)
        elif self.params_type == 'local':
            print('use local params')
        elif self.params_type == 'default':
            print('do not set up for lgb params')
            self.model = lgb.LGBMClassifier(
                metric='auc',
                eval_metric='auc',
                n_estimators=self.num_boost_round,
                is_unbalance=self.is_unbalance,
                #device='gpu'
            )

    def lgb_multiclass(self):

        import lightgbm as lgb
        self.lgb=lgb
        self.model=lgb.LGBMClassifier(  #num_leaves= 15,
            objective='muticlass',
            num_class=self.num_class,
            early_stopping_rounds=100,
            max_depth=40,
            random_state=49,
            metric='softmax',
            eval_metric='softmax',
            min_data=100,
            reg_alpha=.1,
            min_data_in_leaf=30,
            feature_fraction=.8,
            n_jobs=4,
            n_estimators=self.num_boost_round,
            subsample=0.9,
            learning_rate=0.05)

    def DL(self):

        from keras.callbacks import ModelCheckpoint
        from keras.layers import Dense
        from keras.models import Sequential
        import keras
        checkpoint=ModelCheckpoint(model_path,
                                   monitor='loss',
                                   verbose=0,
                                   save_best_only=True,
                                   mode='min')
        callbacks_list=[checkpoint]
        self.model=Sequential()
        self.model.add(Dense(output_dim=32,kernel_initializer='glorot_normal',activation='tanh',
                             input_dim=self.dim))
        self.model.add(Dense(output_dim=1,activation='sigmoid'))
        self.model.compile(keras.optimizers.Adam(lr=0.01),
                           loss='binary_crossentropy',
                           metrics=['accuracy'])

    def RF_model(self):

        from sklearn.ensemble import RandomForestClassifier
        self.model = RandomForestClassifier(max_depth=30,
                                          min_impurity_split=10**(-4),
                                          min_samples_split=50,
                                          max_features=.9,
                                          class_weight='balanced')

    def LR(self):

        from sklearn.linear_model import LogisticRegression,LogisticRegressionCV
        if self.is_unbalance:
            self.model=LogisticRegressionCV(random_state=0, cv=3, class_weight='balanced')
        else:
            self.model = LogisticRegressionCV(random_state=0, cv=3)

    def fillna(self):

        missing=self.data.isnull().sum().sum()
        if missing>0:
            print('fillna by 0. Before processing, num of missing value=',self.data.isnull().sum().sum())
            self.data=self.data.fillna(0)
            print('After processing, num of missing value=',self.data.isnull().sum().sum())

    def train_test_split_(self):
        print('shape of data=', self.data.shape)
        print('split data into train, test and validation parts...')
        self.train_, self.test_, self.y_train_, self.y_test_ = train_test_split(self.data,
                                                                                self.target,
                                                                                test_size=.2,
                                                                                random_state=49,
                                                                                stratify=self.target)
        if self.num_class == 2 and self.up_sample:
            self.up_and_down_sample()
        if self.reduce_memory:
            self.data = None
        return self

    def down_sample(self):

        if not self.use_down_sample or self.train_size == 1 or len(self.train_) < 5*10**5:
            return self
        else:
            print('Using down sample method to correct un balanced data:')
            print('Before down sampling, shape of train=',self.train_.shape)
            print('num of label one=',sum(self.y_train_))
            self.train_['target']=self.y_train_
            train0=self.train_[self.train_.target==0]
            train1=self.train_[self.train_.target==1]
            train0,test=train_test_split(train0,train_size=self.train_size,random_state=49)
            self.train_=train0.append(train1)
            self.train_=self.train_.reset_index(drop=True)
            self.y_train_=self.train_['target']
            self.train_=self.train_.drop(columns=['target'])
            print('After down sampling, shape of train=',self.train_.shape)
            print('num of label one=',sum(self.y_train_))
            self.y_test2=test['target']
            self.test2=test.drop(columns=['target'])

    def simple_up_sample(self):

        frac = sum(self.y_train_==0)/sum(self.y_train_==1)
        if not self.is_up_sample:
            print('do not use method to up sample')
            return self
        if .9 < frac < 1.1:
            return self
        else:
            print('Before up sampling, shape of train=',self.train_.shape)
            print('num of label one=',sum(self.y_train_))
            self.train_['target']=self.y_train_
            train0 = self.train_[self.train_.target == 0]
            train1 = self.train_[self.train_.target == 1]
            if frac > 1.1:
                print('up sample label 1')
                train1=train1.sample(frac=frac,replace=True)
            elif frac < .90:
                print(' up sample label 0')
                train0 = train0.sample(frac=1/frac,replace=True)
            self.train_ = train0.append(train1).sample(frac=1)
            self.y_train_ = self.train_['target']
            self.train_ = self.train_.drop(columns=['target'])
            return self

    def up_and_down_sample(self):

        self.down_sample()
        self.simple_up_sample()
        return self

    def std_selection(self, threshold=0.05):
        list_ = []
        for col in self.data.columns:
            if self.data[col].std() < threshold:
                list_.append(col)
        if len(list_)>0:
            print('will delete %s feat with low variance'%len(list_))
            print(' '.join(list_))
            self.data = self.data.drop(columns = list_)

    def correlation_selection(self, threshold=.95):

        list_ = []
        corr = self.data.corr()
        for col in corr.columns:
            for index in corr.index:
                cond = (col != index) and (corr.loc[index].at[col] > threshold)\
                       and (col not in list_) and (index not in list_)
                if cond:
                    list_.append(col)
        if len(list_)>0:
            print('Will delete %s feat too correlate to other'%len(list_))
            print(' '.join(list_))
            self.data = self.data.drop(columns=list_)

    def feat_selection(self):

        if self.name == 'lr':
            self.data.columns = [col + '*_' + str(i) for col, i in zip(self.data.columns, range(len(self.data.columns)))]
            self.std_selection()
            self.correlation_selection()
            self.data.columns = [col.split('*')[0] for col in self.data.columns]
            self.feature_name = self.data.columns #update
            self.dim = len(self.feature_name)
        return self

    def train(self, is_split=True):

        from collections import Counter
        self.fillna()
        self.feat_selection()
        if is_split:
            self.train_test_split_()
        print('target label %s'%Counter(self.y_train_))
        print('start to training...')
        if self.name=='lgb' or self.name=='multiclass':
            self.model.fit(self.train_,
                           self.y_train_,
                           eval_set=[(self.train_,self.y_train_),(self.test_,self.y_test_)],
                           verbose=-1)
        else:
            self.model.fit(self.train_,self.y_train_)
        print('Done of training')
        return self

    def test(self):
        if self.num_class>2:
            multiclass_score(self.model,self.train_,self.y_train_,name='train')
            multiclass_score(self.model,self.test_,self.y_test_,name='test1')
            if self.test2 is not None:
                multiclass_score(self.model,self.test2,self.y_test2,name='test2')
            #multiclass_score(self.model, self.validation_, self.y_validation_, name='validation')
        else:
            binary_score(self.model,self.train_,self.y_train_,name='train')
            binary_score(self.model,self.test_,self.y_test_,name='test1')
            if self.test2 is not None:
                binary_score(self.model,self.test2,self.y_test2,name='test2')
            #binary_score(self.model, self.validation_, self.y_validation_, name='validation')
        return self

    def create_result_(self):

        self.test_result=pd.DataFrame(index=self.y_test_.index)
        self.test_result['pred_on_test']=self.model.predict_proba(self.test_)[:,1]
        self.test_result['true_on_test']=self.y_test_
        self.train_result=pd.DataFrame(index=self.y_train_.index)
        self.train_result['pred_on_train']=self.model.predict_proba(self.train_)[:,1]
        self.train_result['true_on_train']=self.y_train_
        return self

    def run(self):

        self.num_class=len(np.unique(self.target))
        if self.num_class>2:
            self.lgb_multiclass()
        elif self.name=='lgb':
            print('using lgb model')
            self.lgb_model()
        elif self.name == 'lr':
            print('using linear regression model')
            self.LR()
        elif self.name == 'rf':
            print('using random forest model')
            self.RF_model()
        elif self.name == 'dl':
            print('using deep learning model')
            self.DL()
        else:
            raise Exception("Sorry, please choose a method to training")
        self.train()
        self.test()
        self.plot()
        return self

    def plot(self):

        if 'lgb' in self.name or self.name == 'multiclass':
            plot_ = Plot(name=self.name,
                       model=self.model,
                       feat_name=self.feature_name,
                       lgb=self.lgb)
            plot_.plot_metric_and_importance()
        else:
            plot_ = Plot(name=self.name,model=self.model,feature_name=self.feature_name)
            plot_.plot_rf_or_lr()
        return self
