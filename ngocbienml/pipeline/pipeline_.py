from sklearn.pipeline import Pipeline
from ..data_processing import Fillna, LabelEncoder, FillnaAndDropCatFeat, MinMaxScale, FeatureSelection, \
    AssertGoodHeader
from ..model import ModelWithPipeline, ModelWithPipelineAndKfold
from ..metrics import binary_score_
from ..utils.utils_ import params_prevent_overfit, params

BASE_STEPS = [('AssertGoodHeader', AssertGoodHeader()),
              ('fillna', Fillna()),
              ('label_encoder', LabelEncoder()),
              ('FillnaAndDropCat', FillnaAndDropCatFeat()),
              ('MinMaxScale', MinMaxScale()),
              ('FeatureSelection', FeatureSelection())]

STEPS_KFOLD_LGB = [('AssertGoodHeader', AssertGoodHeader()),
                   ('fillna', Fillna()),
                   ('label_encoder', LabelEncoder()),
                   ('FillnaAndDropCat', FillnaAndDropCatFeat()),
                   ('MinMaxScale', MinMaxScale()),
                   ('FeatureSelection', FeatureSelection()),
                   ('classification', ModelWithPipelineAndKfold(model_name='lgb'))]

STEPS_KFOLD_REGRESSION = [('AssertGoodHeader', AssertGoodHeader()),
                          ('fillna', Fillna()),
                          ('label_encoder', LabelEncoder()),
                          ('FillnaAndDropCat', FillnaAndDropCatFeat()),
                          ('MinMaxScale', MinMaxScale()),
                          ('FeatureSelection', FeatureSelection()),
                          ('classification', ModelWithPipelineAndKfold(model_name='regression'))]


class MyPipeline:

    def __init__(self, steps=BASE_STEPS, model_name='lgb', epochs=200, **kwargs):
        self.model_name = model_name
        if self.model_name is not None:
            self.steps = steps + [('classification', ModelWithPipeline(model_name=model_name, epochs=epochs, **kwargs))]
        else:
            print('This Pipeline to only transform data without using modelling')
            self.steps = steps
        self.pipeline_ = Pipeline(steps=self.steps)

    def fit(self, X, y=None):
        print('start to using pipeline to fit data. Data shape=', X.shape)
        self.pipeline_.fit(X=X, y=y)
        return self

    def transform(self, X, y=None):
        return self.pipeline_.transform(X.copy())

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X, y)

    def predict_proba(self, X, y=None):
        if self.model_name is not None:
            return self.pipeline_.predict_proba(X.copy())
        else:
            return None

    def predict(self, X, y=None, **kwargs):
        if self.model_name is not None:
            return self.pipeline_.predict(X.copy(), **kwargs)
        else:
            return None

    def score(self, X, y=None):
        if self.model_name is not None:
            y_proba = self.pipeline_.predict_proba(X.copy())
            binary_score_(y, y_proba, name='back test')
            return self
        else:
            return None


class PipelineKfold(MyPipeline):

    def __init__(self, name='lgb', params=params, **kwargs):
        super().__init__(**kwargs)
        self.steps = BASE_STEPS + [('classification', ModelWithPipelineAndKfold(model_name=name, params=params))]
        self.pipeline_ = Pipeline(steps=self.steps)
        self.threshold = .5

    def score(self, X, y=None):
        print('pipeline kfold')
        print(self.pipeline_['classification'])
        self.pipeline_.score(X, y)

    def get_score(self, X, y=None):
        return self.pipeline_.transform(X)
