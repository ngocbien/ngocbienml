This is a machine learning tools with pipeline.

Here some example to run machine learning project:

```python
from ngocbienml import MyPipeline
pipeline = MyPipeline()
pipeline.fit(data, target)
pipeline.score(new_data, new_target)
```
Or 
```python
from ngocbienml import PipelineKfold
pipline = PipelineKfold()
pipline.fit(data, target)
pipline.score(new_data, new_target)
```
Where data is pandas dataframe, target is series object.
In the above default settting, principal modules of pipeline are:
- Fillna by mean
- LabelEncoder
- Feature Selection: Use 2 methods: variance and correlation
- MinMaxScale
- LGBClassifier: The default params work well with dataset of 100K rows or more, with minimum of 
20 features. It deals well with unbalanced dataset.
In the above default setting 10% of dataset will be cut for test set if not using kfold or
5 folds in other case.

You can use to save and reload pipeline for a long usage.
```python
from joblib import dump, load
dump(pipeline, path)
pipeline = load(path)
pipeline.score(data, target)
```
You can use include many preprocessing classes  like Fillna, Scale, or Labelencoder 
in your customized pipeline. Note that actually, 
you can not use full label encoder by sklearn

```python
from ngocbienml import Scale, Fillna, Labelencoder, ModelWithPipeline
from sklearn.pipeline import Pipeline
pipeline = Pipeline([('label_encoder', Labelencoder()),
                    ('fillna', Fillna()), 
                    ('scale', Scale()),
                    ('model', ModelWithPipeline())])
pipeline.fit(data, target)
pipline.score(test,  y_test)
```


```python
from ngocbienml import PipelineKfold
pipeline = PipelineKfold()
pipeline.fit(data, target)
pipline.score(test, y_test)
```
We can use only pipeline to transform data, and then use it for other task
```python
from ngocbienml import AssertGoodHeader, Fillna, LabelEncoder,\
FeatureSelection, FillnaAndDropCatFeat, MinMaxScale
from sklearn.pipeline import Pipeline
steps = [('assertGoodheader', AssertGoodHeader()),
        ('Fillna', Fillna()),
        ('LabelEncoder', LabelEncoder()),
        ('FillnaAndDropCatFeat', FillnaAndDropCatFeat()),
        ('MinMaxScale', MinMaxScale()),
        ('FeatureSelection', FeatureSelection())]

pipline = Pipeline(steps = steps)
df_transformed = pipline.fit_transform(df)
```
Or the simplest way is to use the default params
```python
from ngocbienml import MyPipeline
pipline = Pipeline(model_name=None) #do not use model
df_transformed = pipline.fit_transform(df)
```
In the above code, df_tranformed is numeric data frame with the same header of 
df. df_transformed is ready to train by any model.
```python
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(df_transformed, label)
```
You can run deeplearning, for example

```python
classifier = MyPipeline( model_name='dl', 
                             epochs=3000,
                             hidden_layers=[64, 32], 
                             activation=['relu', 'sigmoid'],
                             dropout=.1)
classifier.fit(data, target)
```
model_name is dl or lgb or logistic regression. The default is lgb.

Use search cv for hyper params tuning:
```python
from ngocbienml import SearchCv
SearchCv(n_iter=100).fit(data, target)
```
This tool will break down n_iter to small step and save at the and of these step, to ensure that
you do not loss everything if you shut down your PC before the end of running.
You can re-runing this to refit and fit the better params



What's next:
- More setting in feature extraction and modelling.
- More metric and visualization