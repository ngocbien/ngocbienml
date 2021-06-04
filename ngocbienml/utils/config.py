
path1 = 'C:\\Users\\os_biennn\\Desktop\\bitbucket\\dmprepo\
\\dmprepo\\dmprepo\\source-code\\machine_learning\\ML_Python'
path2 = '\\income_prediction\\data\\dataset_gui_anhBien\\dataset_income\\'
path_month2 = path1+path2+'dataset_2Month.csv'
path_month3 = path1+path2+'dataset_3Month.csv'
picture_path = 'to_add_corect_path'
model_path = path1+'\\income_prediction_project\\data\\model\\'

params = {'feature_fraction': 0.7319044104939286,
          'max_depth': 65,
          'min_child_weight': 1e-05,
          'min_data_in_leaf': 47,
          'n_estimators': 497,
          'num_leaves': 45,
          'reg_alpha': 0,
          'reg_lambda': 50,
          'metric': 'auc',
          'is_unbalance': True,
          'eval_metric': 'auc',
          'subsample': 0.5380272330885912}

params_prevent_overfit = {'feature_fraction': 0.6,
          'max_depth': 10,
          'min_child_weight': 1e-05,
          'min_data_in_leaf': 300,
           'min_split_gain': .001,
          'n_estimators': 100,
          'num_leaves': 30,
          'reg_alpha': 0,
          'reg_lambda': 50,
          'metric': 'auc',
          'is_unbalance': True,
          'eval_metric': 'auc',
          'subsample': 0.5}
