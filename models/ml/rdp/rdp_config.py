__author__ = 'dk'
#超参数
hyper_params={
    'boosting_type': 'rf',
    'objective': 'multiclass',
    'num_leaves': 512,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': 0,
    'num_class':6,
    'lambda_l1':0.05,
    'lambda_l2':0.15,
    'time_threshold':0.3
}
