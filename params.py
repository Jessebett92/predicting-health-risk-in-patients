def param_assign():
    
    rf_params = {
        'n_estimators': 100,
        'max_depth': None,
        'random_state': 123,
        'class_weight': 'balanced'
    }

    xgb_params = {
        'objective': 'multi:softmax',
        'num_class': 3,
        'eval_metric': 'mlogloss',
        'random_state': 123,
        'n_estimators': 100,
        'max_depth': 6,
        'learning_rate': 0.1
    }

 

    return rf_params, xgb_params