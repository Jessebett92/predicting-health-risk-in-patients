def param_assign():
    
    rf_params = {
        'n_estimators': 100,
        'max_depth': None,
        'random_state': 123,
        'class_weight': 'balanced'
    }

    dt_params = {
        'max_depth': None,
        'random_state': 123,
        'class_weight': 'balanced'
    }

    lr_params = {
        'max_iter': 2000,
        'random_state': 123,
        'class_weight': 'balanced',
        'solver': 'lbfgs',
    }

    xgb_params = {
        'objective': 'multi:softmax',
        'num_class': 3,
        'eval_metric': 'mlogloss',
        'use_label_encoder': False,
        'random_state': 123,
        'n_estimators': 100,
        'max_depth': 6,
        'learning_rate': 0.1
    }

 

    return rf_params, dt_params, lr_params, xgb_params