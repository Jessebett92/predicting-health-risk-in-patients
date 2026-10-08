import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

def xgb_model_train(X_train, X_test, y_train, y_test):
    # Encode target labels
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_test_enc = le.transform(y_test)

    model = xgb.XGBClassifier(
        objective='multi:softmax',
        num_class=len(le.classes_),
        eval_metric='mlogloss',
        use_label_encoder=False,
        random_state=123,
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1
    )

    model.fit(X_train, y_train_enc)

    y_pred_enc = model.predict(X_test)
    y_pred = le.inverse_transform(y_pred_enc)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy of the XGBoost Model: {accuracy:.2f}")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    return model
