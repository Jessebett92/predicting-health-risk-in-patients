from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder

def cross_val_with_smote(model_class, X, y, model_params=None, n_splits=5, use_scaler=False):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=123)
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    accuracies = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y_encoded), 1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y_encoded[train_idx], y_encoded[test_idx]

        steps = []
        if use_scaler:
            steps.append(('scaler', StandardScaler()))
        steps.append(('smote', SMOTE(random_state=123)))
        steps.append(('classifier', model_class(**(model_params or {}))))

        pipeline = ImbPipeline(steps=steps)
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        accuracies.append(accuracy)

        print(f"\nFold {fold} Accuracy: {accuracy:.3f}")
        print("Classification Report:")
        print(classification_report(le.inverse_transform(y_test), le.inverse_transform(y_pred)))
        print("Confusion Matrix:")
        print(confusion_matrix(le.inverse_transform(y_test), le.inverse_transform(y_pred)))

    print(f"\nAverage Accuracy over {n_splits} folds: {sum(accuracies)/len(accuracies):.3f}")
    return accuracies
