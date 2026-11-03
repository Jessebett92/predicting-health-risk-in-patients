from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, recall_score

def cross_val_func(
    model_class,
    X,
    y,
    model_params=None,
    n_splits=5,
    use_scaler=False
):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=123)
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    accuracies = []
    recalls = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y_encoded), 1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y_encoded[train_idx], y_encoded[test_idx]

        steps = []
        if use_scaler:
            steps.append(('scaler', StandardScaler()))
        steps.append(('classifier', model_class(**(model_params or {}))))

        pipeline = Pipeline(steps=steps)
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        accuracies.append(accuracy)

        recall = recall_score(y_test, y_pred, average='macro')
        recalls.append(recall)

        print(f"\nFold {fold} Accuracy: {accuracy:.4f}")
        print(f"Fold {fold} Recall: {recall:.4f}")
        print("Classification Report:")
        print(classification_report(
            y_test,
            y_pred,
            target_names=le.classes_.astype(str)
        ))
        print("Confusion Matrix:")
        print(confusion_matrix(y_test, y_pred))

    print(f"\nAverage Accuracy over {n_splits} folds: {sum(accuracies)/len(accuracies):.4f}")
    print(f"Average Recall Score over {n_splits} folds: {sum(recalls)/len(recalls):.4f}")
