from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

def rf_model_train(X_train, X_test, y_train, y_test):
    model = RandomForestClassifier(random_state=123)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = model.score(X_test, y_test)
    print(f"Accuracy of the Random Forest Model: {accuracy:.2f}")

    print("\nClassifaction Report:")
    print(classification_report(y_test, y_pred))

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    return model