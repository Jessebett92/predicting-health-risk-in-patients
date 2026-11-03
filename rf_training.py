import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

def find_optimal_threshold(y_true, y_probs, target_recall=0.65, min_precision=0.25):
    thresholds = np.linspace(0.20, 0.70, 51)
    best_thresh = 0.50
    best_recall = -1
    best_precision = -1
    best_f1 = -1

    for t in thresholds:
        preds = (y_probs >= t).astype(int)
        rec = recall_score(y_true, preds, zero_division=0)
        prec = precision_score(y_true, preds, zero_division=0)
        f1 = f1_score(y_true, preds, zero_division=0)

        if rec >= target_recall and prec >= min_precision:
            if (
                rec > best_recall
                or (rec == best_recall and prec > best_precision)
                or (rec == best_recall and prec == best_precision and f1 > best_f1)
            ):
                best_thresh = t
                best_recall = rec
                best_precision = prec
                best_f1 = f1

    if best_recall == -1:
        for t in thresholds:
            preds = (y_probs >= t).astype(int)
            rec = recall_score(y_true, preds, zero_division=0)
            prec = precision_score(y_true, preds, zero_division=0)
            f1 = f1_score(y_true, preds, zero_division=0)

            if (
                rec > best_recall
                or (rec == best_recall and prec > best_precision)
                or (rec == best_recall and prec == best_precision and f1 > best_f1)
            ):
                best_thresh = t
                best_recall = rec
                best_precision = prec
                best_f1 = f1

    return best_thresh, best_recall, best_precision, best_f1


def rf_model_train(X_train, X_val, X_test, y_train, y_val, y_test):
    model = RandomForestClassifier(
        n_estimators=150,
        max_depth=5,
        min_samples_split=6,
        min_samples_leaf=3,
        max_features='sqrt',
        class_weight='balanced',
        random_state=123,
        n_jobs=-1,
    )

    model.fit(X_train, y_train)

    # Validation threshold tuning
    val_prob = model.predict_proba(X_val)[:, 1]
    optimal_thresh, best_recall, best_precision, best_f1 = find_optimal_threshold(
        y_val,
        val_prob,
        target_recall=0.65,
        min_precision=0.25
    )

    print(f"Optimized Threshold (Validation): {optimal_thresh:.2f}")
    print(f"Validation Recall: {best_recall:.3f}")
    print(f"Validation Precision: {best_precision:.3f}")
    print(f"Validation F1: {best_f1:.3f}")

    # Final test evaluation
    test_prob = model.predict_proba(X_test)[:, 1]
    roc_auc = roc_auc_score(y_test, test_prob)

    default_preds = (test_prob >= 0.50).astype(int)
    tuned_preds = (test_prob >= optimal_thresh).astype(int)

    print(
        '===================== Random Forest: Default Threshold (0.50) '
        '====================='
    )
    print(f'Accuracy: {accuracy_score(y_test, default_preds):.3f}')
    print(f'ROC AUC:  {roc_auc:.3f}')
    print('\nClassification Report:')
    print(
        classification_report(
            y_test, default_preds, target_names=['Not High', 'High'], zero_division=0
        )
    )
    print('Confusion Matrix:')
    print(confusion_matrix(y_test, default_preds))

    print(
        f'\n===================== Random Forest: Tuned Threshold '
        f'({optimal_thresh:.2f}) ====================='
    )
    print(f'Accuracy:             {accuracy_score(y_test, tuned_preds):.3f}')
    print(f'High-Risk Recall:     {recall_score(y_test, tuned_preds, zero_division=0):.3f}')
    print(f'High-Risk Precision:  {precision_score(y_test, tuned_preds, zero_division=0):.3f}')
    print(f'Optimal F1-Score:     {best_f1:.3f}')
    print('\nClassification Report:')
    print(
        classification_report(
            y_test, tuned_preds, target_names=['Not High', 'High'], zero_division=0
        )
    )
    print('Confusion Matrix:')
    print(confusion_matrix(y_test, tuned_preds))

    return model
