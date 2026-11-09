# Age,
# Gender,
# BMI,
# Smoking Status,
# Alcohol Consumption (per week),
# Physical Activity (hours/week),
# Sleep Duration (hours/day),
# Chronic Disease History,
# Stress Level (1-10),
# Health Risk Level

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (

    #percentage of predictions that are correct
    accuracy_score, 
    
    #Number of times the model predicts a 
    # patient as high risk and is correct.
    classification_report, 
   
    
    confusion_matrix,

    #The balance between precision and recall
    f1_score,

    precision_score,

    #The percent of actual high cases that were detected. 
    recall_score,

    #How well the model ranks positive cases above the negative ones. 
    roc_auc_score,

    #Used for visualization
    roc_curve,
    auc,
)

from sklearn.model_selection import RandomizedSearchCV, train_test_split

from cross_vali import cross_val_func
from params import param_assign
from rf_training import rf_model_train

dataset = "healthcare_real_time_dataset.csv"

def main():

    """
    Target column is changed from multi-feature to binary. 
    """
    def ytarget_binary_convert(y, positive_class="High"):
        return (y == positive_class).astype(int)

    def exploratory_step(df):
        """
        Data Exploration
        """
        print(df.info())
        print()
        print(f"Total records: {df.shape[0]}")
        print(f"Number of patients aged 50 or older: {(df['Age'] >= 50).sum()}")
        print(f"Number of patients younger than 50: {(df['Age'] < 50).sum()}")
        print()
        print("CHECK FOR MISSING VALUES")
        print(df.isnull().sum())
        print()
        print("CHECK FOR DUPLICATE ROWS")
        print(df.duplicated().sum())

    def plot_target_distribution(df):
        plt.figure(figsize=(7, 5))
        order = df["Health Risk Level"].value_counts().index
        sns.countplot(data=df, x="Health Risk Level", order=order)
        plt.title("Distribution of Health Risk Levels")
        plt.xlabel("Health Risk Level")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.show()

    def plot_roc_curve(y_true, y_prob, label="Model"):
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)

        plt.figure(figsize=(7, 5))
        plt.plot(fpr, tpr, label=f"{label} (AUC = {roc_auc:.3f})", linewidth=2)
        plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
        plt.title("ROC Curve")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.show()

    def plot_confusion_matrix(y_true, y_pred, title="Confusion Matrix"):
        cm = confusion_matrix(y_true, y_pred)

        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
        plt.title(title)
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.tight_layout()
        plt.show()

    def preprocess_data(df):

        """
        Create copy of the dataset.
        """
        df = df.copy()

        # Fill missing values
        df["Chronic Disease History"] = df["Chronic Disease History"].fillna("None")


        # Categorical feature bins

        #AGE
        df["Age_Group"] = pd.cut(
            df["Age"],
            bins=[0, 30, 50, 100],
            labels=["Young Adult", "Middle Aged", "Senior"],
            include_lowest=True,
        )

        #BMI
        df["BMI_Category"] = pd.cut(
            df["BMI"],
            bins=[0, 18.5, 25, 32, 100],
            labels=["Underweight", "Normal", "Overweight", "Obese"],
            include_lowest=True,
        )

        #Pysical Activity
        df["Is_Active"] = (df["Physical Activity (hours/week)"] >= 2.5).astype(int)

        #BMI/Age and Stress/Sleep
        """
        - Multiply age / bmi together 
        """
        df["Age_BMI"] = df["Age"] * df["BMI"]

        """
        - divides sress level by sleep duration
        - 1e-5 is equal to 0.00001 in python. This avoids division by zero. 
        """
        df["Stress_Sleep_Ratio"] = df["Stress Level (1-10)"] / (
            df["Sleep Duration (hours/day)"] + 1e-5
        )
        
        """
        physical activity is divided by bmi. 
        """
        df["Activity_BMI"] = df["Physical Activity (hours/week)"] / (
            df["BMI"] + 1e-5
        )

        """
        - checks whether the patient consumes at least 7 alcoholic beverages per week
        - this creates 1 if seven or more and 0 if fewer than seven. 
        """

        """
        Checks if the patient has ever been a tobacco user. 
        this creates 1 if ever used and 0 if never. 

        The final score for this funcion will rage from 0 to 2 depending on these features. 
        """
        #Alcohol/Smoking
        df["Alcohol_Smoking_Risk"] = (
            (df["Alcohol Consumption (per week)"] >= 7).astype(int)
                
            + (df["Smoking Status"] != "Never").astype(int)
        )

        # Chronic condition risk flag
        """
        These are high risk conditions. 
        The result is 1 if having diabetes or heart disease. 
        0 if there is anything else. 
        """
        high_risk_conditions = ["Diabetes", "Heart Disease"]
        df["High_Risk_Condition"] = df["Chronic Disease History"].isin(
            high_risk_conditions
        ).astype(int)

        # Lifestyle score
        def lifestyle_score(row):
            """
            Creates a score system for healthy living indicators on lifestyle data
            features. 
            """
            score = 0
            if row["Sleep Duration (hours/day)"] >= 7:
                score += 1
            if row["Alcohol Consumption (per week)"] < 7:
                score += 1
            if row["Physical Activity (hours/week)"] > 5:
                score += 1
            if row["Stress Level (1-10)"] <= 6:
                score += 1
            if row["BMI"] <= 32:
                score += 1
            if row["Smoking Status"] == "Never":
                score += 1
            return score

        df["Lifestyle_Score"] = df.apply(lifestyle_score, axis=1)

        """
        Selects the target column and assigns this to "y"
        Creates the DataFrame X by removing the target column. 
        X = input features and y = the output the model will predict.
        """
        y = df["Health Risk Level"]
        X = df.drop("Health Risk Level", axis=1)

        #Encode multi-category columns (Gender, Smoking status, etc.) 
        # into numerical columns.
        X_encoded = pd.get_dummies(X, drop_first=True)
        return X_encoded, y

    def training_split(X, y):
        """
        performing an 80 / 20 split on the dataframe. 
        """
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=0.2, random_state=123, stratify=y
        )

        """
        Splits again on the temporary 80% into 75 for training / 25 for validation
        """

        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=0.25, random_state=123, stratify=y_temp
        )

        """
        This creates:
        60% - Training
        20% - Validation
        20% - Testing
        """
        print(f"Training set size: {X_train.shape[0]}")
        print(f"Validation set size: {X_val.shape[0]}")
        print(f"Testing set size: {X_test.shape[0]}")

        return X_train, X_val, X_test, y_train, y_val, y_test

    def xgb_hyperparameter_tuning(X_train, y_train):

        """
        To handle imbalance, this function searches for the best XGBoost settings.
        """

        """
        The poitive cases are counted (high-risk examples)
        """
        pos_count = np.sum(y_train == 1)

        """
        Counts the negative cases (not high risk examples)
        """
        neg_count = np.sum(y_train == 0)

        """
        Calculates the balance of the target column and will tell XGBoost to give more 
        importance to the positive examples if the negative outweighs them. 
        """
        calculated_weight = neg_count / pos_count if pos_count > 0 else 1.0


        """
        A dictionary of possible hyperparameter values
        """
        param_dist = {
            "max_depth": [2, 3, 4],
            "learning_rate": [0.01, 0.05, 0.1],
            "n_estimators": [50, 100, 150],
            "subsample": [0.7, 0.8, 1.0],
            "colsample_bytree": [0.6, 0.8, 1.0],
            "scale_pos_weight": [1.0, calculated_weight, 2.0],
            "reg_alpha": [0.0, 0.5, 1.0],
            "reg_lambda": [1.0, 2.0, 5.0],
        }

        """
        Creating the model

        binary classification is expressed, logarithmic loss is used as the evaluation metric
        during training. It penalizes the confident incorrect predictions that the model makes. 
        """
        model = xgb.XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=123,
        )

        random_search = RandomizedSearchCV(
            model,
            param_distributions=param_dist,
            n_iter=20,
            scoring="roc_auc",
            cv=5,
            verbose=1,
            n_jobs=-1,
            random_state=123,
        )

        random_search.fit(X_train, y_train)

        print("Best parameters found:")
        print(random_search.best_params_)
        print(f"Best CV ROC AUC: {random_search.best_score_:.3f}")

        return random_search.best_estimator_

    def tune_threshold(
        y_true,
        y_probs,
        target_recall=0.65,
        min_precision=0.25,
        threshold_low=0.20,
        threshold_high=0.50,
        step=0.01,
    ):
        thresholds = np.arange(threshold_low, threshold_high + step, step)

        best_thresh = 0.50
        best_f1 = -1
        best_precision = -1
        best_recall = -1

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

    def evaluate_predictions(y_true, y_pred, y_prob=None, label="Model"):
        print(f"\n--- {label} ---")
        print(f"Accuracy: {accuracy_score(y_true, y_pred):.3f}")
        if y_prob is not None:
            print(f"ROC AUC: {roc_auc_score(y_true, y_prob):.3f}")
        print(f"High-Risk Recall: {recall_score(y_true, y_pred, zero_division=0):.3f}")
        print(f"High-Risk Precision: {precision_score(y_true, y_pred, zero_division=0):.3f}")
        print(f"F1-Score: {f1_score(y_true, y_pred, zero_division=0):.3f}")
        print(
            classification_report(
                y_true,
                y_pred,
                target_names=["Not High", "High"],
                zero_division=0,
            )
        )
        print(confusion_matrix(y_true, y_pred))

    rf_params, xgb_params = param_assign()

    df = pd.read_csv(dataset)
    exploratory_step(df)

    # Visual 1: target distribution
    plot_target_distribution(df)

    X, y = preprocess_data(df)
    y_binary = ytarget_binary_convert(y, positive_class="High")

    X_train, X_val, X_test, y_train, y_val, y_test = training_split(X, y_binary)

    # Train & Tune XGBoost
    best_xgb_model = xgb_hyperparameter_tuning(X_train, y_train)

    # Validation threshold tuning
    val_prob = best_xgb_model.predict_proba(X_val)[:, 1]
    optimal_thresh, val_rec, val_prec, val_f1 = tune_threshold(
        y_val,
        val_prob,
        target_recall=0.65,
        min_precision=0.25,
    )

    print(f"\nOptimized Threshold (Validation): {optimal_thresh:.2f}")
    print(f"Validation Recall: {val_rec:.3f}")
    print(f"Validation Precision: {val_prec:.3f}")
    print(f"Validation F1: {val_f1:.3f}")

    # Final test evaluation
    test_prob = best_xgb_model.predict_proba(X_test)[:, 1]
    roc_auc = roc_auc_score(y_test, test_prob)

    default_preds = (test_prob >= 0.50).astype(int)
    tuned_preds = (test_prob >= optimal_thresh).astype(int)

    print("\n--- XGBoost Standard Default Threshold ---")
    evaluate_predictions(y_test, default_preds, test_prob, label="XGBoost Standard Default Threshold")
    plot_confusion_matrix(y_test, default_preds, title="XGBoost Confusion Matrix ( Standard Default Threshold)")

    print(f"\n--- XGBoost Tuned Threshold ({optimal_thresh:.2f}) ---")
    evaluate_predictions(y_test, tuned_preds, test_prob, label="XGBoost Tuned Threshold")
    plot_confusion_matrix(y_test, tuned_preds, title=f"XGBoost Confusion Matrix (Tuned Threshold {optimal_thresh:.2f})")

    # Visual 3: ROC curve
    plot_roc_curve(y_test, test_prob, label="XGBoost")

    # Train Random Forest
    print("\n--- Random Forest Training ---")
    rf_model_train(X_train, X_val, X_test, y_train, y_val, y_test)

    # Cross Validation
    print("\n--- Cross Validation ---")
    print("Random Forest Cross Validation:")
    cross_val_func(
        RandomForestClassifier, X, y_binary, model_params=rf_params, n_splits=5
    )

    print("\nXGBoost Cross Validation:")
    cross_val_func(
        xgb.XGBClassifier, X, y_binary, model_params=xgb_params, n_splits=5
    )

    print("\nTraining set target distribution:")
    print(y_train.value_counts())

    # Visual 4: confusion matrix already shown above
    # If you want a second confusion matrix plot for the tuned RF later,
    # you can add it inside rf_training.py or return predictions from rf_model_train.


if __name__ == "__main__":
    main()
