# conda env create -f enviornment.yml
# conda activate myenv

import pandas as pd
from sklearn.model_selection import train_test_split

from rf_training import rf_model_train
from dt_training import decision_tree_train
from lr_training import lr_model_train
from xgboost_training import xgb_model_train
from cross_vali import cross_val_with_smote

dataset = "healthcare_real_time_dataset.csv"

def main():

    def read_data(data):
        df = pd.read_csv(data)
        exploratory_step(df)
        return df
        """
        assigns the raw dataset to a df variable.
        """

    def preprocess_data(df):

        df_processed = pd.get_dummies(df, drop_first=True)

        training_split(df_processed)
        

    def exploratory_step(dataframe):
        """
        Exploring the dataset
        """
        record_count = dataframe.shape[0]
        higher_age_split = dataframe[dataframe['Age'] < 50].shape[0]
        lower_age_split = dataframe[dataframe['Age'] > 50].shape[0]
        print(dataframe.info())
        print("")
        print(f"Total records: {record_count}")
        print(f"Number of patients aged 50 or older: {higher_age_split}")
        print(f"Number of patients younger than 50: {lower_age_split}")
        print("")
        print("CHECK FOR MISSING VALUES")
        print(dataframe.isnull().sum())
        print("")
        print("CHECK FOR DUPLICATE ROWS")
        print(dataframe.duplicated().sum())


        
    def preprocess_data(df):
        # Fill missing values in 'Chronic Disease History' with 'Unknown'
        df.loc[:, 'Chronic Disease History'] = df['Chronic Disease History'].fillna('Unknown')

        # Feature Engineering

        # Age groups
        df['Age_Group'] = pd.cut(df['Age'], bins=[0, 30, 50, 100], labels=['Young Adult', 'Middle Aged', 'Senior'])

        # BMI
        df['BMI_Category'] = pd.cut(df['BMI'], bins=[0, 18.5, 25, 32, 100], labels=['Underweight', 'Normal', 'Overweight', 'Obese'])

        # Sedentary vs Active flag
        df['Is_Active'] = (df['Physical Activity (hours/week)'] >= 2.5).astype(int)



        # Lifestyle risk score
        def lifestyle_score(row):
            score = 0
            if row['Sleep Duration (hours/day)'] >= 7:
                score += 1
            if row['Alcohol Consumption (per week)'] < 7:
                score += 1
            if row['Physical Activity (hours/week)'] > 5:
                score += 1
            if row['Stress Level (1-10)'] <= 4:
                score += 1
            return score

        df['Lifestyle_Score'] = df.apply(lifestyle_score, axis=1)

        # Stress-Sleep ratio
        df['Stress_Sleep_Ratio'] = df['Stress Level (1-10)'] / df['Sleep Duration (hours/day)']

        # Separate target variable
        y = df['Health Risk Level']
        X = df.drop('Health Risk Level', axis=1)

        # One-hot encode categorical features including new categorical features
        X_encoded = pd.get_dummies(X, drop_first=True)

        return X_encoded, y
        

    def training_split(X, y):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=123
        )

        print(f"Training set size: {X_train.shape[0]}")
        print(f"Testing set size: {X_test.shape[0]}")
        
        rf_model_train(X_train, X_test, y_train, y_test)
        decision_tree_train(X_train, X_test, y_train, y_test)
        lr_model_train(X_train, X_test, y_train, y_test)

        # Cross-validation with SMOTE calls
        print("Random Forest CV with SMOTE:")
        cross_val_with_smote(RandomForestClassifier, X, y, model_params=rf_params, n_splits=5)

        print("Decision Tree CV with SMOTE:")
        cross_val_with_smote(DecisionTreeClassifier, X, y, model_params=dt_params, n_splits=5)

        print("Logistic Regression CV with SMOTE:")
        cross_val_with_smote(LogisticRegression, X, y, model_params=lr_params, n_splits=5, use_scaler=True)

        print("XGBoost CV with SMOTE:")
        cross_val_with_smote(xgb.XGBClassifier, X, y, model_params=xgb_params, n_splits=5)

        return X_train, X_test, y_train, y_test


    # execution steps
    df = read_data(dataset)

    X, y = preprocess_data(df)
    X_train, X_test, y_train, y_test = training_split(X, y)

    

    rf_model_train(X_train, X_test, y_train, y_test)
    decision_tree_train(X_train, X_test, y_train, y_test)
    lr_model_train(X_train, X_test, y_train, y_test)


  
    print(y_train.value_counts())

if __name__ == "__main__":
    main()

