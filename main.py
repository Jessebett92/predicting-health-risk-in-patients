# conda env create -f enviornment.yml
# conda activate myenv
import pandas as pd
from sklearn.model_selection import train_test_split
from rf_training import rf_model_train
from dt_training import decision_tree_train
from lr_training import lr_model_train

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

        df['Chronic Disease History'].fillna('Unknown', inplace=True)

        y = df['Health Risk Level']
        X = df.drop('Health Risk Level', axis=1)

        X_encoded = pd.get_dummies(X, drop_first=True)

        return X_encoded, y
        

    def training_split(X, y):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=123
        )

        print(f"Training set size: {X_train.shape[0]}")
        print(f"Testing set size: {X_test.shape[0]}")
        return X_train, X_test, y_train, y_test


    # execution steps
    df = read_data(dataset)
    X, y = preprocess_data(df)
    X_train, X_test, y_train, y_test = training_split(X, y)

    rf_model_train(X_train, X_test, y_train, y_test)
    decision_tree_train(X_train, X_test, y_train, y_test)
    lr_model_train(X_train, X_test, y_train, y_test)

if __name__ == "__main__":
    main()

