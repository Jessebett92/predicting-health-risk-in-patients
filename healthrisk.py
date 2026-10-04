# conda env create -f enviornment.yml
# conda activate myenv
import pandas as pd

dataset = "healthcare_real_time_dataset.csv"

def main():

    def read_data(data):
        df = pd.read_csv(data)
        exploratory_step(df)
        """
        assigns the raw dataset to a df variable.
        """
        

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


    # execution steps
    read_data(dataset)
    


if __name__ == "__main__":
    main()

