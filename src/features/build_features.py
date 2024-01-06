from sklearn.preprocessing import StandardScaler, RobustScaler
import pathlib
import pandas as pd

def scale_data(df):
    std_scaler = StandardScaler()
    rob_scaler = RobustScaler()

    df['scaled_amount'] = rob_scaler.fit_transform(df['Amount'].values.reshape(-1,1))
    df['scaled_time'] = rob_scaler.fit_transform(df['Time'].values.reshape(-1,1))

    df.drop(['Time','Amount'], axis=1, inplace=True)
    return df

def save_data(train, test, output_path):
    # Save the split datasets to the specified output path
    pathlib.Path(output_path).mkdir(parents=True, exist_ok=True)
    train.to_csv(output_path + '/train_scale.csv', index=False)
    test.to_csv(output_path + '/test_scale.csv', index=False)

def main():

    curr_dir = pathlib.Path(__file__)
    home_dir = curr_dir.parent.parent.parent
    output_path = home_dir.as_posix() + '/data/interim'
    train_file_path = home_dir.as_posix() + '/data/processed/train.csv'
    test_file_path = home_dir.as_posix() + '/data/processed/test.csv'

    train_df = pd.read_csv(train_file_path)
    test_df = pd.read_csv(test_file_path)
    train_df = scale_data(train_df)
    test_df = scale_data(test_df)

    save_data(train_df, test_df, output_path)


    


if __name__ == "__main__":
    main()