import pandas as pd
from sklearn.preprocessing import LabelEncoder

def load_data(file_path):
    # Load the dataset
    if file_path.endswith('.csv'):
        df = pd.read_csv(file_path)
    elif file_path.endswith('.data'):
        df = pd.read_csv(file_path, header=None)
    else:
        raise ValueError("Unsupported file format. Please use .csv or .data files.")

    # Initialize a dictionary to store the encoding
    encoding_dict = {}

    # Loop through each column and encode if necessary
    for column in df.columns:
        if df[column].dtype == object:
            le = LabelEncoder()
            df[column] = le.fit_transform(df[column])
            encoding_dict[column] = dict(zip(le.classes_, le.transform(le.classes_)))

    # Print the encoding used for each column
    for column, encoding in encoding_dict.items():
        print(f"Encoding for column '{column}': {encoding}")

    return df


