import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def load_data(file_path):
    data = pd.read_csv(file_path)
    return data

def preprocess_data(data, target_column, test_size=0.2, random_state=42):
    data = data.dropna(subset=[target_column])  # Removes rows with empty target_column

    data = data.fillna(data.mean())  # Replace NaN values with mean value of it's column

    X = data.drop(columns=[target_column])  # Returns 2-dimensional list, data set without target column
    y = data[target_column]  # Drops target column

    scaler = StandardScaler()  # Scaler standardize values that it's values mean is 0 and standard deviation 1
    X_scaled = scaler.fit_transform(X)  # Standardize data

    X_train, y_train, X_test, y_test = train_test_split(X_scaled, y, test_size=test_size, random_state=random_state)  # Train data

    return X_train, y_train, X_test, y_test, scaler

def preprocess_input(input_data, scaler):
    input_data = pd.DataFrame([input_data])
    input_data_scaled = scaler.fit_transform(input_data)
    return input_data_scaled
