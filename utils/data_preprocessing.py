import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split

def load_data(file_path):
    data = pd.read_csv(file_path)
    return data

def preprocess_data(data, target_column, test_size=0.2, random_state=42):
    # Drop rows with missing target values
    data = data.dropna(subset=[target_column])

    # Fill missing values in numeric columns with their mean
    numeric_columns = data.select_dtypes(include=['number']).columns
    data[numeric_columns] = data[numeric_columns].fillna(data[numeric_columns].mean())

    # Separate features (X) and target (y)
    X = data.drop(columns=[target_column])
    y = data[target_column]

    # Identify categorical and numerical columns
    categorical_columns = X.select_dtypes(include=['object']).columns
    numeric_columns = X.select_dtypes(include=['number']).columns

    # Create transformers for numeric and categorical data
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')

    # Create a preprocessing pipeline that applies the transformations
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_columns),
            ('cat', categorical_transformer, categorical_columns)
        ])

    # Apply the transformations
    X_processed = preprocessor.fit_transform(X)

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=test_size, random_state=random_state)

    return X_train, y_train, X_test, y_test, preprocessor

def preprocess_input(input_data, scaler):
    input_data = pd.DataFrame([input_data])
    input_data_scaled = scaler.fit_transform(input_data)
    return input_data_scaled
