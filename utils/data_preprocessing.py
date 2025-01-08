
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
import pickle

def load_data(file_path):
    data = pd.read_csv(file_path)
    return data

def preprocess_data(data, target_column='SalePrice', test_size=0.2, random_state=42):
    # Drop rows with missing target values
    data = data.dropna(subset=[target_column])

    # Drop columns with high percentage of missing values
    missing_threshold = 50  # Threshold: columns with >50% missing values are dropped
    missing_percentage = data.isnull().mean() * 100
    data = data.drop(columns=missing_percentage[missing_percentage > missing_threshold].index)

    # Select features highly correlated with the target
    correlation_matrix = data.select_dtypes(include=['number']).corr()
    target_correlation = correlation_matrix[target_column].abs().sort_values(ascending=False)
    top_features = target_correlation.index[1:6].tolist()  # Top 5 features excluding target
    selected_columns = top_features + [target_column]
    data = data[selected_columns]

    # Fill missing values in numeric columns with their mean
    numeric_columns = data.select_dtypes(include=['number']).columns
    data[numeric_columns] = data[numeric_columns].fillna(data[numeric_columns].mean())

    # Fill missing values in categorical columns with a placeholder
    categorical_columns = data.select_dtypes(include=['object']).columns
    data[categorical_columns] = data[categorical_columns].fillna('Unknown')

    # Save remaining column names to a .pkl file
    column_names = data.columns.tolist()
    with open('remaining_columns.pkl', 'wb') as f:
        pickle.dump(column_names, f)

    # Separate features (X) and target (y)
    X = data.drop(columns=[target_column])
    y = data[target_column]

    # Create transformers for numeric and categorical data
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')

    # Create a preprocessing pipeline that applies the transformations
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, X.select_dtypes(include=['number']).columns),
            ('cat', categorical_transformer, X.select_dtypes(include=['object']).columns)
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