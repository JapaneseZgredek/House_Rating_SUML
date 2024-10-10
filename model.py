from sklearn.linear_model import LinearRegression
import pickle
import numpy as np

def train_model(X_train, y_train, model_filename='house_price_model.pkl'):
    model = LinearRegression()
    model.fit(X_train, y_train)
    with open(model_filename, 'wb') as file:
        pickle.dump(model, file)

# Load model from pickle file
def load_model(model_filename='house_price_model.pkl'):
    with open(model_filename, 'rb') as file:
        model = pickle.load(file)
    return model

def predict(model, input_features):
    input_fetures = np.array(input_features).reshape(1, -1)  # reshape input into vector
    prediction = model.predict(input_fetures)  # returns list with one element
    return prediction[0]
