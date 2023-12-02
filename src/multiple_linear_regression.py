import numpy as np
import pandas as pd
import os
from sklearn.linear_model import LinearRegression

class MultipleLinearRegression:
    def __init__(self, default_intercept:float):
        self._parameters = default_intercept

    def train(self, observations:np.ndarray, target:np.ndarray) -> None:
        '''
        observations is a 2d numpy array of length n and width p (with n = number of observations and p = number of features)
        target is a 1d numpy array of length n
        '''

        base_weight = np.ndarray(shape=(len(observations), 1))
        base_weight[:] = 1
        X = np.hstack((base_weight, observations))
        print(X)

        x_transposed_product = np.matmul(X.transpose(), X)
        x_transposed_output = np.matmul(X.transpose(), target)
        optimal_parameters = np.matmul(np.linalg.inv(x_transposed_product), x_transposed_output)
        self._parameters = optimal_parameters

    def predict(self, data:np.ndarray) -> np.ndarray:
        base_weight = np.ndarray(shape=(len(data), 1))
        base_weight[:] = 1
        X = np.hstack((base_weight, data))
        return np.matmul(X, self._parameters)

def skTest(observations:np.ndarray, output:np.ndarray) -> None:
    mod = LinearRegression()
    mod.fit(observations, output)
    print("sklearn parameters:")
    print(mod.coef_)
    print("\naverage diff by sklearn:")
    pred = mod.predict(observations)

    print(sum(abs(pred - output))/len(pred))


if __name__ == "__main__":
    data = pd.read_csv(os.getcwd() + '/src/test data.csv',  delimiter=';')
    print(data.columns)
    features = data[["Humidity", "Wind Speed (km/h)", "Wind Bearing (degrees)", "Visibility (km)", "Pressure (millibars)"]]
    output = data['Temperature (C)']

    print("test with sklearn: \n")
    skTest(features, output)

    print("\n\n")
    model = MultipleLinearRegression(0)
    model.train(features, output)

    print("\nmodel parameters:")
    print(model._parameters)

    
    res = model.predict(features)
    print("\naverage diff by model:")
    print(sum(abs(res - output))/len(res))
