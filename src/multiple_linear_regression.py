import numpy as np
import pandas as pd
import os

class MultipleLinearRegression:
    def __init__(self, default_intercept:float, default_slope:float):
        self._parameters = default_intercept

    def train(self, observations:np.ndarray, target:np.ndarray) -> None:
        '''
        observations is a 2d numpy array of length n and width p+1 (with n = number of observations and p = number of features)
        target is a 1d numpy array of length n
        '''

        obs_transposed_product = np.matmul(observations.transpose, observations)
        obs_transposed_output = np.matmul(observations.transpose, target)
        optimal_parameters = np.matmul(np.linalg.inv(obs_transposed_product), obs_transposed_output)
        self._parameters = optimal_parameters

    def predict(self, data:np.ndarray) -> np.ndarray:
        return np.matmul(data, self._parameters)

if __name__ == "__main__":
    data = pd.read_csv(os.getcwd() + '/test data.csv')
    data.head()

    model = MultipleLinearRegression(0,0)
    model.train()
    model._parameters