import numpy as np
import pandas as pd
import os
from sklearn.linear_model import LinearRegression


class MultipleLinearRegression:
    def __init__(self, default_intercept: np.ndarray) -> None:
        """
        This function is called when creating a new MultipleLinearRegression
        instance. It stores the model parameters.

        Args:
            default_intercept: numpy array of the default parameters of
            the multiple linear regression model.

        Returns:
            None.

        Raises:
            -
        """
        self._parameters = default_intercept

    def train(self, observations: pd.DataFrame, target: np.ndarray) -> None:
        """
        This function is called when creating a new MultipleLinearRegression
        instance. It stores the model parameters.

        Args:
            observations: pandas DataFrame of the input data
                          with width p and length n.
            target: numpy array of the requested
                    output that the model should find.

        Returns:
            None.

        Raises:
            -
        """

        base_weight = np.ndarray(shape=(len(observations), 1))
        base_weight[:] = 1
        X = np.hstack((base_weight, observations))
        x_transposed_product = np.matmul(X.transpose(), X)
        x_transposed_output = np.matmul(X.transpose(), target)
        optimal_parameters = np.matmul(np.linalg.inv(x_transposed_product),
                                       x_transposed_output)
        self._parameters = optimal_parameters

    def predict(self, data: pd.DataFrame) -> np.ndarray:
        """
        This function uses the parameters of the model
        to predict the output using the input

        Args:
            observations: pandas Dataframe of input
            data that is used to predict the output

        Returns:
            A numpy array of the output

        Raises:
            -
        """
        base_weight = np.ndarray(shape=(len(data), 1))
        base_weight[:] = 1
        X = np.hstack((base_weight, data))
        return np.matmul(X, self._parameters)

    def get_parameters(self) -> np.ndarray:
        """
        This function returns the array of the model parameters

        Args:
            -

        Returns:
            numpy array of the model parameters.

        Raises:
            -
        """
        return self._parameters

    def set_parameters(self, parameters: np.ndarray) -> np.ndarray:
        """
        This function sets the model parameters

        Args:
            the new parameters

        Returns:
            -

        Raises:
            -
        """
        self._parameters = parameters


def skTest(observations: pd.DataFrame, output: np.ndarray) -> None:
    """
    This function calculates and prints the model parameters based on sklearn.

    Args:
        observations: numpy array of the input data with width p and length n.
        target: numpy array of the requested output that the model should find.

    Returns:
        None.

    Raises:
        -
    """
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
    features = data[["Humidity", "Wind Speed (km/h)", "Wind Bearing (degrees)",
                     "Visibility (km)", "Pressure (millibars)"]]
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
