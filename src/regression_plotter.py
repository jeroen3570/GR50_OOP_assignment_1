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

        x_transposed_product = np.matmul(X.transpose(), X)
        x_transposed_output = np.matmul(X.transpose(), target)
        optimal_parameters = np.matmul(np.linalg.inv(x_transposed_product), x_transposed_output)
        self._parameters = optimal_parameters

    def predict(self, data:np.ndarray) -> np.ndarray:
        base_weight = np.ndarray(shape=(len(data), 1))
        base_weight[:] = 1
        X = np.hstack((base_weight, data))
        return np.matmul(X, self._parameters)












import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class RegressionPlotter:
    def __init__(self, input:pd.DataFrame, output:float, parameters:float, plot_features:list[int] =None) -> None:
        self.input = np.array(input)
        self.labels = input.columns
        self.output = output
        self.model_parameters = parameters
        self.feature_array = None

        if plot_features is None:
            self.plot_features = np.arange(0, input.shape[1], 1)
        else:
            self.plot_features = plot_features


    def plot_2D(self) -> None:
        print("2D")
        pass

    def plot_3D(self) -> None:
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.set_xlabel(self.labels[self.plot_features[0]])
        ax.set_ylabel(self.labels[self.plot_features[1]])
        ax.set_zlabel('Output')
        
        x_data = self.input[:, self.plot_features[0]]
        y_data = self.input[:, self.plot_features[1]]
        z_data = self.output

        ax.scatter(x_data, y_data, z_data)
        plt.show()

        """print("\nself.plot_features[0]:")
        print(self.plot_features[0])

        print("\nself.feature_array:")
        print(self.input)

        print("\nx data: ")
        print(x_data)

        print("\ny data: ")
        print(y_data)

        print("\nz data: ")
        print(z_data) """


        

    def plot(self) -> None:
        #choose plot

        if (len(self.plot_features)==2):
            self.plot_3D()
        else:
            self.plot_2D()

        



if __name__ == "__main__":
    ###################################
    # from multiple linear regression #
    ###################################
    data = pd.read_csv(os.getcwd() + '/src/test data.csv',  delimiter=';')
    features = data[["Humidity", "Wind Speed (km/h)", "Wind Bearing (degrees)", "Visibility (km)", "Pressure (millibars)"]]
    output = data['Temperature (C)']

    model = MultipleLinearRegression(0)
    model.train(features, output)
    
    res = model.predict(features)

    ###################################
    # from regression plotter         #
    ###################################
    #a = np.array([[2, 3, 4],[5, 6,7]])

    #b = np.array([1, 2])
    rp = RegressionPlotter(features, res, model._parameters, [1, 3])
    rp.plot()


    #print(rp.plot_features)

    