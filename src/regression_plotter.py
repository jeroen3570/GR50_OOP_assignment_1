import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import multiple_linear_regression as mr


class RegressionPlotter:
    def __init__(self, input: pd.DataFrame, model: mr.MultipleLinearRegression,
                 plot_features: list[int] = None) -> None:
        """
        This function is called when creating a new RegressionPlotter instance.
        It stores the input data, calculated output data from the model,
        the model parameters and which features should be plotted

        Args:
            input: pandas dataframe of the input data with each feature
            in one column.
            model: MultipleLinearRegression instance.
            plot_features: list of index values of features corresponding
            to the input dataframe that should be plotted.

        Returns:
            None.

        Raises:
            -
        """
        self._input = np.array(input)
        self._labels = input.columns
        self._model_parameters = model.get_parameters()
        self._output = model.predict(self._input)
        self._plot_features = None

        if plot_features is None:
            self._plot_features = np.arange(0, input.shape[1], 1)
        else:
            self._plot_features = plot_features

    def _plot_2D(self) -> None:
        """
        This function creates (a series of) 2D plots to show the
        output compared to the input.

        Args:
            None.

        Returns:
            None.

        Raises:
            None.
        """

        number_of_columns = 3
        number_of_rows = int(len(self._plot_features) // number_of_columns)
        position = range(1, len(self._plot_features) + 1)

        if (len(self._plot_features) % number_of_columns) != 0:
            number_of_rows += 1

        fig = plt.figure()
        for i in range(len(self._plot_features)):
            ax = fig.add_subplot(number_of_rows, number_of_columns,
                                 position[i])
            ax.set_xlabel(self._labels[self._plot_features[i]])
            ax.set_ylabel('Output')
            x_data = self._input[:, self._plot_features[i]]
            y_data = self._output
            x_range = np.arange(min(x_data), max(x_data), 1)
            x_param = self._model_parameters[1+self._plot_features[i]]
            y_result = self._model_parameters[0] + x_param*x_range

            ax.plot(y_result, color='r', label='Regression model')
            ax.scatter(x_data, y_data, label='Data points')
            ax.legend()
        plt.show()

    def _plot_3D(self) -> None:
        """
        This function creates one 3D plot of two input
        features and one output value.

        Args:
            None.

        Returns:
            None.

        Raises:
            None.
        """
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.set_xlabel(self._labels[self._plot_features[0]])
        ax.set_ylabel(self._labels[self._plot_features[1]])
        ax.set_zlabel('Output')
        x_data = self._input[:, self._plot_features[0]]
        y_data = self._input[:, self._plot_features[1]]
        z_data = self._output

        base_param = self._model_parameters[0]
        x_range = np.arange(min(x_data), max(x_data), 1)
        y_range = np.arange(min(y_data), max(y_data), 1)
        x_surface, y_surface = np.meshgrid(x_range, y_range)
        x_param = self._model_parameters[1+self._plot_features[0]]
        y_param = self._model_parameters[1+self._plot_features[1]]
        result = base_param + x_param * x_surface + y_param * y_surface
        ax.plot_surface(x_surface, y_surface, result, color='r',
                        label='Regression model')
        ax.scatter(x_data, y_data, z_data, label='Data points')
        ax.legend()
        plt.show()

    def plot(self) -> None:
        """
        This function creates scatter plots for each item
        (is an index for a feature in the input) in the list
        self._plot_features against the output.
        In this plot, the regression model is also shown.
        If there is only one index in the list, one 2D plot is created
        based on the feature and the output. The model is shown as a line.
        If there are two indices in the list, one 3D plot is created based on
        the features and the output. The regression model is shown as a plane.
        If there are more than two indices in the list, one 2D plot is created
        for each feature and the output. The model is shown as a line.

        Args:
            None.

        Returns:
            None.

        Raises:
            None.
        """

        if (len(self._plot_features) == 2):
            self._plot_3D()
        else:
            self._plot_2D()


if __name__ == "__main__":
    ###################################
    # from multiple linear regression #
    ###################################
    data = pd.read_csv(os.getcwd() + '/src/test data.csv',  delimiter=';')
    features = data[["Humidity", "Wind Speed (km/h)", "Wind Bearing (degrees)",
                     "Visibility (km)", "Pressure (millibars)"]]
    output = data['Temperature (C)']

    model = mr.MultipleLinearRegression(0)
    model.train(features, output)
    ###################################
    # from regression plotter         #
    ###################################
    rp = RegressionPlotter(features, model, [1, 2, 3])
    rp.plot()
