import pandas as pd
import os
import multiple_linear_regression as mlr
import regression_plotter as rp

if __name__ == "__main__":
    data = pd.read_csv(os.getcwd() + '/src/test data.csv',  delimiter=';')
    print(data.columns)
    features = data[["Humidity", "Wind Speed (km/h)", "Wind Bearing (degrees)",
                     "Visibility (km)", "Pressure (millibars)"]]
    output = data['Temperature (C)']

    print("test with sklearn: \n")
    mlr.skTest(features, output)

    print("\n\n")
    model = mlr.MultipleLinearRegression(0)
    model.train(features, output)

    print("\nmodel parameters:")
    print(model._parameters)
    res = model.predict(features)
    print("\naverage diff by model:")
    print(sum(abs(res - output))/len(res))

    ###################################
    # from multiple linear regression #
    ###################################
    data = pd.read_csv(os.getcwd() + '/src/test data.csv',  delimiter=';')
    features = data[["Humidity", "Wind Speed (km/h)", "Wind Bearing (degrees)",
                     "Visibility (km)", "Pressure (millibars)"]]
    output = data['Temperature (C)']

    model = mlr.MultipleLinearRegression(0)
    model.train(features, output)
    ###################################
    # from regression plotter         #
    ###################################
    regression_plot = rp.RegressionPlotter(features, model, [1, 2, 3])
    regression_plot.plot()
