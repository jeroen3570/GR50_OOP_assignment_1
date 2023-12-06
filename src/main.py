import pandas as pd
import os
import multiple_linear_regression as mlr
import regression_plotter as rp
import model_saver as ms

if __name__ == "__main__":
    data1 = pd.read_csv(os.getcwd() + '/src/test data1.csv',  delimiter=';')
    print(data1.columns)
    features1 = data1[["Humidity", "Wind Speed (km/h)",
                       "Wind Bearing (degrees)", "Visibility (km)",
                       "Pressure (millibars)"]]
    output1 = data1['Temperature (C)']

    print("test with sklearn: \n")
    mlr.skTest(features1, output1)

    print("\n\n")
    model1 = mlr.MultipleLinearRegression(0)
    model1.train(features1, output1)

    print("\nmodel parameters:")
    print(model1._parameters)
    res1 = model1.predict(features1)
    print("\naverage diff by model:")
    print(sum(abs(res1 - output1))/len(res1))

    ###################################
    # from multiple linear regression #
    ###################################
    data2 = pd.read_csv(os.getcwd() + '/src/test data1.csv',  delimiter=';')
    features2 = data2[["Humidity", "Wind Speed (km/h)",
                       "Wind Bearing (degrees)", "Visibility (km)",
                       "Pressure (millibars)"]]
    output2 = data2['Temperature (C)']

    model2 = mlr.MultipleLinearRegression(0)
    model2.train(features2, output2)
    ###################################
    # from regression plotter         #
    ###################################
    regression_plot = rp.RegressionPlotter(features2, model2, [1, 2, 3])
    regression_plot.plot()
    ###################################
    # Test model saver and loader     #
    ###################################
    print("This are parameters before saving:\n")
    print(model2.get_parameters())
    model_saver = ms.ModelSaver('json')
    model_saver.save_parameters(model2, 'savedParameters')
    print("Parameters are saved\n")
    model_saver.load_parameters(model2, 'savedParameters')
    print("This are parameters after loading:\n")
    print(model2.get_parameters())
