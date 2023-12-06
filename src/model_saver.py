import csv
import json
import numpy as np
from numpy import savetxt


class ModelSaver:
    def __init__(self, format: ["csv", "json"]) -> None:
        """
        This function is called when creating a new ModelSaver. It stores
        the format for the file to be saved/loaded from.

        Args:
            format: a string of either "csv" or "json".

        Returns:
            None.

        Raises:
            -
        """

        self._format = format

    def save_parameters(self, model, file) -> None:
        """
        This function saves the parameters of a given model in a file of the
        format chosen by the user.

        Args:
            model: the model from which the parameters need to be saved. Could
            be any model but the parameters are an np array.
            file: file to save the parameters in.

        Returns:
            None.

        Raises:
            -
        """

        parameters = model.get_parameters()

        if self._format == "csv":
            with open(file, 'w') as csv_file:
                savetxt(csv_file, parameters, delimiter=',')

        elif self._format == "json":
            parameters = parameters.tolist()
            with open(file, 'w') as json_file:
                json.dump(parameters, json_file)

    def load_parameters(self, model, file) -> None:
        """
        This function loads parameters from a file and sets in in the given
        model

        Args:
            model: Model to set the parameters from. Could be any model but
            parameters are a np array.
            file: File to read the parameters from.

        Returns:
            None.

        Raises:
            -
        """
        if self._format == "csv":
            parameters = []
            with open(file, 'r') as csv_file:
                reader = csv.reader(csv_file)
                for row in reader:
                    parameters.append(row)
            np_array = np.array(parameters)
            model.set_parameters(np_array)
        elif self._format == "json":
            with open(file, 'r') as json_file:
                parameters = json.load(json_file)
            np_array = np.array(parameters)
            model.set_parameters(np_array)
