import csv
import json
import numpy as np
from numpy import savetxt


class ModelSaver:
    def __init__(self, format: ["csv", "json"]) -> None:

        self._format = format

    def save_parameters(self, model, file) -> None:

        parameters = model.get_parameters()

        if self._format == "csv":
            with open(file, 'w') as csv_file:
                savetxt(csv_file, parameters, delimiter=',')

        elif self._format == "json":
            parameters = parameters.tolist()
            with open(file, 'w') as json_file:
                json.dump(parameters, json_file)

    def load_parameters(self, model, file) -> None:
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
