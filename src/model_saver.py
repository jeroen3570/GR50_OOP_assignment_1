import csv
import json
import numpy as np
from numpy import savetxt


class ModelSaver:
    def __init__(self, format: ["csv", "json"]) -> None:

        self._format = format

    def save(self, model, file) -> None:

        parameters = model.get_parameters()

        if self._format == "csv":
            with open(file, 'w') as file:
                savetxt(file, parameters, delimiter=',')

        elif self._format == "json":
            with open(file, 'w') as file:
                json.dump(parameters, file)

    def load(self, model, file) -> None:
        parameters = []
        if self._format == "csv":
            with open(file, 'r') as file:
                reader = csv.reader(file)
                for row in reader:
                    parameters.append(row)
            np_array = np.array(parameters)
            model.set_parameters(np_array)
        elif self._format == "json":
            with open(file, 'r') as file:
                reader
