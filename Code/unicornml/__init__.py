import sys
import yaml
import pandas as pd
import numpy as np
from .regression import Regression
from .classification import Classification
from .model import Model
from .preprocessing import Preprocessing, file_split_X_y

from .images import Images


class UnicornML:
    __problem: str
    __algorithms: list
    __metric: str
    model: object
    output_classes: int
    input_shape: tuple
    cv: int
    images: bool
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray

    def __init__(self, input=None, options=None):
        if input is None:
            input = {}
        if options is None:
            options = {}

        if not bool(input):
            sys.exit("Undefined input data")

        if "file" in input:
            data = pd.read_csv(input["file"])
            label_index = input["label_col"] if "label_col" in input else -1
            X, y = file_split_X_y(data, label_index)
        elif "X" in input and "y" in input:
            X, y = input["X"], input["Y"]
        elif "images" in input:
            self.images = True
            directory = input["images"]
            input_shape = (options["height"], options["width"], options["depth"])
            if "fine_tuning" in options:
                fine_tuning = options["fine_tuning"]
            else:
                fine_tuning = False
            self.model = Images(input_shape, directory, fine_tuning=fine_tuning)
        else:
            sys.exit("Invalid options for input")

        if not self.images:
            self.cv = 5

            self.X_train, self.X_test, self.y_train, self.y_test, (self.__problem, self.output_classes) = Preprocessing(X, y, self.cv)
            self.input_shape = self.X_train.shape

            with open("options.yaml") as file:
                config = yaml.full_load(file)

            if "algorithms" in options:
                if not isinstance(options["algorithms"], list):
                    sys.exit("The \"algorithms\" paramater needs to be a list")

                for alg in options["algorithms"]:
                    if not isinstance(alg, str):
                        sys.exit("The algorithm need to be a string")
                    if alg not in config["Problem"][self.__problem]["algorithms"]:
                        sys.exit(
                            "Invalid algorithm %s for a %s problem. Algorithms available:[%s]" % (
                                alg,
                                self.__problem,
                                ", ".join(config["Problem"][self.__problem]["algorithms"])
                            )
                        )
                self.__algorithms = options["algorithms"]
            else:
                self.__algorithms = config["Problem"][self.__problem]["algorithms"]

            if "metric" in options:
                if not isinstance(options["metric"], str):
                    sys.exit("The \"metric\" paramater needs to be a string (choose only one metric, please)")

                if options["metric"] not in config["Problem"][self.__problem]["metrics"]:
                    sys.exit(
                        "Invalid metric %s for a %s problem. Metrics available:[%s]" % (
                            options["metric"],
                            self.__problem,
                            ", ".join(config["Problem"][self.__problem]["metrics"])
                        )
                    )
                self.__metric = options["metric"]
            else:
                self.__metric = config["Problem"][self.__problem]["metrics"][0]

            print("\nIt's a %s problem\nSelected algorithms: [%s]\nSelected metric: [%s]\n" % (
                self.__problem, 
                ",".join(self.__algorithms), self.__metric
            ))

    def Rainbow(self):
        if self.images:
            self.model.train()
        else:
            for algorithm in self.__get_model_algorithms():
                sqrt = True if "sqrt" in algorithm.keys() else False
                self.model.param_tunning_method(
                    algorithm["estimator"],
                    algorithm["desc"],
                    algorithm["params"],
                    sqrt
                )
                if self.__metric == 'mse' and self.get_best_model(False) < 0.01:
                    print("Stopping training early, because a good enough result was achieved")
                    break
                elif  self.__metric == 'accuracy' and self.get_best_model(False) > 0.95:
                    print("Stopping training early, because a good enough result was achieved")
                    break

    def __get_model_algorithms(self):
        if self.__problem == "Classification":
            classificator = Classification(
                self.input_shape,
                self.__algorithms,
                self.__metric,
                self.output_classes
            )
            self.model = Model(
                self.X_train, self.X_test, self.y_train, self.y_test,
                classificator.get_metric(), 1, self.cv
            )
            algorithms = classificator.get_algorithms()
        else:
            regressor = Regression(
                self.input_shape,
                self.__algorithms,
                self.__metric
            )
            self.model = Model(
                self.X_train, self.X_test, self.y_train, self.y_test,
                regressor.get_metric(), regressor.get_metric_sign(), self.cv
            )
            algorithms = regressor.get_algorithms()

        return algorithms

    def get_best_model(self, verbose=True):
        if self.model.metric_sign == -1:
            bestModel = sorted(self.model.results, key=lambda x: x["score"], reverse=False)[0]
        else:
            bestModel = sorted(self.model.results, key=lambda x: x["score"], reverse=True)[0]
        if verbose:
            print("Best model: {0}\t Score: {1}".format(bestModel["name"], bestModel["score"]))
            return bestModel["model"]
        else:
            return bestModel["score"]

    def predict(self, X):
        return self.get_best_model().predict(X)

    def evaluate(self, y, yatt):
        return self.model.metric(y, yatt)
