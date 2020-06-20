import sys
import yaml
import pandas as pd
import numpy as np
from .regression     import Regression
from .classification import Classification
from .model import Model
from .preprocessing import Preprocessing, file_split_X_y

class UnicornML:
    __problem       : str
    __algorithms    : list
    __metrics       : list
    model           : object
    output_classes  : int
    input_shape     : int
    X_train         : np.ndarray
    X_test          : np.ndarray
    y_train         : np.ndarray
    y_test          : np.ndarray

    def __init__(self, input = {}, options = {}):
        if not bool(input):
            sys.exit("Undefined input data")

        X, y = None, None
        if "file" in input:
            data = pd.read_csv(input["file"])
            label_index = input["label_col"] if "label_col" in input else -1
            X, y = file_split_X_y(data, label_index)
        elif "X" in input and "y" in input:
            X, y = input["X"], input["Y"]
        else:
            sys.exit("Invalid options for input")

        self.X_train, self.X_test, self.y_train, self.y_test, (self.__problem, self.output_classes) = Preprocessing(X,y)
        self.input_shape = self.X_train.shape

        config = None
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

        if "metrics" in options:
            if not isinstance(options["metrics"], str):
                sys.exit("The \"metrics\" paramater needs to be a string (choose only one metric, please)")

            if options["metrics"] not in config["Problem"][self.__problem]["metrics"]:
                sys.exit(
                    "Invalid metric %s for a %s problem. Metrics available:[%s]" % (
                        options["metrics"],
                        self.__problem,
                        ", ".join(config["Problem"][self.__problem]["metrics"])
                    )
                )
            self.__metrics = options["metrics"]
        else:
            self.__metrics = config["Problem"][self.__problem]["metrics"]


        print("\nIt's a %s problem\nSelected algorithms: [%s]\nSelected metrics: [%s]\n" % (
            self.__problem,
            ", ".join(self.__algorithms),
            ", ".join(self.__metrics)
        ))

    def Rainbow(self):
        for algorithm in self.__get_model_algorithms():
            sqrt = True if "sqrt" in algorithm.keys() else False
            self.model.param_tunning_method(
                algorithm["estimator"],
                algorithm["desc"],
                algorithm["params"],
                sqrt
            )
            if 'mse' in self.__metrics and self.get_best_model(False) < 0.01:
                print("Stopping training early, because a good enough result was achieved")
                break
            elif 'accuracy' in self.__metrics and self.get_best_model(False) > 0.95:
                print("Stopping training early, because a good enough result was achieved")
                break


    def __get_model_algorithms(self):
        algorithms = None
        if self.__problem == "Classification":
            classificator = Classification(
                self.input_shape,
                self.__algorithms,
                self.__metrics,
                self.output_classes
            )
            self.model = Model(
                self.X_train, self.X_test, self.y_train, self.y_test,
                classificator.get_metrics(), 1
            )
            algorithms = classificator.get_algorithms()
        else:
            regressor = Regression(
                self.input_shape,
                self.__algorithms,
                self.__metrics
            )
            self.model = Model(
                self.X_train, self.X_test, self.y_train, self.y_test,
                regressor.get_metrics(), regressor.get_metrics_sign()
            )
            algorithms = regressor.get_algorithms()

        return algorithms

    def get_best_model(self, verbose=True):
        if self.model.metric_sign == -1:
            model = sorted(self.model.results, key=lambda x: x["score"], reverse=False)[0]
        else:
            model = sorted(self.model.results, key=lambda x: x["score"], reverse=True)[0]
        if verbose:
            print( "Best model: {0}\t Score: {1}".format(model["name"], model["score"]))
            return model["model"]
        else:
            return model["score"]

    def predict(self, X):
        return self.get_best_model().predict(X)

    def evaluate(self, y, yatt):
        return self.model.metric(y, yatt)
