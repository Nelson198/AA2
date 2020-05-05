import sys
import yaml
from .regression     import Regression
from .classification import Classification

class UnicornML:
    __problem    : str
    __algorithms : list
    __metrics    : list

    def __init__(self, X_train, X_test, y_train, y_test, options = {}):
        config = None
        with open("options.yaml") as file:
            config = yaml.full_load(file)

        if "Problem" in options:
            if options["Problem"] not in config["Problem"]:
                sys.exit("Invalid problem defined! Just accepting [%s]" % " ,".join(config["Problem"]))
            self.__problem = options["Problem"]
        else:
            # Check if it's a classification or regression problem
            self.__problem = self.__detect_problem(y_train.tolist())

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
                            " ,".join(config["Problem"][self.__problem]["algorithms"])
                        )
                    )
            self.__algorithms = options["algorithms"]
        else:
            self.__algorithms = config["Problem"][self.__problem]["algorithms"]

        if "metrics" in options:
            if not isinstance(options["metrics"], list):
                sys.exit("The \"metrics\" paramater needs to be a list")

            for metric in options["metrics"]:
                if not isinstance(metric, str):
                    sys.exit("The metric need to be a string")
                if metric not in config["Problem"][self.__problem]["metrics"]:
                    sys.exit(
                        "Invalid metric %s for a %s problem. Metrics available:[%s]" % (
                            metric,
                            self.__problem,
                            " ,".join(config["Problem"][self.__problem]["metrics"])
                        )
                    )
            self.__metrics = options["metrics"]
        else:
            self.__metrics = config["Problem"][self.__problem]["metrics"]

        self.X_train = X_train
        self.X_test  = X_test
        self.y_train = y_train
        self.y_test  = y_test

        print("It's a %s problem\nSelected algorithms: [%s]\nSelected metrics: [%s]" % (
            self.__problem,
            " ,".join(self.__algorithms),
            " ,".join(self.__metrics)
        ))

    def Rainbow(self):
        if self.__problem == "Classification":
            classificator = Classification(
                self.X_train, self.X_test,
                self.y_train, self.y_test,
                self.__algorithms,
                self.__metrics
            )
            classificator.Rainbow()
        else:
            regressor = Regression(
                self.X_train, self.X_test,
                self.y_train, self.y_test,
                self.__algorithms,
                self.__metrics
            )
            regressor.Rainbow()

    def __detect_problem(self, y):
        # Just ints -> Classification
        if all([isinstance(v, int) for v in y]):
            return "Classification"
        # If they are floats -> Regression
        else:
            return "Regression"
