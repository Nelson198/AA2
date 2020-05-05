import sys
import yaml
from .regression     import Regression
from .classification import Classification


class UnicornML:
    def __init__(self, X, y, options = {}):

        config = None
        with open('options.yaml') as file:
            config = yaml.full_load(file)
        print(config, type(config))

        self.__params = {}
        if "Problem" in options:
            if options["Problem"] not in config["Problem"]:
                sys.exit("Invalid problem defined! Just accepting [%s]" % ' ,'.join(config["Problem"]))
            self.__params["Problem"] = options["Problem"]
        else:
            # Check if it's a classification or regression problem
            self.__params["Problem"] = self.__detect_problem(y.tolist())
        # it's unsupervised learning
        '''if bool(re.search('^int', str(y_train.dtype))):
            self.model = Classification(
                    x_train, x_test,
                    y_train, y_test
            )
        else:
            self.model = Regression(
                x_train, x_test,
                y_train, y_test
        )  '''

    #TODO improve
    def __detect_problem(self, y):
        # Just ints -> Classification
        if len(y) == len([v for v in y if isinstance(v,int)]):
            return "Classification"
        # If they are floats -> Regression
        else:
            return "Regression"
