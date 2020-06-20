import numpy as np

from sklearn.metrics       import r2_score, mean_squared_error, mean_absolute_error
from sklearn.linear_model  import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm           import SVR
from sklearn.tree          import DecisionTreeRegressor
from sklearn.ensemble      import RandomForestRegressor
from ..neuralnetwork       import UnicornHyperModel

class Regression:
    __methods : dict

    def __init__(self, input_shape, algorithms = [], metrics = []):
        self.__input_shape = input_shape
        self.__get_methods(algorithms)
        self.__get_metrics(metrics)

    def get_algorithms(self):
        list = []
        for alg in self.__methods:
            list.append(self.__methods[alg]())
        return list

    def get_metrics(self):
        return self.__metrics

    def get_metrics_sign(self):
        return self.__metrics_sign

    def Rainbow(self):
        for method in self.__methods:
            self.__methods[method]()

    def __get_methods(self, algorithms):
        available = {
            "linear"        : self.__linearRegression,
            "svr"           : self.__SVR,
            "decisionTree"  : self.__decisionTreeRegression,
            "randomForest"  : self.__randomForestRegression,
            "neuralNetwork" : self.__neuralNetwork
        }
        self.__methods = available.copy()
        if bool(algorithms):
            for alg in available.keys():
                if alg not in algorithms:
                    del self.__methods[alg]

    def __get_metrics(self, metrics):
        if metrics[0] == "r2":
            self.__metrics = lambda x,y : 1 - (1 - r2_score(x, y)) * (len(y) - 1) / (len(y) - x.shape[1] - 1)
            self.__metrics_sign = 1
        elif metrics[0] == "mae":
            self.__metrics = lambda x,y : mean_absolute_error(x,y)
            self.__metrics_sign = -1
        else: # metrics == "mse" (default metric)
            self.__metrics = lambda x,y : mean_squared_error(x, y)
            self.__metrics_sign = -1

    def __linearRegression(self):
        return {
            "estimator": LinearRegression(),
            "desc": "Linear Regression",
            "params": {}
        }
    
    def __SVR(self):
        return {
            "params": {
                "kernel"  : ["rbf"], # melhor kernel
                "gamma"   : ["scale", "auto"],
                "C"       : list(range(1, 5)),
                "epsilon" : list(np.arange(0, .1, .01))
            },
            "estimator": SVR(),
            "desc": "Support Vector Regression",
        }

    def __decisionTreeRegression(self):
        return {
            "params": {
                "criterion"    : ["mse", "mae", "friedman_mse"],
                "splitter"     : ["best"],
                "max_features" : ["auto", "sqrt", "log2"]
            },
            "estimator": DecisionTreeRegressor(),
            "desc": "Decision Tree Regression"
        }

    def __randomForestRegression(self):
        return {
            "params": {
                "criterion"    : ["mse", "mae"],
                "max_features" : ["auto", "sqrt", "log2"],
                "n_estimators" : list(np.arange(10, 1001, 10))
            },
            "estimator": RandomForestRegressor(),
            "desc": "Random Forest Regression",
            "sqrt": True
        }

    def __neuralNetwork(self):
        return {
            "estimator": UnicornHyperModel(self.__input_shape, 1, "regression"),
            "desc": "Neural Networks",
            "params": {}
        }