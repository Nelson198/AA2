import numpy as np

from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from ..neuralnetwork import UnicornHyperModel


class Regression:
    __methods: dict

    def __init__(self, input_shape, algorithms=None, metric=None):
        if metric is None:
            metric = "mse"
        if algorithms is None:
            algorithms = []

        self.__input_shape = input_shape
        self.__get_methods(algorithms)
        self.__get_metric(metric)

    def get_algorithms(self):
        list = []
        for alg in self.__methods:
            list.append(self.__methods[alg]())
        return list

    def get_metric(self):
        return self.__metric

    def get_metric_sign(self):
        return self.__metric_sign

    def Rainbow(self):
        for method in self.__methods:
            self.__methods[method]()

    def __get_methods(self, algorithms):
        available = {
            "linear": self.__linearRegression,
            "svr": self.__SVR,
            "decisionTree": self.__decisionTreeRegression,
            "randomForest": self.__randomForestRegression,
            "neuralNetwork": self.__neuralNetwork
        }
        self.__methods = available.copy()
        if bool(algorithms):
            for alg in available.keys():
                if alg not in algorithms:
                    del self.__methods[alg]

    def __get_metric(self, metric):
        if metric == "r2":
            self.__metric = lambda x, y: 1 - (1 - r2_score(x, y)) * (len(y) - 1) / (len(y) - x.shape[1] - 1)
            self.__metric_sign = 1
        elif metric == "mae":
            self.__metric = lambda x, y: mean_absolute_error(x, y)
            self.__metric_sign = -1
        else:  # metric == "mse" (default metric)
            self.__metric = lambda x, y: mean_squared_error(x, y)
            self.__metric_sign = -1

    @staticmethod
    def __linearRegression():
        return {
            "estimator": LinearRegression(),
            "desc": "Linear Regression",
            "params": {}
        }

    @staticmethod
    def __SVR():
        return {
            "params": {
                "kernel": ["rbf"],
                "gamma": ["scale", "auto"],
                "C": list(range(1, 5)),
                "epsilon": list(np.arange(0, .1, .01))
            },
            "estimator": SVR(),
            "desc": "Support Vector Regression",
        }

    @staticmethod
    def __decisionTreeRegression():
        return {
            "params": {
                "criterion": ["mse", "mae", "friedman_mse"],
                "splitter": ["best"],
                "max_features": ["auto", "sqrt", "log2"]
            },
            "estimator": DecisionTreeRegressor(),
            "desc": "Decision Tree Regression"
        }

    @staticmethod
    def __randomForestRegression():
        return {
            "params": {
                "criterion": ["mse", "mae"],
                "max_features": ["auto", "sqrt", "log2"],
                "n_estimators": list(np.arange(10, 1001, 10))
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
