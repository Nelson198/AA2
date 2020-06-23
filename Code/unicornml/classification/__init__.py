import numpy as np

from sklearn.metrics import accuracy_score, auc, precision_score, recall_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from ..neuralnetwork import UnicornHyperModel


class Classification:
    __methods: dict

    def __init__(self, input_shape, algorithms=None, metric=None, output_units=2):
        if metric is None:
            metric = "accuracy"
        if algorithms is None:
            algorithms = []

        self.__get_methods(algorithms)
        self.__get_metric(metric)
        self.__output_units = output_units
        self.__input_shape = input_shape

    def get_algorithms(self):
        list = []
        for alg in self.__methods:
            list.append(self.__methods[alg]())
        return list

    def get_metric(self):
        return self.__metric

    def __get_methods(self, algorithms):
        available = {
            "logistic": self.__logisticRegression,
            "knn": self.__KNN,
            "svm": self.__SVM,
            "kernelSVM": self.__kernelSVM,
            "gaussianNB": self.__gaussianNB,
            "bernoulliNB": self.__bernoulliNB,
            "decisionTree": self.__decisonTreeClassification,
            "randomForest": self.__randomForestClassification,
            "neuralNetwork": self.__neuralNetwork
        }
        self.__methods = available.copy()
        if bool(algorithms):
            for alg in available.keys():
                if alg not in algorithms:
                    del self.__methods[alg]

    def __get_metric(self, metric):
        if metric == "recall":
            self.__metric = lambda x, y: recall_score(x, y)
        elif metric == "auc":
            self.__metric = lambda x, y: auc(x, y),
        elif metric == "precision":
            self.__metric = lambda x, y: precision_score(x, y),
        else:  # metric == "accuracy" (default metric)
            self.__metric = lambda x, y: accuracy_score(x, y)

    def __neuralNetwork(self):
        return {
            "estimator": UnicornHyperModel(self.__input_shape, self.__output_units, "classification"),
            "desc": "Neural Networks",
            "params": {}
        }

    @staticmethod
    def __logisticRegression():
        return {
            "params": {
                "solver": ["newton-cg", "sag", "lbfgs"],
                "C": list(np.arange(1, 5)),
                "multi_class": ["auto"]
            },
            "estimator": LogisticRegression(),
            "desc": "Logistic Regression"
        }

    @staticmethod
    def __KNN():
        return {
            "params": {
                "n_neighbors": list(np.arange(1, 21)),  # default = 5
                "leaf_size": list(np.arange(10, 51, 10)),  # default = 30
                "p": [1, 2],  # default = 2
                "weights": ["uniform", "distance"],  # default = "uniform"
                "algorithm": ["auto"]
            },
            "estimator": KNeighborsClassifier(),
            "desc": "K-Nearest Neighbors (KNN)"
        }

    @staticmethod
    def __SVM():
        return {
            "params": {
                "dual": [False],
                "penalty": ["l1", "l2"],
                "C": list(np.arange(1, 5))
            },
            "estimator": LinearSVC(),
            "desc": "Support Vector Machine (SVM)"
        }

    @staticmethod
    def __kernelSVM():
        return {
            "params": {
                "kernel": ["rbf", "sigmoid"],
                "gamma": ["scale", "auto"],
                "C": list(np.arange(1, 5))
            },
            "estimator": SVC(),
            "desc": "kernel Support Vector Machine (kernels rbf and sigmoid)"
        }

    @staticmethod
    def __gaussianNB():
        return {
            "params": {
                "var_smoothing": [1.e-09, 1.e-08, 1.e-07, 1.e-06]
            },
            "estimator": GaussianNB(),
            "desc": "Gaussian Naive Bayes"
        }

    @staticmethod
    def __bernoulliNB():
        return {
            "params": {
                "alpha": [1.0, 0.5, 1.0e-10],
                "fit_prior": [True, False]
            },
            "estimator": BernoulliNB(),
            "desc": "Bernoulli Naive Bayes"
        }

    @staticmethod
    def __decisonTreeClassification():
        return {
            "params": {
                "criterion": ["gini", "entropy"],
                "max_features": [None, "sqrt", "log2"]
            },
            "estimator": DecisionTreeClassifier(),
            "desc": "Decison Tree Classification"
        }

    @staticmethod
    def __randomForestClassification():
        return {
            "params": {
                "criterion": ["gini", "entropy"],
                "max_features": ["sqrt", None, "log2"],
                "n_estimators": list(np.arange(50, 751, 10))
            },
            "estimator": RandomForestClassifier(),
            "desc": "Random Forest Classification",
            "sqrt": True
        }
