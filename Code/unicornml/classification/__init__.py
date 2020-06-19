import numpy as np

from sklearn.metrics         import accuracy_score, auc, precision_score, recall_score

from sklearn.linear_model    import LogisticRegression
from sklearn.neighbors       import KNeighborsClassifier
from sklearn.svm             import LinearSVC, SVC
from sklearn.naive_bayes     import GaussianNB, MultinomialNB, BernoulliNB, ComplementNB
from sklearn.tree            import DecisionTreeClassifier
from sklearn.ensemble        import RandomForestClassifier
from .neuralnetwork          import UnicornHyperModel

class Classification:
    __methods : dict

    def __init__(self, input_shape, algorithms = [], metrics = [], output_units=2):
        self.__get_methods(algorithms)
        self.__get_metrics(metrics)
        self.__output_units = output_units
        self.__input_shape = input_shape

    def get_algorithms(self):
        list = []
        for alg in self.__methods:
            list.append(self.__methods[alg]())
        return list

    def get_metrics(self):
        return self.__metrics

    def __get_methods(self, algorithms):
        available = {
            "logistic"      : self.__logisticRegression,
            "knn"           : self.__KNN,
            "svm"           : self.__SVM,
            "kernelSVM"     : self.__kernelSVM,
            "gaussianNB"    : self.__gaussianNB,
            "bernoulliNB"   : self.__bernoulliNB,
            "decisionTree"  : self.__decisonTreeClassification,
            "randomForest"  : self.__randomForestClassification
             "neuralNetwork" : self.__neuralNetwork
        }
        self.__methods = available.copy()
        if bool(algorithms):
            for alg in available.keys():
                if alg not in algorithms:
                    del self.__methods[alg]

    def __get_metrics(self, metrics):
        if metrics == "recall":
            self.__metrics = lambda x,y : recall_score(x, y)
        elif metrics == "auc":
            self.__metrics = lambda x,y : auc(x, y),
        elif metrics == "precision":
            self.__metrics = lambda x,y : precision_score(x, y),
        else: # metrics == "accuracy" (default metric)
            self.__metrics = lambda x,y : accuracy_score(x,y)

    def __neuralNetwork(self):
        return {
            "estimator": UnicornHyperModel(self.__input_shape, self.__output_units, "classification"),
            "desc": "Neural Networks"
        }



    def __logisticRegression(self):
        return {
            "params": {
                "solver" : ["newton-cg", "sag", "lbfgs"],
                "C"      : list(np.arange(1, 5))
            },
            "estimator": LogisticRegression(),
            "desc":"Logistic Regression with newton-cg, sag and lbfgs"
        }

        #list.append({
        #    "params": {
        #        "solver"   : ["saga"],
        #        "C"        : list(np.arange(1,5)),
        #        "penalty"  : ["elasticnet"],
        #        "l1_ratio" : list(np.arange(0, 1.1, 0.2))
        #    },
        #    "estimator":            LogisticRegression(),
        #    "desc":   "Logistic Regression with saga solver"
        #})

        #list.append({
        #    "params": {
        #        "solver"  : ["saga", "newton-cg", "sag", "lbfgs"],
        #        "penalty" : ["none"]
        #    },
        #    "estimator":    LogisticRegression(),
        #    "desc":    "Logistic Regression with no penalty",
        #})
        #return list

    def __KNN(self):
        return {
            "params": {
                "n_neighbors": list(np.arange(1, 21)),  # default = 5
                "leaf_size": list(np.arange(10, 51, 10)),  # default = 30
                "p": [1, 2],  # default = 2
                "weights": ["uniform", "distance"],  # default = "uniform"
                "algorithm": ["auto"]
            },
            "estimator": KNeighborsClassifier(),
            "desc":"K-Nearest Neighbors (KNN)"
        }

    def __SVM(self):
        return {
            "params": {
                "dual"    : [False],
                "penalty" : ["l1", "l2"],
                "C"       : list(np.arange(1, 5))
            },
            "estimator": LinearSVC(),
            "desc": "Support Vector Machine (SVM)"
        }

    def __kernelSVM(self):
        return {
            "params": {
                "kernel" : ["rbf", "sigmoid"],
                "gamma"  : ["scale", "auto"], # [0.1, 1, 10, 100], better but takes much much longer
                "C"      : list(np.arange(1, 5))
            },
            "estimator": SVC(),
            "desc": "kernel Support Vector Machine (kernels rbf and sigmoid)"
        }


    def __gaussianNB(self):
        return {
            "params": {
                "var_smoothing" : [1.e-09, 1.e-08, 1.e-07, 1.e-06]
            },
            "estimator": GaussianNB(),
            "desc": "Gaussian Naive Bayes"
        }
        
    def __bernoulliNB(self):
        return {
            "params": {
                "alpha"     : [1.0, 0.5, 1.0e-10],
                "fit_prior" : [True, False]
            },
            "estimator": BernoulliNB(),
            "desc": "Bernoulli Naive Bayes"
        }

    
    def __decisonTreeClassification(self):
        return {
            "params": {
                "criterion"    : ["gini", "entropy"],
                "max_features" : [None, "sqrt", "log2"]
            },
            "estimator": DecisionTreeClassifier(),
            "desc": "Decison Tree Classification"
        }

    def __randomForestClassification(self):
        return {
            "params": {
                "criterion"    : ["gini", "entropy"],
                "max_features" : ["sqrt", None, "log2"],
                "n_estimators" : list(np.arange(50, 751, 10))
            },
            "estimator": RandomForestClassifier(),
            "desc": "Random Forest Classification",
            "sqrt": True
        }
