import numpy as np

from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model    import LogisticRegression
from sklearn.neighbors       import KNeighborsClassifier
from sklearn.svm             import LinearSVC, SVC
from sklearn.naive_bayes     import GaussianNB, MultinomialNB, BernoulliNB, ComplementNB
from sklearn.tree            import DecisionTreeClassifier
from sklearn.ensemble        import RandomForestClassifier
from sklearn.metrics         import accuracy_score
from unicornml.model         import Model

# import kerastuner

class Classification:
    def __init__(self, X_train, X_test, y_train, y_test):
        self.methods = {
            "logistic"     : self.__logisticRegression,
            "knn"          : self.__KNN,
            "svm"          : self.__SVM,
            "kernelSVM"    : self.__kernelSVM,
            "naiveBayes"   : self.__naiveBayes,
            "decisionTree" : self.__decisonTreeClassification,
            "randomForest" : self.__randomForestClassification,
            #"neuralNetwork" : self.__neuralNetwork
        }
        self.model = {},
        self.big_model = Model(
            X_train, X_test, y_train, y_test, (lambda x,y: accuracy_score(x,y))
        )

    def Rainbow(self):
        for method in self.methods:
            self.methods[method]()
        return self.model


    def __logisticRegression(self):
        params = {
            "solver"  : ["newton-cg", "sag", "lbfgs"],
            "C"       : list(np.arange(1, 5))
        }
        
        self.big_model.param_tunning_method(
            LogisticRegression(),
            "Logistic Regression with newton-cg, sag and lbfgs",
            params
        )

        params = {
            "solver"   : ["saga"],
            "C"        : list(np.arange(1,5)),
            "penalty"  : ["elasticnet"],
            "l1_ratio" : list(np.arange(0, 1.1, 0.2))
        }

        self.big_model.param_tunning_method(
            LogisticRegression(),
            "Logistic Regression with saga solver",
            params
        )

        params = {
            "solver"  : ["saga", "newton-cg", "sag", "lbfgs"],
            "penalty" : ["none"]
        }

        self.big_model.param_tunning_method(
            LogisticRegression(),
            "Logistic Regression with no penalty",
            params
        )


    def __KNN(self):
        params = {
            "n_neighbors" : list(np.arange(1, 21)), # default = 5
            "leaf_size"   : list(np.arange(10, 51, 10)), # default = 30
            "p"           : [1, 2], # default = 2
            "weights"     : ["uniform", "distance"], # default = "uniform"
            "algorithm"   : ["auto"]
        }
        self.big_model.param_tunning_method(
            KNeighborsClassifier(),
            "K-Nearest Neighbors (KNN)",
            params
        )


    def __SVM(self):
        params = {
            "dual"    : [False],
            "penalty" : ["l1", "l2"],
            "C"       : list(np.arange(1, 5))
        }
        self.big_model.param_tunning_method(
            LinearSVC(),
            "Support Vector Machine (SVM)",
            params
        )

    def __kernelSVM(self):
        params = {
            "kernel"  : ["rbf", "sigmoid"],
            "gamma"   : ["scale", "auto"], # [0.1, 1, 10, 100], better but takes much much longer
            "C"       : list(np.arange(1, 5))
        }
        self.big_model.param_tunning_method(
            SVC(),
            "kernel Support Vector Machine (kernels rbf and sigmoid)",
            params
        )

        params = {
            "kernel"  : ["poly"],
            "gamma"   : ["scale", "auto"], # [0.1, 1, 10, 100], better but takes much much longer
            "degree"  : list(np.arange(2, 5)),
            "C"       : list(np.arange(1, 5))
        }
        self.big_model.param_tunning_method(
            SVC(),
            "kernel Support Vector Machine (kernel poly)",
            params
        )

    def __naiveBayes(self):
        params = {
            "alpha" : [1.0, 0.5, 0.0],
            "fit_prior" : [True, False]
        }

        self.Gaussian()
        #self.Multinomial(params) # rever dados de input
        self.Bernoulli(params)

        params.update({ "norm" : [True, False] })
        #self.Complement(params) # rever dados de input

    
    def __decisonTreeClassification(self):
        params = {
            "criterion"    : ["gini", "entropy"],
            "max_features" : [None, "sqrt", "log2"]
        }

        self.big_model.param_tunning_method(
            DecisionTreeClassifier(),
            "Decison Tree Classification",
            params
        )

    def __randomForestClassification(self):
        params = {
            "criterion"    : ["gini", "entropy"],
            "max_features" : ["sqrt", None, "log2"],
            "n_estimators" : list(np.arange(50, 751, 10))
        }

        self.big_model.param_tunning_method(
            RandomForestClassifier(),
            "Random Forest Classification",
            params,
            True
        )

    def __neuralNetwork(self):
        print("Training with Neural Network")


    ################### Naive Bayes Classifiers Functions ###################
    
    def Gaussian(self):
        params = {
            "var_smoothing" : [1.e-09, 1.e-08, 1.e-07, 1.e-06]
        }

        return self.big_model.param_tunning_method(
            GaussianNB(),
            "Gaussian Naive Bayes",
            params
        )

    def Multinomial(self, params):
        return self.big_model.param_tunning_method(
            MultinomialNB(),
            "Multinomial Naive Bayes",
            params
        )

    def Complement(self, params):
        return self.big_model.param_tunning_method(
            ComplementNB(),
            "Complement Naive Bayes",
            params
        )

    def Bernoulli(self, params):
        return self.big_model.param_tunning_method(
            BernoulliNB(),
            "Bernoulli Naive Bayes",
            params
        )