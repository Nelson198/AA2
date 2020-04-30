import numpy as np

from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model    import LogisticRegression
from sklearn.neighbors       import KNeighborsClassifier
from sklearn.svm             import LinearSVC, SVC
from sklearn.naive_bayes     import GaussianNB, MultinomialNB
from sklearn.tree            import DecisionTreeClassifier
from sklearn.ensemble        import RandomForestClassifier
from sklearn.metrics         import accuracy_score
from unicornml.model         import Model

# import kerastuner

class Classification:
    def __init__(self, X_train, X_test, y_train, y_test):
        self.methods = {
            #"logistic"      : self.__logisticRegression,
            "knn"           : self.__KNN,
            #"svm"           : self.__SVM,
            #"kernelSVM"     : self.__kernelSVM,
            #"naiveBayes"    : self.__naiveBayes,
            #"decisionTree"  : self.__decisonTree,
            #"randomForest"  : self.__randomForest,
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

    #TODO ACABAR !!!
    def __logisticRegression(self):
        print("Training with Logistic Regression")
        classifier = LogisticRegression(solver = "lbfgs")
        classifier.fit(self.X_train, self.Y_train)
        classifier.predict(self.X_test)
        score = classifier.score(self.X_test, self.Y_test)
        print("Score: {0}".format(score))
        if not bool(self.model) or self.model["score"] < score:
            self.model["score"] = score
            self.model["model"] = classifier

    def __KNN(self):
        params = {
            "n_neighbors" : list(np.arange(1, 21)), # default = 5
            "leaf_size"   : list(np.arange(10, 51, 10)), # default = 30
            "p"           : [1, 2], # default = 2
            "weights"     : ["uniform", "distance"] # default = "uniform"
        }

        self.big_model.param_tunning_method(
            KNeighborsClassifier(),
            "K-Nearest Neighbors (KNN)",
            params
        )

    def __SVM(self):
        print("Training with Support Vector Machine (SVM)")

        params = {
            "dual"    : ["primal", "dual"],
            "loss"    : ["hinge", "squared_hinge"],
            "C"       : list(range(1, 5))
        }

        svm = self.__param_tunning(
            LinearSVC(),
            params = params
        )
        
        print("The best params found: " + str(svm.best_params_))

        y_pred = svm.predict(self.X_test)
        score = svm.score(self.X_test, y_pred)
        print("Score: {0}".format(score))
        if not bool(self.model) or self.model["score"] < score:
            self.model["score"] = score
            self.model["model"] = svm


    def __kernelSVM(self):
        print("Training with kernel Support Vector Machine (kernel SVM)")

        params = {
            "kernel"  : ["rbf", "poly", "sigmoid"],
            "gamma"   : ["scale", "auto"], # [0.1,1, 10, 100], better but takes much much longer
            "C"       : list(range(1, 5))
        }

        kernelsvm = self.__param_tunning(
            SVC(),
            params = params
        )
        
        print("The best params found: " + str(kernelsvm.best_params_))

        y_pred = kernelsvm.predict(self.X_test)
        score = kernelsvm.score(self.X_test, y_pred)
        print("Score: {0}".format(score))
        if not bool(self.model) or self.model["score"] < score:
            self.model["score"] = score
            self.model["model"] = kernelsvm


    def __naiveBayes(self):
        print("Training with Naive Bayes")

        models = []
        params = {
            "alpha" : [1.0, 0.5, 0.0],
            "fit_prior" : [True, False]
        }

        models.append(self.Gaussian())
        models.append(self.Multinomial(params))
        models.append(self.Bernoulli(params))

        params.update({ "norm" : [True, False] })
        models.append(self.Complement(params))

        for model in models:
            y_pred = model.predict(self.X_test)
            score = model.score(self.X_test, y_pred)
            print("Score: {0}".format(score))
            if not bool(self.model) or self.model["score"] < score:
                self.model["score"] = score
                self.model["model"] = model

    #TODO ACABAR !!!
    def __decisonTree(self):
        print("Training with Decison Tree Classification")
        tree = DecisionTreeClassifier()
        tree.fit(self.X_train, self.Y_train)
        tree.predict(self.X_test)
        score = tree.score(self.X_test, self.Y_test)
        print("Score: {0}".format(score))
        if not bool(self.model) or self.model["score"] < score:
            self.model["score"] = score
            self.model["model"] = tree

    def __randomForest(self):
        print("Training with Random Forest")

        params = {
            "criterion"    : ["gini", "entropy"],
            "max_features" : ["auto", None, "log2"],
            "n_estimators" : list(np.arange(10, 1001, 10))
        }

        search = self.__param_tunning(
            RandomForestClassifier(),
            params = params,
            sqrt = True
        )
        print("The best params found: " + str(search.best_params_))

        y_pred = search.predict(self.X_test)
        score = search.score(self.Y_test, y_pred)
        print("Score: {0}".format(score))
        if not bool(self.model) or self.model["score"] < score:
            self.model["score"] = score
            # self.model["model"] = regressor

    def __neuralNetwork(self):
        print("Training with Neural Network")


    ################### Naive Bayes Classifiers Functions ###################
    
    def Gaussian(self):
        params = {
            "var_smoothing" : [1.e-09, 1.e-08, 1.e-07, 1.e-06]
        }

        return self.__param_tunning(
            GaussianNB(),
            params = params
        )

    def Multinomial(self, params):
        return self.__param_tunning(
            MultinomialNB(),
            params = params
        )

    def Complement(self, params):
        return self.__param_tunning(
            MultinomialNB(),
            params = params
        )

    def Bernoulli(self, params):
        return self.__param_tunning(
            MultinomialNB(),
            params = params
        )