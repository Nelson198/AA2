import numpy as np

from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model    import LogisticRegression
from sklearn.neighbors       import KNeighborsClassifier
from sklearn.svm             import SVC
from sklearn.naive_bayes     import *
from sklearn.tree            import DecisionTreeClassifier
from sklearn.ensemble        import RandomForestClassifier

# import kerastuner

class Classification:
    def __init__(self, x_train, x_test, y_train, y_test):
        self.X_train = x_train
        self.X_test  = x_test
        self.Y_train = y_train
        self.Y_test  = y_test
        self.methods = {
            "logistic"      : self.__logisticRegression,
            "knn"           : self.__KNN,
            "svm"           : self.__SVM,
            "kernelSVM"     : self.__kernelSVM,
            "naiveBayes"    : self.__naiveBayes,
            "decisionTree"  : self.__decisonTree,
            "randomForest"  : self.__randomForest,
            "neuralNetwork" : self.__neuralNetwork
        }
        self.model = {}

    def Rainbow(self):
        for method in self.methods:
            self.methods[method]()
        return self.model

    def __param_tunning(self, model, params, sqrt = False):
        n_space = np.prod([ len(params[x]) for x in params.keys()])
        if sqrt:
            n_space = np.sqrt(n_space)
        randomized = RandomizedSearchCV (
            estimator = model,
            param_distributions = params,
            random_state = 0,
            cv = 5, #TODO not sure how we can choose the best
            n_jobs = -1, #uses all available processors #TODO this is killing
            n_iter = n_space #TODO this should be dynamic, based on the number of features
        )
        return randomized.fit( self.X_train, self.Y_train)

    #TODO ACABAR !!!
    def __logisticRegression(self):
        print("Training with Logistic Regression")
        classifier = LogisticRegression()
        classifier.fit(self.X_train, self.Y_train)
        classifier.predict(self.X_test)
        score = classifier.score(self.X_test, self.Y_test)
        print("Score: {0}".format(score))
        if not bool(self.model) or self.model["score"] < score:
            self.model["score"] = score
            self.model["model"] = classifier

    def __KNN(self):
        print("Training with K-Nearest Neighbors (KNN)")

        params = {
            "n_neighbors" : list(range(1, 21)), # default = 5
            "leaf_size"   : list(range(10, 51, 10)), # default = 30
            "p"           : [1, 2], # default = 2
            "weights"     : ["uniform", "distance"] # default = "uniform"
        }
        
        knn = self.__param_tunning(
            KNeighborsClassifier(),
            params = params
        )
        
        print("The best params found: " + str(knn.best_params_))

        knn.predict(self.X_test)
        score = knn.score(self.X_test, self.Y_test)
        print("Score: {0}".format(score))
        if not bool(self.model) or self.model["score"] < score:
            self.model["score"] = score
            self.model["model"] = knn

    def __SVM(self):
        print("Training with Support Vector Machine (SVM)")

    def __kernelSVM(self):
        print("Training with kernel Support Vector Machine")

    def __naiveBayes(self):
        print("Training with Naive Bayes")

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

    def __neuralNetwork(self):
        print("Training with Neural Network")
