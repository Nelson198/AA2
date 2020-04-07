from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression

class Regression:
    def __init__(self, x_train, x_test, y_train, y_test):
        self.x_train = x_train
        self.x_test  = x_test
        self.y_train = y_train
        self.y_test  = y_test
        self.methods = {
            'linear'      : self.__linearRegression,
            'poly'        : self.__polynomialRegression,
            #'svr'         : self.__SVR,
            #'decisionTree': self.__decisionTreeRegression,
            #'randomForest': self.__randomForestRegression
        }

    def Rainbow(self):
        for method in self.methods:
            print(method)
            self.methods[method]()
        return 1

    def __linearRegression(self):
        print("inside linear Reg")
        return 1

    def __polynomialRegression(self):
        return 1


        
