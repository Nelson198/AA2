from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, cross_val_score


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
        self.model = {}

    def Rainbow(self):
        for method in self.methods:
            self.methods[method]()
        return self.model

    def __linearRegression(self):
        r2,regressor = self.__cross_validation(LinearRegression())
        if not bool(self.model) or self.model['score'] < r2:
            self.model['score'] = r2
            self.model['model'] = regressor
    

    def __polynomialRegression(self):
        return 1

    def __cross_validation(self, model):
        scores = cross_val_score(
            regressor,
            self.X_train,
            self.y_train,
            scoring = 'r2',
            cv = 5
        )
        y_pred = regressor.predict(X_test) 
        # é necessário comparar o valor dos scores com o do r2 
        # para saber se existe overfitting
        r2 = r2_score(self.y_test, y_pred)
        return r2,regressor

        
