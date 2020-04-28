import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from numpy import arange

class Regression:
    def __init__(self, x_train, x_test, y_train, y_test):
        self.X_train = x_train
        self.X_test  = x_test
        self.Y_train = y_train
        self.Y_test  = y_test
        self.methods = {
            "linear"       : self.__linearRegression,
            "poly"         : self.__polynomialRegression,
            "svr"          : self.__SVR,
            "decisionTree" : self.__decisionTreeRegression,
            "randomForest" : self.__randomForestRegression
        }
        self.model = {}

    def Rainbow(self):
        for method in self.methods:
            self.methods[method]()
        return self.model

    def __linearRegression(self):
        print("Training with Linear Regression")
        regressor = LinearRegression()
        regressor.fit(self.X_train, self.Y_train)
        y_pred = regressor.predict(self.X_test)
        r2 = r2_score(self.Y_test, y_pred)
        print("Score: %f" % r2)
        if not bool(self.model) or self.model["score"] < r2:
            self.model["score"] = r2
            self.model["model"] = regressor
    

    def __polynomialRegression(self):
        for degree in range(2, self.X_train.shape[1]):
            print("Training with polynomial Regression (degree: %d)", degree)
            poly_reg = PolynomialFeatures(degree = degree)
            X_poly = poly_reg.fit_transform(self.X_train)
            poly_reg.fit(X_poly, self.Y_train)
    
            regressor = LinearRegression()
            regressor.fit(X_poly, self.Y_train)
    
            y_pred = regressor.predict(
                poly_reg.fit_transform(self.X_test)
            )
    
            r2 = r2_score(self.Y_test, y_pred)
            print("Score: %f" % r2)
            if not bool(self.model) or self.model["score"] < r2:
                self.model["score"] = r2
                self.model["model"] = regressor

    
    def __SVR(self):
        print("Training with Support Vector Regressor")
        params = {
            'kernel' : ['rbf'], #o melhor kernel é o rbf,
            'gamma'  : ['scale', 'auto'],
            'C'      : list(range(1, 5)),
            'epsilon': list(np.arange(0, .1, .01))
        }

        search = self.__param_tunning(
            SVR(),
            params   = params
        )
        print("The best params found: " + str(search.best_params_))
        #TODO we can use this score >>>>print(search.best_score_)<<<< to check if there's any overfitting

        y_pred = search.predict(self.X_test)
        r2 = r2_score(self.Y_test, y_pred)
        print("Score: %f" % r2)
        if not bool(self.model) or self.model["score"] < r2:
            self.model["score"] = r2
           # self.model["model"] = regressor#TODO how should we return the model
        

    def __decisionTreeRegression(self):
        print("training with Decision Tree Regressor")
        params = {
            'criterion': ['mse', 'mae', 'friedman_mse'],
            'splitter': ['best'],
            'max_features': ['auto', 'sqrt', 'log2'],
        }

        search = self.__param_tunning(
            DecisionTreeRegressor(),
            params   = params,
        )


        print("The best params found: " + str(search.best_params_))

        y_pred = search.predict(self.X_test)
        r2 = r2_score(self.Y_test, y_pred)
        print("Score: %f" % r2)
        if not bool(self.model) or self.model["score"] < r2:
            self.model["score"] = r2
            #self.model["model"] = regressor


    def __randomForestRegression(self):
        print("Training with Random Forest Regressor")
        params={
                'criterion'   : ['mse', 'mae'],
                'max_features': ['auto', 'sqrt', 'log2'],
                'n_estimators': list(arange(10,1001,10))
        }

        search = self.__param_tunning(
            RandomForestRegressor(),
            params   = params,
            sqrt = True
        )
        print("The best params found: " + str(search.best_params_))

        y_pred = search.predict(self.X_test)
        r2 = r2_score(self.Y_test, y_pred)
        print("Score: %f" % r2)
        if not bool(self.model) or self.model["score"] < r2:
            self.model["score"] = r2
        #    self.model["model"] = regressor
        

    def __param_tunning(self, model, params, sqrt = False):
        n_space = np.prod([ len(params[x]) for x in params.keys()])
        if sqrt: n_space = np.sqrt(n_space)
        randomized = RandomizedSearchCV (
            estimator = model,
            param_distributions=params,
            random_state = 0,
            cv = 5, #TODO not sure how we can choose the best
#            n_jobs=-1 #uses all available processors #TODO this is killing
            n_iter=n_space #TODO this should be dynamic, based on the number of features
        )
        return randomized.fit( self.X_train, self.Y_train)

