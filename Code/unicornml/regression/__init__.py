import numpy as np

from sklearn.linear_model    import LinearRegression
from sklearn.metrics         import r2_score
from sklearn.preprocessing   import PolynomialFeatures
from sklearn.svm             import SVR
from sklearn.tree            import DecisionTreeRegressor
from sklearn.ensemble        import RandomForestRegressor
from unicornml.model         import Model

class Regression:
    def __init__(self, X_train, X_test, y_train, y_test):
        self.methods = {
            "linear"       : self.__linearRegression,
            #"poly"         : self.__polynomialRegression,
            "svr"          : self.__SVR,
            "decisionTree" : self.__decisionTreeRegression,
            "randomForest" : self.__randomForestRegression
        }
        self.model = {}
        self.big_model = Model(
            X_train, X_test, y_train, y_test, (lambda x,y: r2_score(x,y))
        )

    def Rainbow(self):
        for method in self.methods:
            self.methods[method]()
        return self.model

    def __linearRegression(self):
        self.big_model.param_tunning_method(
            LinearRegression(),
            "Linear Regression"
        )

    """
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
    """
    
    def __SVR(self):
        params = {
            "kernel"  : ["rbf"], # o melhor kernel é o rbf,
            "gamma"   : ["scale", "auto"],
            "C"       : list(range(1, 5)),
            "epsilon" : list(np.arange(0, .1, .01))
        }
        self.big_model.param_tunning_method(
            SVR(),
            "Support Vector Regressor",
            params
        )

    def __decisionTreeRegression(self):
        params = {
            "criterion"    : ["mse", "mae", "friedman_mse"],
            "splitter"     : ["best"],
            "max_features" : ["auto", "sqrt", "log2"]
        }
        self.big_model.param_tunning_method(
            DecisionTreeRegressor(),
            "Decision Tree Regressor",
            params
        )

    def __randomForestRegression(self):
        params = {
            "criterion"    : ["mse", "mae"],
            "max_features" : ["auto", "sqrt", "log2"],
            "n_estimators" : list(np.arange(10, 1001, 10))
        }

        self.big_model.param_tunning_method(
            RandomForestRegressor(),
            "Random Forest Regressor",
            params,
            True
        )
