import numpy as np

from sklearn.metrics         import r2_score, mean_squared_error

from sklearn.linear_model    import LinearRegression
from sklearn.preprocessing   import PolynomialFeatures
from sklearn.svm             import SVR
from sklearn.tree            import DecisionTreeRegressor
from sklearn.ensemble        import RandomForestRegressor

from unicornml.model         import Model

class Regression:
    __methods : dict
    __metrics : dict

    def __init__(self, X_train, X_test, y_train, y_test, algorithms = [], metrics = []):
        self.__get_methods(algorithms)
        self.__get_metrics(metrics)
        # TODO : Choose the apropriate metric
        self.big_model = lambda x_train, x_test: Model(
            x_train, x_test, y_train, y_test, (lambda x,y : r2_score(x, y))
        )
        self.data = (X_train, X_test, y_train, y_test)

    def Rainbow(self):
        for method in self.__methods:
            self.__methods[method]()

    def __get_methods(self, algorithms):
        available = {
            "linear"       : self.__linearRegression,
            "poly"         : self.__polynomialRegression,
            "svr"          : self.__SVR,
            "decisionTree" : self.__decisionTreeRegression,
            "randomForest" : self.__randomForestRegression
        }
        self.__methods = available.copy()
        if bool(algorithms):
            for alg in available.keys():
                if alg not in algorithms:
                    del self.__methods[alg]

    def __get_metrics(self, metrics):
        available = {
            "r2"  : lambda x,y : r2_score(x, y),
            "mse" : lambda x,y : mean_squared_error(x, y)
        }
        self.__metrics = available.copy()
        if bool(metrics):
            for metric in available.keys():
                if metric not in metrics:
                    del self.__methods[metric]

    def __linearRegression(self):
        (X_train, X_test, _, _) = self.data
        self.big_model(X_train, X_test) \
            .param_tunning_method(
                LinearRegression(),
                "Linear Regression"
            )

    def __polynomialRegression(self):
        (X_train, X_test, y_train, _) = self.data
        for degree in range(2, X_train.shape[1]):
            # Preprocessing
            poly_reg = PolynomialFeatures(degree = degree)
            X_poly = poly_reg.fit_transform(X_train)
            poly_reg.fit(X_poly, y_train)

            self.big_model(X_poly, poly_reg.fit_transform(X_test)) \
                .param_tunning_method(
                    LinearRegression(),
                    "Polynomial Regression (degree: {0})".format(degree)
                )
    
    def __SVR(self):
        (X_train, X_test, _, _) = self.data
        params = {
            "kernel"  : ["rbf"], # o melhor kernel é o rbf,
            "gamma"   : ["scale", "auto"],
            "C"       : list(range(1, 5)),
            "epsilon" : list(np.arange(0, .1, .01)) # rever, pode não ser necessário
        }
        self.big_model(X_train, X_test) \
            .param_tunning_method(
                SVR(),
                "Support Vector Regression",
                params
            )

    def __decisionTreeRegression(self):
        (X_train, X_test, _, _) = self.data
        params = {
            "criterion"    : ["mse", "mae", "friedman_mse"],
            "splitter"     : ["best"],
            "max_features" : ["auto", "sqrt", "log2"]
        }
        self.big_model(X_train, X_test) \
            .param_tunning_method(
                DecisionTreeRegressor(),
                "Decision Tree Regression",
                params
            )

    def __randomForestRegression(self):
        (X_train, X_test, _, _) = self.data
        params = {
            "criterion"    : ["mse", "mae"],
            "max_features" : ["auto", "sqrt", "log2"],
            "n_estimators" : list(np.arange(10, 1001, 10))
        }

        self.big_model(X_train, X_test) \
            .param_tunning_method(
                RandomForestRegressor(),
                "Random Forest Regression",
                params,
                True
            )
