from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, cross_val_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor


class Regression:
    def __init__(self, x_train, x_test, y_train, y_test):
        self.X_train = x_train
        self.X_test  = x_test
        self.Y_train = y_train
        self.Y_test  = y_test
        self.methods = {
            'linear'      : self.__linearRegression,
            'poly'        : self.__polynomialRegression,
            'svr'         : self.__SVR,
            'decisionTree': self.__decisionTreeRegression,
            'randomForest': self.__randomForestRegression
        }
        self.model = {}

    def Rainbow(self):
        for method in self.methods:
            self.methods[method]()
        return self.model

    def __linearRegression(self):
        print("\nTraining with Linear Regression")
        regressor = LinearRegression()
        regressor.fit(self.X_train, self.Y_train)
        y_pred = regressor.predict(self.X_test)
        r2 = r2_score(self.Y_test, y_pred)
        print("Score: %f" % r2)
        if not bool(self.model) or self.model['score'] < r2:
            self.model['score'] = r2
            self.model['model'] = regressor
    

    def __polynomialRegression(self):
        for degree in range(2,self.X_train.shape[1]):
            print("Training with polynomial Regression (degree: %d)", degree)
            poly_reg = PolynomialFeatures( degree = degree)
            X_poly = poly_reg.fit_transform(self.X_train)
            poly_reg.fit(X_poly, self.Y_train)
    
            regressor = LinearRegression()
            regressor.fit(X_poly, self.Y_train)
    
            y_pred = regressor.predict(
                poly_reg.fit_transform(self.X_test)
            )
    
            r2 = r2_score(self.Y_test, y_pred)
            print("Score: %f" % r2)
            if not bool(self.model) or self.model['score'] < r2:
                self.model['score'] = r2
                self.model['model'] = regressor

    
    def __SVR(self):
        print("Training with Support Vector Regressor")
        regressor = SVR(kernel='rbf', gamma = 'scale')
        regressor.fit( self.X_train, self.Y_train)
        y_pred = regressor.predict(self.X_test)

        r2 = r2_score(self.Y_test, y_pred)
        print("Score: %f" % r2)
        if not bool(self.model) or self.model['score'] < r2:
            self.model['score'] = r2
            self.model['model'] = regressor
        

    def __decisionTreeRegression(self):
        print("training with Decision Tree Regressor")
        regressor = DecisionTreeRegressor()
        regressor.fit( self.X_train, self.Y_train)
        y_pred = regressor.predict(self.X_test)

        r2 = r2_score(self.Y_test, y_pred)
        print("Score: %f" % r2)
        if not bool(self.model) or self.model['score'] < r2:
            self.model['score'] = r2
            self.model['model'] = regressor


    def __randomForestRegression(self):
        print("Training with Random Forest Regressor")
        regressor = RandomForestRegressor(n_estimators=10)
        regressor.fit( self.X_train, self.Y_train)
        y_pred = regressor.predict(self.X_test)

        r2 = r2_score(self.Y_test, y_pred)
        print("Score: %f" % r2)
        if not bool(self.model) or self.model['score'] < r2:
            self.model['score'] = r2
            self.model['model'] = regressor

        

    def __cross_validation(self, model):
        scores = cross_val_score(
            regressor,
            self.X_train,
            self.Y_train,
            scoring = 'r2',
            cv = 5
        )
        y_pred = regressor.predict(X_test) 
        # é necessário comparar o valor dos scores com o do r2 
        # para saber se existe overfitting
        r2 = r2_score(self.Y_test, y_pred)
        return r2,regressor

    
    def __param_tunning(self, model, params):
        rsearch = RandomizedSearchCV(
            estimater = model,
            param_distributions=params,
            n_jobs=-1 #uses all available processors
        )
        rsearch.fit

        
