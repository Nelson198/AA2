import random
import numpy as np

from kerastuner.tuners import Hyperband
from sklearn.model_selection import RandomizedSearchCV


class Model:
    def __init__(
        self, X_train, X_test, y_train, y_test,
        metric, metric_sign, cv
    ):
        self.metric = metric
        self.metric_sign = metric_sign
        self.results = []
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.cv = cv

    def param_tunning_method(self, estimator, desc, params=None, sqrt=False):
        if params is None:
            params = {}

        if not bool(params):
            if desc == "Neural Networks":
                trained_model = self.__train_neural_networks(estimator)
            else:
                trained_model = self.__train_without_optimizer(estimator)
        else:
            trained_model = self.__randomized_search(estimator, params, sqrt)

        y_pred = trained_model.predict(self.X_test)
        if desc == "Neural Networks":
            if estimator.get_output_units() == 2:
                y_pred[y_pred > .5] = 1
                y_pred[y_pred <= .5] = 0
            else:
                y_pred = np.argmax(y_pred, axis=1)

        metric = self.metric(self.y_test, y_pred)

        if hasattr(trained_model, "best_params_"):
            print("The best params found: " + str(trained_model.best_params_))

        print("[%s] Score: %f\n" % (desc, metric))
        self.results.append(
            {
                "name": desc,
                "model": trained_model,
                "score": metric
            }
        )

    def __randomized_search(self, estimator, params, sqrt=False):
        n_space = np.prod([len(params[x]) for x in params.keys()])
        if sqrt:
            n_space = np.sqrt(n_space)

        try:
            randomized = RandomizedSearchCV(
                estimator=estimator,
                param_distributions=params,
                random_state=0,
                cv=self.cv,
                n_jobs=-1,  # uses all available processors
                n_iter=n_space
            )
        except:
            randomized = RandomizedSearchCV(
                estimator=estimator,
                param_distributions=params,
                random_state=0,
                cv=self.cv,
                n_iter=n_space
            )
        return randomized.fit(self.X_train, self.y_train)

    def __train_without_optimizer(self, estimator):
        return estimator.fit(self.X_train, self.y_train)

    def __train_neural_networks(self, estimator):
        if estimator.get_metric() == "mse":
            tuner = Hyperband(
                estimator,
                max_epochs=20,
                objective="val_mse",
                executions_per_trial=1,
                directory="regression_nn" + str(random.randint(0, 1000))
            )
        else:
            tuner = Hyperband(
                estimator,
                max_epochs=20,
                objective="val_accuracy",
                executions_per_trial=1,
                directory="classification_nn" + str(random.randint(0, 1000))
            )
        tuner.search(self.X_train, self.y_train, epochs=1, validation_split=0.1, verbose=0)

        return tuner.get_best_models(num_models=1)[0]
