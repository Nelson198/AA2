import sys
import numpy as np
from sklearn.model_selection import RandomizedSearchCV

class Model():
    def __init__(
        self, X_train, X_test, y_train, y_test,
        metric, optmization_method = "randomizedSearch",
        save_results = True
    ):
        if optmization_method not in ["randomizedSearch", "Bayes"]:
            sys.exit("Invalid optmization method")
            
        self.method = optmization_method
        self.save_results = save_results
        self.metric = metric
        self.results = []
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

    def param_tunning_method(self, estimator, desc, params = {}, sqrt = False):
        trained_model = None
        if not bool(params):
            trained_model = self.__train_without_optimizer(estimator)
        elif self.method == "randomizedSearch":
            trained_model = self.__randomized_search(estimator, params, sqrt)
        else:
            trained_model = self.__bayes(estimator, params, sqrt)

        y_pred = trained_model.predict(self.X_test)
        metric = self.metric(self.y_test, y_pred)  # this metric should have a sign

        if hasattr(trained_model, "best_params_"):
            print("The best params found: " + str(trained_model.best_params_))

        print("[%s] Score: %f\n" % (desc, metric))
        if bool(self.save_results) or not bool(self.results):
            self.results.append(
                {
                    "name"  : desc,
                    "model" : trained_model,
                    "score" : metric
                }
            )
        elif bool(self.results) and metric > self.results[0]["score"]:
            self.results[0] = {
                "name"  : desc,
                "model" : trained_model,
                "score" : metric
            }

    def __randomized_search(self, estimator, params, sqrt = False):
        n_space = np.prod([len(params[x]) for x in params.keys()])
        if sqrt:
            n_space = np.sqrt(n_space)

        randomized = RandomizedSearchCV(
            estimator = estimator,
            param_distributions = params,
            random_state = 0,
            cv = 5,  # TODO not sure how we can choose the best
            # n_jobs = -1, #uses all available processors #TODO this is killing
            n_iter = n_space  # TODO this should be dynamic, based on the number of features
        )
        return randomized.fit(self.X_train, self.y_train)

    def __train_without_optimizer(self, estimator):
        return estimator.fit(self.X_train, self.y_train)

    # TODO : Acabar implementação
    def __bayes(self, estimator, params, sqrt):
        pass
