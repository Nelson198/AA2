import sys
import numpy as np
import tensorflow as tf
from kerastuner import HyperModel

class NeuralNetwork:
    def __init__(self, X_train, X_test, y_train, y_test, problem, metric):
        self.X_train = X_train
        self.X_test  = X_test
        self.y_train = y_train
        self.y_test  = y_test

        if problem not in ["classification", "regression"]:
            sys.exit("Invalid problem to solve [%s]" % problem)
        self.problem = problem

        if problem == "regression":
            self.act_output = "linear"
        else:
            self.output_units = len(np.unique(y_train))
            if self.output_units > 2:
                self.act_output = "softmax"
            else:
                self.act_output = "sigmoid"

#    def build(self):


class UnicornHyperModel(HyperModel):
    def __init__(self, input_shape, output_units, output_activation, optimizer):
        self.input_shape = input_shape
        self.output_units = output_units
        self.output_activation = output_activation

    #def build(self, hp):
    #    model = Sequential()
    #    model.add(
    #        layers.Dense(
    #            units=hp.Int('units', 8, 64, 4, default=8),
    #            activation=hp.Choice(
    #                'dense_activation',
    #                values=['relu', 'tanh', 'sigmoid'],
    #                default='relu'),
    #            input_shape=input_shape
    #        )
    #    )

    #    model.add(
    #        layers.Dense(
    #            units=hp.Int('units', 16, 64, 4, default=16),
    #            activation=hp.Choice(
    #                'dense_activation',
    #                values=['relu', 'tanh', 'sigmoid'],
    #                default='relu')
    #        )
    #    )

    #    model.add(
    #        layers.Dropout(
    #            hp.Float(
    #                'dropout',
    #                min_value=0.0,
    #                max_value=0.1,
    #                default=0.005,
    #                step=0.01)
    #        )
    #    )

    #    model.add(layers.Dense(1))

    #    model.compile(
    #        optimizer='rmsprop', loss='mse', metrics=['mse']
    #    )



