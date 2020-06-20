import sys

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from kerastuner              import HyperModel

class UnicornHyperModel(HyperModel):
    def __init__(self, input_shape, output_units, problem):
        self.__input_shape = input_shape
        self.__output_units = output_units

        if problem not in [ "classification", "regression" ]:
            sys.exit("Invalid problem to solve [%s]" % problem)

        if problem == "regression":
            self.__act_output = "linear"
            self.__loss = 'mse'
            self.__metrics = ['mse']
        else:
            if self.__output_units > 2:
                self.__act_output = "softmax"
                self.__loss = "categorical_crossentropy"
                self.__metrics = ["accuracy"]
            else:
                self.__act_output = "sigmoid"
                self.__loss = "binary_crossentropy"
                self.__metrics = ["accuracy"]

    def build(self, hp):
        model = Sequential()
        model.add(
            Dense(
                units=hp.Int('units', 8, 64, 4, default=8),
                activation=hp.Choice(
                    'dense_activation',
                    values=['relu', 'tanh', 'sigmoid'],
                    default='relu'),
                input_shape=self.__input_shape
            )
        )

        model.add(
            Dense(
                units=hp.Int('units', 16, 128, 8, default=16),
                activation=hp.Choice(
                    'dense_activation',
                    values=['relu', 'tanh', 'sigmoid'],
                    default='relu')
            )
        )

        model.add(
            Dropout(
                hp.Float(
                    'dropout',
                    min_value=0.0,
                    max_value=0.1,
                    default=0.005,
                    step=0.01)
            )
        )

        model.add(Dense(self.__output_units, activation=self.__act_output))

        model.compile(
            optimizer='rmsprop', loss=self.__loss, metrics=self.__metrics
        )

        return model

    def get_metrics(self):
        return self.__metrics
