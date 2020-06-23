import sys

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from kerastuner import HyperModel


class UnicornHyperModel(HyperModel):
    def __init__(self, input_shape, output_units, problem):
        super().__init__()
        self.__input_shape = input_shape
        self.__output_units = output_units

        if problem not in ["classification", "regression"]:
            sys.exit("Invalid problem to solve [%s]" % problem)

        if problem == "regression":
            self.__act_output = "linear"
            self.__loss = "mse"
            self.__metric = "mse"
        else:
            if self.__output_units > 2:
                self.__act_output = "softmax"
                self.__loss = "sparse_categorical_crossentropy"
                self.__metric = "accuracy"
            else:
                self.__act_output = "sigmoid"
                self.__loss = "binary_crossentropy"
                self.__metric = "accuracy"

    def build(self, hp):
        model = Sequential()
        model.add(
            Dense(
                units=hp.Int("units", 32, 128, 8, default=32),
                activation=hp.Choice(
                    "dense_activation",
                    values=["relu", "tanh", "sigmoid"],
                    default="relu"),
                input_shape=(self.__input_shape[1],)
            )
        )

        model.add(
            Dense(
                units=hp.Int("units", 64, 512, 16, default=64),
                activation=hp.Choice(
                    "dense_activation",
                    values=["relu", "tanh", "sigmoid"],
                    default="relu")
            )
        )

        model.add(
            Dropout(
                hp.Float(
                    "dropout",
                    min_value=0.0,
                    max_value=0.2,
                    default=0.1,
                    step=0.05)
            )
        )
        
        if self.__act_output == "softmax":
            model.add(Dense(self.__output_units, activation=self.__act_output))
        else:
            model.add(Dense(1, activation=self.__act_output))

        model.compile(
            optimizer="rmsprop", loss=self.__loss, metrics=[self.__metric]
        )

        return model

    def get_metric(self):
        return self.__metric

    def get_output_units(self):
        return self.__output_units
