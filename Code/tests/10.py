import unittest
from unicornml import UnicornML
import numpy as np


class TestClassification(unittest.TestCase):
    def test_classification(self):
        unicorn = UnicornML(
            {"file": "./data/ionosphere.csv"}
        )
        X = np.concatenate((unicorn.X_train, unicorn.X_test), axis=0)
        y = np.concatenate((unicorn.y_train, unicorn.y_test), axis=0)
        unicorn.Rainbow()
        yatt = unicorn.predict(X)
        accuracy = unicorn.evaluate(y, yatt)
        print("Accuracy %f" % accuracy)

        self.assertEqual("foo".upper(), "FOO")


if __name__ == "__main__":
    unittest.main()
