import unittest
import numpy as np
from unicornml import UnicornML


class TestStringMethods(unittest.TestCase):
    def test_linearRegression(self):
        unicorn = UnicornML(
            {"file": "./data/pregnant.csv"}
        )
        X = np.concatenate((unicorn.X_train, unicorn.X_test), axis=0)
        y = np.concatenate((unicorn.y_train, unicorn.y_test), axis=0)
        unicorn.Rainbow()
        yatt = unicorn.predict(X)
        accuracy = unicorn.evaluate(y, yatt)
        print("Accuracy: %f" % accuracy)

        self.assertEqual("foo".upper(), "FOO")


if __name__ == "__main__":
    unittest.main()
