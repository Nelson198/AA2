import unittest
from unicornml import UnicornML
import numpy as np

class TestWine(unittest.TestCase):
    def test_wine(self):
        unicorn = UnicornML(
            { "file": "./data/swedish.csv"}
        )
        X = np.concatenate((unicorn.X_train, unicorn.X_test), axis=0)
        y = np.concatenate((unicorn.y_train, unicorn.y_test), axis=0)
        unicorn.Rainbow()
        yatt = unicorn.predict(X)
        r2 = unicorn.evaluate(y, yatt)
        print("R2: %f" % r2)

        self.assertEqual("foo".upper(), "FOO")

if __name__ == "__main__":
    unittest.main()
