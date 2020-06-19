import unittest
import numpy as np
from unicornml import UnicornML

class TestStringMethods(unittest.TestCase):
    def test_linearRegression(self):
        unicorn = UnicornML(
            { "file": "./data/pregnant.csv"}
        )
        X = np.concatenate((unicorn.X_train, unicorn.X_test), axis=0)
        y = np.concatenate((unicorn.y_train, unicorn.y_test), axis=0)
        unicorn.Rainbow()
        yatt = unicorn.predict(X)
        r2 = unicorn.evaluate(y, yatt)
        print("R2: %f" % r2)

        self.assertEqual("foo".upper(), "FOO")

    '''
    def test_isupper(self):
        print("Hey\n\n\n", file=sys.stderr)
        self.assertTrue('FOO'.isupper())
        self.assertFalse('Foo'.isupper())

    def test_split(self):
        s = 'hello world'
        self.assertEqual(s.split(), ['hello', 'world'])
        # check that s.split fails when the separator is not a string
        with self.assertRaises(TypeError):
            s.split(2)
    '''

if __name__ == "__main__":
    unittest.main()
