import unittest
import pandas as pd
import numpy as np
from unicornml import UnicornML
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

class TestStringMethods(unittest.TestCase):
    def test_linearRegression(self):
        data = pd.read_csv("./data/50_Startups.csv")
        X = data.iloc[:,:-1].values
        Y = data.iloc[:,-1].values

        ct = ColumnTransformer (
            [
                ("encoder", OneHotEncoder(), [3])
            ],
            remainder= "passthrough"
        )
        X = np.array(ct.fit_transform(X), dtype = np.float)
        X = X[:,1:]

        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = .2, random_state = 0)

        unicorn = UnicornML(
            X_train,
            X_test,
            Y_train,
            Y_test
        )
        unicorn.Rainbow()

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
