import unittest
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from unicornml import UnicornML
import numpy as np

class TestClassification(unittest.TestCase):
    def test_classification(self):
        unicorn = UnicornML(
            { "file": "./data/Social_Network_Ads.csv"}
        )
        X = np.concatenate((unicorn.X_train, unicorn.X_test), axis=0)
        y = np.concatenate((unicorn.y_train, unicorn.y_test), axis=0)
        unicorn.Rainbow()
        yatt = unicorn.predict(X)
        accuracy = unicorn.evaluate(y, yatt)
        print("Accuracy %f" % accuracy)

if __name__ == "__main__":
    unittest.main()