import unittest
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from unicornml.classification import Classification

class TestClassification(unittest.TestCase):
    def test_classification(self):
        dataset = pd.read_csv("./data/Social_Network_Ads.csv")
        X = dataset.iloc[:,2:-1].values
        y = dataset.iloc[:,-1].values


        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .2, random_state = 0)

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test  = scaler.transform(X_test)

        classificator = Classification(X_train, X_test, y_train, y_test)

        model = classificator.Rainbow()

if __name__ == "__main__":
    unittest.main()