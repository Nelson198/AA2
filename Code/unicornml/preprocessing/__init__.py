import numpy as np
import pandas as pd
from scipy.stats import mode

from sklearn.preprocessing import OneHotEncoder, LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA


def Preprocessing(X, y, cv):
    X, y = removeSmallCats(X, y, cv)
    
    X = removeNAN(X)

    X = scaling_normalize_x(X)
    y, problem = scaling_normalize_y(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # Principal component analysis
    pca = PCA(0.95)
    pca.fit(X_train)
    pca.transform(X_train)
    pca.transform(X_test)

    return X_train, X_test, y_train, y_test, problem

def removeSmallCats(X, y, cv):
    unique, count = np.unique(y, return_counts=True)
    cats_count = dict(zip(unique, count))

    idx = []
    for i in range(len(y)):
        if cats_count[y[i]] < cv:
            idx.append(i)

    X = np.delete(X, idx, axis=0)
    y = np.delete(y, idx, axis=0)

    return X, y


def removeNAN(X):
    colIdx = []
    for i in range(X.shape[1]):
        totalNAN = pd.isnull(X[:, i]).sum()
        if totalNAN / X.shape[0] > 0.4:
            colIdx.append(i)
        elif totalNAN > 0:
            # Categorical variable
            if isinstance(X[0, i], np.float):
                colMode = mode(X[:, i], axis=0)[0]
                idxs = np.where(np.isnan(X[:, i]))[0]
                X[idxs, i] = colMode
            # Continuous variable
            else:
                colMean = np.nanmean(X[:, i], axis=0)
                idxs = np.where(np.isnan(X[:, i]))[0]
                X[idxs, i] = colMean

    # Remove columns
    X = np.delete(X, colIdx, axis=1)

    return X


def scaling_normalize_y(y):
    if any([not is_digit(v) for v in y]) or all([isinstance(v, np.int64) or isinstance(v, np.int32) for v in y]):
        new_y = LabelEncoder().fit_transform(y.reshape(-1, 1))
        classes = len(np.unique(new_y))
        problem = ("Classification", classes)
    else:
        new_y = MinMaxScaler().fit_transform(y.reshape(-1, 1))
        problem = ("Regression", -1)
    return np.array(new_y).reshape(-1, ), problem


def scaling_normalize_x(X):
    new_X = []
    for col_index in range(X.shape[1]):
        col = X[:, col_index]
        if any([not is_digit(v) for v in col]):
            encoded = one_hot_encoder(col)  # .reshape(-1,1))
            for encoded_col in range(encoded.shape[1]):
                new_X.append(encoded[:, encoded_col].reshape(-1, 1).reshape(-1, ))
        else:
            new_X.append(scaling(col.reshape(-1, 1)).reshape(-1, ))

    return np.array(new_X).T


def one_hot_encoder(col):
    return OneHotEncoder(sparse=False, drop="first", categories="auto") \
        .fit_transform(
        LabelEncoder().fit_transform(col).reshape((-1, 1))
    )


def scaling(col):
    return MinMaxScaler().fit_transform(col)


def file_split_X_y(data, label_index):
    return data.iloc[:, :label_index].values, data.iloc[:, label_index].values


def is_digit(n):
    try:
        int(n)
        return True
    except ValueError:
        return False
