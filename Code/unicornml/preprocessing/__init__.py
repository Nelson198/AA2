import numpy as np
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split

def Preprocessing( X, y):
    new_X = scaling_normalize_x(X)
    new_y, problem = scaling_normalize_y(y)#np.array([[]])
    X_train, X_test, y_train, y_test = train_test_split(new_X, new_y, test_size=.2, random_state=0)
    return X_train, X_test, y_train, y_test, problem


def scaling_normalize_y(y):
    new_y = [] # we are considering y with just one dimension (m,1)
    problem = None
    if any([not is_digit(v) for v in y]):
        problem = "Classification"
        new_y = LabelEncoder.fit_transform(y.reshape(-1,1))
    elif all([isinstance(v,np.int64) or isinstance(v,np.int32) for v in y]):
        problem = "Classification"
        new_y = y
    else:
        new_y = MinMaxScaler().fit_transform(y.reshape(-1,1))
        problem = "Regression"
    return np.array(new_y).reshape(-1,), problem


def scaling_normalize_x(X):
    new_X = []
    for col_index in range(X.shape[1]):
        col = X[:,col_index]
        if any([not is_digit(v) for v in col]):
            encoded = one_hot_encoder(col)#.reshape(-1,1))
            for encoded_col in range(encoded.shape[1]):
                new_X.append(encoded[:,encoded_col].reshape(-1,1).reshape(-1,))
        else:
            new_X.append(scaling(col.reshape(-1,1)).reshape(-1,))

    return np.array(new_X).T


def one_hot_encoder(col):
    return OneHotEncoder(sparse=False, drop="first").fit_transform(
        LabelEncoder().fit_transform(col).reshape((-1,1))
    )


def scaling(col):
    return MinMaxScaler().fit_transform(col)


def file_split_X_y(data, label_index):
    return data.iloc[:,:label_index].values, data.iloc[:,label_index].values


def is_digit(n):
    try:
        int(n)
        return True
    except ValueError:
        return False
