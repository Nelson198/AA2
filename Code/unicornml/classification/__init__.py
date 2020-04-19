class Classification:
    def __init__(self, x_train, x_val, x_test, y_train, y_val, y_test):
        self.X_train = x_train
        self.X_val   = x_val
        self.X_test  = x_test
        self.Y_train = y_train
        self.Y_val   = y_val
        self.Y_test  = y_test

    def Odin(self):
        return 1