import sys
import re
from .regression     import Regression
from .classification import Classification
from .clustering     import Clustering

class UnicornML:
    def __init__(self, x_train, x_val, x_test, y_train, y_val, y_test):
        if not x_train or not x_val or not x_test:
            sys.exit("Undefined data to train, val and test")
        
        # it's unsupervised learning
        if not y_train:
            self.model = Clustering()
        else: 
            # Stupid trick
            if bool(re.search('^int', str(y_train.dtype))):
                self.model = Classification(
                    x_train, x_val, x_test,
                    y_train, y_val, y_test
                )
            else:
                self.model = Regression(
                    x_train, x_val, x_test,
                    y_train, y_val, y_test
                )  
    
    # the Odin is the god model!!!
    def Build(self):
        return self.model.Odin()

        