import sys
from .regression     import Regression
from .classification import Classification


class UnicornML:
    def __init__(self, x_train=None, x_test=None, y_train=None, y_test=None):
        if x_train == None or x_test == None:
            sys.exit("Undefined data to train, val and test")
        
        return 1
        # it's unsupervised learning
        '''if bool(re.search('^int', str(y_train.dtype))):
            self.model = Classification(
                    x_train, x_test,
                    y_train, y_test
            )
        else:
            self.model = Regression(
                x_train, x_test,
                y_train, y_test
        )  '''
    
    # the Odin is the god model!!!
    #def Build(self):
    #    return self.model.Odin()

        