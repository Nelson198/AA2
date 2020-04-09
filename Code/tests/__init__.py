import unittest

def module_tests():
    test_loader = unittest.TestLoader()
    test_suite  = test_loader.discover('.', pattern='*.py')
    return test_suite
