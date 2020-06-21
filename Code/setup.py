from setuptools import setup, find_packages

setup(name="unicornml",
      version="1.0",
      description="An AutoML framework",
      # url = "http://github.com/storborg/funniest",
      author="João Imperadeiro, José Boticas, Nelson Teixeira, Rui Meira",
      # author_email = "flyingcircus@example.com",
      license="MIT",
      packages=find_packages(),
      zip_safe=False,
      test_suite="tests"
      )

# def module_tests():
#    test_loader = unittest.TestLoader()
#    test_suite  = test_loader.discover("tests", pattern="*.py")
#    return test_suite
