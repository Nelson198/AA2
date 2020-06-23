import unittest
from unicornml import UnicornML


class TestStringMethods(unittest.TestCase):
    def test_linearRegression(self):
        unicorn = UnicornML(
            {"images": "/home/joseb/Desktop/aa2/aula6/cats_and_dogs_small"},
            {"height": 150, "width": 150, "depth": 3, "fine_tuning": False}
        )
        unicorn.Rainbow()
        acc = unicorn.evaluate(1, 1)
        print("acc: %f" % acc)

        self.assertEqual("foo".upper(), "FOO")


if __name__ == "__main__":
    unittest.main()
