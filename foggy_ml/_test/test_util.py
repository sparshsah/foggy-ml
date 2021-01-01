# project source code
# assumes that `/path/to/foggy-ml` has been added to `PYTHONPATH`
from foggy_ml.util import *
# testing suite
import unittest

class TestSkeleton(unittest.TestCase):

    def test_skeleton(self):
        self.assertTrue(True)

if __name__ == "__main__":
    unittest.main()
