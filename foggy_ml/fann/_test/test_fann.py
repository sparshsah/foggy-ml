# project source code
# assumes `/path/to/foggy-ml` has been added to `PYTHONPATH`
# preferable `from ..fann import *` relative import, since we need to run this as __main__
from foggy_ml.fann import *
# testing suite
import unittest
from unittest import mock

class TestSkeleton(unittest.TestCase):

    def test_skeleton(self):
        self.assertTrue(True)

if __name__ == "__main__":
    unittest.main()
