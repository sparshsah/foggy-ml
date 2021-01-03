# target source code
# assumes `/path/to/foggy-ml` has been added to `PYTHONPATH`
# preferred over `from ..util import *` relative import, since we need to run this standalone as __main__
# from foggy_ml.util import *
# testing suite
import unittest

class TestSkeleton(unittest.TestCase):

    def test_skeleton(self):
        self.assertTrue(True)

if __name__ == "__main__":
    unittest.main()
