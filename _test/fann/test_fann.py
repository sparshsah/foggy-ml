# TODO(sparshsah): we replicate sklearn so for now we're comfortable, but at some point we should fill this in..
# target source code
# import foggy_ml.fann as fann  # assumes `/path/to/foggy-ml` in `PYTHONPATH`, so pylint: disable=import-error
# unit-testing suite
import unittest


class TestSkeleton(unittest.TestCase):

    def test_skeleton(self):
        self.assertTrue(True)


if __name__ == "__main__":
    unittest.main()
