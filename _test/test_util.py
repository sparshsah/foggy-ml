# target source code
import foggy_ml.util as util  # assumes `/path/to/foggy-ml` in `PYTHONPATH`, so pylint: disable=import-error
# unit-testing suite
import unittest


class TestCheckType(unittest.TestCase):

    def test_primitive_int_succ(self):
        for x in -128, -2, -1, 0, 1, 2, 172:
            self.assertEqual(util.check_type(x, type_=int), x)

    def test_primitive_int_fail(self):
        for x in "a", 3.14, (0, 1), [0, 1]:
            with self.assertRaises(TypeError):
                util.check_type(x, type_=int)


if __name__ == "__main__":
    unittest.main()
