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

    def test_complex_series_succ(self):
        import pandas as pd
        for x in pd.Series(), pd.Series([0, 1]):
            # pd.Series.__equals__() tests elementwise equality, we want to test identity
            self.assertIs(util.check_type(x, type_=pd.Series), x)

    def test_complex_series_fail(self):
        import pandas as pd
        for x in "a", 0, pd.DataFrame(), pd.DataFrame(0, index=["r0", "r1"], columns=["c0", "c1"]):
            with self.assertRaises(TypeError):
                util.check_type(x, type_=pd.Series)


class TestCheckDtype(unittest.TestCase):

    def test_non_container_fail(self):
        # `str` is arguably a container type but OK
        for x in "a", 0, 3.14:
            with self.assertRaises(TypeError):
                util.check_dtype(x, type_=int)

    def test_int_tuple_succ(self):
        x = (0, 1)
        self.assertEqual(util.check_dtype(x, type_=int), x)

    def test_int_tuple_fail(self):
        import pandas as pd
        for x in ("a", "b"), (3.14, 2.71), (0, 0.):
            with self.assertRaises(TypeError):
                util.check_dtype(x, type_=int)

    def test_int_list_succ(self):
        x = [0, 1]
        self.assertEqual(util.check_dtype(x, type_=int), x)

    def test_int_list_fail(self):
        import pandas as pd
        for x in ["a", "b"], [3.14, 2.71], [0, 0.]:
            with self.assertRaises(TypeError):
                util.check_dtype(x, type_=int)

    def test_int_series_succ(self):
        import pandas as pd
        x = pd.Series([0, 1])
        self.assertIs(util.check_dtype(x, type_=int), x)

    def test_int_series_fail(self):
        import pandas as pd
        for x in pd.Series(["a", "b"]), pd.Series([3.14, 2.71]), pd.Series([0, 0.]):
            with self.assertRaises(TypeError):
                util.check_dtype(x, type_=int)

    def test_int_df_succ(self):
        import pandas as pd
        x = pd.DataFrame({"c0": {"r0": 0, "r1": 1}, "c1": {"r0": 2, "r1": 3}})
        self.assertIs(util.check_dtype(x, type_=int), x)

    def test_int_df_fail(self):
        import pandas as pd
        x = pd.DataFrame({"c0": {"r0": 0, "r1": 1}, "c1": {"r0": 2, "r1": 3.}})
        with self.assertRaises(TypeError):
            util.check_dtype(x, type_=int)


if __name__ == "__main__":
    unittest.main()
