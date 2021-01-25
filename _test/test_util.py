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


class TestCheckNotType(unittest.TestCase):
    """Complement of TestCheckType."""

    def test_primitive_int_fail_direct(self):
        for x in -128, -2, -1, 0, 1, 2, 172:
            with self.assertRaises(TypeError):
                util.check_type(x, type_=int, check_not=True)

    def test_primitive_int_fail_wrapped(self):
        for x in -128, -2, -1, 0, 1, 2, 172:
            with self.assertRaises(TypeError):
                util.check_not_type(x, type_=int)

    def test_primitive_int_succ_direct(self):
        for x in "a", 3.14, (0, 1), [0, 1]:
            self.assertEqual(util.check_type(x, type_=int, check_not=True), x)

    def test_primitive_int_succ_wrapped(self):
        for x in "a", 3.14, (0, 1), [0, 1]:
            self.assertEqual(util.check_not_type(x, type_=int), x)

    def test_complex_series_fail_direct(self):
        import pandas as pd
        for x in pd.Series(), pd.Series([0, 1]):
            with self.assertRaises(TypeError):
                util.check_type(x, type_=pd.Series, check_not=True)

    def test_complex_series_fail_wrapped(self):
        import pandas as pd
        for x in pd.Series(), pd.Series([0, 1]):
            with self.assertRaises(TypeError):
                util.check_not_type(x, type_=pd.Series)

    def test_complex_series_succ_direct(self):
        import pandas as pd
        for x in "a", 0, pd.DataFrame(), pd.DataFrame(0, index=["r0", "r1"], columns=["c0", "c1"]):
            self.assertIs(util.check_type(x, type_=pd.Series, check_not=True), x)

    def test_complex_series_succ_wrapped(self):
        import pandas as pd
        for x in "a", 0, pd.DataFrame(), pd.DataFrame(0, index=["r0", "r1"], columns=["c0", "c1"]):
            self.assertIs(util.check_not_type(x, type_=pd.Series), x)


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


class TestCheckNotDtype(unittest.TestCase):
    """Complement of TestCheckDtype."""

    def test_int_tuple_fail_direct(self):
        x = (0, 1)
        with self.assertRaises(TypeError):
            util.check_type(x, type_=int, check_dtype=True, check_not=True)

    def test_int_tuple_fail_wrapped(self):
        x = (0, 1)
        with self.assertRaises(TypeError):
            util.check_dtype(x, type_=int, check_not=True)

    def test_int_tuple_succ_direct(self):
        import pandas as pd
        for x in ("a", "b"), (3.14, 2.71), (0, 0.):
            self.assertEqual(util.check_type(x, type_=int, check_dtype=True, check_not=True), x)

    def test_int_tuple_succ_wrapped(self):
        import pandas as pd
        for x in ("a", "b"), (3.14, 2.71), (0, 0.):
            self.assertEqual(util.check_dtype(x, type_=int, check_not=True), x)

    def test_int_list_fail_direct(self):
        x = [0, 1]
        with self.assertRaises(TypeError):
            util.check_type(x, type_=int, check_dtype=True, check_not=True)

    def test_int_list_fail_wrapped(self):
        x = [0, 1]
        with self.assertRaises(TypeError):
            util.check_dtype(x, type_=int, check_not=True)

    def test_int_list_succ_direct(self):
        import pandas as pd
        for x in ["a", "b"], [3.14, 2.71], [0, 0.]:
            self.assertEqual(util.check_type(x, type_=int, check_dtype=True, check_not=True), x)

    def test_int_list_succ_wrapped(self):
        import pandas as pd
        for x in ["a", "b"], [3.14, 2.71], [0, 0.]:
            self.assertEqual(util.check_dtype(x, type_=int, check_not=True), x)

    def test_int_series_fail_direct(self):
        import pandas as pd
        x = pd.Series([0, 1])
        with self.assertRaises(TypeError):
            util.check_type(x, type_=int, check_dtype=True, check_not=True)

    def test_int_series_fail_wrapped(self):
        import pandas as pd
        x = pd.Series([0, 1])
        with self.assertRaises(TypeError):
            util.check_dtype(x, type_=int, check_not=True)

    def test_int_series_succ_direct(self):
        import pandas as pd
        for x in pd.Series(["a", "b"]), pd.Series([3.14, 2.71]), pd.Series([0, 0.]):
            self.assertIs(util.check_type(x, type_=int, check_dtype=True, check_not=True), x)

    def test_int_series_succ_wrapped(self):
        import pandas as pd
        for x in pd.Series(["a", "b"]), pd.Series([3.14, 2.71]), pd.Series([0, 0.]):
            self.assertIs(util.check_dtype(x, type_=int, check_not=True), x)

    def test_int_df_fail_direct(self):
        import pandas as pd
        x = pd.DataFrame({"c0": {"r0": 0, "r1": 1}, "c1": {"r0": 2, "r1": 3}})
        with self.assertRaises(TypeError):
            util.check_type(x, type_=int, check_dtype=True, check_not=True)

    def test_int_df_fail_wrapped(self):
        import pandas as pd
        x = pd.DataFrame({"c0": {"r0": 0, "r1": 1}, "c1": {"r0": 2, "r1": 3}})
        with self.assertRaises(TypeError):
            util.check_dtype(x, type_=int, check_not=True)

    def test_int_df_succ_direct(self):
        import pandas as pd
        x = pd.DataFrame({"c0": {"r0": 0, "r1": 1}, "c1": {"r0": 2, "r1": 3.}})
        self.assertIs(util.check_type(x, type_=int, check_dtype=True, check_not=True), x)

    def test_int_df_succ_wrapped(self):
        import pandas as pd
        x = pd.DataFrame({"c0": {"r0": 0, "r1": 1}, "c1": {"r0": 2, "r1": 3.}})
        self.assertIs(util.check_dtype(x, type_=int, check_not=True), x)


class TestCheckSubset(unittest.TestCase):

    def check_same_succ(self):
        sub = {"a", 0}
        sup = {"a", 0}
        self.assertEqual(util.check_subset(sub=sub, sup=sup), sub)

    def check_set_succ(self):
        sub = {"a", 0}
        sup = {"a", 0, 1}
        self.assertEqual(util.check_subset(sub=sub, sup=sup), sub)

    def check_set_fail(self):
        sub = {"a", 0}
        sup = {"a", 1, 2}
        with self.assertRaises(ValueError):
            util.check_subset(sub=sub, sup=sup)

    def check_list_succ(self):
        sub = ["a", 0, 0]
        sup = ["a", 0, 1]
        self.assertEqual(util.check_subset(sub=sub, sup=sup), sub)

    def check_list_fail(self):
        sub = ["a", 0]
        sup = ["a", 1, 2]
        with self.assertRaises(ValueError):
            util.check_subset(sub=sub, sup=sup)

    def check_mixed_succ(self):
        sub = ["a", 0, 0]
        sup = {"a", 0, 1}
        self.assertEqual(util.check_subset(sub=sub, sup=sup), sub)


class TestPMF(unittest.TestCase):

    def test_succ(self):
        pmf = [0.2, 0.8]
        self.assertEqual(util.check_pmf(pmf), pmf)

    def test_zero_succ(self):
        pmf = [0.2, 0, 0.8]
        self.assertEqual(util.check_pmf(pmf), pmf)

    def test_nan_zero_succ(self):
        pmf = [0.2, 0, float("nan"), 0.8]
        self.assertEqual(util.check_pmf(pmf, permit_nan=True), pmf)

    def test_neg_fail(self):
        pmf = [0.2, -0.1, 0.1, 0.8]
        with self.assertRaises(ValueError):
            util.check_pmf(pmf)

    def test_fail(self):
        pmf = [0.2, 0.7]
        with self.assertRaises(ValueError):
            util.check_pmf(pmf)


if __name__ == "__main__":
    unittest.main()
