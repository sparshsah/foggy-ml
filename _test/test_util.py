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

    def test_multi_succ(self):
        types = {str, float, tuple, list}
        for x in "a", 3.14, (0, 1), [0, 1]:
            self.assertEqual(util.check_type(x, type_=types), x)

    def test_multi_fail(self):
        types = {str, float, tuple, list}
        x = 0
        with self.assertRaises(TypeError):
            util.check_type(x, type_=types)


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

    def test_multi_fail_direct(self):
        types = [str, float, tuple, list]
        for x in "a", 3.14, (0, 1), [0, 1]:
            with self.assertRaises(TypeError):
                util.check_type(x, type_=types, check_not=True)

    def test_multi_fail_wrapped(self):
        types = [str, float, tuple, list]
        for x in "a", 3.14, (0, 1), [0, 1]:
            with self.assertRaises(TypeError):
                util.check_not_type(x, type_=types)

    def test_multi_succ_direct(self):
        types = {str, float, tuple, list}
        x = 0
        self.assertEqual(util.check_type(x, type_=types, check_not=True), x)

    def test_multi_succ_wrapped(self):
        types = {str, float, tuple, list}
        x = 0
        self.assertEqual(util.check_not_type(x, type_=types), x)


class TestCheckDtype(unittest.TestCase):

    def test_non_collection_fail(self):
        # `str` is arguably a collection type but certainly not a collection of ints
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

    def test_multi_succ_wrapped(self):
        types = {int, str}
        for x in ["a", "b"], [0, 1]:
            self.assertEqual(util.check_dtype(x, type_=types), x)

    def test_multi_fail_wrapped(self):
        types = {int, str}
        x = ["a", 0]
        with self.assertRaises(TypeError):
            util.check_dtype(x, type_=types)


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


class TestCheckOneHot(unittest.TestCase):

    def test_row_succ(self):
        import pandas as pd
        r0 = pd.Series([0, 0, 1, 0])
        self.assertIs(util._check_one_hot(r0), r0)

    def test_row_none_hot_fail(self):
        import pandas as pd
        r0 = pd.Series([0, 0, 0, 0])
        with self.assertRaises(ValueError):
            util._check_one_hot(r0)

    def test_row_un_hot_fail(self):
        import pandas as pd
        r0 = pd.Series([1, 1, 0, 1])
        with self.assertRaises(ValueError):
            util._check_one_hot(r0)

    def test_row_all_hot_fail(self):
        import pandas as pd
        r0 = pd.Series([1, 1, 1, 1])
        with self.assertRaises(ValueError):
            util._check_one_hot(r0)

    def test_row_nan_hot_fail(self):
        import pandas as pd
        r0 = pd.Series([0, 0, float("nan"), 0])
        with self.assertRaises(ValueError):
            util._check_one_hot(r0)

    def test_row_nan_one_hot_fail(self):
        import pandas as pd
        r0 = pd.Series([0, float("nan"), 1, 0])
        with self.assertRaises(ValueError):
            util._check_one_hot(r0)

    def test_df_succ(self):
        import pandas as pd
        import numpy as np
        # both data points belong to category b
        df = pd.DataFrame({"is_category_a": [0, 0], "is_category_b": [1, 1]})
        # returns a copy, not the original -> can't check identity
        self.assertTrue(np.allclose(util.check_one_hot(df), df))

    def test_df_fail(self):
        import pandas as pd
        df = pd.DataFrame({"a": [0, 0], "b": [0, 1]})
        with self.assertRaises(ValueError):
            util.check_one_hot(df)


class TestCheckShapeMatch(unittest.TestCase):

    def test_series(self):
        import pandas as pd
        foo = pd.Series([0, 1, 2])
        bar = pd.Series([0, -1, -1])
        foo_, bar_ = util.check_shape_match(foo, bar)
        self.assertIs(foo_, foo)
        self.assertIs(bar_, bar)

    def test_df(self):
        pass

    def test_series_df(self):
        pass

    def test_df_series(self):
        pass


class TestOneHotify(unittest.TestCase):

    def test_point(self):
        import pandas as pd
        raw = "this"
        options = ["other_a", "this", "other_b"]
        res = util._one_hotify(raw, _y_options=options)

        expected = pd.Series({"other_a": 0, "this": 1, "other_b": 0})
        test = res == expected
        self.assertTrue(test.all())

    def test_points(self):
        import pandas as pd
        raw = pd.Series(["a", "b", "c", "a"])
        res = util.one_hotify(raw)

        expected = pd.DataFrame({
            "a": [1, 0, 0, 1],
            "b": [0, 1, 0, 0],
            "c": [0, 0, 1, 0],
        })
        test = res == expected
        self.assertTrue(test.all().all())


"""
class TestSplitShuffle(unittest.TestCase):

    def test_series(self):
        raise NotImplementedError

    def test_df(self):
        raise NotImplementedError

    def test_series_df(self):
        raise NotImplementedError

    def test_df_series(self):
        raise NotImplementedError
"""


class TestGetCrossEntropy(unittest.TestCase):

    def test_series(self):
        import pandas as pd
        import numpy as np
        _y = pd.Series([0, 0, 1, 0])
        p_y = pd.Series([0.10, 0.30, 0.40, 0.20])
        expected = -np.log(0.40)
        test = np.isclose(util._get_cross_entropy(_y=_y, p_y=p_y), expected)
        self.assertTrue(test)

    def test_series_perfectly_right(self):
        import pandas as pd
        import numpy as np
        _y = pd.Series([0, 0, 1, 0])
        p_y = _y
        res = util._get_cross_entropy(_y=_y, p_y=p_y)
        expected = 0
        test = np.isclose(res, expected)
        self.assertTrue(test)

    def test_series_perfectly_wrong(self):
        import pandas as pd
        import numpy as np
        _y = pd.Series([0, 0, 1, 0])
        p_y = pd.Series([0.10, 0.70, 0, 0.20])
        res = util._get_cross_entropy(_y=_y, p_y=p_y)
        expected = float("inf")
        test = np.isclose(res, expected)
        self.assertTrue(test)

    def test_df(self):
        import pandas as pd
        import numpy as np
        y = util.one_hotify(pd.Series(["a", "b", "a"]))
        p = pd.DataFrame({
            "a": [0.50, 0.20, 0.70],
            "b": [0.50, 0.80, 0.30]
        })
        res = util.get_cross_entropy(y=y, p=p)
        expected = np.log((0.50 * 0.80 * 0.70) ** (-1. / 3))
        test = np.isclose(res, expected)
        self.assertTrue(test)

    def test_df_dup(self):
        """Test that 'duplicating' input doesn't change normalized output (default setting)."""
        import pandas as pd
        import numpy as np
        y = util.one_hotify(pd.Series(["a", "b", "a"]))
        p = pd.DataFrame({
            "a": [0.50, 0.20, 0.70],
            "b": [0.50, 0.80, 0.30]
        })
        res = util.get_cross_entropy(y=pd.concat([y, y]), p=pd.concat([p, p]))
        res_undup = util.get_cross_entropy(y=y, p=p)
        expected = res_undup
        test = np.isclose(res, expected)
        self.assertTrue(test)

    def test_df_dup_unnormed(self):
        """Test that 'duplicating' input doubles un-normalized output."""
        import pandas as pd
        import numpy as np
        y = util.one_hotify(pd.Series(["a", "b", "a"]))
        p = pd.DataFrame({
            "a": [0.50, 0.20, 0.70],
            "b": [0.50, 0.80, 0.30]
        })
        res = util.get_cross_entropy(y=pd.concat([y, y]), p=pd.concat([p, p]), normalize=False)
        res_undup = util.get_cross_entropy(y=y, p=p, normalize=False)
        expected = res_undup * 2
        test = np.isclose(res, expected)
        self.assertTrue(test)

    def test_df_perfectly_right_and_wrong(self):
        import pandas as pd
        import numpy as np
        y = util.one_hotify(pd.Series(["a", "b", "a"]))
        p = pd.DataFrame({
            "a": [1, 1, 0.70],
            "b": [0, 0, 0.30]
        })
        res = util.get_cross_entropy(y=pd.concat([y, y]), p=pd.concat([p, p]))
        expected = float("inf")
        test = np.isclose(res, expected)
        self.assertTrue(test)


if __name__ == "__main__":
    unittest.main()
