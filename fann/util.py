# data structures
import pandas as pd
# calculations and algorithms
import numpy as np

EPSILON: float = 1e-6


def check_type(obj: object, type_: type, check_dtype: bool=False, check_not: bool=False) -> object:
    if check_dtype:
        try:
            # e.g. pd.DataFrame
            type_obj = obj.dtypes
            check = type_obj == type_
        except AttributeError:
            try:
                # e.g. np.ndarray, pd.Series
                type_obj = obj.dtype
                check = type_obj == type_
            except AttributeError:
                # fallback for generic collection e.g. list, set
                type_obj = [type(x) for x in list(obj)]
                check = [isinstance(x, type_) for x in list(obj)]
    else:
        type_obj = type(obj)
        check = isinstance(obj, type_)
    check = np.alltrue(check)
    if check_not:
        check = not check
    if not check:
        raise TypeError("{obj} is (dtype={check_dtype}) {type_obj}, failing against (not={check_not}) {type_}!".format(
            obj=obj, check_dtype=check_dtype, type_obj=type_obj, check_not=check_not, type_=type_))
    return obj


def check_dtype(obj: object, type_: type, check_not: bool=False) -> object:
    return check_type(obj=obj, type_=type_, check_dtype=True, check_not=check_not)


def check_not_type(obj: object, type_: type, check_dtype: bool=False) -> object:
    return check_type(obj=obj, type_=type_, check_dtype=check_dtype, check_not=True)


def check_pmf(pmf: object) -> object:
    # e.g. list, dict, np.ndarray, pd.Series
    _pmf = pd.Series(pmf)
    check_dtype(_pmf, float)
    if not np.alltrue(_pmf >= -EPSILON):
        raise ValueError("{_pmf} not non-negative!".format(_pmf=_pmf))
    if not np.isclose(sum(_pmf), 1.00):
        raise ValueError("{_pmf} sums to {sum_} not 1.00!".format(_pmf=_pmf, sum_=sum(_pmf)))
    return pmf
