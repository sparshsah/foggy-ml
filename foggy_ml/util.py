# syntax utils
from typing import Collection, Optional
# data structures
import pandas as pd
# calculations and algorithms
import numpy as np

EPSILON: float = 1e-6


########################################################################################################################
# CHECKING #############################################################################################################
########################################################################################################################

def _check_type(obj: object, type_: type, check_dtype: bool=False, check_not: bool=False) -> object:
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
    check = np.all(check)
    if check_not:
        check = not check
    if not check:
        raise TypeError("{obj} is (dtype={check_dtype}) {type_obj}, failing against (not={check_not}) {type_}!".format(
            obj=obj, check_dtype=check_dtype, type_obj=type_obj, check_not=check_not, type_=type_))
    return obj


def check_type(obj: object, type_: type, check_dtype: bool=False, check_not: bool=False) -> object:
    """
    Check `obj` against one or more types.

    If `type_` is a collection of types, then
    if `check_not`, check that `obj` isn't any of those types;
    else, check that `obj` is at least one of those types.

    In both of these subcases, if `check_dtype`, then
    the object's dtype must pass against a SINGLE type.
    For example,
    >>> check_type([0, 1], type_={int, str}, check_dtype=True)
    [0, 1]
    >>> check_type(['a', 'b'], type_={int, str}, check_dtype=True)
    ['a', 'b']
    >>> check_type([0, 'a'], type_={int, str}, check_dtype=True)
    TypeError
    """
    if isinstance(type_, type):  # the type being checked against is a type
        return _check_type(obj=obj, type_=type_, check_dtype=check_dtype, check_not=check_not)
    else:  # it must be an collection of possible types
        types = list(type_)
        del type_
        if check_not:
            # if obj IS any of the given types, _check_type will raise mid-comprehension
            _ = [_check_type(obj=obj, type_=type_, check_dtype=check_dtype, check_not=check_not) for type_ in types]
            return obj
        else:
            for type_ in types:
                try:
                    return _check_type(obj=obj, type_=type_, check_dtype=check_dtype, check_not=check_not)
                except TypeError:
                    continue
            raise TypeError("{obj} (dtype={check_dtype}) failing against every (not={check_not}) {type_}!".format(
                obj=obj, check_dtype=check_dtype, check_not=check_not, type_=type_))


def check_dtype(obj: object, type_: type, check_not: bool=False) -> object:
    return check_type(obj=obj, type_=type_, check_dtype=True, check_not=check_not)


def check_not_type(obj: object, type_: type, check_dtype: bool=False) -> object:
    return check_type(obj=obj, type_=type_, check_dtype=check_dtype, check_not=True)


def check_subset(sub: set=set(), sup: set=set()) -> set:
    if not sub.issubset(sup):
        raise ValueError(sub.difference(sup))
    return sub


def check_pmf(pmf: object, permit_nan=False) -> object:
    # e.g. list, dict, np.ndarray, pd.Series
    pmf_ = pd.Series(pmf)
    pmf_ = check_dtype(pmf_, {int, float})
    if not permit_nan and pmf_.isnull().any():
        raise ValueError("{pmf_} not non-null!".format(pmf_=pmf_))
    if (pmf_ < -EPSILON).any():
        raise ValueError("{pmf_} not non-negative!".format(pmf_=pmf_))
    if not np.isclose(pmf_.sum(), 1.00):
        raise ValueError("{pmf_} sums to {sum_} not 1.00!".format(pmf_=pmf_, sum_=sum(pmf_)))
    return pmf


def _check_one_hot(_y: pd.Series) -> pd.Series:
    _y = check_pmf(_y)
    if not np.all(np.isclose(_y, 0) | np.isclose(_y, 1)):
        raise ValueError(_y)
    return _y


def check_one_hot(y: pd.DataFrame) -> pd.DataFrame:
    return y.apply(_check_one_hot, axis="columns")


########################################################################################################################
# DATA WRANGLING #######################################################################################################
########################################################################################################################

def _one_hotify(_y: object, _y_options: list) -> pd.Series:
    """
    Convert a single category label to one-hot vector representation.

    E.g.
    >>> _one_hotify(2, range(4)) == pd.Series({0: 0, 1: 0, 2: 1, 3: 0})
    or
    >>> _one_hotify("Away", ["Home", "Away"]) == pd.Series({"Home": 0, "Away": 1})

    input
    -----
    _y: object, e.g. int or str, the actual category label.

    _y_options: list, the possible options of category label.

    output
    ------
    pd.Series, the one-hot vector representation.
    """
    _y = pd.Series({_y: 1}, index=_y_options).fillna(0)
    return _check_one_hot(_y)


def one_hotify(y: pd.Series, _y_options: Optional[list]=None) -> pd.DataFrame:
    """
    Convert a flat vector of category labels to a one-hot DataFrame.

    input
    -----
    y: pd.Series, the flat vector.

    _y_options: Optional[list] (default None), use if you want to enforce a particular
        column order for the output. Default column order follows default sort.
        For example, if your category labels are ["Home", "Away"], that might be a more
        natural order than the default alphabetical sort ["Away", "Home"].
    """
    _y_options = sorted(y.unique()) if _y_options is None else _y_options
    _ = check_subset(sub=set(y.unique()), sup=set(_y_options))

    y = y.apply(_one_hotify, _y_options=_y_options)
    return check_one_hot(y)


########################################################################################################################
# LOSS | |- || |_ ######################################################################################################
########################################################################################################################

"""
One choice (not implemented) that can help combat overfitting is
to "regularize" parameters by penalizing deviations from zero.
This is like LASSO or Ridge or ElasticNet OLS regression.
"""

def _get_neg_llh(p_y: Collection, normalize: bool=True) -> float:
    """
    Negative log likelihood.

    input
    -----
    p_y: Collection, how much probability mass we assigned to the correct label for each point.
        E.g. if p_y = [0.10, 0.85, 0.50], then there were 3 data points. For the first point,
        we distributed 1-0.10=0.90 probability mass among incorrect labels. Depending on
        how many categories there are, this is pretty poor. In particular, if this is a
        binary classification task, then a simple coin flip would distribute only 0.50
        probability mass to the incorrect label (whichever that might be).
        For the second point, we distributed only 1-0.85=0.15 probability mass
        among incorrect labels. This is pretty good. For the last point,
        we distributed exactly half the probability mass among incorrect labels.

    normalize: bool (default True), whether to divide by |dataset|;
        False -> joint log-likelihood, True -> mean log-likelihood.

    output
    ------
    float, the negative (joint or mean) log-likelihood.
    """
    p_y = check_type(p_y, type_=Collection)

    # convert to negative joint log-likelihood
    neg_llh = -np.sum(np.log(p_y))
    if normalize:
        r"""
        negative arithmetic mean log-likelihood
        = $ - \sum_i{\log p_i} / n $
        = $ - \log(\prod_i p_i) / n $
        = $ - \log((\prod_i p_i)^{1/n}) $
        = negative log geometric mean likelihood.
        """
        neg_llh /= len(p_y)
    return check_type(neg_llh, type_=float)


def get_neg_llh(y: pd.DataFrame, p: pd.DataFrame, normalize: bool=True) -> float:
    """
    Get negative (joint or mean) log-likelihood of the set of independent outcomes specified by `y`,
    given the "model" specified by `p`.
    Log-likelihood is more numerically stable than raw likelihood,
    and negative makes this suitable for use as loss function in minization problem formulation.

    input
    -----
    y: pd.DataFrame (one-hot), the correct category label for each point.

    p: pd.DataFrame (index = observations, columns = category labels),
        how much probability mass we assigned to each category label for each point.
        Each row should be a well-formed probability mass function
        AKA discrete probability distribution.

    normalize: bool (default True), whether to divide by |dataset|;
        False -> joint log-likelihood, True -> mean log-likelihood.

    output
    ------
    float, the negative (joint or mean) log-likelihood.
    """
    y = check_one_hot(y=y)
    p = p.apply(check_pmf, axis="columns")

    """
    y is one-hot, so this picks out the entry corresponding to
    the probability assigned to the correct label in each row
    """
    p_y = (p * y).sum(axis="columns")
    return _get_neg_llh(p_y=p_y, normalize=normalize)
