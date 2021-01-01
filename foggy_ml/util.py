# syntax utils
from typing import Optional
# data structures
import pandas as pd
# calculations and algorithms
import numpy as np

EPSILON: float = 1e-6


########################################################################################################################
# CHECKING #############################################################################################################
########################################################################################################################

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


def check_subset(sub: set=set(), sup: set=set()) -> set:
    if not sub.issubset(sup):
        raise ValueError(sub.difference(sup))
    return sub


def check_pmf(pmf: object) -> object:
    # e.g. list, dict, np.ndarray, pd.Series
    pmf_ = pd.Series(pmf)
    check_dtype(pmf_, float)
    if not np.alltrue(pmf_ >= -EPSILON):
        raise ValueError("{pmf_} not non-negative!".format(pmf_=pmf_))
    if not np.isclose(sum(pmf_), 1.00):
        raise ValueError("{pmf_} sums to {sum_} not 1.00!".format(pmf_=pmf_, sum_=sum(pmf_)))
    return pmf


def check_one_hot(y: pd.DataFrame) -> pd.DataFrame:
    y = check_type(y, type_=pd.DataFrame)
    y = y.apply(check_pmf, axis="columns")
    if not np.alltrue(np.isclose(y, 0) | np.isclose(y, 1)):
        raise ValueError
    return y


########################################################################################################################
# DATA WRANGLING #######################################################################################################
########################################################################################################################

def one_hotify(y: pd.Series, y_choices: Optional[list]=None) -> pd.DataFrame:
    """
    Convert a flat vector of category labels to a one-hot DataFrame.

    input
    -----
    y: pd.Series, the flat vector.

    y_choices: Optional[list] (default None), if you want to enforce a particular
        column order for the output. Default column order follows default sort.
        For example, if your category labels are ["Home", "Away"], that might be a more
        natural order than the default alphabetical sort ["Away", "Home"].
    """
    y_choices = sorted(y.unique()) if y_choices is None else y_choices
    _ = check_subset(sub=set(y.unique()), sup=set(y_choices))
    assert False, "Implement this!"
    y = check_one_hot(y)
    return y


########################################################################################################################
# LOSS | |- || |_ ######################################################################################################
########################################################################################################################

"""
One choice (not implemented) that can help combat overfitting is
to "regularize" parameters by penalizing deviations from zero.
This is like LASSO or Ridge or ElasticNet OLS regression.
"""

def get_neg_llh(y: pd.DataFrame, p_hat: pd.DataFrame, normalize: bool=True) -> float:
    """
    Get negative (joint or mean) log-likelihood of the set of independent outcomes specified by `y`,
    given the "model" specified by `p_hat`.
    Log-likelihood is more numerically stable than raw likelihood,
    and negative makes this suitable for use as loss function in minization problem formulation.

    input
    -----
    y: pd.DataFrame (one-hot), the correct category label for each point.

    p_hat: pd.DataFrame (index = observations, columns = category labels),
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
    p_hat = check_type(p_hat, type_=pd.DataFrame)
    p_hat = p_hat.apply(check_pmf, axis="columns")

    # pick out the entry corresponding to the probability assigned to the correct label in each row
    p_hat_for_correct_label_per_row = (y * p_hat).sum(axis="columns")
    p_hat_for_correct_label_per_row = check_type(p_hat_for_correct_label_per_row, type_=pd.Series)
    # convert to negative joint log-likelihood
    neg_llh = -np.sum(np.log(p_hat_for_correct_label_per_row))
    neg_llh = check_type(neg_llh, type_=float)
    if normalize:
        r"""
        negative arithmetic mean log-likelihood
        = $ - \sum_i{\log p_i} / n $
        = $ - \log(\prod_i{p_i}) / n $
        = $ - \log((\prod_i{p_i})^{1/n}) $
        = negative log geometric mean likelihood.
        """
        neg_llh /= len(y.index)
    return neg_llh
