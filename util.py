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
    raise NotImplementedError


########################################################################################################################
# DATA WRANGLING #######################################################################################################
########################################################################################################################

def one_hotify(y: pd.Series) -> pd.DataFrame:
    raise NotImplementedError


########################################################################################################################
# LOSS #################################################################################################################
########################################################################################################################

"""
One choice (not implemented) that can help combat overfitting is
to "regularize" parameters by penalizing deviations from zero.
This is like LASSO or Ridge or ElasticNet OLS regression.
"""

def get_llh(p: pd.Series) -> float:
    """
    Log likelihood of a series of independent outcomes.
    More numerically stable than raw likelihood.

    input
    -----
    p: pd.Series, the probability of each outcome.

    output
    ------
    float, the joint log likelihood of the outcomes.
    """
    p = check_pmf(pmf=p)

    llh = np.log(p).sum()
    llh = check_type(llh, float)
    return llh


def _get_loss(p_y: pd.Series) -> float:
    """
    Negative log likelihood loss.

    input
    -----
    p_y: pd.Series, how much probability mass we assigned to the correct label for each point.
        E.g. if p_y = [0.10, 0.85, 0.50], then there were 3 data points. For the first point,
        we distributed 1-0.10=0.90 probability mass among incorrect labels. Depending on
        how many categories there are, this is pretty poor. In particular, if this is a
        binary classification task, then a simple coin flip would distribute only 0.50
        probability mass to the incorrect label (whichever that might be).
        For the second point, we distributed only 1-0.85=0.15 probability mass
        among incorrect labels. This is pretty good. For the last point,
        we distributed exactly half the probability mass among incorrect labels.

    output
    ------
    float, the calculated loss.
    """
    loss = -get_llh(p=p_y)
    loss = check_type(loss, float)
    return loss


def get_loss(y: pd.Series, p_hat: pd.DataFrame) -> float:
    """
    Negative log likelihood loss (normalized by |training data|).

    input
    -----
    y: pd.Series, the correct category label for each point.

    p_hat: pd.DataFrame (index = observations, columns = category labels),
        how much probability mass we assigned to each category label for each point.
        Each row should be a well-formed probability mass function
        AKA discrete probability distribution.

    output
    ------
    float, the calculated loss.
    """
    y = check_type(y, pd.Series)
    p_hat = check_type(p_hat, pd.DataFrame)
    p_hat = p_hat.apply(check_pmf, axis="columns")

    # pick out the entry for the correct label in each row
    p_y = pd.Series({n: p_hat.loc[n, label] for n, label in y.items()})
    loss = _get_loss(p_y=p_y) / y.count()
    loss = check_type(loss, float)
    return loss


def get_neg_llh(y: pd.DataFrame, p_hat: pd.DataFrame, normalize: bool=True) -> float:
    neg_llh = -np.sum(np.log(y * p_hat))
    if normalize:
        # negative log geom mean likelihood
        # since geom mean corresponds to _**(1/n)
        # and log(_**(1/n)) corresponds to (1/n)*log(_)
        neg_llh /= len(y.index)
    return neg_llh
