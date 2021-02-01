# syntax utils
from typing import Tuple, List, Iterable, Callable, Union, Optional
# data structures
import pandas as pd
# data wrangling
from sklearn.utils import shuffle
# calculations and algorithms
import numpy as np
from scipy.special import expit, softmax  # linter can't see C funcs, so pylint: disable=no-name-in-module

Floatvec = Union[float, Iterable[float]]
Arraylike = Union[np.ndarray, pd.Series, pd.DataFrame]

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

    note
    ----
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


def check_shape_match(x0: object, x1: object, axis=0) -> Tuple[object, object]:
    if x1.shape[axis] != x0.shape[axis]:
        msg = "{n0} != {n1}!".format(n0=x0.shape[axis], n1=x1.shape[axis])
        raise ValueError(msg)
    return x0, x1


########################################################################################################################
# DATA WRANGLING #######################################################################################################
########################################################################################################################

def _one_hotify(_y: object, _y_options: list) -> pd.Series:
    """
    Convert a single category label to one-hot vector representation.

    E.g.
    >>> _one_hotify(2, range(4))
    pd.Series({0: 0, 1: 0, 2: 1, 3: 0})
    >>> _one_hotify('Away', ['Home', 'Away'])
    pd.Series({'Home': 0, 'Away': 1})

    input
    -----
    _y: object, e.g. int or str, the correct category label.

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
        For example, if your category labels are ['Home', 'Away'], that might be a more
        natural order than the default alphabetical sort ['Away', 'Home'].
    """
    _y_options = sorted(y.unique()) if _y_options is None else _y_options
    _ = check_subset(sub=set(y.unique()), sup=set(_y_options))

    y = y.apply(_one_hotify, _y_options=_y_options)
    return check_one_hot(y)


def split_shuffle(*arrays: Tuple[Arraylike], n: int, random_seed: int=1337) -> Tuple[List[Arraylike]]:
    """
    Shuffle then split each array, consistently with the others.

    E.g.
    >>> y, X = split_shuffle(y, X, n=3)
    >>> y_batch_0, X_batch_0 = y[0], X[0]
    >>> y_batch_2, X_batch_2 = y[-1], X[-1]
    """
    arrays = shuffle(*arrays, random_state=random_seed)
    # tuple more space-efficient
    arrays = tuple([np.array_split(ary=ary, indices_or_sections=n) for ary in arrays])
    return arrays


########################################################################################################################
# | |- || |_ ###########################################################################################################
########################################################################################################################

r"""
Two common loss functions are MSE and cross-entropy AKA "log loss"[1].

One reason we find the MSE intuitively appealing is that
it more severely penalizes "confident" wrong answers.
E.g. suppose the correct category label is A, the possible labels are [A, B, C],
and we assign Pr[A] := 50% and Pr[B] := 25% =: Pr[C].
The MSE will get worse if we instead assign Pr[B] := 49% and Pr[C] := 1%.
On the other hand the cross-entropy would stay the same.
I'd argue that the second assignment is truly "worse", in that
although we'd still pick the correct label (A) if forced to choose,
we're now willing to wager almost as much on an incorrect label (B)
as we are on the correct label.

With that said, in general maximum-likelihood estimators have nice properties and whereas
in the linear regression setting minimizing MSE (the basis of OLS) yields an MLE, in the
classification setting minimizing cross-entropy[2, 3] yields an MLE.

One legitimate additional option (not implemented) that can help combat
overfitting is to also "regularize" weights by penalizing deviations from zero.
This is like LASSO or Ridge or ElasticNet OLS regression.

Footnote
--------

[1] Why do people sometimes call cross-entropy as "log loss"?
This is to stand for "negative log likelihood". Indeed, in the standard
classification setting, for a single data point, cross-entropy is
the same as negative log likelihood. E.g. if we encode the ground-truth
correct label as a one-hot Kronecker delta (a degenerate probability mass function),
and assign the correct label a probability of 40%, our cross-entropy
is simply -log(0.4), since the incorrect terms have a ground-truth
probability factor of zero hence drop out of the sum.

In suggestively-aligned psuedocode,
ground_truth  :=      [0          , 0          , 1          , 0        ]
prediction    :=      [      0.1  ,       0.3  ,       0.4  ,       0.2]
cross_entropy := -sum([0*log(0.1) + 0*log(0.3) + 1*log(0.4) + 0*log(0.2)])
               = -                                 log(0.4).

Over many data points, the conventional definition for cross-entropy
is not exactly the negative log-likelihood, but rather the
negative arithmetic-mean log-likelihood, which is the negative
log geometric-mean likelihood, which is the log reciprocal geometric-mean likelihood.

In pseudoTeX,
negative arithmetic-mean log-likelihood := - (1/N) \sum_n { \log p_n }
                                         = - (1/N) \log { \prod_n p_n }
                                         = - \log { (\prod_n p_n) ^ {1/N} }
                                         = \log { (\prod_n p_n) ^ {-1/N} }
                                         = \log { 1 / (\prod_n p_n) ^ {1/N} }
                                         =: log reciprocal geometric-mean likelihood.


[2] Why not simply maximize the likelihood? Well, conventional
optimization problem formulations in CS tend to be couched in terms
of minimization, so we just negate it.


[3] Why not simply operate on the raw likelhood? Raw likelihoods
can be numerically unstable, since many small probabilities
multiplied together can become impractically small. One solution
is to scale each probability by some constant factor before taking the product,
but the conventional solution is to operate on the log of the product, which takes in
values below unity and "squashes them out" from zero down to negative infinity.
"""

def _get_cross_entropy(_y: Union[pd.Series, pd.DataFrame], p_y: Union[pd.Series, pd.DataFrame]) -> float:
    """
    Cross entropy AKA negative log likelihood.
    For either a single or many data points.
    For many data points, pass DataFrames (index = observations).

    input
    -----
    _y: Union[pd.Series, pd.DataFrame], the ground-truth probability distribution.
        For use as NN loss, this should be one-hot encoding of correct category label.

    p_y: Union[pd.Series, pd.DataFrame], the probability assigned to each category label.

    output
    ------
    float, the cross-entropy.
    """
    ce = -sum(_y * np.log(p_y))  # for single data point, same as -_y.dot(np.log(p_y))
    return check_type(ce, type_=float)


def get_cross_entropy(y: pd.DataFrame, p: pd.DataFrame, normalize: bool=True) -> float:
    """
    Cross entropy AKA negative log likelihood.

    input
    -----
    y: pd.DataFrame (index = observations, columns = category labels) (one-hot),
        the correct category label for each point.

    p: pd.DataFrame (index = observations, columns = category labels),
        how much probability mass we assigned to each category label for each point.
        Each row should be a well-formed probability mass function
        AKA discrete probability distribution.

    normalize: bool (default True), whether to divide by |dataset|;
        False -> joint cross-entropy, True -> mean cross-entropy.

    output
    ------
    float, the (joint or mean) cross-entropy.
    """
    y = check_one_hot(y=y)
    p = p.apply(check_pmf, axis="columns")

    ce = _get_cross_entropy(_y=y, p_y=p)
    return ce / y.shape[0] if normalize else ce


def crossmax(_y: pd.Series, x: pd.Series) -> float:
    """
    Get cross-entropy of softmax of `x`.

    input
    -----
    _y: pd.Series, the ground-truth probability distribution.
        For use as NN loss, this should be one-hot encoding of correct category label.

    x: pd.Series, incoming data.

    output
    ------
    float, the cross-entropy of the softmax.
    """
    return _get_cross_entropy(_y=_y, p_y=softmax(x))


########################################################################################################################
# GRADIENT #############################################################################################################
########################################################################################################################

def d_dx(x: Floatvec, fn: Callable, _y: Optional=None) -> Floatvec:
    """
    Vectorized (partial) derivative, i.e. d(x, fn) = [d/dx_i fn(x)_i for i in range(len(x))].

    proof
    -----
    expit(x)                 := (1 + exp(-x))^{-1}
    d/dx expit(x)             = -1 (1 + exp(-x))^{-2} * exp(-x) * -1  # chain rule
                              = (1 + exp(-x))^{-2} * exp(-x)
                              = (1 + exp(-x))^{-1} * exp(-x) / (1 + exp(-x))
                              = expit(x) * (exp(-x) + 1 - 1) / (1 + exp(-x))
                              = expit(x) * [ (exp(-x) + 1) / (1 + exp(-x)) - 1 / (1 + exp(-x)) ]
                              = expit(x) * [ (1 + exp(-x)) / (1 + exp(-x)) - (1 + exp(-x))^{-1} ]
                              = expit(x) * (1 - expit(x)).

    softmax(x)               := exp(x) / sum(exp(x))
    d/dx softmax(x)           = [sum(exp(x)) d/dx exp(x) - exp(x) d/dx sum(exp(x))] / sum(exp(x))^2  # quotient rule
                              = [sum(exp(x)) * exp(x) - exp(x) * exp(x)] / sum(exp(x))^2
                              = exp(x) * [sum(exp(x)) - exp(x)] / sum(exp(x))^2
                              = exp(x) * [sum(exp(x)) / sum(exp(x))^2 - exp(x) / sum(exp(x))^2]
                              = exp(x) * [1 / sum(exp(x)) - softmax(x) / sum(exp(x))]
                              = exp(x) / sum(exp(x)) * (1 - softmax(x))
                              = softmax(x) * (1 - softmax(x)).

    cross_entropy(_y, a)     := -sum(_y * log(a))
    d/da cross_entropy(_y, a) = -_y/a.

    crossmax(_y, x)          := cross_entropy(_y, a)  # letting a:= softmax(x)
    d/dx crossmax(_y, x)      = d/da cross_entropy(_y, a) * d/dx a  # chain rule
                              = -_y/a * a * (1 - a)
                              = -_y * (1 - a)
                              = _y * (a - 1)
                              = _ya - _y
                              = [a_i - _y_i if i == true_label else 0 for i in range(len(x))].
    This last one is *almost* right.. we're just failing to account for the cross-derivative:
    This is a vector-valued function with vector-valued inputs.
    Above, when we calculated d/dx softmax(x), we calculated only [d/dx_i softmax(x)_i for i in range(len(x))], but
    in reality we of course have a Jacobian [[d/dx_i softmax(x)_j for i in range(len(x))] for j in range(len(x))].
    If you do out that calculation, you arrive at a - _y, which is simply [a_i - _y_i for i in range(len(x))].
    """
    if fn in (expit, softmax):
        return fn(x) * (1 - fn(x))
    elif fn is crossmax:
        return softmax(x) - _y
    else:
        raise NotImplementedError(fn)
