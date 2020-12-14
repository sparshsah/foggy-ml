"""
One choice (not implemented) that can help combat overfitting is
to "regularize" parameters by penalizing deviations from zero.
This is like LASSO or Ridge or ElasticNet OLS regression.
"""

# syntax utils
import util
# data structures
import pandas as pd
# calculations and algorithms
import numpy as np


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
    p = util.check_pmf(pmf=p)

    llh = np.log(p).sum()
    llh = util.check_type(llh, float)
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
    loss = util.check_type(loss, float)
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
    y = util.check_type(y, pd.Series)
    p_hat = util.check_type(p_hat, pd.DataFrame)
    p_hat = p_hat.apply(util.check_pmf, axis="columns")

    # pick out the entry for the correct label in each row
    p_y = pd.Series({n: p_hat.loc[n, label] for n, label in y.items()})
    loss = _get_loss(p_y=p_y) / y.count()
    loss = util.check_type(loss, float)
    return loss
