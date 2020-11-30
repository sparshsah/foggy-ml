"""
Arbitrary-depth, arbitrary-width feedforward artificial neural network.
Step-by-step Python implementation of simple deep learning for binary classification.
Source code in `fann.py`, demo in `fann.ipynb`.
"""

# data structures
import pandas as pd
# algorithms
import numpy as np
from scipy.special import expit


########################################################################################################################
# NN OBJECT ############################################################################################################
########################################################################################################################

"""
Fixing the neuron activation---in our case, logistic---and output squashing---in our case, softmax---function,
an ANN is essentially identified by its forward-propagation weights.

We store the model as a pd.DataFrame with MultiIndex. Each "super-row" (axis=0, level=0) represents a layer.
Each row (axis=0, level=1) represents the weights feeding into a single neuron on that layer.
(E.g. the first row represents the weights feeding from the input layer into
the first neuron on the first hidden layer.) The first column is always reserved for the bias term.

Because different layers can have different widths, some rows may not be completely filled across.
But obviously, for neurons on the same layer, the number of neurons on the previous layer is also the same.
Hence, any two rows on the same "super-row" (i.e. sharing a key on axis=0 & level=0), will be filled to the same width.
"""

def _fprop(w: pd.Series, x: pd.Series) -> pd.Series:
    """Recursive helper function."""
    raise NotImplementedError


def fprop(nn: pd.DataFrame, x: pd.Series) -> pd.Series:
    """
    Forward propagate.

    input
    -----
    nn: pd.DataFrame, the model.

    x: pd.Series, a single input data point.

    output
    ------
    pd.Series, the probability the model assigns to each category label.
    """
    raise NotImplementedError


def _predict(nn: pd.DataFrame, x: pd.Series) -> int:
    """
    Predict which category the input point belongs to.

    input
    -----
    nn: pd.DataFrame, the model.

    x: pd.Series, a single input data point.

    output
    ------
    label (usually int), a scalar value identifiying which category we predict.
    """
    return fprop(nn=nn, x=x).idxmax()  # essentially argmax


def predict(nn: pd.DataFrame, X: pd.DataFrame) -> pd.Series:
    """Predict which category each input point belongs to."""
    _predict = lambda x: _predict(nn=nn, x=x)
    return X.apply(_predict, axis="columns")


########################################################################################################################
# ACTIVATION FUNCTION ##################################################################################################
########################################################################################################################

"""
Three common neuron activation functions are logistic (AKA sigmoid), tanh, and ReLU.
We like the logistic for (binary) classification tasks, for the slightly (maybe even misleadingly) arbitrary reason
that its output lies in [0, 1], so it can be interpreted as this neuron's "best guess"
of the probability that the input belongs to Category 1. As further support, the logistic's generalization, the softmax,
is a commonly-used "output squashing" function for multinomial classification tasks: It transforms a `n`-vector
of Real numbers into a probability mass distribution. (And tanh is simply a scaled-and-shifted version of logistic.)
ReLU is cool too, and has another cool connection, this time to the hinge loss function.
"""


########################################################################################################################
# LOSS FUNCTION ########################################################################################################
########################################################################################################################

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
    return np.log(p).sum()


def _get_loss(p_y: pd.Series) -> float:
    """
    Negative log likelihood loss.

    input
    -----
    p_y: pd.Series, how much probability mass we assigned to the correct label for each point.
        E.g. if p_y = [0.10, 0.85, 0.50], then there were 2 data points. For the first point,
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
    return -get_llh(p=p_y)


def get_loss(p_hat: pd.DataFrame, y: pd.Series) -> float:
    """
    Negative log likelihood loss.

    input
    -----
    p_hat: pd.DataFrame (columns = category labels, index = observations),
        how much probability mass we assigned to each category label for each point.
        Each row should be a well-formed probability mass function
        AKA discrete probability distribution.

    y: pd.Series, the correct category label for each point.

    output
    ------
    float, the calculated loss.
    """
    # pick out the entry for the correct label in each row
    p_y = pd.Series({n: p_hat.loc[n, label] for n, label in y.items()})
    return _get_loss(p_y=p_y)
