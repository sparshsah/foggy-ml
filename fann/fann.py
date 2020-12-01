"""
Arbitrary-depth, arbitrary-width feedforward artificial neural network.
Step-by-step Python implementation of simple deep learning for multinomial classification.
Source code in `fann.py`, demo in `fann.ipynb`.
"""

# data structures
import pandas as pd
# calculations and algorithms
import numpy as np
from scipy.special import expit, softmax
# syntax utils
from typing import Callable


########################################################################################################################
# NN OBJECT ############################################################################################################
########################################################################################################################

"""
Fixing the neuron activation---in our case, logistic---and output squashing---in our case, softmax---function,
a NN model is essentially identified by its forward-propagation weights.

We store the model as a pd.DataFrame with MultiIndex. Each "super-row" (axis=0, level=0) represents a layer.
Each row (axis=0, level=1) represents the weights feeding into a single neuron on that layer.
(E.g. the first row represents the weights feeding from the input layer into
the first neuron on the first hidden layer.) The first column (indexed as -1 for compatibility reasons)
is always reserved for the bias term.

Because different layers can have different widths, some rows may not be completely filled across.
But obviously, for neurons on the same layer, the number of neurons on the previous layer is also the same.
Hence, any two rows on the same "super-row" (i.e. sharing a key on axis=0 & level=0), will be filled to the same width.
"""

NN = pd.DataFrame  # w/ MultiIndex[layers, neurons]


########################################################################################################################
# ACTIVATION AND SQUASHING #############################################################################################
########################################################################################################################

"""
Three common neuron activation functions are logistic (AKA expit AKA inverse-logit AKA sigmoid), tanh, and ReLU.

We like the logistic for (binary) classification tasks, for the slightly (maybe even misleadingly) arbitrary reason
that its output lies in [0, 1], so it can be interpreted as this neuron's "best guess"
of the probability that the input belongs to Category 1. As further support, the logistic's generalization, the softmax,
is a commonly-used "output squashing" function for multinomial classification tasks: It transforms a `n`-vector
of Real numbers into a probability mass distribution. (And tanh is simply a scaled-and-shifted version of logistic.)

ReLU is cool too, and has another cool connection, this time to the hinge loss function.
"""

activate = expit

squash = softmax


########################################################################################################################
# FORWARD PROPAGATION ##################################################################################################
########################################################################################################################

# controversial, but we like the clarity and modularity of a bunch of underscored one-liner helpers

def ___fprop(x: pd.Series, w_neuron: pd.Series, fn: Callable[[float], float]=activate) -> float:
    """
    Forward-propagate the previous layer's output with the current neuron's weights and activation function.

    input
    -----
    x: pd.Series, the previous layer's output (possibly a single input data point,
        which can be seen as the first layer's "output").

    w_neuron: pd.Series, the current neuron's weights.

    fn: function, the current neuron's activation function.

    output
    ------
    float: the current neuron's output.
    """
    assert isinstance(x, pd.Series), type(x)
    assert isinstance(w_neuron, pd.Series), type(w_neuron)
    bias = w_neuron[-1]
    assert pd.notnull(bias), "Weights {w} missing bias!".format(w=w_neuron)
    w_neuron = w_neuron.reindex(index=x.index)
    assert w_neuron.notnull().all(), "Weights {w} are not completely filled!".format(w=w_neuron)

    return fn(bias + x.dot(w_neuron))


def __fprop(x: pd.Series, w_layer: pd.DataFrame) -> pd.Series:
    """
    Forward-propagate the previous layer's output with the current layer's weights and activation function.

    input
    -----
    x: pd.Series, the previous layer's output (possibly a single input data point,
        which can be seen as the first layer's "output").

    w_layer: pd.DataFrame, the current layer's weights where each row corresponds to a neuron.

    output
    ------
    pd.Series, the current layer's output.
    """
    return w_layer.apply(lambda w_neuron: ___fprop(x=x, w_neuron=w_neuron), axis="columns")


def _fprop(x: pd.Series, nn: NN) -> pd.Series:
    """
    Forward-propagate the input through the network.

    input
    -----
    x: pd.Series, a single input data point.

    nn: NN AKA pd.DataFrame w/ MultiIndex, the model.

    output
    ------
    pd.Series, the last layer's output.
    """
    layers = nn.index.levels[0]
    curr_layer = layers[0]
    x = __fprop(x=x, w_layer=nn.loc[pd.IndexSlice[curr_layer, :], :])
    # recurse
    next_layers = layers[1:]
    return _fprop(x=x, nn=nn.loc[pd.IndexSlice[next_layers, :], :]) if len(next_layers) > 0 else x


def fprop(x: pd.Series, nn: NN) -> pd.Series:
    """
    Forward-propagate the input through the network.

    input
    -----
    x: pd.Series, a single input data point.

    nn: NN AKA pd.DataFrame w/ MultiIndex, the model.

    output
    ------
    pd.Series, the probability the model assigns to each category label.
    """
    return squash(_fprop(x=x, nn=nn))


def _predict(x: pd.Series, nn: NN) -> int:
    """
    Predict which category the input point belongs to.

    input
    -----
    x: pd.Series, a single input data point.

    nn: NN AKA pd.DataFrame w/ MultiIndex, the model.

    output
    ------
    int (or other scalar value, or whatever the user has used), a label identifiying which category we predict.
    """
    return fprop(nn=nn, x=x).idxmax()  # essentially argmax


def predict(X: pd.DataFrame, nn: NN) -> pd.Series:
    """
    Predict which category each input point belongs to.

    input
    -----
    X: pd.DataFrame, the input data points where each row is a single observation.

    nn: NN AKA pd.DataFrame w/ MultiIndex, the model.

    output
    ------
    pd.Series of int (or other scalar value, or whatever the user has used),
        labels identifiying which category we predict for each point.
    """
    _predict = lambda x: _predict(x=x, nn=nn)  # intentionally shadows name from outer scope
    return X.apply(_predict, axis="columns")


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


def get_loss(y: pd.Series, p_hat: pd.DataFrame) -> float:
    """
    Negative log likelihood loss (normalized by |training data|).

    input
    -----
    y: pd.Series, the correct category label for each point.

    p_hat: pd.DataFrame (columns = category labels, index = observations),
        how much probability mass we assigned to each category label for each point.
        Each row should be a well-formed probability mass function
        AKA discrete probability distribution.

    output
    ------
    float, the calculated loss.
    """
    # pick out the entry for the correct label in each row
    p_y = pd.Series({n: p_hat.loc[n, label] for n, label in y.items()})
    return _get_loss(p_y=p_y) / y.count()


########################################################################################################################
# TRAINING AKA BACK-PROPAGATION ########################################################################################
########################################################################################################################
