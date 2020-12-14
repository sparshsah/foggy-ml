"""
Arbitrary-depth, arbitrary-width feedforward artificial neural network.
Easy-to-read Python implementation of deep learning for multinomial classification.
"""

# syntax utils
from typing import List, Callable, Union
import util
# data structures
import pandas as pd
# calculations and algorithms
import numpy as np
from scipy.special import expit, softmax
import loss


########################################################################################################################
# NN OBJECT ############################################################################################################
########################################################################################################################

"""
Fixing the neuron activation---in our case, logistic---and output squashing---in our case, softmax---function,
a NN model is essentially identified by its feed-forward AKA forward-pass AKA forward-propagation weights.

We store the model as a pd.DataFrame with MultiIndex. Each "super-row" (axis=0, level=0) represents a layer.
Each row (axis=0, level=1) represents the weights feeding into a single neuron on that layer.
(E.g. the first row represents the weights feeding from the input layer into
the first hidden layer's first neuron; the second row---if applicable---could represent
the weights feeding from the input layer into the first hidden layer's second neuron.)
The first column (indexed as "_bias_" for compatibility reasons) is always reserved for the bias term.

Because different layers can have different widths, some rows may not be completely filled across.
But obviously, for neurons on the same layer, the number of neurons on the previous layer is also the same.
Hence, any two rows on the same "super-row" or "block" (i.e. sharing a key on axis=0),
will be filled to the same width.
"""

# types
Neuron: type = pd.Series
Layer: type = pd.DataFrame
NN: type = pd.DataFrame  # w/ MultiIndex[layers, neurons]

# magic numbers
NN_INDEX_NLEVELS: int = 2  # MultiIndex[layers, neurons]
BIAS_INDEX: Union[int, str] = "_bias_"


# type checkers

def check_data_point(x: object) -> pd.Series:
    util.check_type(x, pd.Series)
    util.check_not_type(x.index, pd.MultiIndex)
    if BIAS_INDEX in x.index:
        raise ValueError("Data point \n{x}\n contains reserved index {i}!".format(x=x, i=BIAS_INDEX))
    util.check_dtype(x, float)
    return x


def check_neuron(neuron: object) -> Neuron:
    util.check_type(neuron, Neuron)
    util.check_not_type(neuron.index, pd.MultiIndex)
    if BIAS_INDEX not in neuron.index:
        raise ValueError("Neuron \n{neuron}\n missing bias index {i}!".format(neuron=neuron, i=BIAS_INDEX))
    util.check_dtype(neuron, float)
    return neuron


def check_layer(layer: object) -> Layer:
    util.check_type(layer, Layer)
    util.check_not_type(layer.index, pd.MultiIndex)
    util.check_not_type(layer.columns, pd.MultiIndex)
    util.check_dtype(layer, float)
    return layer


def check_nn(nn: object) -> NN:
    util.check_type(nn, NN)
    util.check_type(nn.index, pd.MultiIndex)
    # levels[0] indexes the layers, levels[1] indexes the neurons on each layer
    if nn.index.nlevels != NN_INDEX_NLEVELS:
        raise ValueError("NN \n{nn}\n index nlevels = {nlevels} not {nlevels_}!".format(
            nn=nn, nlevels=nn.index.nlevels, nlevels_=NN_INDEX_NLEVELS))
    util.check_not_type(nn.columns, pd.MultiIndex)
    util.check_dtype(nn, float)
    return nn


# check-and-return calculation utils

def nnify(nn: List[Layer]) -> NN:
    nn = [check_layer(layer=layer) for layer in nn]
    nn = pd.concat(nn, keys=range(len(nn)))
    nn = check_nn(nn=nn)
    return nn


def get_bias(neuron: Neuron) -> float:
    """Get bias weight from neuron."""
    # neuron = check_neuron(neuron=neuron)

    bias = neuron[BIAS_INDEX]
    bias = util.check_type(bias, float)
    if pd.isnull(bias):
        raise ValueError("Neuron \n{neuron}\n missing bias!".format(neuron=neuron))
    return bias


def get_w_in(x: pd.Series, neuron: Neuron) -> pd.Series:
    """Get feed-in weights from neuron."""
    # x = check_data_point(x=x)
    # neuron = check_neuron(neuron=neuron)

    w_in = neuron.reindex(index=x.index)
    if w_in.isnull().any():
        raise ValueError("Feed-in weights \n{w_in}\n not completely filled!".format(w_in=w_in))

    w_pad = neuron.drop(labels=BIAS_INDEX).drop(labels=x.index)
    if w_pad.notnull().any():
        raise ValueError("\n{w_pad}\n contains value(s), but should be just padding!".format(w_pad=w_pad))

    return w_in


def get_a_in(x: pd.Series, w_in: pd.Series) -> float:
    """Get incoming activation."""
    # x = check_data_point(x=x)
    # w_in = util.check_type(w_in, pd.Series)

    a_in = x.dot(w_in)
    a_in = util.check_type(a_in, float)
    return a_in


def get_a_out(bias: float, a_in: float, fn: Callable[[float], float]) -> float:
    """Get outgoing activation."""
    # bias = util.check_type(bias, float)
    # a_in = util.check_type(a_in, float)
    # fn = util.check_type(fn, Callable[[float], float])

    a_out = fn(bias + a_in)
    a_out = util.check_type(a_out, float)
    return a_out


########################################################################################################################
# ACTIVATION AND SQUASHING #############################################################################################
########################################################################################################################

"""
Three common neuron activation functions are logistic (AKA expit AKA inverse-logit AKA sigmoid), tanh, and ReLU.

We like the logistic for (binary) classification tasks, for the slightly arbitrary reason
that its output lies in [0, 1], so it can potentially be interpreted as this neuron's "best guess"
of the probability that the input belongs to Category 1. As further support, the logistic's generalization, the softmax,
is a commonly-used "output squashing" function for multinomial classification tasks: It transforms a `n`-vector
of Real numbers into a probability mass distribution. (And tanh is simply a scaled-and-shifted version of logistic.)

ReLU (not implemented) is cool too, and has another cool connection, this time to the hinge loss function.
"""

activate: Callable[[float], float] = expit

squash: Callable[[pd.Series], pd.Series] = softmax  # function[[pd.Series[float]] -> pd.Series[float]]


########################################################################################################################
# FEED FORWARD AKA FORWARD PASS AKA FORWARD PROPAGATION ################################################################
########################################################################################################################

def ___fprop(x: pd.Series, neuron: Neuron, fn: Callable[[float], float]=activate) -> float:
    """
    Forward-propagate the previous layer's output with the current neuron's weights and activation function.

    input
    -----
    x: pd.Series, the previous layer's output (possibly a raw data point,
        which can be seen as the input layer's "output"), where each entry correponds to
        a neuron on the previous layer.

    neuron: Neuron, the current neuron's weights, where each entry corresponds to
        (the bias or) a neuron on the previous layer.

    fn: function, the current neuron's activation function.

    output
    ------
    float: the current neuron's output.
    """
    x = check_data_point(x=x)
    neuron = check_neuron(neuron=neuron)
    # fn = util.check_type(fn, Callable[[float], float])

    bias = get_bias(neuron=neuron)
    w_in = get_w_in(x=x, neuron=neuron)
    a_in = get_a_in(x=x, w_in=w_in)
    a_out = get_a_out(bias=bias, a_in=a_in, fn=fn)
    return a_out


def __fprop(x: pd.Series, layer: Layer) -> pd.Series:
    """
    Forward-propagate the previous layer's output with the current layer's weights and activation function.

    input
    -----
    x: pd.Series, the previous layer's output (possibly a raw data point,
        which can be seen as the input layer's "output"), where each entry correponds to
        a neuron on the previous layer.

    layer: Layer, the current layer's weights, where each row corresponds to
        a neuron on the current layer and each column corresponds to
        (the bias or) a neuron on the previous layer.

    output
    ------
    pd.Series (index = layer.index), the current layer's output, where each entry corresponds to
        a neuron on the current layer.
    """
    x = check_data_point(x=x)
    layer = check_layer(layer=layer)
    return layer.apply(lambda neuron: ___fprop(x=x, neuron=neuron), axis="columns")


def _fprop(x: pd.Series, nn: NN) -> pd.Series:
    """
    Forward-propagate the input through the network.

    input
    -----
    x: pd.Series, a single raw data point.

    nn: NN, the model.

    output
    ------
    pd.Series, the final layer's output.
    """
    x = check_data_point(x=x)
    nn = check_nn(nn=nn)

    # levels[0] indexes the layers, levels[1] indexes the neurons on each layer, so
    # this is basically a list of (names of) layers in this NN e.g. [0, 1, 2, ..].
    # pd.MultiIndex "remembers" old, unused levels even after you drop all rows that used those levels.
    layers = nn.index.remove_unused_levels().levels[0]

    curr_layer = layers[0]
    # we want to "squeeze" the MultiIndex i.e. we want indices to be
    # not `[(curr_layer, 0), (curr_layer, 1), (curr_layer, 2), ..]` but rather `[0, 1, 2, ..]`,
    # so don't use `curr_layer = nn.loc[pd.IndexSlice[curr_layer, :], :]`.
    curr_layer = nn.loc[curr_layer]
    curr_layer = check_layer(layer=curr_layer)
    x = __fprop(x=x, layer=curr_layer)

    # recurse
    remainining_layers = layers[1:]
    if len(remainining_layers) > 0:
        remainining_layers = nn.loc[pd.IndexSlice[remainining_layers, :], :]
        remaining_layers = check_nn(nn=remainining_layers)
        return _fprop(x=x, nn=remainining_layers)
    else:
        # this was the final i.e. output layer
        return x


def fprop(x: pd.Series, nn: NN) -> pd.Series:
    """
    Forward-propagate the input through the network.

    input
    -----
    x: pd.Series, a single raw data point.

    nn: NN, the model.

    output
    ------
    pd.Series, the probability the model assigns to each category label.
    """
    x = check_data_point(x=x)
    nn = check_nn(nn=nn)
    return squash(_fprop(x=x, nn=nn))


def fprop_(X: pd.DataFrame, nn: NN) -> pd.DataFrame:
    """
    Forward-propagate each input through the network.
    Could be done more efficiently with some clever linear algebra, but this does the job.

    input
    -----
    x: pd.DataFrame, the raw data points where each row is an observation.

    nn: NN, the model.

    output
    ------
    pd.DataFrame (index = observations, columns = category labels),
        how much probability mass we assigned to each category label for each point.
        Each row is a well-formed probability mass function AKA discrete probability distribution.
    """
    X = X.apply(check_data_point, axis="columns")
    nn = check_nn(nn=nn)

    p_hat = X.apply(lambda x: fprop(x=x, nn=nn), axis="columns")
    p_hat = p_hat.apply(util.check_pmf, axis="columns")
    return p_hat


def predict(X: pd.DataFrame, nn: NN) -> pd.Series:
    """
    Predict which category each input point belongs to.

    input
    -----
    X: pd.DataFrame, the raw data points where each row is a single observation.

    nn: NN, the model.

    output
    ------
    pd.Series of int (or other scalar value, or whatever the user has used),
        a single label per point identifiying which category we predict for that point.
    """
    X = X.apply(check_data_point, axis="columns")
    nn = check_nn(nn=nn)

    p_hat = fprop_(X=X, nn=nn)
    p_hat = p_hat.apply(util.check_pmf, axis="columns")
    return p_hat.apply(lambda _p_hat: _p_hat.idxmax(), axis="columns")  # argmax of each row


########################################################################################################################
# TRAINING AKA BACK-PROPAGATION ########################################################################################
########################################################################################################################

"""
some tutorials for basic 2-layer perceptrons:
https://towardsdatascience.com/how-to-build-your-own-neural-network-from-scratch-in-python-68998a08e4f6
https://towardsdatascience.com/step-by-step-guide-to-building-your-own-neural-network-from-scratch-df64b1c5ab6e
https://towardsdatascience.com/how-to-build-a-deep-neural-network-without-a-framework-5d46067754d5
"""

def bprop():
    # This function learns parameters for the neural network and returns the model.
    # - nn_hdim: Number of nodes in the hidden layer
    # - num_passes: Number of passes through the training data for gradient descent
    # Initialize the parameters to random values. We need to learn these.
    nn_hdim = 3
    num_passes = 20_000
    np.random.seed(1337)
    W1 = np.random.randn(nn_input_dim, nn_hdim) / np.sqrt(nn_input_dim)
    b1 = np.zeros((1, nn_hdim))
    W2 = np.random.randn(nn_hdim, nn_output_dim) / np.sqrt(nn_hdim)
    b2 = np.zeros((1, nn_output_dim))

    # This is what we return at the end
    model = {}

    # Gradient descent. For each batch...
    for i in range(0, num_passes):

        # Forward propagation
        z1 = X.dot(W1) + b1
        a1 = np.tanh(z1)
        z2 = a1.dot(W2) + b2
        exp_scores = np.exp(z2)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

        # Backpropagation
        delta3 = probs
        delta3[range(num_examples), y] -= 1
        dW2 = (a1.T).dot(delta3)
        db2 = np.sum(delta3, axis=0, keepdims=True)
        delta2 = delta3.dot(W2.T) * (1 - np.power(a1, 2))
        dW1 = np.dot(X.T, delta2)
        db1 = np.sum(delta2, axis=0)

        # Gradient descent parameter update
        W1 += -epsilon * dW1
        b1 += -epsilon * db1
        W2 += -epsilon * dW2
        b2 += -epsilon * db2

        # Assign new parameters to the model
        model = { 'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}

    return model
