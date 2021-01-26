"""
Arbitrary-depth, arbitrary-width feedforward artificial neural network.
Easy-to-read, step-by-step Python implementation of deep learning for multinomial classification.

See style notes in top-level repository README.md.
"""

# syntax utils
from typing import List, Iterable, Callable, Union
# data structures
import pandas as pd
# data wrangling
from .. import util
# calculations and algorithms
import numpy as np
from scipy.special import expit, softmax  # linter can't see C funcs, so pylint: disable=no-name-in-module
from ..util import get_neg_llh as get_loss

__all__ = [
    # data structures
    "Neuron", "Layer", "NN",
    # magic numbers
    "NN_INDEX_NLEVELS", "BIAS_INDEX",
    # data wrangling
    "check_data_point", "check_neuron", "check_layer", "check_nn",
    "nnify", "get_bias", "get_w_in", "get_a_in", "get_a_out",
    # calculations and algorithms
    "activate", "squash",
    "____fprop", "___fprop", "__fprop", "_fprop", "fprop", "predict",
]

########################################################################################################################
# NN OBJECT ############################################################################################################
########################################################################################################################

"""
Fixing the neuron activation---in our case, logistic---and output squashing---in our case, softmax---function,
a FANN is essentially identified by its feed-forward AKA forward-pass AKA forward-propagation weights.

We store the model as a pd.DataFrame with MultiIndex.
Each "block" or "super-row" (i.e. collection of rows sharing an index key on axis=0, level=0)
represents a layer.
Each row (having a unique pair of index keys on axis=0, levels=[0, 1])
represents the weights feeding into a single neuron on that layer.
The first column (having index key '_bias_' on axis=1)
is always reserved for the bias term.

E.g. the first row represents the bias/weights feeding from the input layer into
the first hidden layer's first neuron; the second row---if applicable---could represent
the bias/weights feeding from the input layer into the first hidden layer's second neuron.
"""

# types
Neuron: type = pd.Series
Layer: type = pd.DataFrame
NN: type = pd.DataFrame  # w/ MultiIndex[layers, neurons]

# magic numbers
NN_INDEX_NLEVELS: int = 2  # MultiIndex[layers, neurons]
BIAS_INDEX: Union[int, str] = "_bias_"


# initialize

def init_neuron():
    raise NotImplementedError


def init_layer():
    raise NotImplementedError


def init_nn(input_width: int, layer_width: Union[int, Iterable[int]], output_width=int, random_seed=1337) -> NN:
    """
    Initialize an NN with Standard Normal random weights.

    input
    -----
    intput_width: int, how many features. E.g. `3` for points in three-dimensional space.

    layer_width: Union[int, Iterable[int]], the width(s) of the hidden layer(s).
        E.g. `4` for a single hidden layer with 4 neurons,
        or `(4, 5)` for a hidden layer of 4 neurons followed by a hidden layer of 5 neurons.

    output_width: int, how many categories. E.g. `2` for binomial classification.

    random_seed: int, the random seed.

    output
    ------
    NN, a new Neural Network object.
    """
    # setup loop control
    layer_width = [layer_width,] if isinstance(layer_width, int) else list(layer_width)
    layer_width = [input_width,] + layer_width + [output_width,]
    del output_width, input_width
    layer_width = pd.Series(layer_width)
    layer_width = pd.concat([layer_width, layer_width.shift()],
                            axis="columns", keys=["curr", "prev"])

    rng = np.random.default_rng(seed=random_seed)
    layers = [Layer(rng.standard_normal(size=(width["curr"], width["prev"])))
              for _, width in layer_width.iloc[1:].iterrows()]
    return nnify(nn=layers)


# type checkers

def check_data_point(x: object) -> pd.Series:
    util.check_type(x, pd.Series)
    util.check_not_type(x.index, pd.MultiIndex)
    if BIAS_INDEX in x.index:
        raise ValueError("Data point \n{x}\n contains reserved index `{i}`!".format(x=x, i=BIAS_INDEX))
    util.check_dtype(x, float)
    return x


def check_neuron(neuron: object) -> Neuron:
    util.check_type(neuron, Neuron)
    util.check_not_type(neuron.index, pd.MultiIndex)
    if BIAS_INDEX not in neuron.index:
        raise ValueError("Neuron \n{neuron}\n missing bias index `{i}`!".format(neuron=neuron, i=BIAS_INDEX))
    util.check_dtype(neuron, float)
    return neuron


def check_layer(layer: object) -> Layer:
    util.check_type(layer, Layer)
    util.check_not_type(layer.index, pd.MultiIndex)
    util.check_not_type(layer.columns, pd.MultiIndex)
    layer.apply(check_neuron, axis="columns")
    """
    Because different layers can have different widths, some rows may not be completely filled across.
    But obviously, for neurons on the same layer, the number of neurons on the previous layer is also the same.
    Hence, any two rows on the same layer must be filled to the same width.
    """
    layer_cols = layer.dropna(how="all", axis="columns").columns  # names of neurons on incoming Layer
    for _, neuron in layer.iterrows():
        neuron = check_neuron(neuron=neuron)
        neuron_cols = neuron.dropna().index
        if not neuron_cols.equals(layer_cols):
            raise ValueError(
                "Neuron's weights not filled across its full Layer.. {neuron_cols} != {layer_cols}!".format(
                    neuron_cols=neuron_cols, layer_cols=layer_cols))
        del neuron_cols
    del layer_cols
    return layer


def check_nn(nn: object) -> NN:
    util.check_type(nn, NN)
    util.check_type(nn.index, pd.MultiIndex)
    # levels[0] indexes the layers, levels[1] indexes the neurons on each layer
    if nn.index.nlevels != NN_INDEX_NLEVELS:
        raise ValueError("NN \n{nn}\n index nlevels = {nlevels} not {nlevels_}!".format(
            nn=nn, nlevels=nn.index.nlevels, nlevels_=NN_INDEX_NLEVELS))
    util.check_not_type(nn.columns, pd.MultiIndex)
    for layer in nn.index.remove_unused_levels().levels[0]:
        check_layer(layer=nn.loc[layer])
    return nn


# check-and-return calculation utils

def nnify(nn: List[Layer]) -> NN:
    nn = [check_layer(layer=layer) for layer in nn]
    nn = pd.concat(nn, keys=range(len(nn)))
    return check_nn(nn=nn)


def get_bias(neuron: Neuron) -> float:
    """Get bias weight from neuron."""
    # neuron = check_neuron(neuron=neuron)

    bias = neuron[BIAS_INDEX]
    bias = util.check_type(bias, float)
    if pd.isnull(bias):
        raise ValueError("Neuron \n{neuron}\n missing bias!".format(neuron=neuron))
    return bias


def get_w_in(x: pd.Series, neuron: Neuron) -> pd.Series:
    """Extract feed-in weights from neuron, conforming to x's shape."""
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
    # w_in = check_type(w_in, pd.Series)

    a_in = x.dot(w_in)
    return util.check_type(a_in, float)


def get_a_out(bias: float, a_in: float, fn: Callable[[float], float]) -> float:
    """Get outgoing activation."""
    # bias = check_type(bias, float)
    # a_in = check_type(a_in, float)
    # fn = check_type(fn, Callable[[float], float])

    a_out = fn(bias + a_in)
    return util.check_type(a_out, float)


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

def ____fprop(x: pd.Series, neuron: Neuron, fn: Callable[[float], float]=activate) -> float:
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
    # fn = check_type(fn, Callable[[float], float])

    bias = get_bias(neuron=neuron)
    w_in = get_w_in(x=x, neuron=neuron)
    a_in = get_a_in(x=x, w_in=w_in)
    return get_a_out(bias=bias, a_in=a_in, fn=fn)


def ___fprop(x: pd.Series, layer: Layer) -> pd.Series:
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
    return layer.apply(lambda neuron: ____fprop(x=x, neuron=neuron), axis="columns")


def __fprop(x: pd.Series, nn: NN) -> pd.Series:
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
    # this is basically a list of (names of) layers in this NN e.g. [0, 1, 2, ..]
    layers = nn.index.remove_unused_levels().levels[0]

    curr_layer = layers[0]
    # we want to "squeeze" the MultiIndex i.e. we want indices to be
    # not `[(curr_layer, 0), (curr_layer, 1), (curr_layer, 2), ..]` but simply `[0, 1, 2, ..]`,
    # so don't use `curr_layer = nn.loc[pd.IndexSlice[curr_layer, :], :]`
    curr_layer = nn.loc[curr_layer]
    curr_layer = check_layer(layer=curr_layer)
    x = ___fprop(x=x, layer=curr_layer)

    # recurse
    remaining_layers = layers[1:]
    if len(remaining_layers) > 0:
        remaining_layers = nn.loc[pd.IndexSlice[remaining_layers, :], :]
        remaining_layers = check_nn(nn=remaining_layers)
        return __fprop(x=x, nn=remaining_layers)
    else:
        # this was the final i.e. output layer
        return x


def _fprop(x: pd.Series, nn: NN) -> pd.Series:
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
    return squash(__fprop(x=x, nn=nn))


def fprop(X: pd.DataFrame, nn: NN) -> pd.DataFrame:
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

    p_hat = X.apply(lambda x: _fprop(x=x, nn=nn), axis="columns")
    return p_hat.apply(util.check_pmf, axis="columns")


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

    p_hat = fprop(X=X, nn=nn)
    p_hat = p_hat.apply(util.check_pmf, axis="columns")
    return p_hat.apply(lambda _p_hat: _p_hat.idxmax(), axis="columns")  # argmax of each row


########################################################################################################################
# TRAINING AKA BACKPROPAGATION #########################################################################################
########################################################################################################################

"""
When I was first studying machine learning algorithms in college, the whole concept of gradient descent
(which is the basis of back-propagation) seemed unnecessary to me. I was accustomed to the OLS method
from high-school statistics classes, where too we have a prediction problem and a loss function
(in that case, the MSE loss), and simply calculate (X'X)^{-1}X'y to fit our linear model's coefficients.
Why not do the same here, use calculus (first- and second-order conditions) plus some algebra
to solve for the neural network's optimal weights in closed form?

Then my professor reminded me that even in high-school statistics, logistic regression requires
iterative "gradient descent"[1]. This reminds me,
TODO(sparshsah): why can't a logistic regression model be fitted in closed form?
Point is, some models just aren't amenable to fitting in closed form.

[1] Possibly-wrong Math Lesson: More precisely, ordinary logistic regression uses the Newton-Raphson method,
which is like gradient descent in spirit but finds roots by constructing a first-order Taylor approximation of
a function in the neighborhood of a suspected root. Incidentally, the "finds roots"
(instead of "finds local minima") part means that for the Newton-Raphson method we must operate on the
first derivative of the loss function, finding its root(s), and therefore must know and calculate the derivative of
the first derivative i.e. the second-partial and cross-derivatives of the loss function. Assuming a slow-enough
learning rate (or fine-enough step size, if that's how you want to specify the update rule), gradient descent
can find local minima using the first derivative alone.
"""

def bprop(mini_batch_size=None, random_seed=1337):
    if mini_batch_size is not None:
        # technically, batch gradient descent is like trivial SGD where each epoch
        # learns from a single mini-batch containing all the training data but OK
        raise NotImplementedError("Don't yet support Stochastic Gradient Descent!")
    # TODO(sparshsah)
    raise NotImplementedError
