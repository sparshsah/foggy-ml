"""See description & style notes in top-level repository README.md."""

# syntax utils
from typing import Tuple, Iterable, Callable, Union, Optional
# data structures
import pandas as pd
# data wrangling
from .. import util
# calculations and algorithms
import numpy as np

# TODO(sparshsah): not all (no pun intended!) of this stuff is strictly useful as a public API..
__all__ = [
    # data structures
    "Neuron", "Layer", "NN",
    # magic numbers
    "NN_INDEX_NLEVELS", "BIAS_INDEX", "LEARN_R_DEFAULT", "BATCH_SZ_DEFAULT", "MAX_EPOCH_DEFAULT",
    # initialization
    "init_neuron", "init_layer", "init_nn",
    # data wrangling
    "check_data_point", "check_neuron", "check_layer", "check_nn",
    "layerify", "nnify",
    "get_bias", "get_w_in", "get_a_in", "get_a_out",
    # calculations and algorithms
    "activate", "squash",
    "_____fprop", "____fprop", "___fprop", "__fprop", "_fprop", "fprop", "predict",
    "__bprop", "_bprop", "bprop", "__train", "_train", "train",
]

########################################################################################################################
# NN OBJECT ############################################################################################################
########################################################################################################################

"""
Fixing the neuron activation---in our case, expit---and output-squashing---in our case, softmax---function,
a (trained) FANN is essentially identified by its forward-propagation (AKA forward-pass AKA feedforward) weights.

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
RNG: type = np.random.Generator

# magic numbers
NN_INDEX_NLEVELS: int = 2  # MultiIndex[layers, neurons]
BIAS_INDEX: Union[int, str] = "_bias_"
LEARN_R_DEFAULT: float = 0.20
BATCH_SZ_DEFAULT: float = 32  # unrelated but inspired by the size of "infinity" for Central Limit Thm
MAX_EPOCH_DEFAULT: int = 2048


# initialize

def init_neuron(prev_layer_width: int, rng: RNG) -> Neuron:
    neuron = Neuron(index=(BIAS_INDEX,) + tuple(range(prev_layer_width)))
    # generate bias
    neuron.loc[BIAS_INDEX] = rng.normal()
    # generate weights
    # normalize so ex-ante stdev of sum is exactly unity -> neuron saturation less likely -> learns faster
    # skip the bias index
    neuron.iloc[1:] = rng.normal(scale=1. / prev_layer_width**0.5, size=prev_layer_width)
    return check_neuron(neuron=neuron)


def init_layer(prev_layer_width: int, layer_width: int, rng: RNG) -> Layer:
    layer = [init_neuron(prev_layer_width=prev_layer_width, rng=rng) for _ in range(layer_width)]
    return layerify(layer=layer)


def init_nn(input_width: int, layer_width: Union[int, Iterable[int]], output_width=int, random_seed=1337) -> NN:
    """
    Initialize an NN with random weights.

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
    layer_width = (layer_width,) if isinstance(layer_width, int) else tuple(layer_width)
    layer_width = (input_width,) + layer_width + (output_width,)
    del output_width, input_width
    layer_width = pd.Series(layer_width)
    layer_width = pd.concat((layer_width.shift(), layer_width), axis="columns", keys=("prev", "curr"))
    # the shift introduced a NaN, forcing the dtype to float.. fix that here
    layer_width = layer_width.loc[1:].astype(int)

    rng = np.random.default_rng(seed=random_seed)

    nn = [init_layer(prev_layer_width=width["prev"], layer_width=width["curr"], rng=rng)
          for _, width in layer_width.iterrows()]
    return nnify(nn=nn)


# type checkers

def check_data_point(x: object) -> pd.Series:
    util.check_type(x, pd.Series)
    util.check_not_type(x.index, pd.MultiIndex)
    if BIAS_INDEX in x.index:
        raise ValueError(f"Data point \n{x}\n contains reserved index `{BIAS_INDEX}`!")
    util.check_dtype(x, type_={int, float})
    return x


def check_neuron(neuron: object) -> Neuron:
    util.check_type(neuron, Neuron)
    util.check_not_type(neuron.index, pd.MultiIndex)
    if BIAS_INDEX not in neuron.index:
        raise ValueError(f"Neuron \n{neuron}\n missing bias index `{BIAS_INDEX}`!")
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
            raise ValueError(f"Neuron's weights not filled across its full Layer.. {neuron_cols} != {layer_cols}!")
        del neuron_cols
    del layer_cols
    return layer


def check_nn(nn: object) -> NN:
    util.check_type(nn, NN)
    util.check_type(nn.index, pd.MultiIndex)
    # levels[0] indexes the layers, levels[1] indexes the neurons on each layer
    if nn.index.nlevels != NN_INDEX_NLEVELS:
        raise ValueError(f"NN \n{nn}\n index nlevels = {nn.index.nlevels} not {NN_INDEX_NLEVELS}!")
    util.check_not_type(nn.columns, pd.MultiIndex)
    for layer in nn.index.remove_unused_levels().levels[0]:
        check_layer(layer=nn.loc[layer])
    return nn


# check-and-return calculation utils

def layerify(layer: Iterable[Neuron]) -> Layer:
    layer = pd.concat(layer, axis="columns").T
    return check_layer(layer=layer)


def nnify(nn: Iterable[Layer]) -> NN:
    nn = [check_layer(layer=layer) for layer in nn]
    nn = pd.concat(nn, keys=range(len(nn)))
    return check_nn(nn=nn)


def get_bias(neuron: Neuron) -> float:
    """Get bias weight from neuron."""
    # neuron = check_neuron(neuron=neuron)

    bias = neuron[BIAS_INDEX]
    bias = util.check_type(bias, float)
    if pd.isnull(bias):
        raise ValueError(f"Neuron \n{neuron}\n missing bias!")
    return bias


def get_w_in(x: pd.Series, neuron: Neuron) -> pd.Series:
    """Extract feed-in weights from neuron, conforming to x's shape."""
    # x = check_data_point(x=x)
    # neuron = check_neuron(neuron=neuron)

    w_in = neuron.reindex(index=x.index)
    if w_in.isnull().any():
        raise ValueError(f"Feed-in weights \n{w_in}\n not completely filled!")

    w_pad = neuron.drop(labels=BIAS_INDEX).drop(labels=x.index)
    if w_pad.notnull().any():
        raise ValueError(f"\n{w_pad}\n contains value(s), but should be just padding!")

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
Three common neuron-activation functions are ReLU, tanh, and expit (AKA inverse-logit AKA logistic AKA sigmoid).

ReLU (not implemented) is quite versatile, and has a cool connection to the hinge loss function.

Tanh is simply a scaled-and-shifted version of expit.

We choose the expit, for the slightly arbitrary reason that its output lies in [0, 1],
so (at least for binary classification) it can potentially be interpreted as this neuron's "best guess"
of the probability that the input belongs to Category 1.
As further support, the expit's generalization, the softmax,
is a commonly-used output-squashing function for multinomial classification:
It transforms a `n`-vector of Real numbers into a probability mass distribution.
"""

activate = util.expit
squash = util.softmax


########################################################################################################################
# FORWARD PROPAGATION AKA FORWARD PASS AKA FEEDFORWARD (PREDICTION) ####################################################
########################################################################################################################

def _____fprop(x: pd.Series, neuron: Neuron, fn: Callable[[float], float]=activate,
               expand: bool=False) -> Union[float, pd.Series]:
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

    expand: bool, whether to return both incoming and outgoing activation.

    output
    ------
    if not `expand`:
        float, current neuron's outgoing activation.
    else:
        pd.Series (index = ['a_in', 'a_out']),
        current neuron's incoming and outgoing activation.
    """
    x = check_data_point(x=x)
    neuron = check_neuron(neuron=neuron)
    # fn = check_type(fn, Callable[[float], float])

    bias = get_bias(neuron=neuron)
    w_in = get_w_in(x=x, neuron=neuron)
    a_in = get_a_in(x=x, w_in=w_in)
    del w_in, neuron, x
    a_out = get_a_out(bias=bias, a_in=a_in, fn=fn)
    del bias, fn
    if expand:
        return pd.Series({"a_in": a_in, "a_out": a_out}, index=("a_in", "a_out"))
    else:
        return a_out


def ____fprop(x: pd.Series, layer: Layer, fn: Callable[[float], float]=activate,
              expand: bool=False) -> Union[pd.Series, pd.DataFrame]:
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

    fn: function, the activation function.

    expand: bool, whether to return both incoming and outgoing activation per neuron.

    output
    ------
    if not `expand`:
        pd.Series (index = layer.index), the current layer's outgoing activation per neuron,
        where each entry corresponds to a neuron on the current layer.
    else:
        pd.DataFrame (index = layer.index, columns = ['a_in', 'a_out']),
        current layer's incoming and outgoing activation per neuron,
        where each row corresponds to a neuron on the current layer.
    """
    x = check_data_point(x=x)
    layer = check_layer(layer=layer)
    """
    Layer.apply() will smartly return pd.DataFrame if ____fprop() returns pd.Series,
    or pd.Series if ___fprop() returns scalar.
    In either case, returned object's index will match layer.index.
    """
    return layer.apply(lambda neuron: _____fprop(x=x, neuron=neuron, fn=fn, expand=expand), axis="columns")


def ___fprop(x: pd.Series, nn: NN, fn: Callable[[float], float]=activate,
             expand: bool=False) -> Union[pd.Series, Tuple[pd.DataFrame]]:
    """
    Forward-propagate the input through the network.

    input
    -----
    x: pd.Series, a single raw data point.

    nn: NN, the model.

    fn: function, the activation function.

    expand: bool, whether to return incoming and outgoing activation per neuron, per layer.

    output
    ------
    if not `expand`:
        pd.Series (index = layers[-1].index), the output layer's outgoing activation per neuron,
        where each entry corresponds to a neuron on the output layer. Not yet squashed!
    else:
        Tuple[pd.DataFrame]
        pd.concat()-able to pd.DataFrame (index = nn.index, columns = ['a_in', 'a_out']),
        each layer's incoming and outgoing activation per neuron,
        where each "super-row" (axis=0, level=0) corresponds to a layer and
        each row (axis=0, level=1) corresponds to a neuron on that layer.
        Output layer outgoing activations not yet squashed!
    """
    x = check_data_point(x=x)
    nn = check_nn(nn=nn)
    """
    levels[0] indexes the layers, levels[1] indexes the neurons on each layer, so
    this is basically a list of (names of) layers in this NN e.g. [0, 1, 2, ..]
    """
    layers = nn.index.remove_unused_levels().levels[0]
    curr_layer = layers[0]
    """
    we want to "squeeze" the MultiIndex i.e. we want indices to be
    not `[(curr_layer, 0), (curr_layer, 1), (curr_layer, 2), ..]` but simply `[0, 1, 2, ..]`,
    so don't use `curr_layer = nn.loc[pd.IndexSlice[curr_layer, :], :]`
    """
    curr_layer = nn.loc[curr_layer]
    curr_layer = check_layer(layer=curr_layer)
    a = ____fprop(x=x, layer=curr_layer, fn=fn, expand=expand)  # curr layer activations
    a_out = a["a_out"] if expand else a  # curr layer outgoing activations
    del x

    # recurse
    remaining_layers = layers[1:]
    if len(remaining_layers) > 0:
        remaining_layers = nn.loc[pd.IndexSlice[remaining_layers, :], :]
        remaining_layers = check_nn(nn=remaining_layers)
        # remaining layers activations
        a_ = ___fprop(x=a_out, nn=remaining_layers,
                      # if next layer is output layer, don't pass thru activation fn (we will later squash)
                      fn=(lambda x: x) if len(remaining_layers) == 1 else fn,
                      expand=expand)
        # std CPython doesn't implement tail-call optimization, hence we prefer this more legible syntax
        return (a,) + a_ if expand else a_
    else:  # this was the final i.e. output layer
        return (a,) if expand else a


def __fprop(x: pd.Series, nn: NN, expand: bool=False) -> Union[pd.Series, pd.DataFrame]:
    """
    Forward-propagate the input through the network.

    input
    -----
    x: pd.Series, a single raw data point.

    nn: NN, the model.

    expand: bool, whether to return incoming and outgoing activation per neuron, per layer.

    output
    ------
    if not `expand`:
        pd.Series (index = layers[-1].index), the output layer's outgoing activation per neuron,
        where each entry corresponds to a neuron on the output layer. Not yet squashed!
    else:
        pd.DataFrame (index = nn.index, columns = ['a_in', 'a_out']),
        each layer's incoming and outgoing activation per neuron,
        where each "super-row" (axis=0, level=0) corresponds to a layer and
        each row (axis=0, level=1) corresponds to a neuron on that layer.
        Output layer outgoing activations not yet squashed!
    """
    a = ___fprop(x=x, nn=nn, expand=expand)
    return pd.concat(a, keys=nn.index.remove_unused_levels().levels[0]) if expand else a


def _fprop(x: pd.Series, nn: NN, expand: bool=False) -> Union[pd.Series, pd.DataFrame]:
    """
    Forward-propagate the input through the network.

    input
    -----
    x: pd.Series, a single raw data point.

    nn: NN, the model.

    expand: bool, whether to return incoming and outgoing activation per neuron, per layer.

    output
    ------
    if not `expand`:
        pd.Series (index = layers[-1].index), the probability the model assigns to each category label,
        where each entry corresponds to a neuron on the output layer.
    else:
        pd.DataFrame (index = nn.index, columns = ['a_in', 'a_out']),
        each layer's incoming and outgoing activation per neuron,
        where each "super-row" (axis=0, level=0) corresponds to a layer and
        each row (axis=0, level=1) corresponds to a neuron on that layer.
        Output layer outgoing activations are the probability the model assigns
        to each category label, where each entry corresponds to a neuron on the output layer.
    """
    x = check_data_point(x=x)
    nn = check_nn(nn=nn)

    a = __fprop(x=x, nn=nn, expand=expand)  # activations
    del x
    if expand:  # isinstance(a, pd.DataFrame)
        # squash only the output layer's outgoing activations into a probability mass function
        # along a's MultiIndex's first level, get the last label
        output_layer = a.index.remove_unused_levels().levels[0][-1]
        a.loc[pd.IndexSlice[output_layer, :], "a_out"] = squash(a.loc[pd.IndexSlice[output_layer, :], "a_out"])
    else:  # isinstance(a, pd.Series)
        a = squash(a)
    return a


def fprop(X: pd.DataFrame, nn: NN) -> pd.DataFrame:
    """
    Forward-propagate each input through the network.
    Could be done more efficiently with some clever linear algebra, but this is more step-by-step.

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
# | |- || |_ ###########################################################################################################
########################################################################################################################

"""
TODO(sparshsah): logic for "early termination" of training if loss "doesn't change", i.e.
if np.isclose(curr_epoch_loss, prev_epoch_loss, atol=tol), then we're done training
"""


########################################################################################################################
# BACKPROPAGATION (TRAINING) ###########################################################################################
########################################################################################################################

"""
When I was first studying machine learning algorithms in college, the whole concept of gradient descent
(which is the basis of NN training) seemed unnecessary to me. I was accustomed to the OLS method
from high-school statistics classes, where too we have a prediction problem and a loss function
(in that case, the MSE loss), and simply calculate (X'X)^{-1}X'y to fit our linear model's coefficients.
Why not do the same here, use calculus (first- and second-order conditions) plus some algebra
to solve for the neural network's optimal weights in closed form?

Then my professor reminded me that even in high-school statistics, logistic regression requires
iterative "gradient descent"[1]. This reminds me,
TODO(sparshsah): why can't a logistic regression model be fitted in closed form?
Point is, take as given that some models just aren't amenable to fitting in closed form.

So, we use (stochastic) gradient descent, stepping a bit in the most efficient direction each time.

Batch size[2] here is a tradeoff between (A) faster learning from small batches, and
(B) more stable learning from large batches. A conventional choice is 32 training points per batch.
Notice that the "optimal" batch size doesn't necessarily scale with |training data|:
The speed/stability of convergence doesn't change just because you add more data to the training sample.

With that in mind, backpropagation is just a fast way to compute the gradient we want to descend,
using dynamic programming to implement the Chain Rule from calculus.


Footnote
--------

[1] Possibly-wrong Math Lesson: More precisely, ordinary logistic regression uses the Newton-Raphson method,
which is like gradient descent in spirit but finds roots by constructing a first-order Taylor approximation of
a function in the neighborhood of a suspected root. Incidentally, the "finds roots"
(instead of "finds local minima") part means that for the Newton-Raphson method we must operate on the
first derivative of the loss function, finding its root(s), and therefore must know and calculate the derivative of
the first derivative i.e. the second-partial and cross-derivatives of the loss function. Assuming a slow-enough
learning rate (or fine-enough step size, if that's how you want to specify the update rule), gradient descent
can find local minima using the first derivative alone. The tradeoff is that Newton-Raphson can be faster.

[2] Most authors call batch_sz=1 as "online" learning (which is a bit of a misnomer
unless you also set max_epoch=1), batch_sz=|dataset| as "batch" learning, and anything in between as
"mini-batch" or "stochastic" learning. We don't understand why this annoying and confusing
"mini-batch" vs "batch" terminology distinction persists.
"""

def __bprop(_y: object, x: pd.Series, nn: NN) -> pd.DataFrame:
    A = _fprop(x=x, nn=nn, expand=True)  # pylint: disable=unused-variable
    # get loss

    # get gradient
    # e.g. output layer's gradient could be based on the "crossmax" activation
    raise NotImplementedError


def _bprop(y: pd.Series, X: pd.DataFrame, nn: NN) -> pd.DataFrame:
    """Get the gradient, averaged over the batch."""
    raise NotImplementedError


def bprop(y: pd.Series, X: pd.DataFrame, nn: NN) -> pd.DataFrame:
    """Get the gradient, shaped like the NN."""
    raise NotImplementedError


def __train(y: pd.Series, X: pd.DataFrame, nn: NN, learn_r: float) -> NN:
    """Descend a step along the gradient."""
    grad = bprop(y=y, X=X, nn=nn)
    return nn - learn_r * grad


def _train(y: pd.Series, X: pd.DataFrame, nn: NN, num_batches: int,
           learn_r: float=LEARN_R_DEFAULT, max_epoch: int=MAX_EPOCH_DEFAULT,
           random_seed: int=1337)-> NN:
    for _ in range(max_epoch):
        """
        TODO(sparshsah): PEP 572 -- Assignment Expressions (Python 3.8) supports e.g.
        `split_shuffle(.., random_seed=random_seed := random_seed + 1)`, to mean
        `split_shuffle(.., random_seed = ++random_seed)`.
        """
        random_seed += 1
        y_batches, X_batches = util.split_shuffle(y, X, n=num_batches, random_seed=random_seed)
        for batch in range(num_batches):
            nn = __train(y=y_batches[batch], X=X_batches[batch], nn=nn, learn_r=learn_r)
    return check_nn(nn=nn)


def train(y: pd.Series, X: pd.DataFrame,
          layer_width: Union[int, Iterable[int]],
          learn_r: float=LEARN_R_DEFAULT, batch_sz: float=BATCH_SZ_DEFAULT, max_epoch: int=MAX_EPOCH_DEFAULT,
          random_seed: int=1337)-> NN:
    _ = util.check_shape_match(y, X)
    y = util.one_hotify(y=y)

    nn = init_nn(output_width=y.shape[1], input_width=X.shape[1],
                 layer_width=layer_width,
                 random_seed=random_seed)
    return _train(y=y, X=X,
                  nn=nn, learn_r=learn_r, num_batches=y.shape[0]//batch_sz, max_epoch=max_epoch,
                  random_seed=random_seed)
