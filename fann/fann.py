"""
Arbitrary-depth, arbitrary-width feedforward artificial neural network.
Easy-to-read Python implementation of deep learning for multinomial classification.
Source code in `fann.py`, unit tests in `test_fann.py`, demo in `demo_fann.ipynb`.

Style notes
-----------
1. Many readers will see this code and instinctively want to refactor
from functional to OOP. Resist this urge.

2. Obviously, we've added a lot of data checking and assertions,
which tend to be slow, especially relative to the speed of
some of the vectorized functions that perform the actual
substance of the calculations. However, they make the code
easier to reason about---in both writing and reading it---and
anyway it's not like this would be PyTorch-rivalling code
~if only~ we removed the assertions.. as far as we know,
we're not giving up any Turing Awards by leaving them in.

3. We make liberal use of `del` statements. This is not because
we're C programmers who never learned how to use `free()` calls properly,
or because we don't trust Python's garbage collector,
but rather to enforce scope for certain variables,
decluttering the namespace and preventing accidental misuse.

4. We permit ourselves our personal predilection for underscores,
to the point of controversy and perhaps even overuse.
    For example, if we have a 2D array `arr`,
we will iterate over each row within that array as `_arr`.
Similarly, if we have a function `foo`,
we will call its helper `_foo`, and in turn its helper `__foo`.
    This nomenclature takes a little getting-used-to,
but we vigorously defend the modularity and clarity it promotes. For example,
building a nested series of one-liner helpers becomes second nature,
so that each individual function is easy to digest
and its position in the hierarchy is immediately obvious.
    In fact, if you think of a row within an array (or a helper to a function)
as a "private attribute" or "subcomponent", you might even call this
at-first-glance-unidiomatic nomenclature truly Pythonic!

5. We stick to the first-person plural in comments ("we", "us", "our").
This isn't the "Royal We", it just helps
make reading the code feel like a journey,
a shared experience between author and reader,
and also has the knock-on benefit of making any mistakes you find
in my code seem like at least partially your fault.
"""

# syntax utils
from typing import Callable, Union
# data structures
import pandas as pd
# calculations and algorithms
import numpy as np
from scipy.special import expit, softmax


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
EPSILON: float = 1e-6
NN_INDEX_NLEVELS: int = 2  # MultiIndex[layers, neurons]
BIAS_INDEX: Union[int, str] = "_bias_"


# data structure checkers
def check_type(obj: object, type_: type, check_dtype: bool=False, check_not: bool=False) -> type(None):
    if check_dtype:
        try:
            # e.g. pd.DataFrame
            type_obj = obj.dtypes
            check = type_obj == type_
        except AttributeError:
            try:
                # e.g. pd.Series
                type_obj = obj.dtypes
                check = type_obj == type_
            except AttributeError:
                # e.g. list
                type_obj = [type(x) for x in obj]
                check = [isinstance(x, type_) for x in obj]
    else:
        type_obj = type(obj)
        check = isinstance(obj, type_)
    check = np.alltrue(check)
    if check_not:
        check = not check
    if not check:
        raise TypeError("{obj} is (dtype={check_dtype}) {type_obj}, failing against (not={check_not}) {type_}!".format(
            obj=obj, check_dtype=check_dtype, type_obj=type_obj, check_not=check_not, type_=type_))


def check_dtype(obj: object, type_: type, check_not: bool=False) -> type(None):
    return check_type(obj=obj, type_=type_, check_dtype=True, check_not=check_not)


def check_not_type(obj: object, type_: type, check_dtype: bool=False) -> type(None):
    return check_type(obj=obj, type_=type_, check_dtype=check_dtype, check_not=True)


def check_data_point(x: object) -> type(None):
    check_type(x, pd.Series)
    check_not_type(x.index, pd.MultiIndex)
    if BIAS_INDEX in x.index:
        raise ValueError("Data point \n{x}\n contains reserved index {i}!".format(x=x, i=BIAS_INDEX))
    check_dtype(x, float)


def check_neuron(neuron: object) -> type(None):
    check_type(neuron, Neuron)
    check_not_type(neuron.index, pd.MultiIndex)
    if BIAS_INDEX not in neuron.index:
        raise ValueError("Neuron \n{neuron}\n missing bias index {i}!".format(neuron=neuron, i=BIAS_INDEX))
    check_dtype(neuron, float)


def check_layer(layer: object) -> type(None):
    check_type(layer, Layer)
    check_not_type(layer.index, pd.MultiIndex)
    check_not_type(layer.columns, pd.MultiIndex)
    check_dtype(layer, float)


def check_nn(nn: object) -> type(None):
    check_type(nn, NN)
    check_type(nn.index, pd.MultiIndex)
    # levels[0] indexes the layers, levels[1] indexes the neurons on each layer
    if nn.index.nlevels != NN_INDEX_NLEVELS:
        raise ValueError("NN \n{nn}\n index nlevels = {nlevels} not {nlevels_}!".format(
            nn=nn, nlevels=nn.index.nlevels, nlevels_=NN_INDEX_NLEVELS))
    check_not_type(nn.columns, pd.MultiIndex)
    check_dtype(nn, float)


def check_pmf(pmf: object) -> type(None):
    check_dtype(pmf, float)
    if not np.alltrue(pmf >= -EPSILON):
        raise ValueError("{pmf} not non-negative!".format(pmf=pmf))
    if not np.isclose(sum(pmf), 1.00):
        raise ValueError("{pmf} sums to {sum_} not 1.00!".format(pmf=pmf, sum_=sum(pmf)))


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
    check_data_point(x=x)
    check_neuron(neuron=neuron)

    bias = neuron[BIAS_INDEX]
    assert pd.notnull(bias), "\n{w}\n missing bias!".format(w=neuron)
    assert isinstance(bias, float), type(bias)

    w_in = neuron.reindex(index=x.index)
    assert w_in.notnull().all(), "Feed-in weights \n{w}\n are not completely filled!".format(w=w_in)

    a_in = x.dot(w_in)
    assert isinstance(a_in, float), type(a_in)
    a_out = fn(bias + a_in)
    assert isinstance(a_out, float), type(a_out)
    return a_out


def __fprop(x: pd.Series, nn_layer: Layer) -> pd.Series:
    """
    Forward-propagate the previous layer's output with the current layer's weights and activation function.

    input
    -----
    x: pd.Series, the previous layer's output (possibly a raw data point,
        which can be seen as the input layer's "output"), where each entry correponds to
        a neuron on the previous layer.

    nn_layer: Layer, the current layer's weights, where each row corresponds to
        a neuron on the current layer and each column corresponds to
        (the bias or) a neuron on the previous layer.

    output
    ------
    pd.Series(index = nn_layer.index), the current layer's output, where each entry corresponds to
        a neuron on the current layer.
    """
    check_data_point(x=x)
    check_layer(nn_layer)
    return nn_layer.apply(lambda neuron: ___fprop(x=x, neuron=neuron), axis="columns")


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
    check_data_point(x=x)
    check_nn(nn)

    # levels[0] indexes the layers, levels[1] indexes the neurons on each layer, so
    # this is basically a list of (names of) layers in this NN e.g. [0, 1, 2, ..].
    # pd.MultiIndex "remembers" old, unused levels even after you drop all rows that used those levels.
    layers = nn.index.remove_unused_levels().levels[0]

    curr_layer = layers[0]
    # we want to "squeeze" the MultiIndex i.e. we want indices to be
    # not `[(curr_layer, 0), (curr_layer, 1), (curr_layer, 2), ..]` but rather `[0, 1, 2, ..]`,
    # so don't use `curr_layer = nn.loc[pd.IndexSlice[curr_layer, :], :]`.
    curr_layer = nn.loc[curr_layer]
    check_layer(curr_layer)
    x = __fprop(x=x, nn_layer=curr_layer)

    # recurse
    remainining_layers = layers[1:]
    if len(remainining_layers) > 0:
        remainining_layers = nn.loc[pd.IndexSlice[remainining_layers, :], :]
        check_nn(remainining_layers)
        return _fprop(x=x, nn=remainining_layers)
    else:
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
    check_data_point(x=x)
    check_nn(nn)
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
    pd.DataFrame (columns = category labels, index = observations),
        how much probability mass we assigned to each category label for each point.
        Each row is a well-formed probability mass function AKA discrete probability distribution.
    """
    X.apply(check_data_point, axis="columns")
    check_nn(nn)
    return X.apply(lambda x: fprop(x=x, nn=nn), axis="columns")


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
    X.apply(check_data_point, axis="columns")
    check_nn(nn)
    p_hat = fprop_(X=X, nn=nn)
    return p_hat.apply(lambda _p_hat: _p_hat.idxmax(), axis="columns")  # argmax of each row


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
