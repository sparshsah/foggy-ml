# data structures
import pandas as pd
# algorithms
import numpy as np
import scipy as sp
import sklearn
import sklearn.linear_model
# visualization
import matplotlib.pyplot as plt
# sample data
import scipy.stats as stats
import sklearn.datasets as datasets

# Three common neuron activation functions are
# logistic (AKA sigmoid), tanh, and ReLU.
# I like the logistic for (binary) classification tasks,
# for the slightly (maybe even misleadingly) arbitrary reason
# that its output lies in [0, 1],
# so it can be interpreted as this neuron's "best guess"
# of the probability that the input belongs to Category 1.
# As further support, the logistic's generalization, the softmax,
# is a commonly-used "output squashing" function for
# multinomial classification tasks: It transforms a `n`-vector
# of Real numbers into a probability mass distribution.
# Tanh is simply a scaled-and-shifted version of logistic.
# ReLU is cool too, and has another cool connection,
# this time to the hinge loss function.
