# ml
homebrew machine learning
(inspired by a variety of lessons, blog posts, and StackOverflow answers on the Web)

## fann
Arbitrary-depth, arbitrary-width feedforward artificial neural network.
Easy-to-read Python implementation of deep learning for multinomial classification.
Source code in `foggy_ml/fann/fann.py`, unit tests in `foggy_ml/fann/test/test_fann.py`, demo in `foggy_ml/fann/demo/demo_fann.ipynb`.

# Style notes

- Many readers will see this code and instinctively want to
refactor from functional to OOP. Resist this urge.

- Obviously, we've added a lot of sometimes-redundant
type checking and assertions, which tend to be slow,
often 10-100x slower than the actual calculations.
However, they (we hope) make the code easier to reason about,
which is the whole point of a project like this.
We worked hard on our code, but it's not like it would be a viable
PyTorch alternative \~if only\~ we removed the type checks.

- We make liberal use of `del` statements. This is not because
we're C programmers who never learned how to use `free()` calls properly,
or because we don't trust Python's garbage collector,
but rather to enforce scope for certain variables,
decluttering the namespace and preventing accidental misuse.

- We permit ourselves our personal predilection for underscores,
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
at-first-glance unidiomatic nomenclature truly Pythonic!

- We stick to the first-person plural in comments ("we", "us", "our").
This isn't the "Royal We", it just helps
make reading the code feel like a journey,
a shared experience between author and reader,
and also has the knock-on benefit of making any mistakes you find
in my code seem like at least partially your fault.
