# PyDSM


PyDSM is a lightweight Python 3 framework for building and exploring distributional semantic models with focus on extensibility and ease of use. While mostly developed as a personal project, I hope it could still be found useful for others. 

Building a DSM with PyDSM is easy:

    In [1]: import pydsm, plainstream

    In [2]: wikitext = plainstream.get_text(language='en', max_words=1000, tokenize=True)

    In [3]: cooc = pydsm.build(pydsm.CooccurrenceDSM, window_size=(2,2), corpus=wikitext, lower_threshold=3)
    Building collocation matrix from corpus....
    Total time: 1.75 s

    In [4]: cooc
    Out[4]:
    CooccurrenceDSM
    Vocab size: 445
    [61, 61]  a    been  the  ...
    that      0.0  0.0   2.0  ...
    some      0.0  0.0   0.0  ...
    ''        1.0  0.0   2.0  ...
    ...       ...  ...   ...  ...

Please see [the tutorial](http://nbviewer.ipython.org/github/jimmycallin/pydsm/blob/master/docs/tutorial/Tutorial.ipynb) for a quick introduction of the package.

# Features

- Build distributional semantic models from text corpora (Cooccurrence matrix and Random Indexing models included).
- Find nearest neighbors using common similarity measures.
- Apply common weighting techniques, such as positive pointwise mutual information.
- Simple DSM visualizations.

# Installation
Download the package, and type:

    $ python setup.py install

The package is only tested on python 3.4.

# Acknowledgements

A lot of inspiration comes from the [DISSECT toolkit](http://clic.cimec.unitn.it/composes/toolkit/), a part of the [COMPOSES](http://clic.cimec.unitn.it/composes/) project. Many headaches were avoided from inspecting their work. 
