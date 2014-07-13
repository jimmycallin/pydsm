# PyDSM


PyDSM is a lightweight Python 3 framework for building and exploring distributional semantic models with focus on extensibility and ease of use. While mostly developed as a personal project, I hope it could still be found useful for others. 

Building a DSM with PyDSM is easy:

    In [1]: import pydsm

    In [2]: cooc = pydsm.build(pydsm.CooccurrenceDSM, corpus='wiki.20k', window_size=(2,2), language='en')   98% [================================================================ ]

    Total time: 137.82 s

    In [4]: cooc
    Out[4]: 
    CooccurrenceDSM
    Vocab size: 356916
    [356916, 356916]  enforcement  outbids  audiorecording  ...
    enforcement       4.0          0.0      0.0             ...
    outbids           0.0          0.0      0.0             ...
    faurissons        0.0          0.0      0.0             ...
    ...               ...          ...      ...             ...

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
