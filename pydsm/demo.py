import pydsm

model = pydsm.build(model=pydsm.CooccurrenceDSM,
                    window_size=(2, 2),
                    corpus='/Users/jimmy/dev/resources/wiki/en/',
                    store='/tmp/',
                    language='en',
                    lemmatize=False,
                    min_freq=2,
                    max_freq=100000,
                    min_ratio=2.0e-10,
                    max_ratio=0.9)


# Store somewhere else
model.store('/tmp/')

# List built dsms
pydsm.list(path='/tmp/')

# Load built pydsm
pydsm.load(id='hdsfid334d')

# Load pydsm
model = pydsm.load(window_size=(2, 2), language='en', lemmatize=False, min_freq=2, max_freq=100000)

# Apply PPMI weighting
model = model.apply_weighting('ppmi', in_place=False)

# Find closest neighbors
neighbors = model.nearest_neighbors('dog', n=20)
equivalent = model.nn('dog')

# Return distributional vector
vector = model['dog']

# Return context vector
context = model[:, 'dog']

# Return word-context value
wordcontext = model['dog', 'cat']

