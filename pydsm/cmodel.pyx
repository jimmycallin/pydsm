from __future__ import print_function
from cpython cimport bool
from collections import defaultdict
from itertools import chain
from pydsm.utils import timer, tokenize
import io

def _read_documents(corpus):
    """
    If text file, treats each line as a sentence.
    If list of list, treats each list as sentence of words
    """
    if isinstance(corpus, str):
        corpus = open(corpus)

    for sentence in corpus:
        if isinstance(sentence, list):
            yield sentence
        else:
            yield list(_tokenize(sentence))

    if isinstance(corpus, io.TextIOBase):
        corpus.close()


def _tokenize(s):
    """
    Removes all URL's replacing them with 'URL'. Only keeps A-Ö 0-9.
    """
    return tokenize(s)


cdef tuple _build_contexts(self, focus, list sentence, int i):
    cdef bool ordered, directed, is_ngrams
    cdef int lower_threshold, left, right
    cdef tuple window_size
    cdef list context_left, context_right

    ordered = self.config.get('ordered', False)
    directed = self.config.get('directed', False)
    lower_threshold = self.config.get('lower_threshold', 0)
    window_size = self.config['window_size']
    is_ngrams = self.config.get('is_ngrams', False)

    left = i - window_size[0] if i - window_size[0] > 0 else 0
    right = i + window_size[1] + 1 if i + window_size[1] + 1 <= len(sentence) else len(sentence)
    #flatten lists if contains ngrams
    if is_ngrams:
        context_left = [w for ngram in sentence[left:i] for w in ngram]
        context_right = [w for ngram in sentence[i+1:right] for w in ngram]
    else:
        context_left = sentence[left:i]
        context_right = sentence[i + 1:right]

    if lower_threshold:
        context_left = [w for w in context_left if self.vocabulary[w] > lower_threshold]
        context_right = [w for w in context_right if self.vocabulary[w] > lower_threshold]
    if directed:
        context_left = [w + '_left' for w in context_left]
        context_right = [w + '_right' for w in context_right]
    if ordered:
        context_left = [w + '_{}'.format(i+1) for i, w in enumerate(context_left)]
        context_right = [w + '_{}'.format(i+1) for i, w in enumerate(context_right)]
    context_left.extend(context_right)
    return focus, context_left

def _vocabularize(self, corpus):
    """
    Wraps the corpus object creating a generator that counts the vocabulary, 
    and yields the focus word along with left and right context.
    Lists as replacements of words are treated as one unit and iterated through (good for ngrams).
    """
    cdef bool is_ngrams
    cdef int lower_threshold, n, i
    cdef list sentence
    cdef str focus

    is_ngrams = self.config.get('is_ngrams', False)
    lower_threshold =self.config.get('lower_threshold', 0)

    for n, sentence in enumerate(_read_documents(corpus)):
        if n % 1000 == 0:
            print(".", end=" ", flush=True)
        for i, focus in enumerate(sentence):
            self.vocabulary[focus] += 1
            if self.vocabulary[focus] < lower_threshold:
                continue
            if is_ngrams:
                for f in focus:
                    yield _build_contexts(self, f, sentence, i)
            else:
                yield _build_contexts(self, focus, sentence, i)

def build(focus_words, context_words):
    """
    Builds a dict of dict of collocation frequencies. This is to be cythonized.
    """
    # Collect word collocation frequencies in dict of dict
    colfreqs = defaultdict(lambda: defaultdict(int))
    for focus, contexts in zip(focus_words, context_words):
        for context in contexts:
            colfreqs[focus][context] += 1

    return colfreqs
