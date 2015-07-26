from __future__ import print_function
import time
import regex
import sys
from collections import defaultdict


find_url = regex.compile(r'(https?:\/\/)?([\da-z\.-]+)\.([a-z\.]{2,6})([\/\p{Alpha}\.-]*)*\/?')
find_words = regex.compile(r"[\p{Alpha}']+|[:=]-?[\)\(dsc/opx*|]|[-_^]{3,}|</?3")


def tokenize(s):
    return find_words.findall(find_url.sub('URL', s.lower()))


def count_rows(filepath):
    n = 0
    with open(filepath) as f:
        for line in f:
            n += 1
    return n


def to_dict_tree(digraph, root):
    import networkx as nx
    """
    Convert a networkx.DiGraph to a tree.
    (king, (governor,(editor,)), (state,(collapse,),(head,)(lord,)))
    """
    assert nx.is_tree(digraph)
    str2node = {root: []}
    for parent, child in nx.traversal.dfs_edges(digraph, root):
        score = digraph.edge[parent][child]['score']
        childnode = {child: [], 'score': score}
        children = str2node[parent]
        children.append(childnode)
        str2node[child] = childnode[child]

    return {root: str2node[root]}


class frozendict(dict):

    """
    A frozendict can be used as a key in a dict, or in a set.
    """
    __slots__ = ('_hash',)

    def __hash__(self):
        rval = getattr(self, '_hash', None)
        if rval is None:
            rval = self._hash = hash(frozenset(self.iteritems()))
        return rval


def tree():
    """
    Creates an infinite dict
    """
    return defaultdict(tree)


def timeit(method):

    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()

        print('Total time of {0}: {1:.2f} sec'.format(method.__name__, te - ts))
        return result

    return timed


class wrap_file_function(object):

    """
    Wrap a function which takes a file or a str as it's first argument.
    If a str is provided, replace the first argument of the wrapped function
    with a file handle, and close the file afterwards

    Example:

    @wrap_file_function('w')
    def write_hi(f):
      f.write('hi!\n')

    # This will write to already open file handle.
    f = open('f1.txt', 'w')
    write_hi(f)
    f.close()

    # This will open file f2.txt with mode 'w', write to it, and close the file.
    write_hi('f2.txt')
    """

    def __init__(self, *args):
        self.modes = args if args else ('r',)

    def __call__(self, func):

        def wrapped(*args, **kwargs):

            close = []
            files = []
            num_files = len(self.modes)

            for i, mode in enumerate(self.modes):

                fp = args[i]
                should_close = False

                if isinstance(fp, str):
                    fp = open(fp, mode)
                    should_close = True

                files.append(fp)
                close.append(should_close)

            try:

                # Replace the files in args when calling func
                args = files + list(args[num_files:])

                ret = func(*args, **kwargs)

            finally:

                for fp, should_close in zip(files, close):
                    if should_close:
                        fp.close()

            return ret

        return wrapped


class ProgressBar():
    DEFAULT_BAR_LENGTH = float(65)

    def __init__(self, end, start=0):
        self.end = end
        self.start = start
        self._barLength = ProgressBar.DEFAULT_BAR_LENGTH

        self.setLevel(self.start)
        self._plotted = False

    def setLevel(self, level, initial=False):
        self._level = level
        if level < self.start:
            self._level = self.start
        if level > self.end:
            self._level = self.end

        self._ratio = float(self._level - self.start) / float(self.end - self.start)
        self._levelChars = int(self._ratio * self._barLength)

    def plotProgress(self):
        sys.stdout.write("\r  %3i%% [%s%s]" % (
            int(self._ratio * 100.0),
            '=' * int(self._levelChars),
            ' ' * int(self._barLength - self._levelChars),
        ))
        self._plotted = True

    def setAndPlot(self, level):
        oldChars = self._levelChars
        self.setLevel(level)
        if (not self._plotted) or (oldChars != self._levelChars):
            self.plotProgress()

    def __del__(self):
        sys.stdout.write("\n")
