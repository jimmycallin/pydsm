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

        print('Total time of {0}: {1:.2f} sec'.format(method.__name__, te-ts))
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
        self.end    = end
        self.start  = start
        self._barLength = ProgressBar.DEFAULT_BAR_LENGTH

        self.setLevel(self.start)
        self._plotted = False

    def setLevel(self, level, initial=False):
        self._level = level
        if level < self.start:  self._level = self.start
        if level > self.end:    self._level = self.end

        self._ratio = float(self._level - self.start) / float(self.end - self.start)
        self._levelChars = int(self._ratio * self._barLength)

    def plotProgress(self):
        sys.stdout.write("\r  %3i%% [%s%s]" %(
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

# if __name__ == "__main__":

#   # Demonstration of wrapping a function which writes to a file.
#   print '- *50
#   @wrap_file_function('w')
#   def write_hi(f):
#     f.write('hi!\n')

#   f = open('temp.txt', 'w')
#   write_hi(f)
#   write_hi(f)
#   write_hi(f)
#   f.close()

#   write_hi('temp2.txt')


#   # Demonstration of wrapping a function which reads from a file.
#   print '-'*50
#   @wrap_file_function()
#   def read_file(f):
#       print f.read()

#   f = open('temp.txt')
#   print 'Reading file temp.txt from handle f:'
#   read_file(f)
#   print 'Reading file temp2.txt'
#   read_file('temp2.txt')

#   # Demonstration of wrapping a function takes multiple files
#   print '-'*50
#   @wrap_file_function('r', 'r')
#   def read_files(f1, f2):
#       print 'reading f1: '
#       print f1.read()
#       print 'reading f2: '
#       print f2.read()

#   @wrap_file_function('r', 'w')
#   def read_write(f1, f2):
#     f2.write(f1.read())

#   read_files(open('temp.txt'), open('temp2.txt'))
#   read_files('temp.txt', 'temp2.txt')
#   read_write('temp.txt', 'temp.copy.txt')

#   print 'Contents of temp.copy.txt:'
#   read_file('temp.copy.txt')
