"""A bounded memo that evicts, rather than one that jams when it fills.

A plain dict guarded by ``if len(memo) < limit`` stops accepting once it is full,
so it keeps whatever it happened to see first and refuses everything after.  That
is the wrong way round: what it then declines to cache are the positions the run
in front of it is about to reuse, and the loss is not gradual.  Playing every
answer behind one fixed opening reaches at most 243 distinct positions, so at
turn two alone the memo collapses a 12,947-wide computation to a 240-wide one;
the deeper turns and the endgame search collapse further again, and measured
over a whole sweep the three caches together are worth ~100x
(``analysis/ablations.py``).  A full memo gives all of it back at once.

Least-recently-used is worth its read-side bookkeeping here.  Relinking costs
~100 ns against a miss that costs a scoring pass over ~13k guesses, and these
tables memoize pure functions of a position, so what is cached only ever changes
how long a run takes and never what it answers.
"""

from collections import OrderedDict


class Memo:
    """A mapping with a cap, dropping the least recently used entry when full.

    ``get`` counts as a use, which is what separates this from insertion order.
    """

    def __init__(self, limit):
        self.limit = int(limit)
        self._items = OrderedDict()

    def get(self, key):
        """The cached value, or None.  No caller stores None, so the two cases
        do not need telling apart."""
        value = self._items.get(key)
        if value is not None:
            self._items.move_to_end(key)
        return value

    def put(self, key, value):
        self._items[key] = value
        if len(self._items) > self.limit:
            self._items.popitem(last=False)

    def clear(self):
        self._items.clear()

    def __len__(self):
        return len(self._items)
