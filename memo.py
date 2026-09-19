"""A bounded memo that evicts its least recently used entry when full.

A dict that refuses new entries once full keeps whatever it saw first and
declines the positions the run in front of it is about to reuse.  Every table
here memoizes a pure function of a position, so eviction changes how long a
run takes and never what it answers.  analysis/README.md has what the memos
are worth.
"""

from collections import OrderedDict


class Memo:
    """A mapping with a cap, dropping the least recently used entry when full.

    ``get`` counts as a use, so a hot entry stays resident.
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
