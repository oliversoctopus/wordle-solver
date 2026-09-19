//! A bounded memo that evicts its least recently used entries when full.
//!
//! A map that refuses new entries once full keeps whatever it saw first and
//! declines the positions the run in front of it is about to reuse. What is
//! cached is a pure function of a position, so eviction changes how long a
//! run takes and never what it answers.

use std::collections::HashMap;
use std::hash::Hash;

/// How much of the map to drop when it fills. Evicting one entry per insert
/// would make every insert a scan of the whole map once it is full; clearing a
/// slice at a time amortizes that away, at the cost of throwing out a little
/// more than strictly necessary.
const EVICT_FRACTION: usize = 8;

/// A map with a cap, dropping the least recently used entries when full.
///
/// `get` counts as a use, so a position looked up constantly stays resident
/// however long ago it was first computed.
pub struct Memo<K, V> {
    limit: usize,
    clock: u64,
    items: HashMap<K, (V, u64)>,
}

impl<K: Eq + Hash + Clone, V: Clone> Memo<K, V> {
    pub fn new(limit: usize) -> Memo<K, V> {
        Memo {
            limit: limit.max(1),
            clock: 0,
            items: HashMap::new(),
        }
    }

    pub fn get(&mut self, key: &K) -> Option<V> {
        self.clock += 1;
        let clock = self.clock;
        let (value, last_used) = self.items.get_mut(key)?;
        *last_used = clock;
        Some(value.clone())
    }

    pub fn put(&mut self, key: K, value: V) {
        self.clock += 1;
        self.items.insert(key, (value, self.clock));
        if self.items.len() > self.limit {
            self.evict();
        }
    }

    /// Drops the oldest slice of the map in one pass.
    fn evict(&mut self) {
        let drop_count = (self.limit / EVICT_FRACTION).max(1);
        let mut ages: Vec<u64> = self.items.values().map(|(_, age)| *age).collect();
        // nth_element rather than a full sort: only the cutoff matters
        let (_, &mut cutoff, _) = ages.select_nth_unstable(drop_count);
        self.items.retain(|_, (_, age)| *age > cutoff);
    }

    pub fn len(&self) -> usize {
        self.items.len()
    }

    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }

    pub fn clear(&mut self) {
        self.items.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn remembers_what_it_was_given() {
        let mut memo: Memo<u32, u32> = Memo::new(10);
        memo.put(1, 100);
        assert_eq!(memo.get(&1), Some(100));
        assert_eq!(memo.get(&2), None);
    }

    #[test]
    fn stays_under_its_limit() {
        let mut memo: Memo<u32, u32> = Memo::new(100);
        for i in 0..1000 {
            memo.put(i, i);
        }
        assert!(memo.len() <= 100);
    }

    #[test]
    fn keeps_what_is_being_read() {
        let mut memo: Memo<u32, u32> = Memo::new(64);
        memo.put(0, 0);
        // touch it constantly while pushing far more than the cap through
        for i in 1..500 {
            memo.put(i, i);
            assert_eq!(memo.get(&0), Some(0), "evicted a hot entry at {i}");
        }
    }
}
