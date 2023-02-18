use std::num::NonZeroUsize;

use crate::{Cache, DistanceCmp, Key};
use lru::LruCache;

pub struct DistanceCache {
    lru: LruCache<Key, DistanceCmp>,
}

impl DistanceCache {
    pub fn new(cap: usize) -> Self {
        DistanceCache {
            lru: LruCache::new(NonZeroUsize::new(cap).unwrap()),
        }
    }
}

impl Cache for DistanceCache {
    fn get(&mut self, key: &Key) -> Option<DistanceCmp> {
        self.lru.get(key).map(|&res| res)
    }

    fn put(&mut self, key: Key, value: DistanceCmp) {
        self.lru.put(key, value);
    }
}

pub struct NoCache {}

impl Cache for NoCache {
    fn get(&mut self, _key: &Key) -> Option<DistanceCmp> {
        None
    }

    fn put(&mut self, _key: Key, _value: DistanceCmp) {}
}
