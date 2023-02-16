use std::num::NonZeroUsize;

use crate::{Cache, Key};
use lru::LruCache;

pub struct DistanceCache {
    lru: LruCache<Key, f64>,
}

impl DistanceCache {
    pub fn new(cap: usize) -> Self {
        DistanceCache {
            lru: LruCache::new(NonZeroUsize::new(cap).unwrap()),
        }
    }
}

impl Cache for DistanceCache {
    fn get(&mut self, key: &Key) -> Option<f64> {
        self.lru.get(key).map(|&res| res)
    }

    fn put(&mut self, key: Key, value: f64) {
        self.lru.put(key, value);
    }
}

pub struct NoCache {}

impl Cache for NoCache {
    fn get(&mut self, _key: &Key) -> Option<f64> {
        None
    }

    fn put(&mut self, _key: Key, _value: f64) {}
}
