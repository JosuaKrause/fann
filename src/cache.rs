use std::{collections::HashMap, num::NonZeroUsize};

use crate::{Cache, Distance, DistanceCmp, Embedding, Key, LocalCache, LocalCacheFactory};
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

#[derive(Debug, Clone, Copy)]
pub struct NoCache {}

pub fn no_cache() -> NoCache {
    NoCache {}
}

impl Cache for NoCache {
    fn get(&mut self, _key: &Key) -> Option<DistanceCmp> {
        None
    }

    fn put(&mut self, _key: Key, _value: DistanceCmp) {}
}

#[derive(Debug, Clone, Copy)]
pub struct NoLocalCache<'a, T>
where
    T: 'a,
{
    embed: &'a Embedding<T>,
}

impl<'a, D, T> LocalCache<'a, D, T> for NoLocalCache<'a, T>
where
    D: Distance<T>,
    T: 'a,
{
    fn get(&mut self, _index: usize) -> Option<DistanceCmp> {
        None
    }

    fn put(&mut self, _index: usize, _value: DistanceCmp) {}

    fn embedding(&self) -> &'a Embedding<T> {
        self.embed
    }
}

#[derive(Debug, Clone, Copy)]
pub struct NoLocalCacheFactory {}

pub fn no_local_cache() -> NoLocalCacheFactory {
    NoLocalCacheFactory {}
}

impl<'a, D, T> LocalCacheFactory<'a, D, NoLocalCache<'a, T>, T> for NoLocalCacheFactory
where
    D: Distance<T>,
    T: 'a,
{
    fn create(&self, embed: &'a Embedding<T>) -> NoLocalCache<'a, T> {
        NoLocalCache { embed }
    }
}

pub struct DistanceLocalCache<'a, T>
where
    T: 'a,
{
    map: HashMap<usize, DistanceCmp>,
    embed: &'a Embedding<T>,
}

impl<'a, D, T> LocalCache<'a, D, T> for DistanceLocalCache<'a, T>
where
    D: Distance<T>,
    T: 'a,
{
    fn get(&mut self, index: usize) -> Option<DistanceCmp> {
        self.map.get(&index).map(|&res| res)
    }

    fn put(&mut self, index: usize, value: DistanceCmp) {
        self.map.insert(index, value);
    }

    fn embedding(&self) -> &'a Embedding<T> {
        self.embed
    }
}

#[derive(Debug, Clone, Copy)]
pub struct DistanceLocalCacheFactory {}

impl DistanceLocalCacheFactory {
    pub fn new() -> Self {
        DistanceLocalCacheFactory {}
    }
}

impl<'a, D, T> LocalCacheFactory<'a, D, DistanceLocalCache<'a, T>, T> for DistanceLocalCacheFactory
where
    D: Distance<T>,
    T: 'a,
{
    fn create(&self, embed: &'a Embedding<T>) -> DistanceLocalCache<'a, T> {
        DistanceLocalCache {
            embed,
            map: HashMap::new(),
        }
    }
}
