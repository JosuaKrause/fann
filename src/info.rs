use std::collections::{hash_map::IntoIter, HashMap};

use bitvec::vec::BitVec;
use polars::export::num::ToPrimitive;

pub trait Info {
    fn log_cache_access(&mut self, is_miss: bool);
    fn log_scan(&mut self, index: usize, is_outer: bool);
    fn log_dist(&mut self, index: &Option<usize>);

    fn cache_hits_miss(&self) -> (u64, u64);
    fn cache_hit_rate(&self) -> f64 {
        let (hits, miss) = self.cache_hits_miss();
        hits.to_f64().unwrap() / (hits + miss).to_f64().unwrap()
    }

    fn scan_map(&self) -> IntoIter<usize, &str>;
    fn dist_vec(&self) -> Vec<usize>;
    fn clear(&mut self);
}

#[derive(Debug, Clone, Copy)]
pub struct NoInfo {}

pub fn no_info() -> NoInfo {
    NoInfo {}
}

impl Info for NoInfo {
    fn log_cache_access(&mut self, _is_miss: bool) {}
    fn log_scan(&mut self, _index: usize, _is_outer: bool) {}
    fn log_dist(&mut self, _index: &Option<usize>) {}

    fn cache_hits_miss(&self) -> (u64, u64) {
        (0, 0)
    }

    fn scan_map(&self) -> IntoIter<usize, &str> {
        HashMap::new().into_iter()
    }

    fn dist_vec(&self) -> Vec<usize> {
        Vec::new()
    }

    fn clear(&mut self) {}
}

pub struct BaseInfo {
    hits: u64,
    miss: u64,
    scan_map: HashMap<usize, &'static str>,
    dist_vec: BitVec,
}

impl BaseInfo {
    pub fn new(size: usize) -> BaseInfo {
        BaseInfo {
            hits: 0,
            miss: 0,
            scan_map: HashMap::new(),
            dist_vec: BitVec::repeat(false, size),
        }
    }
}

impl Info for BaseInfo {
    fn log_cache_access(&mut self, is_miss: bool) {
        match is_miss {
            true => self.miss += 1,
            false => self.hits += 1,
        }
    }

    fn log_scan(&mut self, index: usize, is_outer: bool) {
        match is_outer {
            true => self.scan_map.insert(index, "O"),
            false => self.scan_map.insert(index, "I"),
        };
    }

    fn log_dist(&mut self, index: &Option<usize>) {
        if let Some(ix) = index {
            self.dist_vec.set(*ix, true);
        }
    }

    fn cache_hits_miss(&self) -> (u64, u64) {
        (self.hits, self.miss)
    }

    fn scan_map(&self) -> IntoIter<usize, &str> {
        self.scan_map.clone().into_iter()
    }

    fn dist_vec(&self) -> Vec<usize> {
        self.dist_vec.iter_ones().collect()
    }

    fn clear(&mut self) {
        self.hits = 0;
        self.miss = 0;
        self.scan_map = HashMap::new();
        self.dist_vec = BitVec::repeat(false, self.dist_vec.len());
    }
}
