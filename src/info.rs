use std::collections::{hash_map::IntoIter, HashMap, HashSet};

pub trait Info {
    fn log_cache_access(&mut self, is_miss: bool);
    fn log_scan(&mut self, index: usize, is_outer: bool);
    fn log_dist(&mut self, index: usize);

    fn cache_hits_miss(&self) -> (u64, u64);
    fn cache_hit_rate(&self) -> f64 {
        let (hits, miss) = self.cache_hits_miss();
        (hits as f64) / ((hits + miss) as f64)
    }

    fn scan_map(&self) -> IntoIter<usize, &str>;
    fn dist_vec(&self) -> Vec<usize>;
    fn dist_count(&self) -> usize;
    fn clear(&mut self);
}

#[derive(Debug, Clone, Copy)]
pub struct NoInfo;

pub fn no_info() -> NoInfo {
    NoInfo
}

impl Info for NoInfo {
    fn log_cache_access(&mut self, _is_miss: bool) {}
    fn log_scan(&mut self, _index: usize, _is_outer: bool) {}
    fn log_dist(&mut self, _index: usize) {}

    fn cache_hits_miss(&self) -> (u64, u64) {
        (0, 0)
    }

    fn scan_map(&self) -> IntoIter<usize, &str> {
        HashMap::new().into_iter()
    }

    fn dist_vec(&self) -> Vec<usize> {
        Vec::new()
    }

    fn dist_count(&self) -> usize {
        0
    }

    fn clear(&mut self) {}
}

pub struct BaseInfo {
    hits: u64,
    miss: u64,
    scan_map: HashMap<usize, &'static str>,
    dist_set: HashSet<usize>,
}

impl BaseInfo {
    pub fn new() -> BaseInfo {
        BaseInfo {
            hits: 0,
            miss: 0,
            scan_map: HashMap::new(),
            dist_set: HashSet::new(),
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

    fn log_dist(&mut self, index: usize) {
        self.dist_set.insert(index);
    }

    fn cache_hits_miss(&self) -> (u64, u64) {
        (self.hits, self.miss)
    }

    fn scan_map(&self) -> IntoIter<usize, &str> {
        self.scan_map.clone().into_iter()
    }

    fn dist_vec(&self) -> Vec<usize> {
        self.dist_set.iter().map(|&ix| ix).collect()
    }

    fn dist_count(&self) -> usize {
        self.dist_set.len()
    }

    fn clear(&mut self) {
        self.hits = 0;
        self.miss = 0;
        self.scan_map = HashMap::new();
        self.dist_set = HashSet::new();
    }
}
