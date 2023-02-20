use blake2::Blake2s256;
use digest::Digest;
use serde::{Deserialize, Serialize};

use crate::info::Info;

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct DistanceCmp(f64);

impl DistanceCmp {
    pub fn zero() -> Self {
        DistanceCmp(0.0)
    }

    pub fn of(v: f64) -> Self {
        DistanceCmp(v)
    }

    pub fn to(&self) -> f64 {
        self.0
    }

    pub fn combine<F>(&self, other: &Self, map: F) -> Self
    where
        F: FnOnce(f64, f64) -> f64,
    {
        DistanceCmp::of(map(self.to(), other.to()))
    }
}

impl PartialEq for DistanceCmp {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other).is_eq()
    }
}

impl Eq for DistanceCmp {}

impl PartialOrd for DistanceCmp {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for DistanceCmp {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0.total_cmp(&other.0)
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Embedding<T> {
    pub embed: T,
    pub index: Option<usize>,
}

impl<T> Embedding<T> {
    pub fn wrap(embed: T, index: usize) -> Embedding<T> {
        Embedding {
            embed,
            index: Some(index),
        }
    }

    pub fn as_embedding(embed: T) -> Embedding<T> {
        Embedding { embed, index: None }
    }
}

pub trait Distance<T> {
    fn distance_cmp(&self, a: &Embedding<T>, b: &Embedding<T>) -> DistanceCmp;
    fn finalize_distance(&self, dist_cmp: &DistanceCmp) -> f64;
    fn name(&self) -> &str;
}

pub trait EmbeddingProvider<'a, D, T>
where
    D: Distance<T> + Copy,
{
    fn get_embed(&'a self, index: usize) -> T;
    fn all(&self) -> std::ops::Range<usize>;
    fn distance(&self) -> D;

    fn get(&'a self, index: usize) -> Embedding<T> {
        Embedding::wrap(self.get_embed(index), index)
    }

    fn hash_embed<H>(&self, index: usize, hasher: &mut H)
    where
        H: Digest;

    fn compute_hash(&self) -> String {
        let mut hasher = Blake2s256::new();
        let all = self.all();
        hasher.update(all.start.to_be_bytes());
        hasher.update(all.end.to_be_bytes());
        all.into_iter()
            .for_each(|ix| self.hash_embed(ix, &mut hasher));
        format!("{hash:x}", hash = hasher.finalize())
    }
}

#[derive(Hash, Eq, PartialEq, Debug)]
pub struct Key {
    lower_index: usize,
    upper_index: usize,
}

impl Key {
    pub fn new(index_a: usize, index_b: usize) -> Self {
        Key {
            lower_index: index_a.min(index_b),
            upper_index: index_a.max(index_b),
        }
    }
}

pub trait Cache {
    fn get(&mut self, key: &Key) -> Option<DistanceCmp>;
    fn put(&mut self, key: Key, value: DistanceCmp);

    fn cached_dist<T, F, I>(
        &mut self,
        a: &Embedding<T>,
        b: &Embedding<T>,
        dist: F,
        info: &mut I,
    ) -> DistanceCmp
    where
        F: Fn(&Embedding<T>, &Embedding<T>) -> DistanceCmp,
        I: Info,
    {
        let mut compute = |a, b| {
            info.log_cache_access(true);
            dist(a, b)
        };

        match (a.index, b.index) {
            (None, _) => compute(a, b),
            (_, None) => compute(a, b),
            (Some(index_a), Some(index_b)) => {
                let key = Key::new(index_a, index_b);
                match self.get(&key) {
                    Some(res) => {
                        info.log_cache_access(false);
                        res
                    }
                    None => {
                        let res = compute(a, b);
                        self.put(Key::new(index_a, index_b), res);
                        res
                    }
                }
            }
        }
    }

    fn cached_distance<'a, D, T, I>(
        &mut self,
        a: &Embedding<T>,
        b: &Embedding<T>,
        distance: D,
        info: &mut I,
    ) -> DistanceCmp
    where
        D: Distance<T> + Copy,
        T: 'a,
        I: Info,
    {
        info.log_dist(&a.index);
        info.log_dist(&b.index);
        self.cached_dist(&a, &b, |a, b| distance.distance_cmp(a, b), info)
    }
}

pub trait LocalCache<'a, D, T>
where
    D: Distance<T>,
    T: 'a,
{
    fn get(&mut self, index: usize) -> Option<DistanceCmp>;
    fn put(&mut self, index: usize, value: DistanceCmp);
    fn embedding(&self) -> &'a Embedding<T>;

    fn cached_distance<I>(&mut self, embed: &Embedding<T>, distance: D, info: &mut I) -> DistanceCmp
    where
        I: Info,
    {
        info.log_dist(&embed.index);
        match embed.index {
            None => {
                info.log_cache_access(true);
                distance.distance_cmp(self.embedding(), embed)
            }
            Some(index) => match self.get(index) {
                Some(res) => {
                    info.log_cache_access(false);
                    res
                }
                None => {
                    info.log_cache_access(true);
                    let res = distance.distance_cmp(self.embedding(), embed);
                    self.put(index, res);
                    res
                }
            },
        }
    }
}

pub trait LocalCacheFactory<'a, D, L, T>
where
    D: Distance<T>,
    L: LocalCache<'a, D, T>,
    T: 'a,
{
    fn create(&self, embed: &'a Embedding<T>) -> L;
}

pub trait NearestNeighbors<'a, F, D, L, T>
where
    F: LocalCacheFactory<'a, D, L, T>,
    D: Distance<T>,
    L: LocalCache<'a, D, T>,
    T: 'a,
{
    fn get_closest<I>(
        &self,
        other: &'a Embedding<T>,
        count: usize,
        cache_factory: &F,
        info: &mut I,
    ) -> Vec<(usize, f64)>
    where
        I: Info;
}
