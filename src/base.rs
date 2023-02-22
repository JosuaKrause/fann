use std::{fmt, marker::PhantomData};

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

pub trait Distance<T>
where
    Self: Copy,
{
    fn distance_cmp(&self, a: T, b: T) -> DistanceCmp;
    fn finalize_distance(&self, dist_cmp: &DistanceCmp) -> f64;
    fn name(&self) -> &str;
}

#[derive(Debug, Clone)]
pub struct InvalidRangeError;

impl fmt::Display for InvalidRangeError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "given range is invalid")
    }
}

pub trait EmbeddingProvider<'a, D, T>
where
    D: Distance<T>,
    Self: Sized + 'a,
    T: 'a,
{
    fn with_embed<F, R>(&'a self, index: &usize, op: F) -> R
    where
        F: Fn(T) -> R;

    fn with_pair<F, R>(&'a self, a: &usize, b: &usize, op: F) -> R
    where
        F: Fn(T, T) -> R;

    fn dist_internal<C, I>(
        &'a self,
        &aindex: &usize,
        &bindex: &usize,
        cache: &mut C,
        info: &mut I,
    ) -> DistanceCmp
    where
        C: Cache,
        I: Info,
    {
        info.log_dist(aindex);
        info.log_dist(bindex);
        let key = Key::new(aindex, bindex);
        match cache.get(&key) {
            Some(res) => {
                info.log_cache_access(false);
                res
            }
            None => {
                info.log_cache_access(true);
                let res = self.with_pair(&aindex, &bindex, |embed_a, embed_b| {
                    self.distance().distance_cmp(embed_a, embed_b)
                });
                cache.put(Key::new(aindex, bindex), res);
                res
            }
        }
    }

    fn all(&self) -> std::ops::Range<usize>;

    fn distance(&self) -> D;

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

    fn subrange(&'a self, new_range: std::ops::Range<usize>) -> Result<Self, InvalidRangeError>;
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
}

pub struct LocalDistance<'a, E, D, T>
where
    E: EmbeddingProvider<'a, D, T>,
    D: Distance<T>,
    T: 'a + Copy,
{
    provider: &'a E,
    embed: T,
    distance_type: PhantomData<D>,
}

impl<'a, E, D, T> LocalDistance<'a, E, D, T>
where
    E: EmbeddingProvider<'a, D, T>,
    D: Distance<T>,
    T: 'a + Copy,
{
    pub fn new(provider: &'a E, embed: T) -> Self {
        LocalDistance {
            provider,
            embed,
            distance_type: PhantomData,
        }
    }

    pub fn distance_cmp<I>(&self, index: usize, info: &mut I) -> DistanceCmp
    where
        I: Info,
    {
        info.log_dist(index);
        let distance = self.provider.distance();
        self.provider
            .with_embed(&index, |other| distance.distance_cmp(self.embed, other))
    }

    pub fn finalize_distance(&self, dist_cmp: &DistanceCmp) -> f64 {
        let distance = self.provider.distance();
        distance.finalize_distance(dist_cmp)
    }
}

pub trait NearestNeighbors<'a, E, D, T>
where
    E: EmbeddingProvider<'a, D, T>,
    D: Distance<T> + Copy,
    T: 'a,
{
    fn get_closest<I>(&'a self, other: T, count: usize, info: &mut I) -> Vec<(usize, f64)>
    where
        I: Info;
}
