#[derive(Debug, Clone, Copy)]
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
pub struct Embedding<T>
where
    T: Copy,
{
    pub value: T,
    pub index: Option<usize>,
}

pub trait Distance<T>
where
    T: Copy,
{
    fn distance_cmp(&self, a: Embedding<T>, b: Embedding<T>) -> DistanceCmp;
    fn finalize_distance(&self, dist_cmp: &DistanceCmp) -> f64;
}

pub trait EmbeddingProvider<'a, D, T>
where
    D: Distance<T>,
    T: Copy,
{
    fn get_embed(&'a self, index: usize) -> T;
    fn all(&self) -> std::ops::Range<usize>;
    fn distance(&self) -> &D;

    fn get(&'a self, index: usize) -> Embedding<T> {
        self.wrap(self.get_embed(index), index)
    }

    fn wrap(&'a self, embed: T, index: usize) -> Embedding<T> {
        Embedding {
            value: embed,
            index: Some(index),
        }
    }

    fn as_embedding(&self, embed: T) -> Embedding<T> {
        Embedding {
            value: embed,
            index: None,
        }
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

    fn cached_dist<T, F>(&mut self, a: &Embedding<T>, b: &Embedding<T>, dist: F) -> DistanceCmp
    where
        F: Fn(&Embedding<T>, &Embedding<T>) -> DistanceCmp,
        T: Copy,
    {
        match (a.index, b.index) {
            (None, _) => dist(a, b),
            (_, None) => dist(a, b),
            (Some(index_a), Some(index_b)) => {
                let key = Key::new(index_a, index_b);
                match self.get(&key) {
                    Some(res) => res,
                    None => {
                        let res = dist(a, b);
                        self.put(Key::new(index_a, index_b), res);
                        res
                    }
                }
            }
        }
    }

    fn cached_distance<'a, D, T: 'a>(
        &mut self,
        a: Embedding<T>,
        b: Embedding<T>,
        distance: &'a D,
    ) -> DistanceCmp
    where
        D: Distance<T>,
        T: Copy,
    {
        self.cached_dist(&a, &b, |a, b| distance.distance_cmp(*a, *b))
    }
}

pub trait NearestNeighbors<T>
where
    T: Copy,
{
    fn get_closest(&mut self, embed: Embedding<T>, count: usize) -> Vec<(usize, f64)>;
}
