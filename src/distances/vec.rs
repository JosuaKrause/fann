use crate::{
    cache::{NoLocalCache, NoLocalCacheFactory},
    info::Info,
    Distance, DistanceCmp, Embedding, EmbeddingProvider, NearestNeighbors,
};

#[derive(Debug, Clone, Copy)]
pub struct VecDotDistance {}

pub const VEC_DOT_DISTANCE: VecDotDistance = VecDotDistance {};

impl Distance<&Vec<f64>> for VecDotDistance {
    fn distance_cmp(&self, a: &Embedding<&Vec<f64>>, b: &Embedding<&Vec<f64>>) -> DistanceCmp {
        let res: f64 = a
            .embed
            .iter()
            .zip(b.embed.iter())
            .map(|(&cur_a, &cur_b)| cur_a * cur_b)
            .sum();
        DistanceCmp::of((-res).exp())
    }

    fn finalize_distance(&self, dist_cmp: &DistanceCmp) -> f64 {
        dist_cmp.to()
    }
}

#[derive(Debug, Clone, Copy)]
pub struct VecL2Distance {}

pub const VEC_L2_DISTANCE: VecL2Distance = VecL2Distance {};

impl Distance<&Vec<f64>> for VecL2Distance {
    fn distance_cmp(&self, a: &Embedding<&Vec<f64>>, b: &Embedding<&Vec<f64>>) -> DistanceCmp {
        let res: f64 = a
            .embed
            .iter()
            .zip(b.embed.iter())
            .map(|(&cur_a, &cur_b)| (cur_a - cur_b) * (cur_a - cur_b))
            .sum();
        DistanceCmp::of(res)
    }

    fn finalize_distance(&self, dist_cmp: &DistanceCmp) -> f64 {
        dist_cmp.to().sqrt()
    }
}

pub struct VecProvider<'a, D>
where
    D: Distance<&'a Vec<f64>>,
{
    embeddings: &'a Vec<Vec<f64>>,
    distance: D,
}

impl<'a, D> VecProvider<'a, D>
where
    D: Distance<&'a Vec<f64>>,
{
    pub fn new(embeddings: &'a Vec<Vec<f64>>, distance: D) -> Self {
        VecProvider {
            embeddings,
            distance,
        }
    }
}

impl<'a, D> EmbeddingProvider<'a, D, &'a Vec<f64>> for VecProvider<'a, D>
where
    D: Distance<&'a Vec<f64>> + Copy,
{
    fn get_embed(&'a self, index: usize) -> &'a Vec<f64> {
        &self.embeddings[index]
    }

    fn all(&self) -> std::ops::Range<usize> {
        0..self.embeddings.len()
    }

    fn distance(&self) -> D {
        self.distance
    }
}

impl<'a, D>
    NearestNeighbors<'a, NoLocalCacheFactory, D, NoLocalCache<'a, &'a Vec<f64>>, &'a Vec<f64>>
    for VecProvider<'a, D>
where
    D: Distance<&'a Vec<f64>>,
{
    fn get_closest<I>(
        &self,
        other: &Embedding<&'a Vec<f64>>,
        count: usize,
        _cache_factory: &NoLocalCacheFactory,
        _info: &mut I,
    ) -> Vec<(usize, f64)>
    where
        I: Info,
    {
        let mut dists: Vec<(usize, DistanceCmp)> = self
            .embeddings
            .iter()
            .enumerate()
            .map(|(ix, cur)| {
                let val = Embedding::wrap(cur, ix);
                (ix, self.distance.distance_cmp(&val, other))
            })
            .collect();
        dists.sort_unstable_by(|(_, a), (_, b)| a.cmp(b));
        dists
            .iter()
            .take(count)
            .map(|(ix, dist)| (*ix, self.distance.finalize_distance(dist)))
            .collect()
    }
}
