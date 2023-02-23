use crate::{
    info::Info, Distance, DistanceCmp, EmbeddingProvider, InvalidRangeError, NearestNeighbors,
};
use digest::Digest;

#[derive(Debug, Clone, Copy)]
pub struct VecDotDistance;

pub const VEC_DOT_DISTANCE: VecDotDistance = VecDotDistance;

impl Distance<Vec<f64>> for VecDotDistance {
    fn distance_cmp(&self, a: &Vec<f64>, b: &Vec<f64>) -> DistanceCmp {
        let res: f64 = a
            .iter()
            .zip(b.iter())
            .map(|(&cur_a, &cur_b)| cur_a * cur_b)
            .sum();
        DistanceCmp::of((-res).exp())
    }

    fn finalize_distance(&self, dist_cmp: &DistanceCmp) -> f64 {
        dist_cmp.to()
    }

    fn name(&self) -> &str {
        "dot"
    }
}

#[derive(Debug, Clone, Copy)]
pub struct VecL2Distance;

pub const VEC_L2_DISTANCE: VecL2Distance = VecL2Distance;

impl Distance<Vec<f64>> for VecL2Distance {
    fn distance_cmp(&self, a: &Vec<f64>, b: &Vec<f64>) -> DistanceCmp {
        let res: f64 = a
            .iter()
            .zip(b.iter())
            .map(|(&cur_a, &cur_b)| (cur_a - cur_b) * (cur_a - cur_b))
            .sum();
        DistanceCmp::of(res)
    }

    fn finalize_distance(&self, dist_cmp: &DistanceCmp) -> f64 {
        dist_cmp.to().sqrt()
    }

    fn name(&self) -> &str {
        "l2"
    }
}

pub struct VecProvider<'a, D>
where
    D: Distance<Vec<f64>>,
{
    embeddings: &'a Vec<Vec<f64>>,
    range: std::ops::Range<usize>,
    distance: D,
}

impl<'a, D> VecProvider<'a, D>
where
    D: Distance<Vec<f64>>,
{
    pub fn new(embeddings: &'a Vec<Vec<f64>>, distance: D) -> Self {
        VecProvider {
            embeddings,
            range: 0..embeddings.len(),
            distance,
        }
    }
}

impl<'a, D> EmbeddingProvider<D, Vec<f64>> for VecProvider<'a, D>
where
    D: Distance<Vec<f64>> + Copy,
{
    fn with_embed<F, R>(&self, index: usize, op: F) -> R
    where
        F: FnOnce(&Vec<f64>) -> R,
    {
        assert!(self.range.contains(&index));
        op(&self.embeddings[index])
    }

    fn with_pair<F, R>(&self, a: usize, b: usize, op: F) -> R
    where
        F: FnOnce(&Vec<f64>, &Vec<f64>) -> R,
    {
        assert!(self.range.contains(&a));
        assert!(self.range.contains(&b));
        op(&self.embeddings[a], &self.embeddings[b])
    }

    fn all(&self) -> std::ops::Range<usize> {
        self.range.clone()
    }

    fn distance(&self) -> D {
        self.distance
    }

    fn hash_embed<H>(&self, index: usize, hasher: &mut H)
    where
        H: Digest,
    {
        assert!(self.range.contains(&index));
        self.embeddings[index]
            .iter()
            .for_each(|v| hasher.update(v.to_be_bytes()));
    }

    fn subrange(&self, new_range: std::ops::Range<usize>) -> Result<Self, InvalidRangeError> {
        if new_range.start < self.range.start || new_range.end > self.range.end {
            return Err(InvalidRangeError);
        }
        Ok(Self {
            embeddings: self.embeddings,
            range: new_range,
            distance: self.distance,
        })
    }
}

impl<'a, D> NearestNeighbors<VecProvider<'a, D>, D, Vec<f64>> for VecProvider<'a, D>
where
    D: Distance<Vec<f64>> + Copy,
{
    fn get_closest<I>(&self, other: &Vec<f64>, count: usize, _info: &mut I) -> Vec<(usize, f64)>
    where
        I: Info,
    {
        let mut dists: Vec<(usize, DistanceCmp)> = self
            .embeddings
            .iter()
            .enumerate()
            .map(|(ix, cur)| (ix, self.distance.distance_cmp(cur, &other)))
            .collect();
        dists.sort_unstable_by(|(_, a), (_, b)| a.cmp(b));
        dists
            .iter()
            .take(count)
            .map(|(ix, dist)| (*ix, self.distance.finalize_distance(dist)))
            .collect()
    }
}
