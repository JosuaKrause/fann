use digest::Digest;
use ndarray::{Array1, ArrayView1, ArrayView2};

use crate::{info::Info, Distance, DistanceCmp, Embedding, EmbeddingProvider, NearestNeighbors};

#[derive(Debug, Clone, Copy)]
pub struct NdDotDistance {}

pub const ND_DOT_DISTANCE: NdDotDistance = NdDotDistance {};

impl<'a> Distance<ArrayView1<'a, f64>> for NdDotDistance {
    fn distance_cmp(
        &self,
        a: &Embedding<ArrayView1<'a, f64>>,
        b: &Embedding<ArrayView1<'a, f64>>,
    ) -> DistanceCmp {
        DistanceCmp::of((-a.embed.dot(&b.embed)).exp())
    }

    fn finalize_distance(&self, dist_cmp: &DistanceCmp) -> f64 {
        dist_cmp.to()
    }

    fn name(&self) -> &str {
        "dot"
    }
}

#[derive(Debug, Clone, Copy)]
pub struct NdL2Distance {}

pub const ND_L2_DISTANCE: NdL2Distance = NdL2Distance {};

impl<'a> Distance<ArrayView1<'a, f64>> for NdL2Distance {
    fn distance_cmp(
        &self,
        a: &Embedding<ArrayView1<'a, f64>>,
        b: &Embedding<ArrayView1<'a, f64>>,
    ) -> DistanceCmp {
        let diff = &a.embed - &b.embed.view();
        let res = (&diff * &diff).sum();
        DistanceCmp::of(res)
    }

    fn finalize_distance(&self, dist_cmp: &DistanceCmp) -> f64 {
        dist_cmp.to().sqrt()
    }

    fn name(&self) -> &str {
        "l2"
    }
}

pub struct NdProvider<'a, D>
where
    D: Distance<ArrayView1<'a, f64>>,
{
    arr: ArrayView2<'a, f64>,
    distance: D,
}

impl<'a, D> NdProvider<'a, D>
where
    D: Distance<ArrayView1<'a, f64>>,
{
    pub fn new(arr: ArrayView2<'a, f64>, distance: D) -> Self {
        NdProvider { arr, distance }
    }
}

impl<'a, D> EmbeddingProvider<'a, D, ArrayView1<'a, f64>> for NdProvider<'a, D>
where
    D: Distance<ArrayView1<'a, f64>> + Copy,
{
    fn get_embed(&'a self, index: usize) -> ArrayView1<'a, f64> {
        self.arr.row(index)
    }

    fn all(&self) -> std::ops::Range<usize> {
        0..self.arr.shape()[0]
    }

    fn distance(&self) -> D {
        self.distance
    }

    fn hash_embed<H>(&self, index: usize, hasher: &mut H)
    where
        H: Digest,
    {
        self.arr
            .row(index)
            .iter()
            .for_each(|v| hasher.update(v.to_be_bytes()));
    }
}

impl<'a> NearestNeighbors<'a, ArrayView1<'a, f64>> for NdProvider<'a, NdDotDistance> {
    fn get_closest<I>(
        &self,
        other: &Embedding<ArrayView1<'a, f64>>,
        count: usize,
        _info: &mut I,
    ) -> Vec<(usize, f64)>
    where
        I: Info,
    {
        let dists: Array1<DistanceCmp> = self
            .arr
            .dot(&other.embed)
            .map(|v| DistanceCmp::of((-v).exp()));
        let mut indices: Vec<usize> = (0..dists.len()).collect();
        indices.sort_unstable_by_key(|&ix| dists[ix]);
        indices
            .iter()
            .take(count)
            .map(|&ix| (ix, self.distance.finalize_distance(&dists[ix])))
            .collect()
    }
}
