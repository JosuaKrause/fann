use ndarray::{Array1, ArrayView1, ArrayView2};

use crate::{
    cache::NoCache, Distance, DistanceCmp, Embedding, EmbeddingProvider, NearestNeighbors,
};

#[derive(Debug, Clone, Copy)]
pub struct NdDotDistance {}

pub const ND_DOT_DISTANCE: NdDotDistance = NdDotDistance {};

impl<'a> Distance<ArrayView1<'a, f64>> for NdDotDistance {
    fn distance_cmp(
        &self,
        a: &Embedding<ArrayView1<'a, f64>>,
        b: &Embedding<ArrayView1<'a, f64>>,
    ) -> DistanceCmp {
        DistanceCmp::of((-a.value.dot(&b.value)).exp())
    }

    fn finalize_distance(&self, dist_cmp: &DistanceCmp) -> f64 {
        dist_cmp.to()
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
        let diff = &a.value - &b.value.view();
        let res = (&diff * &diff).sum();
        DistanceCmp::of(res)
    }

    fn finalize_distance(&self, dist_cmp: &DistanceCmp) -> f64 {
        dist_cmp.to().sqrt()
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
}

impl<'a> NearestNeighbors<NoCache, ArrayView1<'a, f64>> for NdProvider<'a, NdDotDistance> {
    fn get_closest(
        &self,
        embed: &Embedding<ArrayView1<'a, f64>>,
        count: usize,
        _cache: &mut NoCache,
    ) -> Vec<(usize, f64)> {
        let dists: Array1<DistanceCmp> = self
            .arr
            .dot(&embed.value)
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
