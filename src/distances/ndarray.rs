use digest::Digest;
use ndarray::{s, Array1, ArrayView1, ArrayView2};

use crate::{
    info::Info, Distance, DistanceCmp, EmbeddingProvider, InvalidRangeError, NearestNeighbors,
};

#[derive(Debug, Clone, Copy)]
pub struct NdDotDistance;

pub const ND_DOT_DISTANCE: NdDotDistance = NdDotDistance;

impl<'a> Distance<ArrayView1<'a, f64>> for NdDotDistance {
    fn distance_cmp(&self, a: ArrayView1<'a, f64>, b: ArrayView1<'a, f64>) -> DistanceCmp {
        DistanceCmp::of((-a.dot(&b)).exp())
    }

    fn finalize_distance(&self, dist_cmp: &DistanceCmp) -> f64 {
        dist_cmp.to()
    }

    fn name(&self) -> &str {
        "dot"
    }
}

#[derive(Debug, Clone, Copy)]
pub struct NdL2Distance;

pub const ND_L2_DISTANCE: NdL2Distance = NdL2Distance;

impl<'a> Distance<ArrayView1<'a, f64>> for NdL2Distance {
    fn distance_cmp(&self, a: ArrayView1<'a, f64>, b: ArrayView1<'a, f64>) -> DistanceCmp {
        let diff = &a - &b;
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
    offset: usize,
    distance: D,
}

impl<'a, D> NdProvider<'a, D>
where
    D: Distance<ArrayView1<'a, f64>>,
{
    pub fn new(arr: ArrayView2<'a, f64>, distance: D) -> Self {
        NdProvider {
            arr,
            offset: 0,
            distance,
        }
    }
}

impl<'a, D> EmbeddingProvider<'a, D, ArrayView1<'a, f64>> for NdProvider<'a, D>
where
    D: Distance<ArrayView1<'a, f64>>,
    Self: 'a,
{
    fn with_embed<F, R>(&'a self, &index: &usize, op: F) -> R
    where
        F: Fn(ArrayView1<'a, f64>) -> R,
    {
        op(self.arr.row(index - self.offset))
    }

    fn with_pair<F, R>(&'a self, &a: &usize, &b: &usize, op: F) -> R
    where
        F: Fn(ArrayView1<'a, f64>, ArrayView1<'a, f64>) -> R,
    {
        op(self.arr.row(a - self.offset), self.arr.row(b - self.offset))
    }

    fn all(&self) -> std::ops::Range<usize> {
        self.offset..(self.arr.shape()[0] + self.offset)
    }

    fn distance(&self) -> D {
        self.distance
    }

    fn hash_embed<H>(&self, index: usize, hasher: &mut H)
    where
        H: Digest,
    {
        self.arr
            .row(index - self.offset)
            .iter()
            .for_each(|v| hasher.update(v.to_be_bytes()));
    }

    fn subrange(
        &'a self,
        new_range: std::ops::Range<usize>,
    ) -> Result<Self, crate::InvalidRangeError> {
        let all = self.all();
        if new_range.start < all.start || new_range.end > all.end {
            return Err(InvalidRangeError);
        }
        let tmp_range = (new_range.start - all.start)..(new_range.end - all.start);
        let new_offset = new_range.start;
        Ok(Self {
            arr: self.arr.slice(s![tmp_range, ..]),
            offset: new_offset,
            distance: self.distance,
        })
    }
}

impl<'a> NearestNeighbors<'a, NdProvider<'a, NdDotDistance>, NdDotDistance, ArrayView1<'a, f64>>
    for NdProvider<'a, NdDotDistance>
{
    fn get_closest<I>(
        &self,
        other: ArrayView1<'a, f64>,
        count: usize,
        _info: &mut I,
    ) -> Vec<(usize, f64)>
    where
        I: Info,
    {
        let dists: Array1<DistanceCmp> = self.arr.dot(&other).map(|v| DistanceCmp::of((-v).exp()));
        let mut indices: Vec<usize> = (0..dists.len()).collect();
        indices.sort_unstable_by_key(|&ix| dists[ix]);
        indices
            .iter()
            .take(count)
            .map(|&ix| (ix, self.distance.finalize_distance(&dists[ix])))
            .collect()
    }
}
