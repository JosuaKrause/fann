use std::marker::PhantomData;

use digest::Digest;
use ndarray::{Array1, ArrayBase, ArrayView1, Axis, Data, Ix2, Slice};

use crate::{
    info::Info, Distance, DistanceCmp, EmbeddingProvider, InvalidRangeError, NearestNeighbors,
};

#[derive(Debug, Clone, Copy)]
pub struct NdDotDistance;

pub const ND_DOT_DISTANCE: NdDotDistance = NdDotDistance;

impl<'a> Distance<ArrayView1<'a, f64>> for NdDotDistance {
    fn distance_cmp(&self, a: &ArrayView1<'a, f64>, b: &ArrayView1<'a, f64>) -> DistanceCmp {
        DistanceCmp::of((-a.dot(b)).exp())
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
    fn distance_cmp(&self, a: &ArrayView1<'a, f64>, b: &ArrayView1<'a, f64>) -> DistanceCmp {
        let diff = a - b;
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

pub struct NdProvider<'a, 'b, S, D>
where
    S: Data<Elem = f64>,
    D: Distance<ArrayView1<'a, f64>>,
{
    arr: &'b ArrayBase<S, Ix2>,
    range: std::ops::Range<usize>,
    distance: D,
    view_lifetime: PhantomData<&'a D>,
}

impl<'a, 'b, S, D> NdProvider<'a, 'b, S, D>
where
    S: Data<Elem = f64>,
    D: Distance<ArrayView1<'a, f64>>,
{
    pub fn new(arr: &'b ArrayBase<S, Ix2>, distance: D) -> Self {
        Self {
            arr,
            range: 0..arr.shape()[0],
            distance,
            view_lifetime: PhantomData,
        }
    }
}

impl<'a, 'b, S, D> EmbeddingProvider<D, ArrayView1<'a, f64>> for NdProvider<'a, 'b, S, D>
where
    S: Data<Elem = f64>,
    D: Distance<ArrayView1<'a, f64>> + Copy,
    'b: 'a,
{
    fn with_embed<F, R>(&self, index: usize, op: F) -> R
    where
        F: Fn(&ArrayView1<'a, f64>) -> R,
    {
        assert!(self.range.contains(&index));
        op(&self.arr.row(index))
    }

    fn with_pair<F, R>(&self, a: usize, b: usize, op: F) -> R
    where
        F: Fn(&ArrayView1<'a, f64>, &ArrayView1<'a, f64>) -> R,
    {
        assert!(self.range.contains(&a));
        assert!(self.range.contains(&b));
        op(&self.arr.row(a), &self.arr.row(b))
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
        self.arr
            .row(index)
            .iter()
            .for_each(|v| hasher.update(v.to_be_bytes()));
    }

    fn subrange(
        &self,
        new_range: std::ops::Range<usize>,
    ) -> Result<Self, crate::InvalidRangeError> {
        if new_range.start < self.range.start || new_range.end > self.range.end {
            return Err(InvalidRangeError);
        }
        Ok(Self {
            arr: self.arr,
            range: new_range,
            distance: self.distance,
            view_lifetime: self.view_lifetime,
        })
    }
}

impl<'a, 'b, S>
    NearestNeighbors<NdProvider<'a, 'b, S, NdDotDistance>, NdDotDistance, ArrayView1<'a, f64>>
    for NdProvider<'a, 'b, S, NdDotDistance>
where
    S: Data<Elem = f64>,
    'b: 'a,
{
    fn get_closest<I>(
        &self,
        other: &ArrayView1<'a, f64>,
        count: usize,
        _info: &mut I,
    ) -> Vec<(usize, f64)>
    where
        I: Info,
    {
        let dists: Array1<DistanceCmp> = self
            .arr
            .slice_axis(Axis(0), Slice::from(self.all()))
            .dot(other)
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
