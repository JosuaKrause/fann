use std::marker::PhantomData;

use crate::{DotDistance, DotDistanceCmp, DotEmbeddingProvider};

pub trait DotEmbeddingAccess<E, D, T>
where
    E: DotEmbeddingProvider<D, T>,
    D: DotDistance<T>,
{
    fn add(self: &mut Self, index: usize);

    fn get_closest(self: &Self, other: &T, count: usize) -> Vec<(usize, DotDistanceCmp)>;
}

pub struct ConvexSet<'a, A, E, D, T>
where
    A: DotEmbeddingAccess<E, D, T>,
    E: DotEmbeddingProvider<D, T>,
    D: DotDistance<T>,
{
    hull: &'a A,
    inner: Option<&'a ConvexSet<'a, A, E, D, T>>,
    provider_type: PhantomData<E>,
    distance_type: PhantomData<D>,
    embed_type: PhantomData<T>,
}

impl<'a, A, E, D, T> ConvexSet<'a, A, E, D, T>
where
    A: DotEmbeddingAccess<E, D, T>,
    E: DotEmbeddingProvider<D, T>,
    D: DotDistance<T>,
{
    pub fn get_closest(self: &Self, other: &T, count: usize) -> Vec<(usize, DotDistanceCmp)> {
        let mut res: Vec<(usize, DotDistanceCmp)> = Vec::with_capacity(2 * count - 1);

        let mut remain = count;
        let mut convex = self;
        while remain > 0 {
            res.extend(convex.hull.get_closest(other, remain));
            res.sort_unstable_by(|(_, a), (_, b)| a.cmp(b));
            res.truncate(count);
            convex = match convex.inner {
                Some(inner) => inner,
                None => break,
            };
            remain -= 1;
        }
        res
    }
}
