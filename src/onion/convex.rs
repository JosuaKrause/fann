use std::marker::PhantomData;

use crate::{DotDistance, DotDistanceCmp, DotEmbeddingProvider};

pub struct Point {
    index: usize,
    own_dist: Option<DotDistanceCmp>,
}

pub struct ConvexSet<'a, E, D, T>
where
    E: DotEmbeddingProvider<D, T>,
    D: DotDistance<T>,
{
    hull: Vec<&'a Point>,
    inner: Vec<&'a Point>,
    provider_type: PhantomData<E>,
    distance_type: PhantomData<D>,
    embed_type: PhantomData<T>,
}

impl<'a, E, D, T> ConvexSet<'a, E, D, T>
where
    E: DotEmbeddingProvider<D, T>,
    D: DotDistance<T>,
{
}
