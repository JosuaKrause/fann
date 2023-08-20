use std::marker::PhantomData;

use crate::{info::Info, DotCache, DotDistance, DotDistanceCmp, DotEmbeddingProvider};

pub trait DotEmbeddingAccess<E, D, T>
where
    E: DotEmbeddingProvider<D, T>,
    D: DotDistance<T>,
{
    fn add(self: &mut Self, provider: &E, index: usize);

    fn get_closest<C, I>(
        self: &Self,
        provider: &E,
        other: &T,
        count: usize,
        cache: &mut C,
        info: &mut I,
    ) -> Vec<(usize, DotDistanceCmp)>
    where
        C: DotCache,
        I: Info;
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
    pub fn on_hull<C, I>(
        self: &Self,
        provider: &E,
        other: usize,
        cache: &mut C,
        info: &mut I,
    ) -> bool
    where
        C: DotCache,
        I: Info,
    {
        let own = provider.dist_internal(other, other, cache, info);
        provider.with_embed(other, |embed| {
            let res = self.get_closest(provider, embed, 1, Some((other, own)), cache, info);
            match res.get(0) {
                Some(&(index, _)) => index == other,
                None => true,
            }
        })
    }

    pub fn get_closest<C, I>(
        self: &Self,
        provider: &E,
        other: &T,
        count: usize,
        include_self: Option<(usize, DotDistanceCmp)>,
        cache: &mut C,
        info: &mut I,
    ) -> Vec<(usize, DotDistanceCmp)>
    where
        C: DotCache,
        I: Info,
    {
        let mut res: Vec<(usize, DotDistanceCmp)> =
            Vec::with_capacity(2 * count - (if include_self.is_some() { 0 } else { 1 }));
        include_self.map(|elem| res.push(elem));
        let mut remain = count;
        let mut convex = self;
        while remain > 0 {
            res.extend(
                convex
                    .hull
                    .get_closest(provider, other, remain, cache, info),
            );
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
