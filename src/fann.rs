use std::marker::PhantomData;

use crate::{
    info::Info, Cache, Distance, Embedding, EmbeddingProvider, LocalCache, LocalCacheFactory,
    NearestNeighbors,
};

pub mod kmed;

pub trait Tree<'a, E, D, T>
where
    E: EmbeddingProvider<'a, D, T>,
    D: Distance<T> + Copy,
    T: 'a,
{
    fn build<C, I>(
        provider: &'a E,
        max_node_size: Option<usize>,
        pre_cluster: Option<usize>,
        cache: &mut C,
        info: &mut I,
    ) -> Self
    where
        C: Cache,
        I: Info;

    fn draw<I>(
        &self,
        high_ix: usize,
        info: Option<&I>,
        res: Option<Vec<(usize, f64)>>,
        prune: bool,
        radius: bool,
    ) -> String
    where
        I: Info;

    fn get_closest<L, I>(
        &self,
        count: usize,
        provider: &'a E,
        cache: &mut L,
        info: &mut I,
    ) -> Vec<(usize, f64)>
    where
        I: Info,
        L: LocalCache<'a, D, T>;
}

pub struct Fann<'a, E, D, N, T>
where
    E: EmbeddingProvider<'a, D, T>,
    D: Distance<T> + Copy,
    N: Tree<'a, E, D, T>,
    T: 'a,
{
    provider: &'a E,
    root: Option<N>,
    distance_type: PhantomData<D>,
    embed_type: PhantomData<T>,
}

impl<'a, E, D, N, T> Fann<'a, E, D, N, T>
where
    E: EmbeddingProvider<'a, D, T>,
    D: Distance<T> + Copy,
    N: Tree<'a, E, D, T>,
    T: 'a,
{
    pub fn new(provider: &'a E) -> Fann<'a, E, D, N, T> {
        Fann {
            provider,
            root: None,
            distance_type: PhantomData,
            embed_type: PhantomData,
        }
    }

    pub fn get_tree(&self) -> &Option<N> {
        &self.root
    }

    pub fn set_tree(&mut self, tree: N) {
        self.root = Some(tree);
    }

    pub fn clear_tree(&mut self) {
        self.root = None;
    }

    pub fn build<C, I>(
        &mut self,
        max_node_size: Option<usize>,
        pre_cluster: Option<usize>,
        cache: &mut C,
        info: &mut I,
    ) where
        C: Cache,
        I: Info,
    {
        self.root = Some(N::build(
            self.provider,
            max_node_size,
            pre_cluster,
            cache,
            info,
        ));
    }
}

impl<'a, F, E, D, L, N, T> NearestNeighbors<'a, F, D, L, T> for Fann<'a, E, D, N, T>
where
    F: LocalCacheFactory<'a, D, L, T>,
    E: EmbeddingProvider<'a, D, T>,
    D: Distance<T> + Copy,
    L: LocalCache<'a, D, T>,
    N: Tree<'a, E, D, T>,
    T: 'a,
{
    fn get_closest<I>(
        &self,
        other: &'a Embedding<T>,
        count: usize,
        cache_factory: &F,
        info: &mut I,
    ) -> Vec<(usize, f64)>
    where
        I: Info,
    {
        let mut cache = cache_factory.create(other);
        self.get_tree()
            .as_ref()
            .unwrap()
            .get_closest(count, self.provider, &mut cache, info)
    }
}
