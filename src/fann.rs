use std::marker::PhantomData;

use crate::{
    info::Info, BuildParams, Buildable, Cache, Distance, EmbeddingProvider, Forest, LocalDistance,
    NearestNeighbors, Tree,
};

pub mod kmed;

pub struct Fann<P, N, E, D, T>
where
    P: BuildParams,
    N: Tree<P, E, D, T>,
    E: EmbeddingProvider<D, T>,
    D: Distance<T>,
{
    provider: E,
    root: Option<N>,
    is_dirty: bool,
    param_type: PhantomData<P>,
    provider_type: PhantomData<E>,
    distance_type: PhantomData<D>,
    embed_type: PhantomData<T>,
}

impl<P, N, E, D, T> Fann<P, N, E, D, T>
where
    P: BuildParams,
    N: Tree<P, E, D, T>,
    E: EmbeddingProvider<D, T>,
    D: Distance<T>,
{
    pub fn new(provider: E) -> Self {
        Self {
            provider,
            root: None,
            is_dirty: false,
            param_type: PhantomData,
            provider_type: PhantomData,
            distance_type: PhantomData,
            embed_type: PhantomData,
        }
    }

    pub fn draw<I>(
        &self,
        info: Option<&I>,
        res: Option<Vec<(usize, f64)>>,
        prune: bool,
        radius: bool,
    ) -> String
    where
        I: Info,
    {
        self.root.as_ref().unwrap().draw(
            self.provider.all().last().unwrap(),
            info,
            res,
            prune,
            radius,
        )
    }
}

impl<P, N, E, D, T> NearestNeighbors<E, D, T> for Fann<P, N, E, D, T>
where
    P: BuildParams,
    N: Tree<P, E, D, T>,
    E: EmbeddingProvider<D, T>,
    D: Distance<T>,
{
    fn get_closest<I>(&self, other: &T, count: usize, info: &mut I) -> Vec<(usize, f64)>
    where
        I: Info,
    {
        let ldist = LocalDistance::new(&self.provider, other);
        self.get_tree()
            .as_ref()
            .unwrap()
            .get_closest(count, &ldist, info)
    }
}

impl<P, N, E, D, T> Buildable<P, N, E, D, T> for Fann<P, N, E, D, T>
where
    P: BuildParams,
    N: Tree<P, E, D, T>,
    E: EmbeddingProvider<D, T>,
    D: Distance<T>,
{
    fn build<C, I>(&mut self, params: &P, cache: &mut C, info: &mut I)
    where
        C: Cache,
        I: Info,
    {
        self.root = Some(N::build(&self.provider, params, cache, info));
        self.is_dirty = true;
    }

    fn is_ready(&self) -> bool {
        self.root.is_some()
    }

    fn is_dirty(&self) -> bool {
        self.is_dirty
    }

    fn provider(&self) -> &E {
        &self.provider
    }

    fn get_tree(&self) -> &Option<N> {
        &self.root
    }

    fn raw_set_tree(&mut self, tree: N, is_dirty: bool) {
        self.root = Some(tree);
        self.is_dirty = is_dirty;
    }

    fn clear_tree(&mut self) {
        self.root = None;
    }
}

pub struct FannForest<P, N, E, D, T>
where
    P: BuildParams,
    N: Tree<P, E, D, T>,
    E: EmbeddingProvider<D, T> + NearestNeighbors<E, D, T>,
    D: Distance<T>,
{
    trees: Vec<Fann<P, N, E, D, T>>,
    remain: E,
    param_type: PhantomData<P>,
}

impl<P, N, E, D, T> Forest<P, N, E, D, T, Fann<P, N, E, D, T>> for FannForest<P, N, E, D, T>
where
    P: BuildParams,
    N: Tree<P, E, D, T>,
    E: EmbeddingProvider<D, T> + NearestNeighbors<E, D, T>,
    D: Distance<T>,
{
    fn create_from(trees: Vec<Fann<P, N, E, D, T>>, remain: E) -> Self {
        Self {
            trees,
            remain,
            param_type: PhantomData,
        }
    }

    fn create_builder_from(provider: E) -> Fann<P, N, E, D, T> {
        Fann::new(provider)
    }

    fn get_trees(&self) -> &Vec<Fann<P, N, E, D, T>> {
        &self.trees
    }

    fn get_trees_mut(&mut self) -> &mut Vec<Fann<P, N, E, D, T>> {
        &mut self.trees
    }

    fn get_remain(&self) -> &E {
        &self.remain
    }
}
