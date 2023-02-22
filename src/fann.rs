use std::marker::PhantomData;

use crate::{
    info::Info, BuildParams, Buildable, Cache, Distance, Embedding, EmbeddingProvider, Forest,
    LocalDistance, NearestNeighbors, Tree,
};

pub mod kmed;

pub struct Fann<'a, P, N, E, D, T>
where
    P: BuildParams,
    N: Tree<'a, P, E, D, T>,
    E: EmbeddingProvider<'a, D, T>,
    D: Distance<T>,
    T: 'a,
{
    provider: E,
    root: Option<N>,
    is_dirty: bool,
    param_type: PhantomData<P>,
    provider_type: PhantomData<&'a E>,
    distance_type: PhantomData<D>,
    embed_type: PhantomData<T>,
}

impl<'a, P, N, E, D, T> Fann<'a, P, N, E, D, T>
where
    P: BuildParams,
    N: Tree<'a, P, E, D, T>,
    E: EmbeddingProvider<'a, D, T>,
    D: Distance<T>,
    T: 'a,
{
    pub fn new(provider: E) -> Fann<'a, P, N, E, D, T> {
        Fann {
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

impl<'a, P, N, E, D, T> NearestNeighbors<'a, T> for Fann<'a, P, N, E, D, T>
where
    P: BuildParams,
    N: Tree<'a, P, E, D, T>,
    E: EmbeddingProvider<'a, D, T>,
    D: Distance<T>,
    T: 'a,
{
    fn get_closest<I>(
        &'a self,
        other: &'a Embedding<T>,
        count: usize,
        info: &mut I,
    ) -> Vec<(usize, f64)>
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

impl<'a, P, N, E, D, T> Buildable<'a, P, N, E, D, T> for Fann<'a, P, N, E, D, T>
where
    P: BuildParams,
    N: Tree<'a, P, E, D, T>,
    E: EmbeddingProvider<'a, D, T>,
    D: Distance<T>,
    T: 'a,
{
    fn build<C, I>(&'a mut self, params: &P, cache: &mut C, info: &mut I)
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

pub struct FannForest<'a, P, N, E, D, T>
where
    P: BuildParams,
    N: Tree<'a, P, E, D, T>,
    E: EmbeddingProvider<'a, D, T> + NearestNeighbors<'a, T>,
    D: Distance<T>,
    T: 'a,
{
    trees: Vec<Fann<'a, P, N, E, D, T>>,
    remain: E,
    param_type: PhantomData<P>,
}

impl<'a, P, N, E, D, T> Forest<'a, P, N, E, D, T, Fann<'a, P, N, E, D, T>>
    for FannForest<'a, P, N, E, D, T>
where
    P: BuildParams,
    N: Tree<'a, P, E, D, T>,
    E: EmbeddingProvider<'a, D, T> + NearestNeighbors<'a, T>,
    D: Distance<T>,
    T: 'a,
    Self: 'a,
{
    fn create_from(trees: Vec<Fann<'a, P, N, E, D, T>>, remain: E) -> Self {
        Self {
            trees,
            remain,
            param_type: PhantomData,
        }
    }

    fn create_builder_from(provider: E) -> Fann<'a, P, N, E, D, T> {
        Fann::new(provider)
    }

    fn get_trees(&'a self) -> &'a Vec<Fann<'a, P, N, E, D, T>> {
        &self.trees
    }

    fn get_trees_mut(&'a mut self) -> &'a mut Vec<Fann<'a, P, N, E, D, T>> {
        &mut self.trees
    }

    fn get_remain(&self) -> &E {
        &self.remain
    }
}
