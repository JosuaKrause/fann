use std::fmt;

use crate::{
    info::Info, kmed::Node, Cache, Distance, EmbeddingProvider, LocalDistance, NearestNeighbors,
};
use rayon::prelude::*;
use serde::{de::DeserializeOwned, Serialize};
use zip::{result::ZipError, write::FileOptions};

#[derive(Debug, Clone)]
pub struct MisconfiguredTreeError;

impl fmt::Display for MisconfiguredTreeError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "tree was created for different provider")
    }
}

#[derive(Debug)]
pub enum TreeLoadError {
    MisconfiguredTreeError(MisconfiguredTreeError),
    ZipError(ZipError),
    SerdeError(serde_json::Error),
}

impl From<MisconfiguredTreeError> for TreeLoadError {
    fn from(value: MisconfiguredTreeError) -> Self {
        TreeLoadError::MisconfiguredTreeError(value)
    }
}

impl From<ZipError> for TreeLoadError {
    fn from(value: ZipError) -> Self {
        TreeLoadError::ZipError(value)
    }
}

impl From<serde_json::Error> for TreeLoadError {
    fn from(value: serde_json::Error) -> Self {
        TreeLoadError::SerdeError(value)
    }
}

#[derive(Debug, Clone)]
pub struct TreeNotBuiltError;

impl fmt::Display for TreeNotBuiltError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "tree has not been built")
    }
}

#[derive(Debug)]
pub enum TreeWriteError {
    TreeNotBuiltError(TreeNotBuiltError),
    ZipError(ZipError),
    SerdeError(serde_json::Error),
}

impl From<TreeNotBuiltError> for TreeWriteError {
    fn from(value: TreeNotBuiltError) -> Self {
        TreeWriteError::TreeNotBuiltError(value)
    }
}

impl From<ZipError> for TreeWriteError {
    fn from(value: ZipError) -> Self {
        TreeWriteError::ZipError(value)
    }
}

impl From<serde_json::Error> for TreeWriteError {
    fn from(value: serde_json::Error) -> Self {
        TreeWriteError::SerdeError(value)
    }
}

pub trait BuildParams {}

pub trait Tree<P, E, D, T>
where
    P: BuildParams,
    E: EmbeddingProvider<D, T>,
    D: Distance<T>,
    Self: Serialize + DeserializeOwned,
{
    fn build<C, I>(provider: &E, params: &P, cache: &mut C, info: &mut I) -> Self
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

    fn get_closest<'a, I>(
        &self,
        count: usize,
        ldist: &LocalDistance<'a, E, D, T>,
        info: &mut I,
    ) -> Vec<(usize, f64)>
    where
        I: Info;

    fn fingerprint(&self) -> (&str, &str);

    fn load<R>(archive: &mut zip::ZipArchive<R>, name: &str) -> Result<Self, TreeLoadError>
    where
        R: std::io::Read + std::io::Seek,
    {
        let zip_file = archive.by_name(name)?;
        let res: Self = serde_json::from_reader(zip_file)?;
        Ok(res)
    }

    fn save<W>(&self, writer: &mut zip::ZipWriter<W>, name: &str) -> Result<(), TreeWriteError>
    where
        W: std::io::Write + std::io::Seek,
    {
        let options = FileOptions::default()
            .compression_method(zip::CompressionMethod::Bzip2)
            .unix_permissions(0o755);
        writer.start_file(name, options)?;
        serde_json::to_writer(writer, self)?;
        Ok(())
    }

    fn get_root(&self) -> &Node;
}

pub trait Buildable<P, N, E, D, T>
where
    P: BuildParams,
    N: Tree<P, E, D, T>,
    E: EmbeddingProvider<D, T>,
    D: Distance<T>,
    Self: NearestNeighbors<E, D, T>,
{
    fn build<C, I>(&mut self, params: &P, cache: &mut C, info: &mut I)
    where
        C: Cache,
        I: Info;

    fn is_ready(&self) -> bool;

    fn is_dirty(&self) -> bool;

    fn get_tree(&self) -> &Option<N>;

    fn provider(&self) -> &E;

    fn raw_set_tree(&mut self, tree: N, is_dirty: bool);

    fn set_tree(
        &mut self,
        tree: N,
        ignore_provider: bool,
        is_dirty: bool,
    ) -> Result<(), MisconfiguredTreeError> {
        if !ignore_provider {
            let (phash, dname) = tree.fingerprint();
            let provider = self.provider();
            if dname != provider.distance().name() {
                return Err(MisconfiguredTreeError);
            }
            if phash != &provider.compute_hash() {
                return Err(MisconfiguredTreeError);
            }
        }
        self.raw_set_tree(tree, is_dirty);
        Ok(())
    }

    fn clear_tree(&mut self);
}

pub trait Forest<P, N, E, D, T, B>
where
    P: BuildParams,
    N: Tree<P, E, D, T>,
    E: EmbeddingProvider<D, T> + NearestNeighbors<E, D, T>,
    D: Distance<T>,
    B: Buildable<P, N, E, D, T>,
    Self: Sized,
{
    fn create_from(root_provider: E, trees: Vec<B>, remain: E) -> Self;

    fn create_builder_from(provider: E) -> B;

    fn create(root_provider: E, min_tree_size: usize, max_tree_size: usize) -> Self {
        let all = root_provider.all();
        let is_large_end = all.len() % max_tree_size >= min_tree_size;
        let expected_trees = all.len() / max_tree_size + if is_large_end { 1 } else { 0 };
        let mut trees = Vec::with_capacity(expected_trees);
        let mut start = all.start;
        while start + min_tree_size <= all.end {
            let size = (all.end - start).min(max_tree_size);
            let cur_provider = root_provider.subrange(start..(start + size)).unwrap();
            trees.push(Self::create_builder_from(cur_provider));
            start += size;
        }
        let remain = root_provider.subrange(start..all.end).unwrap();
        Self::create_from(root_provider, trees, remain)
    }

    fn build_all<C, I>(&mut self, params: &P, cache: &mut C, info: &mut I)
    where
        C: Cache,
        I: Info,
    {
        self.get_trees_mut().iter_mut().for_each(|tree| {
            tree.build(params, cache, info);
        });
    }

    fn get_name(provider: &E) -> String {
        let range = provider.all();
        format!(
            "tree{start}-{size}.json",
            start = range.start,
            size = range.len()
        )
    }

    fn load_all<R, C, I>(
        &mut self,
        file: &mut R,
        ignore_provider: bool,
        params: &P,
        cache: &mut C,
        info: &mut I,
        force: bool,
    ) -> Result<(), TreeLoadError>
    where
        R: std::io::Read + std::io::Seek,
        C: Cache,
        I: Info,
    {
        let mut archive = zip::ZipArchive::new(file)?;
        self.get_trees_mut().iter_mut().fold(Ok(()), |res, tree| {
            if res.is_err() {
                return res;
            }
            let name = Self::get_name(tree.provider());
            let need_build = force || {
                if !archive.file_names().any(|fname| fname == name.as_str()) {
                    true
                } else {
                    let root = N::load(&mut archive, &name)?;
                    tree.set_tree(root, ignore_provider, false)?;
                    false
                }
            };
            if need_build {
                tree.build(params, cache, info);
            }
            res
        })
    }

    fn save_all<W>(&mut self, file: &mut W) -> Result<(), TreeWriteError>
    where
        W: std::io::Write + std::io::Seek,
    {
        let mut writer = zip::ZipWriter::new(file);
        self.get_trees_mut().iter_mut().fold(Ok(()), |res, tree| {
            if res.is_err() || !tree.is_dirty() {
                return res;
            }
            match tree.get_tree() {
                Some(root) => {
                    let name = Self::get_name(tree.provider());
                    root.save(&mut writer, &name)
                }
                None => Err(From::from(TreeNotBuiltError)),
            }
        })
    }

    fn get_trees(&self) -> &Vec<B>;

    fn get_trees_mut(&mut self) -> &mut Vec<B>;

    fn get_remain(&self) -> &E;

    fn get_root_provider<'a>(&'a self) -> &'a E;

    fn get_closest<I>(&self, other: &T, count: usize, info: &mut I) -> Vec<(usize, f64)>
    where
        I: Info,
    {
        let mut res: Vec<(usize, f64)> = self
            .get_trees()
            .iter()
            .map(|tree| tree.get_closest(other, count, info))
            .flatten()
            .collect();
        // res.extend(self.get_remain().get_closest(other, count, info));
        res.par_sort_unstable_by(|(_, a), (_, b)| a.total_cmp(b));
        res.truncate(count);
        res
    }
}
