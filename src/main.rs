use clap::{arg, Parser};
use fann::cache::DistanceCache;
use fann::distances::vec::{VecProvider, VEC_DOT_DISTANCE};
use fann::info::{no_info, BaseInfo, Info};
use fann::kmed::{FannBuildParams, FannTree};
use std::marker::PhantomData;
use std::time::Instant;

use ndarray::{s, Array2, ArrayView1, ArrayView2};
use polars::io::prelude::*;
use polars::prelude::Float64Type;

use fann::distances::ndarray::{NdProvider, ND_DOT_DISTANCE};
use fann::{EmbeddingProvider, FannForest, Forest, NearestNeighbors};

fn load_embed(path: &str) -> Array2<f64> {
    let mut file = std::fs::File::open(path).unwrap();
    let df = ParquetReader::new(&mut file).finish().unwrap();
    df.to_ndarray::<Float64Type>().unwrap()
}

fn to_vec_vec(arr: ArrayView2<f64>) -> Vec<Vec<f64>> {
    arr.rows().into_iter().map(|row| row.to_vec()).collect()
}

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[arg(short, long, default_value_t = String::from("embeds.pq"))]
    file: String,
    #[arg(short, long, default_value_t = 10)]
    min_tree: usize,
    #[arg(short, long, default_value_t = 1000)]
    max_tree: usize,
    #[arg(short, long, default_value_t = 1000)]
    total: usize,
    #[arg(long, default_value_t = false)]
    force: bool,
    #[arg(short, long, default_value_t = false)]
    info: bool,
}

fn main0() {
    let args = Args::parse();
    let min_tree = args.min_tree;
    let max_tree = args.max_tree;
    let total_size = args.total;
    let force = args.force;
    let print_info = args.info;
    println!("size: {}", total_size);

    let t_load = Instant::now();
    let df = load_embed(args.file.as_str());
    println!("load took {:?}", t_load.elapsed());
    println!("{shape:?}", shape = df.shape());
    let mut info = BaseInfo::new(total_size);

    let provider = NdProvider::new(df.slice(s![0..total_size, ..]), ND_DOT_DISTANCE);
    let vv = to_vec_vec(df.slice(s![0..total_size, ..]));
    let vv_provider = VecProvider::new(&vv, VEC_DOT_DISTANCE);

    println!("{size:?}", size = provider.all());
    let main_provider = provider.subrange(0..total_size).unwrap();

    let t_build = Instant::now();
    let tfilename = format!("tree-{}.zip", total_size);
    let tfile = std::path::Path::new(&tfilename);
    let mut cache = DistanceCache::new(1000000);
    let params = FannBuildParams {
        max_node_size: None,
    };
    let forest: FannForest<_, FannTree, _, _, _> =
        FannForest::create(&main_provider, min_tree, max_tree);
    if tfile.exists() {
        let mut file = std::fs::File::open(tfile).unwrap();
        forest.load_all(&mut file, false, &params, &mut cache, &mut info, force);
    } else {
        forest.build_all(&params, &mut cache, &mut info);
    }
    println!("build took {:?}", t_build.elapsed());
    let (hits, miss) = info.cache_hits_miss();
    println!(
        "cache[rate: {:.2}% hits: {} miss: {} total: {}]",
        info.cache_hit_rate() * 100.0,
        hits,
        miss,
        hits + miss,
    );
    info.clear();

    let embed = df.row(total_size);

    let t_search = Instant::now();
    let closest = forest.get_closest(embed, 10, &mut info);
    println!("search took {:?}", t_search.elapsed());
    println!("{:?}", closest);

    println!("cache[total: {}]", info.dist_count());
    if print_info {
        println!(
            "{draw}",
            draw =
                forest
                    .get_trees()
                    .first()
                    .unwrap()
                    .draw(Some(&info), Some(closest), true, false)
        );
    }

    let t_base_search = Instant::now();
    let base_closest = provider.get_closest(embed, 10, &mut no_info());
    println!("baseline search took {:?}", t_base_search.elapsed());
    println!("{:?}", base_closest);

    let t_vv_base_search = Instant::now();
    let vv_base_closest = vv_provider.get_closest(&embed.to_vec(), 10, &mut no_info());
    println!("vv baseline search took {:?}", t_vv_base_search.elapsed());
    println!("{:?}", vv_base_closest);
}

// EXPERIMENTS

use ndarray::{arr1, arr2};

trait Op<'a, T: 'a> {
    fn op(&self, a: &T, b: &T) -> f64;
}

trait Viewer<'a, O, T: 'a>
where
    O: Op<'a, T> + Copy,
{
    fn all(&self) -> std::ops::Range<usize>;
    fn get(&'a self, index: usize) -> T;
    fn get_op(&self) -> O;
}

struct Foo<'a, O>
where
    O: Op<'a, ArrayView1<'a, f64>> + Copy,
{
    full: ArrayView2<'a, f64>,
    op: O,
}

impl<'a, O> Foo<'a, O>
where
    O: Op<'a, ArrayView1<'a, f64>> + Copy,
{
    fn new(full: ArrayView2<'a, f64>, op: O) -> Self {
        Self { full, op }
    }
}

impl<'a, O> Viewer<'a, O, ArrayView1<'a, f64>> for Foo<'a, O>
where
    O: Op<'a, ArrayView1<'a, f64>> + Copy,
{
    fn all(&self) -> std::ops::Range<usize> {
        0..self.full.shape()[0]
    }

    fn get(&'a self, index: usize) -> ArrayView1<'a, f64> {
        self.full.row(index)
    }

    fn get_op(&self) -> O {
        self.op
    }
}

#[derive(Clone, Copy)]
struct FooOp;

impl<'a> Op<'a, ArrayView1<'a, f64>> for FooOp {
    fn op(&self, a: &ArrayView1<'a, f64>, b: &ArrayView1<'a, f64>) -> f64 {
        let diff = a - b;
        (&diff * &diff).sum()
    }
}

struct Bar<'a, O>
where
    O: Op<'a, &'a Vec<f64>> + Copy,
{
    full: Vec<Vec<f64>>,
    op: O,
    lifetime: PhantomData<&'a Vec<f64>>,
}

impl<'a, O> Bar<'a, O>
where
    O: Op<'a, &'a Vec<f64>> + Copy,
{
    fn new(full: Vec<Vec<f64>>, op: O) -> Self {
        Self {
            full,
            op,
            lifetime: PhantomData,
        }
    }
}

impl<'a, O> Viewer<'a, O, &'a Vec<f64>> for Bar<'a, O>
where
    O: Op<'a, &'a Vec<f64>> + Copy,
{
    fn all(&self) -> std::ops::Range<usize> {
        0..self.full.len()
    }

    fn get(&'a self, index: usize) -> &'a Vec<f64> {
        &self.full[index]
    }

    fn get_op(&self) -> O {
        self.op
    }
}

#[derive(Clone, Copy)]
struct BarOp;

impl<'a> Op<'a, &'a Vec<f64>> for BarOp {
    fn op(&self, a: &&'a Vec<f64>, b: &&'a Vec<f64>) -> f64 {
        a.iter()
            .zip(b.iter())
            .map(|(va, vb)| (va - vb) * (va - vb))
            .sum()
    }
}

trait Main<'a, V, O, T: 'a>
where
    V: Viewer<'a, O, T>,
    O: Op<'a, T> + Copy,
{
    fn create(viewer: &'a V) -> Self;
    fn build(&'a mut self);
    fn algo(&self, arr: &T) -> usize;
}

struct MaxMain<'a, V, O, T: 'a>
where
    V: Viewer<'a, O, T>,
    O: Op<'a, T> + Copy,
{
    view: &'a V,
    pres: Option<Vec<f64>>,
    lifetime: PhantomData<&'a T>,
    op_type: PhantomData<O>,
}

impl<'a, V, O, T: 'a> Main<'a, V, O, T> for MaxMain<'a, V, O, T>
where
    V: Viewer<'a, O, T>,
    O: Op<'a, T> + Copy,
{
    fn create(view: &'a V) -> Self {
        Self {
            view,
            pres: None,
            lifetime: PhantomData,
            op_type: PhantomData,
        }
    }

    fn build(&mut self) {
        let op = self.view.get_op();
        let base = self.view.get(0);
        self.pres = Some(
            self.view
                .all()
                .map(|ix| {
                    let row = self.view.get(ix);
                    op.op(&row, &base)
                })
                .collect(),
        );
    }

    fn algo(&self, arr: &T) -> usize {
        let op = self.view.get_op();
        let (res, _) = self
            .pres
            .as_ref()
            .unwrap()
            .iter()
            .enumerate()
            .map(|(ix, base_op)| (ix, (op.op(&self.view.get(ix), arr)) * base_op))
            .max_by(|(_, a), (_, b)| a.total_cmp(b))
            .unwrap();
        res
    }
}

struct Partition<'a, M, V, O, T: 'a>
where
    M: Main<'a, V, O, T>,
    V: Viewer<'a, O, T>,
    O: Op<'a, T> + Copy,
{
    mains: Vec<M>,
    lifetime: PhantomData<&'a T>,
    view_type: PhantomData<V>,
    op_type: PhantomData<O>,
}

impl<'a, M, V, O, T: 'a> Partition<'a, M, V, O, T>
where
    M: Main<'a, V, O, T>,
    V: Viewer<'a, O, T>,
    O: Op<'a, T> + Copy,
{
    fn create(views: Vec<&'a V>) -> Self {
        Self {
            mains: views.iter().map(|view| M::create(view)).collect(),
            lifetime: PhantomData,
            view_type: PhantomData,
            op_type: PhantomData,
        }
    }

    fn all_mains(&'a mut self) -> &'a mut Vec<M> {
        &mut self.mains
    }

    fn build_all(&'a mut self) {
        self.all_mains().iter_mut().for_each(|main| main.build());
    }

    fn algo_all(&self, arr: &T) -> usize {
        self.mains.iter().map(|main| main.algo(arr)).sum()
    }
}

fn main() {
    {
        let arr = arr2(&[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]);
        let foo = Foo::new(arr.view(), FooOp);
        let bar = Bar::new(
            vec![
                vec![1.0, 2.0, 3.0],
                vec![4.0, 5.0, 6.0],
                vec![7.0, 8.0, 9.0],
            ],
            BarOp,
        );
        let mut mfoo = MaxMain::create(&foo);
        mfoo.build();
        let tfoo = arr1(&[1.0, 5.0, 9.0]);
        println!("{}", mfoo.algo(&tfoo.view()));

        let mut mbar = MaxMain::create(&bar);
        mbar.build();
        let tbar = vec![1.0, 5.0, 9.0];
        println!("{}", mbar.algo(&&tbar));
    }
    {
        let a1 = arr2(&[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]);
        let a2 = arr2(&[[9.0, 8.0, 7.0], [6.0, 5.0, 4.0], [3.0, 2.0, 1.0]]);
        let foo1 = Foo::new(a1.view(), FooOp);
        let foo2 = Foo::new(a2.view(), FooOp);
        let mut part: Partition<MaxMain<_, _, _>, _, _, _> = Partition::create(vec![&foo1, &foo2]);
        part.build_all();
        let tfoo = arr1(&[1.0, 5.0, 9.0]);
        println!("{}", part.algo_all(&tfoo.view()));
    }
    main0();
}
