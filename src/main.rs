use clap::{arg, Parser};
use fann::cache::DistanceCache;
use fann::distances::vec::{VecProvider, VEC_DOT_DISTANCE};
use fann::info::{no_info, BaseInfo, Info};
use fann::kmed::{FannBuildParams, FannTree};
use std::time::Instant;

use ndarray::{s, Array2, ArrayView2};
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
    #[arg(long, default_value_t = 10)]
    min_tree: usize,
    #[arg(long, default_value_t = 1000)]
    max_tree: usize,
    #[arg(short, long, default_value_t = 1000)]
    total: usize,
    #[arg(long, default_value_t = false)]
    force: bool,
    #[arg(short, long, default_value_t = false)]
    info: bool,
}

fn main() {
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
    let mut info = BaseInfo::new();

    let all = df.slice(s![0..total_size, ..]);
    let provider = NdProvider::new(&all, ND_DOT_DISTANCE);
    let vv = to_vec_vec(all);
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
    let mut forest: FannForest<_, FannTree, _, _, _> =
        FannForest::create(&main_provider, min_tree, max_tree);
    if tfile.exists() {
        let mut file = std::fs::File::open(tfile).unwrap();
        forest
            .load_all(&mut file, false, &params, &mut cache, &mut info, force)
            .unwrap();
    } else {
        forest.build_all(&params, &mut cache, &mut info);
        let mut file = std::fs::File::create(tfile).unwrap();
        forest.save_all(&mut file).unwrap();
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
    let closest = forest.get_closest(&embed, 10, &mut info);
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
    let base_closest = provider.get_closest(&embed, 10, &mut no_info());
    println!("baseline search took {:?}", t_base_search.elapsed());
    println!("{:?}", base_closest);

    let t_vv_base_search = Instant::now();
    let vv_base_closest = vv_provider.get_closest(&embed.to_vec(), 10, &mut no_info());
    println!("vv baseline search took {:?}", t_vv_base_search.elapsed());
    println!("{:?}", vv_base_closest);
}
