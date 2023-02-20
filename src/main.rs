use clap::{arg, Parser};
use fann::distances::vec::{VecProvider, VEC_DOT_DISTANCE};
use fann::info::{no_info, BaseInfo, Info};
use fann::kmed::FannTree;
use std::time::Instant;

use fann::cache::DistanceCache;
use ndarray::{s, Array2, ArrayView2};
use polars::io::prelude::*;
use polars::prelude::Float64Type;

use fann::distances::ndarray::{NdProvider, ND_DOT_DISTANCE};
use fann::{Embedding, EmbeddingProvider, Fann, NearestNeighbors};

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
    #[arg(short, long, default_value_t = 1000)]
    total: usize,
    #[arg(short, long)]
    precluster: Option<usize>,
    #[arg(long, default_value_t = false)]
    force: bool,
    #[arg(short, long, default_value_t = false)]
    info: bool,
}

fn main() {
    let args = Args::parse();
    let total_size = args.total;
    let pre_cluster = args.precluster;
    let force = args.force;
    let print_info = args.info;
    println!("size: {} pre_cluster: {:?}", total_size, pre_cluster);

    let t_load = Instant::now();
    let df = load_embed(args.file.as_str());
    println!("load took {:?}", t_load.elapsed());
    println!("{shape:?}", shape = df.shape());
    let mut info = BaseInfo::new(total_size);

    let provider = NdProvider::new(df.slice(s![0..total_size, ..]), ND_DOT_DISTANCE);
    let vv = to_vec_vec(df.slice(s![0..total_size, ..]));
    let vv_provider = VecProvider::new(&vv, VEC_DOT_DISTANCE);

    println!("{size:?}", size = provider.all());

    let mut fann = Fann::new(&provider);
    let t_build = Instant::now();
    let tfilename = format!("tree-{}.zip", total_size);
    let tfile = std::path::Path::new(&tfilename);
    if !force && tfile.exists() {
        fann.set_tree(
            FannTree::load(&std::fs::File::open(tfile).unwrap()).unwrap(),
            false,
        )
        .unwrap();
        println!("load took {:?}", t_build.elapsed());
    } else {
        let mut cache = DistanceCache::new(100000);
        fann.build(None, pre_cluster, &mut cache, &mut info);
        fann.get_tree()
            .as_ref()
            .unwrap()
            .save(&std::fs::File::create(tfile).unwrap())
            .unwrap();
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
    }

    let embed_v = df.row(total_size);
    let embed = Embedding::as_embedding(embed_v);

    let t_search = Instant::now();
    let closest = fann.get_closest(&embed, 10, &mut info);
    println!("search took {:?}", t_search.elapsed());
    println!("{:?}", closest);

    println!("cache[total: {}]", info.dist_count());
    if print_info {
        println!(
            "{draw}",
            draw = fann.draw(Some(&info), Some(closest), true, false)
        );
    }

    let t_base_search = Instant::now();
    let base_closest = provider.get_closest(&embed, 10, &mut no_info());
    println!("baseline search took {:?}", t_base_search.elapsed());
    println!("{:?}", base_closest);

    let vv_embed_v = embed_v.to_vec();
    let vv_embed = Embedding::as_embedding(&vv_embed_v);

    let t_vv_base_search = Instant::now();
    let vv_base_closest = vv_provider.get_closest(&vv_embed, 10, &mut no_info());
    println!("vv baseline search took {:?}", t_vv_base_search.elapsed());
    println!("{:?}", vv_base_closest);
}
