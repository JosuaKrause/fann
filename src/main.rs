use clap::{arg, Parser};
use std::time::Instant;

use fann::cache::DistanceCache;
use ndarray::{s, Array2};
use polars::io::prelude::*;
use polars::prelude::Float64Type;

use fann::distances::ndarray::{NdProvider, ND_DOT_DISTANCE};
use fann::{EmbeddingProvider, Fann, NearestNeighbors};

fn load_embed(path: &str) -> Array2<f64> {
    let mut file = std::fs::File::open(path).unwrap();
    let df = ParquetReader::new(&mut file).finish().unwrap();
    df.to_ndarray::<Float64Type>().unwrap()
}

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[arg(short, long, default_value_t = String::from("embeds.pq"))]
    file: String,
    #[arg(short, long, default_value_t = 1000)]
    total: usize,
}

fn main() {
    let args = Args::parse();

    let t_load = Instant::now();
    let df = load_embed(args.file.as_str());
    println!("load took {:?}", t_load.elapsed());
    println!("{shape:?}", shape = df.shape());
    let total_size: usize = args.total;
    let provider = NdProvider::new(df.slice(s![0..total_size, ..]), &ND_DOT_DISTANCE);
    let mut cache = DistanceCache::new(10000);
    println!("{size:?}", size = provider.all());
    let mut fann = Fann::new(&provider, &mut cache);
    let t_build = Instant::now();
    fann.build(None);
    println!("build took {:?}", t_build.elapsed());
    println!("{draw}", draw = fann.draw(false, false));
    let embed_v = df.row(total_size);
    let embed = provider.as_embedding(embed_v);
    let t_search = Instant::now();
    let closest = fann.get_closest(embed, 10);
    println!("search took {:?}", t_search.elapsed());
    println!("{:?}", closest);
}
