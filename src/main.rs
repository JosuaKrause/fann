use std::time::Instant;

use fann::cache::DistanceCache;
use ndarray::{s, Array2};
use polars::io::prelude::*;
use polars::prelude::Float64Type;

use fann::distances::ndarray::DotDistance;
use fann::{Distance, Fann};

fn load_embed(path: &str) -> Array2<f64> {
    let mut file = std::fs::File::open(path).unwrap();
    let df = ParquetReader::new(&mut file).finish().unwrap();
    df.to_ndarray::<Float64Type>().unwrap()
}

fn main() {
    let t_load = Instant::now();
    let df = load_embed("embeds.pq");
    println!("load took {:?}", t_load.elapsed());
    println!("{shape:?}", shape = df.shape());
    let total_size: usize = 1000;
    let dist = DotDistance::new(df.slice(s![0..total_size, ..]));
    let mut cache = DistanceCache::new(10000);
    println!("{size:?}", size = dist.all());
    let mut fann = Fann::new(&dist, &mut cache);
    let t_build = Instant::now();
    fann.build(None);
    println!("build took {:?}", t_build.elapsed());
    println!("{draw}", draw = fann.draw(false, false));
    let embed_v = df.row(total_size);
    let embed = fann.as_embedding(embed_v);
    let t_search = Instant::now();
    let closest = fann.get_closest(&embed, 10);
    println!("search took {:?}", t_search.elapsed());
    println!("{:?}", closest);
}
