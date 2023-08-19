use std::{fmt::Debug, time::Instant};

use ndarray::{s, Array2};
use polars::prelude::{Float64Type, ParquetReader, SerReader};

use crate::{
    algo::StreamingNeighbors,
    cache::DistanceCache,
    distances::ndarray::{NdProvider, ND_DOT_DISTANCE},
    info::{no_info, BaseInfo, Info},
    kmed::{FannBuildParams, FannTree},
    EmbeddingProvider, FannForest, Forest, NearestNeighbors,
};

fn load_embed(path: &str) -> Array2<f64> {
    let mut file = std::fs::File::open(path).unwrap();
    let df = ParquetReader::new(&mut file).finish().unwrap();
    df.to_ndarray::<Float64Type>().unwrap()
}

fn assert_eq_fst<A, B>(a: &Vec<(A, B)>, b: &Vec<(A, B)>)
where
    A: Copy + Eq + Debug,
{
    let ax: Vec<A> = a.iter().map(|(v, _)| *v).collect();
    let bx: Vec<A> = b.iter().map(|(v, _)| *v).collect();
    assert_eq!(ax, bx);
}

fn eq_fst<A, B>(a: &Vec<(A, B)>, b: &Vec<(A, B)>) -> bool
where
    A: Eq,
{
    if a.len() != b.len() {
        return false;
    }
    a.iter().zip(b.iter()).all(|((va, _), (vb, _))| *va == *vb)
}

#[test]
fn fann_ndarray() {
    let df = load_embed("data/test.pq");
    let all = df.slice(s![0..1000usize, ..]);
    let provider = NdProvider::new(&all, ND_DOT_DISTANCE);
    let main_provider = provider.subrange(0..1000).unwrap();
    let mut forest: FannForest<_, FannTree, _, _, _> = FannForest::create(main_provider, 100, 100);
    let params = FannBuildParams {
        max_node_size: None,
    };
    let mut cache = DistanceCache::new(1000000);
    forest.build_all(&params, &mut cache, &mut no_info());
    let mut info = BaseInfo::new();

    // for count in [1, 2, 5, 10, 30] {
    for count in [10] {
        for ix in 0..df.shape()[0] {
            let embed = df.row(ix);

            println!("row: {} count: {}", ix, count);
            let base_timing = Instant::now();
            let base_closest = provider.get_closest(&embed, count, &mut no_info());
            println!("base {:?}", base_timing.elapsed());

            // let closest_timing = Instant::now();
            // let closest = forest.get_closest(&embed, count, &mut info);
            // println!("closest (w/ info) {:?}", closest_timing.elapsed());
            // if !eq_fst(&base_closest, &closest) {
            //     println!(
            //         "{draw}",
            //         draw = forest.get_trees().first().unwrap().draw(
            //             Some(&info),
            //             Some(closest.clone()),
            //             true,
            //             false
            //         )
            //     );
            // }
            // assert_eq_fst(&base_closest, &closest);
            // info.clear();

            // let closest_ni_timing = Instant::now();
            // let closest_ni = forest.get_closest(&embed, count, &mut no_info());
            // println!("closest {:?}", closest_ni_timing.elapsed());
            // assert_eq_fst(&base_closest, &closest_ni);

            let stream_timing = Instant::now();
            let closest_stream = forest.get_closest_stream(&embed, count, &mut info);
            println!("stream (w/ info) {:?}", stream_timing.elapsed());
            if !eq_fst(&base_closest, &closest_stream) {
                println!(
                    "{draw}",
                    draw = forest.get_trees().first().unwrap().draw(
                        Some(&info),
                        Some(closest_stream.clone()),
                        true,
                        true
                    )
                );
                println!(
                    "baseline: {:?}\nstream:   {:?}",
                    base_closest, closest_stream
                );
            }
            assert_eq_fst(&base_closest, &closest_stream);
            info.clear();

            let stream_ni_timing = Instant::now();
            let closest_stream_ni = forest.get_closest_stream(&embed, count, &mut no_info());
            println!("stream {:?}", stream_ni_timing.elapsed());
            assert_eq_fst(&base_closest, &closest_stream_ni);

            println!("=====");
            println!("");
        }
    }
}
